"""
Deep Learning Model Calibration Evaluation for HPC.
Evaluates GRU, Text-only, and Fusion models on calibration metrics.

Usage:
    python code/evaluation/run_dl_calibration_hpc.py

Outputs:
    - results/calibration/calibration_summary.csv (appended with DL models)
    - results/calibration/calibration_dl_summary.json
    - results/calibration/reliability_diagrams/reliability_*.png
"""

import sys
import os
from pathlib import Path

# Setup paths
CODE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / 'evaluation'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from calibration_metrics import (
    evaluate_calibration,
    plot_reliability_diagram,
    plot_multi_model_reliability
)
from config import (
    TIMESERIES_FILE, NOTE_TIME_FILE, LLM_FEATURES_FILE, COHORT_FILE,
    RESULTS_DIR, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE,
    TEST_SIZE, RANDOM_STATE, LLM_COLS, PROCESSED_DIR
)

# Output paths
OUTPUT_DIR = RESULTS_DIR / 'calibration'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'reliability_diagrams').mkdir(exist_ok=True)

N_BINS = 10

# ================================================================
# Model definitions (mirror train_temporal_gru_v2.py)
# ================================================================

class ClinicalGRU(nn.Module):
    """Clinical GRU model - must match training architecture."""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class MIMICDataset(Dataset):
    """MIMIC dataset wrapper."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


# ================================================================
# Data loading functions
# ================================================================

def load_cohort() -> pd.DataFrame:
    """Load cohort data with label mapping."""
    df = pd.read_csv(COHORT_FILE)
    if 'label_mortality' in df.columns and 'mortality' not in df.columns:
        df['mortality'] = df['label_mortality']
    # Prolonged LOS label: keep consistent with the benchmark definition (>7 days).
    if 'prolonged_los' not in df.columns:
        if 'prolonged_los_7d' in df.columns:
            df['prolonged_los'] = df['prolonged_los_7d']
        elif 'prolonged_los_3d' in df.columns:
            df['prolonged_los'] = df['prolonged_los_3d']
    if 'has_sepsis_final' in df.columns:
        df['has_sepsis'] = df['has_sepsis_final']
    if 'has_aki_final' in df.columns:
        df['has_aki'] = df['has_aki_final']
    return df


def get_cohort_mask(cohort_df: pd.DataFrame, cohort_name: str) -> np.ndarray:
    """Get cohort filter mask."""
    if cohort_name == 'all':
        return np.ones(len(cohort_df), dtype=bool)
    elif cohort_name == 'sepsis':
        return cohort_df['has_sepsis'].values == 1
    elif cohort_name == 'aki':
        return cohort_df['has_aki'].values == 1
    return np.ones(len(cohort_df), dtype=bool)


def load_gru_data_and_model():
    """
    Load GRU model data and the best checkpoint.
    Returns: (X_values, X_mask, y, subjects, input_dim, model_checkpoint)
    """
    print("\n[GRU] Loading data pipeline...")

    # Load cohort
    df_cohort = pd.read_csv(COHORT_FILE)
    df_cohort['stay_id'] = df_cohort['stay_id'].astype(int)
    df_keys = df_cohort[['stay_id', 'subject_id']].copy()
    df_keys['label'] = df_cohort['label_mortality']
    df_clean = df_keys.dropna(subset=['stay_id', 'subject_id', 'label']).reset_index(drop=True)
    valid_stay_ids = df_clean['stay_id'].unique()

    # Load timeseries
    ts_df = pd.read_csv(TIMESERIES_FILE)
    ts_df['stay_id'] = pd.to_numeric(ts_df['stay_id'], errors='coerce').fillna(-1).astype(int)
    ts_df = ts_df[ts_df['stay_id'].isin(valid_stay_ids)]

    # Load LLM features
    note_time_df = pd.read_csv(NOTE_TIME_FILE)
    note_time_df['hour_offset'] = pd.to_numeric(note_time_df['hour_offset'], errors='coerce')
    note_time_df = note_time_df[(note_time_df['hour_offset'] >= 0) & (note_time_df['hour_offset'] < 24)]
    llm_df = pd.read_csv(LLM_FEATURES_FILE)
    llm_df['stay_id'] = pd.to_numeric(llm_df['stay_id'], errors='coerce').fillna(-1).astype(int)
    note_merged = note_time_df.merge(llm_df, on='stay_id', how='inner')

    # Build tensor
    feature_cols = [c for c in ts_df.columns
                   if c not in ['stay_id', 'hour', 'subject_id', 'hadm_id', 'intime']]

    N = len(df_clean)
    T = 24
    D_physio = len(feature_cols)
    D_llm = len(LLM_COLS)
    D = D_physio + D_llm

    print(f"[GRU] N={N}, T={T}, D_physio={D_physio}, D_llm={D_llm}, D={D}")

    id_map = {sid: i for i, sid in enumerate(df_clean['stay_id'])}
    mux = pd.MultiIndex.from_product([df_clean['stay_id'], range(T)], names=['stay_id', 'hour'])
    ts_df = ts_df.set_index(['stay_id', 'hour'])
    ts_df = ts_df[~ts_df.index.duplicated(keep='first')]
    ts_df = ts_df.reindex(mux)

    X_tensor = np.full((N, T, D), np.nan)
    X_tensor[:, :, :D_physio] = ts_df[feature_cols].values.reshape(N, T, D_physio)

    # Inject LLM features
    for row in note_merged.itertuples():
        if row.stay_id in id_map:
            idx = id_map[row.stay_id]
            h = int(row.hour_offset)
            feats = [getattr(row, c, 0) for c in LLM_COLS]
            if 0 <= h < T:
                X_tensor[idx, h:, D_physio:] = feats

    # Forward fill + mask
    obs_mask = (~np.isnan(X_tensor)).astype(np.float32)
    nan_mask = np.isnan(X_tensor)
    idx_ffill = np.where(~nan_mask, np.arange(nan_mask.shape[1])[None, :, None], 0)
    np.maximum.accumulate(idx_ffill, axis=1, out=idx_ffill)
    X_tensor = X_tensor[np.arange(N)[:, None, None],
                       idx_ffill,
                       np.arange(D)[None, None, :]]
    X_tensor = np.nan_to_num(X_tensor, nan=0.0)

    # Find best model checkpoint
    model_dir = RESULTS_DIR / 'Output_temporal_gru' / 'models'
    checkpoints = sorted(model_dir.glob('best_model_fold*.pt'))

    if not checkpoints:
        print("[GRU] No model checkpoints found!")
        return None

    # Load the best checkpoint (by val_auroc)
    best_ckpt = None
    best_auroc = -1
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            if ckpt.get('val_auroc', 0) > best_auroc:
                best_auroc = ckpt['val_auroc']
                best_ckpt = ckpt
                best_ckpt['_path'] = str(ckpt_path)
        except Exception as e:
            print(f"  Warning: Could not load {ckpt_path}: {e}")

    if best_ckpt is None:
        print("[GRU] No valid checkpoint found!")
        return None

    print(f"[GRU] Best checkpoint: {best_ckpt['_path']} (val_auroc={best_auroc:.4f})")

    return {
        'X_values': X_tensor,
        'X_mask': obs_mask,
        'y': df_clean['label'].values,
        'subjects': df_clean['subject_id'].values,
        'stay_ids': df_clean['stay_id'].values,
        'D': D,
        'checkpoint': best_ckpt
    }


def load_text_features(cohort_df: pd.DataFrame):
    """Load text-only features from episode JSON files."""
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    episode_dirs = [
        PROJECT_ROOT / 'episodes' / 'episodes_enhanced',
        PROJECT_ROOT / 'final_release' / 'episodes_enhanced',
    ]

    episode_dir = None
    for d in episode_dirs:
        if d.exists():
            episode_dir = d
            break

    if episode_dir is None:
        print("[Text] No episode directory found. Trying to load from saved features...")
        # Try loading pre-computed text features
        text_features_file = RESULTS_DIR / 'text_only_baselines' / 'text_features.csv'
        if text_features_file.exists():
            return pd.read_csv(text_features_file)
        return None

    print(f"[Text] Loading episodes from {episode_dir}...")

    valid_stay_ids = set(pd.to_numeric(cohort_df['stay_id'], errors='coerce').fillna(-1).astype(int).tolist())
    features_list = []
    episode_files = list(episode_dir.glob('*.json'))
    print(f"[Text] Found {len(episode_files)} episode files")

    for ep_file in episode_files:
        try:
            with open(ep_file) as f:
                episode = json.load(f)

            stay_id = episode.get('stay_id')
            if stay_id is None:
                continue
            stay_id = int(stay_id)
            if stay_id not in valid_stay_ids:
                continue

            # Extract text features (match episode schema + train_text_only/train_fusion).
            clinical = episode.get('clinical_text', {}) or {}
            notes = clinical.get('notes', []) or []
            n_notes = int(clinical.get('n_notes', len(notes)) or 0)

            def _note_text(n: dict) -> str:
                # Episodes may store multiple text fields; prefer full text if present.
                return n.get('text_full') or n.get('text_relevant') or n.get('text') or ''

            total_text_length = int(sum(len(_note_text(n)) for n in notes))
            avg_text_length = float(total_text_length) / float(max(n_notes, 1))

            # Annotation-derived reasoning features
            reasoning = episode.get('reasoning', {}) or {}
            detected_patterns = reasoning.get('detected_patterns', []) or []
            n_patterns = int(reasoning.get('n_patterns_detected', len(detected_patterns)) or len(detected_patterns))
            n_alignments = int(reasoning.get('n_alignments', 0) or 0)
            supportive = int(reasoning.get('n_supportive', 0) or 0)
            contradictory = int(reasoning.get('n_contradictory', 0) or 0)

            total_annot = supportive + contradictory
            if total_annot > 0:
                supportive_ratio = supportive / total_annot
                contradictory_ratio = contradictory / total_annot
            else:
                supportive_ratio = 0.5
                contradictory_ratio = 0.5

            if n_alignments > 0:
                annotation_density = total_annot / n_alignments
            else:
                annotation_density = 0.0

            features_list.append({
                'stay_id': stay_id,
                'n_notes': n_notes,
                'total_text_length': total_text_length,
                'avg_text_length': avg_text_length,
                'n_patterns': n_patterns,
                'n_alignments': n_alignments,
                'n_supportive': supportive,
                'n_contradictory': contradictory,
                'supportive_ratio': supportive_ratio,
                'contradictory_ratio': contradictory_ratio,
                'annotation_density': annotation_density,
            })
        except Exception:
            continue

    if not features_list:
        return None

    return pd.DataFrame(features_list)


def load_aggregated_features(window: str) -> pd.DataFrame:
    """Load aggregated features for a specific window."""
    features_file = PROCESSED_DIR / 'data_windows' / f'window_{window}' / 'features_aggregated.csv'
    if not features_file.exists():
        return None
    return pd.read_csv(features_file)


# ================================================================
# GRU Calibration Evaluation
# ================================================================

def evaluate_gru_calibration(gru_data: dict) -> list:
    """Run GRU model inference and compute calibration metrics."""
    results = []

    X_values = gru_data['X_values']
    X_mask = gru_data['X_mask']
    y = gru_data['y']
    subjects = gru_data['subjects']
    stay_ids = gru_data['stay_ids']
    D = gru_data['D']
    checkpoint = gru_data['checkpoint']

    # Split into train/test (same split as training)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X_values, y, groups=subjects))

    X_test = X_values[test_idx]
    X_test_mask = X_mask[test_idx]
    y_test = y[test_idx]
    test_stay_ids = stay_ids[test_idx]

    print(f"[GRU] Test set: {len(y_test)} samples, positive rate: {y_test.mean():.3f}")

    # Scale using checkpoint scaler params
    N_test, T, D_feat = X_test.shape
    scaler = StandardScaler()
    scaler.mean_ = checkpoint['scaler_mean']
    scaler.scale_ = checkpoint['scaler_scale']

    X_test_2d = X_test.reshape(-1, D_feat)
    X_test_2d = scaler.transform(X_test_2d)
    X_test_scaled = X_test_2d.reshape(N_test, T, D_feat)

    # Concat mask
    X_test_input = np.concatenate([X_test_scaled, X_test_mask], axis=2)
    input_dim = D_feat * 2

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ClinicalGRU(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Run inference
    test_loader = DataLoader(MIMICDataset(X_test_input, y_test), batch_size=BATCH_SIZE)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            preds = model(bx).squeeze().cpu().numpy()
            all_preds.extend(preds if preds.ndim > 0 else [preds.item()])
            all_targets.extend(by.numpy())

    y_pred = np.array(all_preds)
    y_true = np.array(all_targets)

    print(f"[GRU] Inference complete. AUROC: {roc_auc_score(y_true, y_pred):.4f}")

    # Get cohort info for subgroup analysis
    cohort_df = load_cohort()
    test_cohort = cohort_df[cohort_df['stay_id'].isin(test_stay_ids)].set_index('stay_id')

    # Evaluate on each cohort
    for cohort_name in ['all', 'sepsis', 'aki']:
        if cohort_name == 'all':
            mask = np.ones(len(y_true), dtype=bool)
        else:
            cohort_mask_series = test_cohort.reindex(test_stay_ids)
            if cohort_name == 'sepsis':
                mask = (cohort_mask_series['has_sepsis'].values == 1)
            else:
                mask = (cohort_mask_series['has_aki'].values == 1)

        if mask.sum() < 50:
            print(f"  [GRU] Skipping cohort={cohort_name}, only {mask.sum()} samples")
            continue

        y_t = y_true[mask]
        y_p = y_pred[mask]

        metrics = evaluate_calibration(y_t, y_p, N_BINS)
        result = {
            'window': '24h',
            'task': 'mortality',
            'cohort': cohort_name,
            'model': 'ClinicalGRU',
            'n_samples': int(mask.sum()),
            'n_test': int(mask.sum()),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v)
               for k, v in metrics.items()}
        }
        results.append(result)
        print(f"  [GRU] {cohort_name}: ECE={metrics['ece']:.4f}, "
              f"Brier={metrics['brier_score']:.4f}, "
              f"AUROC={roc_auc_score(y_t, y_p):.4f}")

        # Save reliability diagram for 'all' cohort
        if cohort_name == 'all':
            plot_reliability_diagram(
                y_t, y_p, N_BINS,
                title=f"ClinicalGRU - Mortality (24h, all)",
                save_path=OUTPUT_DIR / 'reliability_diagrams' / 'reliability_ClinicalGRU_mortality_24h_all.png'
            )
            plt.close('all')

    return results, {'y_true': y_true, 'y_pred': y_pred}


# ================================================================
# Text-only Calibration Evaluation
# ================================================================

def evaluate_text_calibration(cohort_df: pd.DataFrame) -> list:
    """Evaluate text-only model calibration."""
    results = []

    text_df = load_text_features(cohort_df)
    if text_df is None:
        print("[Text] Could not load text features. Skipping.")
        return results

    # Merge with cohort
    merged = text_df.merge(
        cohort_df[['stay_id', 'subject_id', 'mortality', 'has_sepsis', 'has_aki']],
        on='stay_id', how='inner'
    )
    print(f"[Text] Merged text features: {len(merged)} samples")

    if len(merged) < 100:
        print("[Text] Not enough samples. Skipping.")
        return results

    # Prepare features
    feature_cols = [c for c in text_df.columns if c != 'stay_id']
    X = merged[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = merged['mortality'].values
    groups = merged['subject_id'].values

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Train XGBoost on text features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print(f"[Text] XGBoost AUROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Evaluate calibration on each cohort
    test_merged = merged.iloc[test_idx].reset_index(drop=True)

    for cohort_name in ['all', 'sepsis', 'aki']:
        mask = get_cohort_mask(test_merged, cohort_name)

        if mask.sum() < 50:
            continue

        y_t = y_test[mask]
        y_p = y_prob[mask]

        metrics = evaluate_calibration(y_t, y_p, N_BINS)
        result = {
            'window': '24h',
            'task': 'mortality',
            'cohort': cohort_name,
            'model': 'TextOnly_XGBoost',
            'n_samples': int(mask.sum()),
            'n_test': int(mask.sum()),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v)
               for k, v in metrics.items()}
        }
        results.append(result)
        print(f"  [Text] {cohort_name}: ECE={metrics['ece']:.4f}, Brier={metrics['brier_score']:.4f}")

        if cohort_name == 'all':
            plot_reliability_diagram(
                y_t, y_p, N_BINS,
                title=f"TextOnly XGBoost - Mortality (24h, all)",
                save_path=OUTPUT_DIR / 'reliability_diagrams' / 'reliability_TextOnly_XGBoost_mortality_24h_all.png'
            )
            plt.close('all')

    return results


# ================================================================
# Fusion Calibration Evaluation (Early + Late)
# ================================================================

def evaluate_fusion_calibration(cohort_df: pd.DataFrame) -> list:
    """Evaluate fusion model calibration using structured + text features."""
    results = []

    # Load structured features (24h window)
    struct_df = load_aggregated_features('24h')
    if struct_df is None:
        print("[Fusion] No structured features for 24h. Skipping.")
        return results

    text_df = load_text_features(cohort_df)
    if text_df is None:
        print("[Fusion] No text features. Running structured-only late fusion.")
        return results

    # Merge all
    merged = struct_df.merge(
        cohort_df[['stay_id', 'subject_id', 'mortality', 'has_sepsis', 'has_aki']],
        on='stay_id', how='inner'
    )

    # Early Fusion: concat structured + text features
    text_features_only = text_df.drop(columns=['stay_id'], errors='ignore')
    text_feature_cols = [f'text_{c}' for c in text_features_only.columns]
    text_features_only.columns = text_feature_cols

    text_df_renamed = text_df[['stay_id']].copy()
    for col in text_feature_cols:
        text_df_renamed[col] = text_features_only[col.replace('text_', '')].values if col.replace('text_', '') in text_features_only.columns else text_features_only[col].values

    # Actually just merge by stay_id
    text_for_merge = text_df.copy()
    text_for_merge.columns = ['stay_id'] + [f'text_{c}' for c in text_df.columns if c != 'stay_id']

    merged_full = merged.merge(text_for_merge, on='stay_id', how='inner')
    print(f"[Fusion] Merged: {len(merged_full)} samples (structured + text)")

    if len(merged_full) < 100:
        print("[Fusion] Not enough merged samples. Skipping.")
        return results

    # Prepare features
    exclude_cols = ['stay_id', 'subject_id', 'mortality', 'prolonged_los', 'has_sepsis', 'has_aki']
    feature_cols = [c for c in merged_full.columns if c not in exclude_cols]

    X = merged_full[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = merged_full['mortality'].values
    groups = merged_full['subject_id'].values

    # Split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Early Fusion XGBoost
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=RANDOM_STATE, use_label_encoder=False,
        eval_metric='logloss', n_jobs=-1
    )
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    print(f"[Fusion] Early Fusion AUROC: {roc_auc_score(y_test, y_prob):.4f}")

    # Evaluate calibration
    test_merged = merged_full.iloc[test_idx].reset_index(drop=True)

    for cohort_name in ['all', 'sepsis', 'aki']:
        mask = get_cohort_mask(test_merged, cohort_name)

        if mask.sum() < 50:
            continue

        y_t = y_test[mask]
        y_p = y_prob[mask]

        metrics = evaluate_calibration(y_t, y_p, N_BINS)
        result = {
            'window': '24h',
            'task': 'mortality',
            'cohort': cohort_name,
            'model': 'EarlyFusion_XGBoost',
            'n_samples': int(mask.sum()),
            'n_test': int(mask.sum()),
            **{k: float(v) if isinstance(v, (np.floating, float)) else int(v)
               for k, v in metrics.items()}
        }
        results.append(result)
        print(f"  [EarlyFusion] {cohort_name}: ECE={metrics['ece']:.4f}, Brier={metrics['brier_score']:.4f}")

        if cohort_name == 'all':
            plot_reliability_diagram(
                y_t, y_p, N_BINS,
                title=f"Early Fusion XGBoost - Mortality (24h, all)",
                save_path=OUTPUT_DIR / 'reliability_diagrams' / 'reliability_EarlyFusion_XGBoost_mortality_24h_all.png'
            )
            plt.close('all')

    return results


# ================================================================
# Main
# ================================================================

def convert_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def main():
    print("=" * 60)
    print("TIMELY-Bench DL Model Calibration Evaluation")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Time: {datetime.now().isoformat()}")
    print("=" * 60)

    cohort_df = load_cohort()
    print(f"Loaded cohort: {len(cohort_df)} samples")

    all_results = []
    all_predictions = {}

    # 1. GRU Calibration
    print("\n" + "=" * 40)
    print("1. ClinicalGRU Calibration")
    print("=" * 40)
    gru_data = load_gru_data_and_model()
    if gru_data is not None:
        gru_results, gru_preds = evaluate_gru_calibration(gru_data)
        all_results.extend(gru_results)
        all_predictions['ClinicalGRU'] = (gru_preds['y_true'], gru_preds['y_pred'])
    else:
        print("[GRU] Skipped - no model data available.")

    # 2. Text-only Calibration
    print("\n" + "=" * 40)
    print("2. Text-only Model Calibration")
    print("=" * 40)
    text_results = evaluate_text_calibration(cohort_df)
    all_results.extend(text_results)

    # 3. Fusion Calibration
    print("\n" + "=" * 40)
    print("3. Fusion Model Calibration")
    print("=" * 40)
    fusion_results = evaluate_fusion_calibration(cohort_df)
    all_results.extend(fusion_results)

    # ================================================================
    # Save results
    # ================================================================
    if all_results:
        new_df = pd.DataFrame(all_results)

        # Merge with existing calibration results
        existing_file = OUTPUT_DIR / 'calibration_summary.csv'
        if existing_file.exists():
            existing_df = pd.read_csv(existing_file)
            # Remove any existing DL model rows to avoid duplicates
            dl_models = new_df['model'].unique()
            existing_df = existing_df[~existing_df['model'].isin(dl_models)]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df

        combined_df.to_csv(existing_file, index=False)
        print(f"\nResults appended to {existing_file}")
        print(f"Total rows: {len(combined_df)}")

        # Save DL-specific JSON
        dl_json = {
            'generated_at': datetime.now().isoformat(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'n_bins': N_BINS,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'models_evaluated': list(new_df['model'].unique()),
            'results': convert_to_native(all_results)
        }

        with open(OUTPUT_DIR / 'calibration_dl_summary.json', 'w') as f:
            json.dump(dl_json, f, indent=2)
        print(f"DL results saved to {OUTPUT_DIR / 'calibration_dl_summary.json'}")

        # Generate multi-model comparison plot
        if all_predictions:
            fig = plot_multi_model_reliability(
                all_predictions, N_BINS,
                title='DL Model Calibration Comparison (24h, Mortality)',
                save_path=OUTPUT_DIR / 'reliability_diagrams' / 'dl_models_comparison.png'
            )
            plt.close('all')

        # Print summary
        print("\n" + "=" * 60)
        print("DL Calibration Summary (24h, mortality)")
        print("=" * 60)
        summary = new_df[
            (new_df['task'] == 'mortality') &
            (new_df['cohort'] == 'all')
        ]
        if not summary.empty:
            print(summary[['model', 'ece', 'mce', 'brier_score', 'n_test']].to_string(index=False))

    else:
        print("\nNo DL calibration results generated.")
        print("Check if model checkpoints and data files exist.")

    print(f"\nDone at {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
