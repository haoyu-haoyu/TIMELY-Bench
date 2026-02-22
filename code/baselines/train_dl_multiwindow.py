"""
Multi-Window Deep Learning Model Training for Robustness Analysis.
Trains ClinicalGRU and EarlyFusion models across configured windows
(6h, 12h, 24h, D0).

Usage:
    python code/baselines/train_dl_multiwindow.py --window 6h --task mortality
    python code/baselines/train_dl_multiwindow.py --window 12h --task prolonged_los
    python code/baselines/train_dl_multiwindow.py --all  # Run all combinations

Outputs:
    - results/robustness/dl_window_performance.csv
    - results/Output_temporal_gru/models/best_model_{window}_{task}.pt
"""

import sys
from pathlib import Path
import argparse

# Setup paths
CODE_DIR = Path(__file__).parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PROJECT_ROOT / 'data' / 'processed' / 'data_windows'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import json
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    COHORT_FILE, RESULTS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE,
    HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE, EPOCHS, LR,
    USE_HOLDOUT_TEST, PROCESSED_DIR, WINDOWS as CONFIG_WINDOWS, TASKS as CONFIG_TASKS
)

# Alias for clarity
NUM_EPOCHS = EPOCHS
LEARNING_RATE = LR

# Output paths
OUTPUT_DIR = RESULTS_DIR / 'robustness'
MODEL_DIR = RESULTS_DIR / 'Output_temporal_gru' / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

WINDOWS = list(CONFIG_WINDOWS)
TASKS = list(CONFIG_TASKS)
COHORTS = ['all', 'sepsis', 'aki']


# ================================================================
# Model Definitions
# ================================================================

class ClinicalGRU(nn.Module):
    """Clinical GRU model for temporal prediction."""
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class MIMICDataset(Dataset):
    """PyTorch Dataset wrapper."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


# ================================================================
# Data Loading
# ================================================================

def load_cohort():
    """Load cohort data with label mapping."""
    df = pd.read_csv(COHORT_FILE)
    df['stay_id'] = df['stay_id'].astype(int)

    # Map label columns
    if 'label_mortality' in df.columns:
        df['mortality'] = df['label_mortality']
    if 'prolonged_los_7d' in df.columns:
        df['prolonged_los'] = df['prolonged_los_7d']
    elif 'prolonged_los_3d' in df.columns:
        df['prolonged_los'] = df['prolonged_los_3d']
    if 'has_sepsis_final' in df.columns:
        df['has_sepsis'] = df['has_sepsis_final']
    if 'has_aki_final' in df.columns:
        df['has_aki'] = df['has_aki_final']

    return df


def load_temporal_features(window: str):
    """Load temporal features for a specific window."""
    data_dir = PROCESSED_DIR / 'data_windows' / f'window_{window}'

    # Load features
    features_path = data_dir / 'features_temporal.npy'
    metadata_path = data_dir / 'metadata.json'

    if not features_path.exists():
        raise FileNotFoundError(f"Temporal features not found: {features_path}")

    X = np.load(features_path)

    # Load metadata for stay_ids
    with open(metadata_path) as f:
        metadata = json.load(f)

    stay_ids = np.array(metadata['stay_ids'])

    print(f"[{window}] Loaded temporal features: {X.shape}")
    return X, stay_ids


def load_aggregated_features(window: str):
    """Load aggregated features for EarlyFusion."""
    data_dir = PROCESSED_DIR / 'data_windows' / f'window_{window}'
    features_path = data_dir / 'features_aggregated.csv'

    if not features_path.exists():
        return None

    return pd.read_csv(features_path)


def get_cohort_mask(cohort_df: pd.DataFrame, cohort_name: str, stay_ids: np.ndarray):
    """Get cohort filter mask aligned with stay_ids."""
    # Create mapping from stay_id to index
    cohort_df = cohort_df.set_index('stay_id')

    mask = np.ones(len(stay_ids), dtype=bool)

    if cohort_name == 'sepsis':
        for i, sid in enumerate(stay_ids):
            if sid in cohort_df.index:
                mask[i] = cohort_df.loc[sid, 'has_sepsis'] == 1
            else:
                mask[i] = False
    elif cohort_name == 'aki':
        for i, sid in enumerate(stay_ids):
            if sid in cohort_df.index:
                mask[i] = cohort_df.loc[sid, 'has_aki'] == 1
            else:
                mask[i] = False

    return mask


def prepare_labels(cohort_df: pd.DataFrame, stay_ids: np.ndarray, task: str):
    """Prepare labels aligned with stay_ids."""
    cohort_df = cohort_df.set_index('stay_id')

    y = np.full(len(stay_ids), np.nan)
    subjects = np.zeros(len(stay_ids), dtype=int)

    for i, sid in enumerate(stay_ids):
        if sid in cohort_df.index:
            row = cohort_df.loc[sid]
            y[i] = row[task] if task in row else np.nan
            subjects[i] = row['subject_id'] if 'subject_id' in row else 0

    # Filter valid samples
    valid_mask = ~np.isnan(y)

    return y, subjects, valid_mask


# ================================================================
# Training Functions
# ================================================================

def train_gru_model(X, y, subjects, stay_ids, window, task, cohort_df, device):
    """Train ClinicalGRU model with cross-validation."""
    print(f"\n{'='*50}")
    print(f"Training ClinicalGRU: window={window}, task={task}")
    print(f"{'='*50}")

    N, T, D = X.shape
    print(f"Data shape: N={N}, T={T}, D={D}")

    # Prepare mask for observation
    obs_mask = (~np.isnan(X)).astype(np.float32)

    # Forward fill NaN values
    for i in range(N):
        for d in range(D):
            last_valid = 0.0
            for t in range(T):
                if np.isnan(X[i, t, d]):
                    X[i, t, d] = last_valid
                else:
                    last_valid = X[i, t, d]

    X = np.nan_to_num(X, nan=0.0)

    # Split: train+val vs test
    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X, y, groups=subjects))
    else:
        train_val_idx = np.arange(len(y))
        test_idx = np.array([], dtype=int)

    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    mask_tv = obs_mask[train_val_idx]
    subjects_tv = subjects[train_val_idx]

    # Scale features
    scaler = StandardScaler()
    X_tv_2d = X_tv.reshape(-1, D)
    X_tv_2d = scaler.fit_transform(X_tv_2d)
    X_tv_scaled = X_tv_2d.reshape(len(train_val_idx), T, D)

    # Concat mask
    X_tv_input = np.concatenate([X_tv_scaled, mask_tv], axis=2)
    input_dim = D * 2

    # Cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    best_model_state = None
    best_auroc = -1

    for fold, (tr_rel, val_rel) in enumerate(gkf.split(X_tv_input, y_tv, groups=subjects_tv), 1):
        print(f"\n  Fold {fold}/{N_FOLDS}")

        X_train = X_tv_input[tr_rel]
        y_train = y_tv[tr_rel]
        X_val = X_tv_input[val_rel]
        y_val = y_tv[val_rel]

        # Create model
        model = ClinicalGRU(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # Create data loaders
        train_loader = DataLoader(
            MIMICDataset(X_train, y_train),
            batch_size=BATCH_SIZE, shuffle=True
        )
        val_loader = DataLoader(
            MIMICDataset(X_val, y_val),
            batch_size=BATCH_SIZE
        )

        # Training loop
        best_val_auroc = 0
        patience_counter = 0
        patience = 10

        for epoch in range(NUM_EPOCHS):
            model.train()
            train_loss = 0
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                pred = model(bx).squeeze()
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation
            model.eval()
            val_preds, val_targets = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    bx = bx.to(device)
                    pred = model(bx).squeeze().cpu().numpy()
                    val_preds.extend(pred if pred.ndim > 0 else [pred.item()])
                    val_targets.extend(by.numpy())

            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)

            try:
                val_auroc = roc_auc_score(val_targets, val_preds)
            except:
                val_auroc = 0.5

            if val_auroc > best_val_auroc:
                best_val_auroc = val_auroc
                patience_counter = 0
                if val_auroc > best_auroc:
                    best_auroc = val_auroc
                    best_model_state = {
                        'model_state_dict': model.state_dict(),
                        'scaler_mean': scaler.mean_,
                        'scaler_scale': scaler.scale_,
                        'val_auroc': val_auroc,
                        'input_dim': input_dim,
                        'window': window,
                        'task': task
                    }
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        # Compute final metrics for this fold
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                pred = model(bx).squeeze().cpu().numpy()
                val_preds.extend(pred if pred.ndim > 0 else [pred.item()])
                val_targets.extend(by.numpy())

        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)

        auroc = roc_auc_score(val_targets, val_preds)
        auprc = average_precision_score(val_targets, val_preds)

        fold_results.append({
            'fold': fold,
            'auroc': auroc,
            'auprc': auprc
        })
        print(f"    AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    # Test set evaluation
    test_metrics = {}
    if len(test_idx) > 0 and best_model_state is not None:
        X_test = X[test_idx]
        mask_test = obs_mask[test_idx]
        y_test = y[test_idx]

        # Scale
        X_test_2d = X_test.reshape(-1, D)
        X_test_2d = scaler.transform(X_test_2d)
        X_test_scaled = X_test_2d.reshape(len(test_idx), T, D)
        X_test_input = np.concatenate([X_test_scaled, mask_test], axis=2)

        # Load best model
        model = ClinicalGRU(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(device)
        model.load_state_dict(best_model_state['model_state_dict'])
        model.eval()

        test_loader = DataLoader(MIMICDataset(X_test_input, y_test), batch_size=BATCH_SIZE)
        test_preds, test_targets = [], []

        with torch.no_grad():
            for bx, by in test_loader:
                bx = bx.to(device)
                pred = model(bx).squeeze().cpu().numpy()
                test_preds.extend(pred if pred.ndim > 0 else [pred.item()])
                test_targets.extend(by.numpy())

        test_preds = np.array(test_preds)
        test_targets = np.array(test_targets)

        test_metrics['auroc'] = roc_auc_score(test_targets, test_preds)
        test_metrics['auprc'] = average_precision_score(test_targets, test_preds)
        print(f"\n  Test AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}")

    # Save best model
    if best_model_state is not None:
        model_path = MODEL_DIR / f'best_model_{window}_{task}.pt'
        torch.save(best_model_state, model_path)
        print(f"  Model saved to: {model_path}")

    # Aggregate results
    aurocs = [r['auroc'] for r in fold_results]
    auprcs = [r['auprc'] for r in fold_results]

    return {
        'cv_auroc_mean': float(np.mean(aurocs)),
        'cv_auroc_std': float(np.std(aurocs)),
        'cv_auprc_mean': float(np.mean(auprcs)),
        'cv_auprc_std': float(np.std(auprcs)),
        'test_auroc': test_metrics.get('auroc', np.nan),
        'test_auprc': test_metrics.get('auprc', np.nan),
        'fold_details': fold_results
    }


def train_early_fusion(X_temporal, stay_ids, cohort_df, window, task, device):
    """Train EarlyFusion XGBoost model."""
    print(f"\n{'='*50}")
    print(f"Training EarlyFusion: window={window}, task={task}")
    print(f"{'='*50}")

    # Load aggregated features
    agg_df = load_aggregated_features(window)
    if agg_df is None:
        print(f"  No aggregated features for {window}. Skipping EarlyFusion.")
        return None

    # Prepare labels
    y, subjects, valid_mask = prepare_labels(cohort_df, stay_ids, task)

    # Align with aggregated features
    agg_df = agg_df.set_index('stay_id')

    feature_cols = [c for c in agg_df.columns if c not in ['subject_id', 'mortality', 'prolonged_los']]

    valid_indices = []
    X_features = []
    y_labels = []
    subject_ids = []

    for i, sid in enumerate(stay_ids):
        if valid_mask[i] and sid in agg_df.index:
            valid_indices.append(i)
            X_features.append(agg_df.loc[sid, feature_cols].values)
            y_labels.append(y[i])
            subject_ids.append(subjects[i])

    X = np.array(X_features)
    y = np.array(y_labels)
    groups = np.array(subject_ids)

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"  Samples: {len(y)}, Features: {X.shape[1]}")

    # Split
    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    else:
        train_val_idx = np.arange(len(y))
        test_idx = np.array([], dtype=int)

    X_tv, y_tv = X[train_val_idx], y[train_val_idx]
    groups_tv = groups[train_val_idx]

    # Cross-validation
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []

    for fold, (tr_rel, val_rel) in enumerate(gkf.split(X_tv, y_tv, groups=groups_tv), 1):
        X_train, X_val = X_tv[tr_rel], X_tv[val_rel]
        y_train, y_val = y_tv[tr_rel], y_tv[val_rel]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_val_s = scaler.transform(X_val)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
        model.fit(X_train_s, y_train)
        pred = model.predict_proba(X_val_s)[:, 1]

        auroc = roc_auc_score(y_val, pred)
        auprc = average_precision_score(y_val, pred)

        fold_results.append({'fold': fold, 'auroc': auroc, 'auprc': auprc})
        print(f"    Fold {fold}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    # Test evaluation
    test_metrics = {}
    if len(test_idx) > 0:
        X_test, y_test = X[test_idx], y[test_idx]

        scaler = StandardScaler()
        X_tv_s = scaler.fit_transform(X_tv)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
        model.fit(X_tv_s, y_tv)
        test_pred = model.predict_proba(X_test_s)[:, 1]

        test_metrics['auroc'] = roc_auc_score(y_test, test_pred)
        test_metrics['auprc'] = average_precision_score(y_test, test_pred)
        print(f"\n  Test AUROC={test_metrics['auroc']:.4f}, AUPRC={test_metrics['auprc']:.4f}")

    aurocs = [r['auroc'] for r in fold_results]
    auprcs = [r['auprc'] for r in fold_results]

    return {
        'cv_auroc_mean': float(np.mean(aurocs)),
        'cv_auroc_std': float(np.std(aurocs)),
        'cv_auprc_mean': float(np.mean(auprcs)),
        'cv_auprc_std': float(np.std(auprcs)),
        'test_auroc': test_metrics.get('auroc', np.nan),
        'test_auprc': test_metrics.get('auprc', np.nan),
        'fold_details': fold_results
    }


# ================================================================
# Main
# ================================================================

def run_single(window: str, task: str):
    """Run training for a single window-task combination."""
    print(f"\n{'='*60}")
    print(f"TIMELY-Bench DL Training: window={window}, task={task}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Time: {datetime.now().isoformat()}")
    print(f"{'='*60}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    cohort_df = load_cohort()
    X_temporal, stay_ids = load_temporal_features(window)

    # Prepare labels
    y, subjects, valid_mask = prepare_labels(cohort_df, stay_ids, task)

    # Filter valid samples
    X_valid = X_temporal[valid_mask]
    y_valid = y[valid_mask]
    subjects_valid = subjects[valid_mask]
    stay_ids_valid = stay_ids[valid_mask]

    print(f"Valid samples: {len(y_valid)}, Positive rate: {y_valid.mean():.3f}")

    results = []

    # Train ClinicalGRU
    gru_result = train_gru_model(
        X_valid.copy(), y_valid, subjects_valid, stay_ids_valid,
        window, task, cohort_df, device
    )

    results.append({
        'window': window,
        'task': task,
        'cohort': 'all',
        'model': 'ClinicalGRU',
        'n_samples': len(y_valid),
        'positive_rate': float(y_valid.mean()),
        **gru_result
    })

    # Train EarlyFusion
    fusion_result = train_early_fusion(
        X_temporal, stay_ids, cohort_df, window, task, device
    )

    if fusion_result is not None:
        results.append({
            'window': window,
            'task': task,
            'cohort': 'all',
            'model': 'EarlyFusion_XGBoost',
            'n_samples': len(y_valid),
            'positive_rate': float(y_valid.mean()),
            **fusion_result
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-window DL training for Robustness')
    parser.add_argument('--window', type=str, choices=WINDOWS, help='Time window')
    parser.add_argument('--task', type=str, choices=TASKS, help='Prediction task')
    parser.add_argument('--all', action='store_true', help='Run all combinations')
    args = parser.parse_args()

    all_results = []

    if args.all:
        # Run all window-task combinations
        for window in WINDOWS:
            for task in TASKS:
                try:
                    results = run_single(window, task)
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error in {window}/{task}: {e}")
                    continue
    elif args.window and args.task:
        results = run_single(args.window, args.task)
        all_results.extend(results)
    else:
        parser.print_help()
        return

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Merge with existing results if any
        output_file = OUTPUT_DIR / 'dl_window_performance.csv'
        if output_file.exists():
            existing_df = pd.read_csv(output_file)
            # Remove duplicates based on window, task, model
            for _, row in results_df.iterrows():
                mask = (
                    (existing_df['window'] == row['window']) &
                    (existing_df['task'] == row['task']) &
                    (existing_df['model'] == row['model']) &
                    (existing_df['cohort'] == row['cohort'])
                )
                existing_df = existing_df[~mask]
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
        else:
            combined_df = results_df

        combined_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        print(f"Total rows: {len(combined_df)}")

        # Also save detailed JSON
        json_output = {
            'generated_at': datetime.now().isoformat(),
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'config': {
                'hidden_dim': HIDDEN_DIM,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'batch_size': BATCH_SIZE,
                'num_epochs': NUM_EPOCHS,
                'learning_rate': LEARNING_RATE
            },
            'results': all_results
        }

        json_file = OUTPUT_DIR / 'dl_window_performance.json'
        with open(json_file, 'w') as f:
            json.dump(json_output, f, indent=2, default=str)
        print(f"JSON saved to: {json_file}")

        # Print summary
        print("\n" + "="*60)
        print("Summary")
        print("="*60)
        print(results_df[['window', 'task', 'model', 'cv_auroc_mean', 'cv_auroc_std', 'test_auroc']].to_string(index=False))

    print(f"\nDone at {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
