"""
Early Fusion 和 Late Fusion 模型
结合时序特征和标注特征的两种融合策略

Early Fusion: 特征级别拼接，然后送入单一模型
Late Fusion: 分别训练时序模型和标注模型，然后集成预测
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMESERIES_FILE, NOTE_TIME_FILE, LLM_FEATURES_FILE, COHORT_FILE,
    RESULTS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE, LLM_COLS,
    USE_HOLDOUT_TEST, get_features_file
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'fusion_baselines'
OUTPUT_JSON = OUTPUT_DIR / 'fusion_results_folds.json'
OUTPUT_JSON_LATE_XGB = OUTPUT_DIR / 'fusion_results_late_xgb.json'
OUTPUT_CSV_LATE_XGB = OUTPUT_DIR / 'fusion_results_late_xgb.csv'
PRED_DIR = OUTPUT_DIR / 'predictions'
STANDARD_DIR = RESULTS_DIR / 'standardized'


def load_annotation_features():
    """从 Episode 加载标注特征"""
    print("加载标注特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    annotations = []
    
    for ep_file in tqdm(episode_files, desc="Loading annotations"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            stay_id = ep.get('stay_id')
            reasoning = ep.get('reasoning', {})
            
            n_supportive = reasoning.get('n_supportive', 0)
            n_contradictory = reasoning.get('n_contradictory', 0)
            n_alignments = reasoning.get('n_alignments', 0)
            
            total_annot = n_supportive + n_contradictory
            supportive_ratio = n_supportive / total_annot if total_annot > 0 else 0.5
            annotation_density = total_annot / n_alignments if n_alignments > 0 else 0
            
            annotations.append({
                'stay_id': stay_id,
                'n_supportive': n_supportive,
                'n_contradictory': n_contradictory,
                'supportive_ratio': supportive_ratio,
                'annotation_density': annotation_density
            })
        except:
            pass
    
    return pd.DataFrame(annotations)


def load_tabular_features():
    """加载表格化的时序特征"""
    print("加载时序特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    vitals_cols = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2']
    
    features = []
    for ep_file in tqdm(episode_files, desc="Loading timeseries"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            feat = {'stay_id': ep.get('stay_id')}
            ts = ep.get('timeseries', {})
            vitals = ts.get('vitals', [])
            
            if vitals:
                df = pd.DataFrame(vitals)
                for col in vitals_cols:
                    if col in df.columns:
                        values = pd.to_numeric(df[col], errors='coerce').dropna()
                        if len(values) > 0:
                            feat[f'{col}_mean'] = values.mean()
                            feat[f'{col}_std'] = values.std() if len(values) > 1 else 0
                            feat[f'{col}_min'] = values.min()
                            feat[f'{col}_max'] = values.max()
            
            features.append(feat)
        except:
            pass
    
    return pd.DataFrame(features)


def _filter_cohort_ids(cohort_df, cohort_name):
    if cohort_name == 'all':
        return cohort_df['stay_id'].values
    if cohort_name == 'sepsis':
        return cohort_df[cohort_df['has_sepsis_final'] == 1]['stay_id'].values
    if cohort_name == 'aki':
        return cohort_df[cohort_df['has_aki_final'] == 1]['stay_id'].values
    if cohort_name == 'sepsis_aki':
        mask = (cohort_df['has_sepsis_final'] == 1) & (cohort_df['has_aki_final'] == 1)
        return cohort_df[mask]['stay_id'].values
    raise ValueError(f"Unknown cohort: {cohort_name}")


def _label_column(task):
    label_map = {
        'mortality': 'label_mortality',
        'prolonged_los': 'prolonged_los_7d',
        'readmission': 'readmission_30d'
    }
    return label_map[task]


def load_structured_features(window='24h', cohort='all', task='mortality'):
    """Load structured features consistent with run_baselines.py."""
    features_file = get_features_file(window)
    if not Path(features_file).exists():
        raise FileNotFoundError(f"Missing structured features: {features_file}")

    cohort_df = pd.read_csv(COHORT_FILE)
    cohort_df['stay_id'] = cohort_df['stay_id'].astype(int)
    stay_ids = _filter_cohort_ids(cohort_df, cohort)

    feat = pd.read_csv(features_file)
    feat['stay_id'] = feat['stay_id'].astype(int)
    feat = feat[feat['stay_id'].isin(stay_ids)].copy()

    label_col = _label_column(task)
    cohort_labels = cohort_df[['stay_id', 'subject_id', label_col]].rename(columns={label_col: 'label'})
    df = feat.merge(cohort_labels, on='stay_id', how='inner')
    df = df[df['label'].notna()].copy()
    return df


def _extract_text_features(ep):
    features = {
        'stay_id': ep.get('stay_id'),
        'subject_id': ep.get('patient', {}).get('subject_id')
    }

    clinical = ep.get('clinical_text', {})
    features['n_notes'] = clinical.get('n_notes', 0)
    notes = clinical.get('notes', [])
    if notes:
        features['total_text_length'] = sum(len(n.get('text', '')) for n in notes)
        features['avg_text_length'] = features['total_text_length'] / len(notes)
    else:
        features['total_text_length'] = 0
        features['avg_text_length'] = 0

    reasoning = ep.get('reasoning', {})
    features['n_patterns'] = len(reasoning.get('detected_patterns', []))
    features['n_supportive'] = reasoning.get('n_supportive', 0)
    features['n_contradictory'] = reasoning.get('n_contradictory', 0)
    features['n_alignments'] = reasoning.get('n_alignments', 0)

    total_annot = features['n_supportive'] + features['n_contradictory']
    if total_annot > 0:
        features['supportive_ratio'] = features['n_supportive'] / total_annot
        features['contradictory_ratio'] = features['n_contradictory'] / total_annot
    else:
        features['supportive_ratio'] = 0.5
        features['contradictory_ratio'] = 0.5

    if features['n_alignments'] > 0:
        features['annotation_density'] = total_annot / features['n_alignments']
    else:
        features['annotation_density'] = 0

    return features


def load_text_features(task='mortality'):
    """Load text-only features consistent with train_text_only.py."""
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    rows = []
    for ep_file in tqdm(episode_files, desc="Extracting text features"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            rows.append(_extract_text_features(ep))
        except Exception:
            continue
    text_df = pd.DataFrame(rows)

    cohort_df = pd.read_csv(COHORT_FILE)
    cohort_df['stay_id'] = cohort_df['stay_id'].astype(int)
    label_col = _label_column(task)
    cohort_labels = cohort_df[['stay_id', 'subject_id', label_col]].rename(columns={label_col: 'label'})
    text_df['stay_id'] = text_df['stay_id'].astype(int)
    text_df = text_df.merge(cohort_labels, on='stay_id', how='inner')
    text_df = text_df[text_df['label'].notna()].copy()
    return text_df


def _align_struct_text(struct_df, text_df):
    info = {}
    info['structured_rows'] = int(len(struct_df))
    info['text_rows'] = int(len(text_df))
    info['structured_dup_stay'] = int(struct_df['stay_id'].duplicated().sum())
    info['text_dup_stay'] = int(text_df['stay_id'].duplicated().sum())

    common = sorted(set(struct_df['stay_id']) & set(text_df['stay_id']))
    struct_df = struct_df[struct_df['stay_id'].isin(common)].copy()
    text_df = text_df[text_df['stay_id'].isin(common)].copy()

    struct_df = struct_df.sort_values('stay_id').reset_index(drop=True)
    text_df = text_df.sort_values('stay_id').reset_index(drop=True)

    if not np.array_equal(struct_df['stay_id'].values, text_df['stay_id'].values):
        raise ValueError("stay_id alignment failed between structured and text features.")

    info['merged_rows'] = int(len(struct_df))
    info['merged_dup_stay'] = int(struct_df['stay_id'].duplicated().sum())
    info['structured_only'] = int(len(set(struct_df['stay_id']) - set(common)))
    info['text_only'] = int(len(set(text_df['stay_id']) - set(common)))
    return struct_df, text_df, info


def _fit_xgb(X_train, y_train, X_val, max_depth, scale_pos_weight=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    if scale_pos_weight is not None:
        model.set_params(scale_pos_weight=scale_pos_weight)

    model.fit(X_train, y_train)
    pred = model.predict_proba(X_val)[:, 1]
    return pred, model, scaler


def _compute_metrics(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return 0.5, float(np.mean(y_true))
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)


def _build_pred_df(stay_ids, subject_ids, labels, preds, split_name, fold_ids=None):
    df = pd.DataFrame({
        'stay_id': stay_ids,
        'subject_id': subject_ids,
        'label': labels,
        'pred': preds,
        'split': split_name
    })
    if fold_ids is not None:
        df['fold'] = fold_ids
    return df


def train_late_fusion_from_preds(task='mortality', window='24h', cohort='all', alpha_grid=None):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 11)

    struct_df = load_structured_features(window=window, cohort=cohort, task=task)
    text_df = load_text_features(task=task)
    struct_df, text_df, align_info = _align_struct_text(struct_df, text_df)

    struct_cols = [c for c in struct_df.columns if c not in ['stay_id', 'subject_id', 'label']]
    text_cols = [c for c in text_df.columns if c not in ['stay_id', 'subject_id', 'label']]

    X_struct = struct_df[struct_cols].values
    X_text = text_df[text_cols].values
    y = struct_df['label'].values
    groups = struct_df['subject_id'].values
    stay_ids = struct_df['stay_id'].values

    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(np.zeros(len(y)), y, groups=groups))
    else:
        train_val_idx = np.arange(len(y))
        test_idx = np.array([], dtype=int)

    gkf = GroupKFold(n_splits=N_FOLDS)
    best_alphas = []
    fold_rows = []

    val_pred_struct = np.full(len(y), np.nan)
    val_pred_text = np.full(len(y), np.nan)
    fold_ids = np.full(len(y), -1, dtype=int)

    best_struct_model = None
    best_struct_scaler = None
    best_struct_auroc = -1

    for fold, (tr_rel, val_rel) in enumerate(gkf.split(X_struct[train_val_idx], y[train_val_idx], groups=groups[train_val_idx]), start=1):
        tr_idx = train_val_idx[tr_rel]
        val_idx = train_val_idx[val_rel]

        # structured (XGBoost max_depth=5, run_baselines)
        neg_count = (y[tr_idx] == 0).sum()
        pos_count = (y[tr_idx] == 1).sum()
        scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0
        pred_s, model_s, scaler_s = _fit_xgb(X_struct[tr_idx], y[tr_idx], X_struct[val_idx], 5, scale_pos_weight)

        # text-only (XGBoost max_depth=6, train_text_only)
        pred_t, model_t, scaler_t = _fit_xgb(X_text[tr_idx], y[tr_idx], X_text[val_idx], 6, None)

        val_pred_struct[val_idx] = pred_s
        val_pred_text[val_idx] = pred_t
        fold_ids[val_idx] = fold

        auroc_s, auprc_s = _compute_metrics(y[val_idx], pred_s)
        auroc_t, auprc_t = _compute_metrics(y[val_idx], pred_t)

        if auroc_s > best_struct_auroc:
            best_struct_auroc = auroc_s
            best_struct_model = model_s
            best_struct_scaler = scaler_s

        best_alpha = None
        best_score = -1
        best_pred = None
        for alpha in alpha_grid:
            pred = alpha * pred_s + (1 - alpha) * pred_t
            score, _ = _compute_metrics(y[val_idx], pred)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_pred = pred

        auroc_best, auprc_best = _compute_metrics(y[val_idx], best_pred)
        fold_rows.append({
            'fold': fold,
            'val_auroc_alpha1': auroc_s,
            'val_auprc_alpha1': auprc_s,
            'val_auroc_alpha0': auroc_t,
            'val_auprc_alpha0': auprc_t,
            'best_alpha': float(best_alpha),
            'best_val_auroc': auroc_best,
            'best_val_auprc': auprc_best
        })
        best_alphas.append(best_alpha)

    alpha_final = float(np.mean(best_alphas)) if best_alphas else 0.5

    # Test predictions
    test_metrics = {}
    test_pred_struct = None
    test_pred_text = None
    if len(test_idx) > 0:
        X_struct_test = X_struct[test_idx]
        X_text_test = X_text[test_idx]

        # structured: use best fold model (match run_baselines behavior)
        X_struct_test_s = best_struct_scaler.transform(X_struct_test)
        X_struct_test_s = np.nan_to_num(X_struct_test_s, nan=0.0, posinf=0.0, neginf=0.0)
        test_pred_struct = best_struct_model.predict_proba(X_struct_test_s)[:, 1]

        # text: train on all train_val (match train_text_only behavior)
        pred_t_full, model_t_full, scaler_t_full = _fit_xgb(
            X_text[train_val_idx], y[train_val_idx], X_text_test, 6, None
        )
        test_pred_text = pred_t_full

        test_metrics['test_alpha1'] = _compute_metrics(y[test_idx], test_pred_struct)
        test_metrics['test_alpha0'] = _compute_metrics(y[test_idx], test_pred_text)
        pred_best = alpha_final * test_pred_struct + (1 - alpha_final) * test_pred_text
        test_metrics['test_best'] = _compute_metrics(y[test_idx], pred_best)

    # save prediction files
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    struct_pred_df = _build_pred_df(
        stay_ids[train_val_idx], groups[train_val_idx], y[train_val_idx], val_pred_struct[train_val_idx], 'val', fold_ids[train_val_idx]
    )
    text_pred_df = _build_pred_df(
        stay_ids[train_val_idx], groups[train_val_idx], y[train_val_idx], val_pred_text[train_val_idx], 'val', fold_ids[train_val_idx]
    )

    if len(test_idx) > 0:
        struct_pred_df = pd.concat([
            struct_pred_df,
            _build_pred_df(stay_ids[test_idx], groups[test_idx], y[test_idx], test_pred_struct, 'test')
        ], ignore_index=True)
        text_pred_df = pd.concat([
            text_pred_df,
            _build_pred_df(stay_ids[test_idx], groups[test_idx], y[test_idx], test_pred_text, 'test')
        ], ignore_index=True)

    structured_pred_path = PRED_DIR / f'structured_xgb_{window}_{cohort}_{task}.csv'
    text_pred_path = PRED_DIR / f'text_only_xgb_{window}_{cohort}_{task}.csv'
    struct_pred_df.to_csv(structured_pred_path, index=False)
    text_pred_df.to_csv(text_pred_path, index=False)

    sanity_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'task': task,
        'window': window,
        'cohort': cohort,
        'structured_pred_path': str(structured_pred_path),
        'text_pred_path': str(text_pred_path),
        'n_rows_structured': int(len(struct_pred_df)),
        'n_rows_text': int(len(text_pred_df)),
        'merge_key': 'stay_id',
        'alignment_info': align_info,
        'best_alpha_per_fold': [float(a) for a in best_alphas],
        'alpha_final': alpha_final,
        'folds': fold_rows,
        'test_metrics': test_metrics
    }

    STANDARD_DIR.mkdir(parents=True, exist_ok=True)
    sanity_path = STANDARD_DIR / 'late_fusion_sanity_xgb.json'
    with open(sanity_path, 'w') as f:
        json.dump(sanity_payload, f, indent=2, ensure_ascii=True)
    print(f"Late fusion sanity saved to: {sanity_path}")
    print(f"Structured pred path: {structured_pred_path}")
    print(f"Text pred path: {text_pred_path}")

    # summarize results
    auroc_alpha1 = [r['val_auroc_alpha1'] for r in fold_rows]
    auprc_alpha1 = [r['val_auprc_alpha1'] for r in fold_rows]
    auroc_alpha0 = [r['val_auroc_alpha0'] for r in fold_rows]
    auprc_alpha0 = [r['val_auprc_alpha0'] for r in fold_rows]
    auroc_best = [r['best_val_auroc'] for r in fold_rows]
    auprc_best = [r['best_val_auprc'] for r in fold_rows]

    results_rows = [
        {
            'model': 'Late Fusion (Alpha=1.0 Structured XGB)',
            'task': task,
            'cohort': cohort,
            'window': window,
            'n_samples': int(len(y)),
            'positive_rate': float(np.mean(y)) if len(y) > 0 else 0.0,
            'cv_auroc_mean': float(np.mean(auroc_alpha1)),
            'cv_auroc_std': float(np.std(auroc_alpha1)),
            'cv_auprc_mean': float(np.mean(auprc_alpha1)),
            'cv_auprc_std': float(np.std(auprc_alpha1)),
            'test_auroc': float(test_metrics.get('test_alpha1', (np.nan, np.nan))[0]) if test_metrics else np.nan,
            'test_auprc': float(test_metrics.get('test_alpha1', (np.nan, np.nan))[1]) if test_metrics else np.nan,
            'alpha_final': 1.0,
            'fold_details': fold_rows
        },
        {
            'model': 'Late Fusion (Tuned Alpha XGB Preds)',
            'task': task,
            'cohort': cohort,
            'window': window,
            'n_samples': int(len(y)),
            'positive_rate': float(np.mean(y)) if len(y) > 0 else 0.0,
            'cv_auroc_mean': float(np.mean(auroc_best)),
            'cv_auroc_std': float(np.std(auroc_best)),
            'cv_auprc_mean': float(np.mean(auprc_best)),
            'cv_auprc_std': float(np.std(auprc_best)),
            'test_auroc': float(test_metrics.get('test_best', (np.nan, np.nan))[0]) if test_metrics else np.nan,
            'test_auprc': float(test_metrics.get('test_best', (np.nan, np.nan))[1]) if test_metrics else np.nan,
            'alpha_final': alpha_final,
            'fold_details': fold_rows
        }
    ]

    # include alpha=0 in sanity only
    sanity_payload['val_alpha0_mean'] = float(np.mean(auroc_alpha0)) if auroc_alpha0 else np.nan
    sanity_payload['val_alpha0_auprc_mean'] = float(np.mean(auprc_alpha0)) if auprc_alpha0 else np.nan
    if test_metrics:
        sanity_payload['test_alpha0'] = {
            'auroc': float(test_metrics['test_alpha0'][0]),
            'auprc': float(test_metrics['test_alpha0'][1])
        }
        with open(sanity_path, 'w') as f:
            json.dump(sanity_payload, f, indent=2, ensure_ascii=True)

    return results_rows


def train_early_fusion(X_ts, X_annot, y, groups):
    """Early Fusion: 特征拼接"""
    print("\n=== Early Fusion (XGBoost) ===")
    
    # 拼接特征
    X = np.concatenate([X_ts, X_annot], axis=1)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_tv, X_test = X[train_val_idx], X[test_idx]
    y_tv, y_test = y[train_val_idx], y[test_idx]
    groups_tv = groups[train_val_idx]
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tv, y_tv, groups=groups_tv)):
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, use_label_encoder=False,
                                  eval_metric='logloss', n_jobs=-1)
        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_val)[:, 1]
        
        auroc = roc_auc_score(y_val, pred)
        auprc = average_precision_score(y_val, pred)
        fold_results.append({
            'fold': fold + 1,
            'auroc': auroc,
            'auprc': auprc
        })
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    
    # 测试
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_test_s = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=RANDOM_STATE, use_label_encoder=False,
                              eval_metric='logloss', n_jobs=-1)
    model.fit(X_tv_s, y_tv)
    test_pred = model.predict_proba(X_test_s)[:, 1]
    test_auroc = roc_auc_score(y_test, test_pred)
    test_auprc = average_precision_score(y_test, test_pred)

    aurocs = [r['auroc'] for r in fold_results]
    auprcs = [r['auprc'] for r in fold_results]
    print(f"\n   CV AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"   CV AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    print(f"   Test AUPRC: {test_auprc:.4f}")
    
    return {
        'cv_auroc_mean': float(np.mean(aurocs)),
        'cv_auroc_std': float(np.std(aurocs)),
        'cv_auprc_mean': float(np.mean(auprcs)),
        'cv_auprc_std': float(np.std(auprcs)),
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'fold_details': fold_results
    }


def train_late_fusion_fixed(X_ts, X_annot, y, groups, alpha=0.7):
    """Late Fusion: 固定权重加权"""
    print(f"\n=== Late Fusion (Fixed alpha={alpha:.2f}) ===")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X_ts, y, groups=groups))
    
    X_ts_tv, X_ts_test = X_ts[train_val_idx], X_ts[test_idx]
    X_annot_tv, X_annot_test = X_annot[train_val_idx], X_annot[test_idx]
    y_tv, y_test = y[train_val_idx], y[test_idx]
    groups_tv = groups[train_val_idx]
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_ts_tv, y_tv, groups=groups_tv)):
        # 时序模型
        scaler_ts = StandardScaler()
        X_ts_tr = scaler_ts.fit_transform(X_ts_tv[tr_idx])
        X_ts_val = scaler_ts.transform(X_ts_tv[val_idx])
        
        model_ts = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                                     use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        model_ts.fit(X_ts_tr, y_tv[tr_idx])
        pred_ts = model_ts.predict_proba(X_ts_val)[:, 1]
        
        # 标注模型
        scaler_annot = StandardScaler()
        X_annot_tr = scaler_annot.fit_transform(X_annot_tv[tr_idx])
        X_annot_val = scaler_annot.transform(X_annot_tv[val_idx])
        
        model_annot = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                                        use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        model_annot.fit(X_annot_tr, y_tv[tr_idx])
        pred_annot = model_annot.predict_proba(X_annot_val)[:, 1]
        
        # 集成
        pred_fusion = alpha * pred_ts + (1 - alpha) * pred_annot
        auroc = roc_auc_score(y_tv[val_idx], pred_fusion)
        auprc = average_precision_score(y_tv[val_idx], pred_fusion)
        fold_results.append({
            'fold': fold + 1,
            'auroc': auroc,
            'auprc': auprc,
            'alpha': float(alpha)
        })
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    
    # 测试
    scaler_ts = StandardScaler()
    X_ts_tv_s = scaler_ts.fit_transform(X_ts_tv)
    X_ts_test_s = scaler_ts.transform(X_ts_test)
    
    scaler_annot = StandardScaler()
    X_annot_tv_s = scaler_annot.fit_transform(X_annot_tv)
    X_annot_test_s = scaler_annot.transform(X_annot_test)
    
    model_ts = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                                 use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model_ts.fit(X_ts_tv_s, y_tv)
    
    model_annot = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model_annot.fit(X_annot_tv_s, y_tv)
    
    test_pred_ts = model_ts.predict_proba(X_ts_test_s)[:, 1]
    test_pred_annot = model_annot.predict_proba(X_annot_test_s)[:, 1]
    test_pred = alpha * test_pred_ts + (1 - alpha) * test_pred_annot
    test_auroc = roc_auc_score(y_test, test_pred)
    test_auprc = average_precision_score(y_test, test_pred)
    
    aurocs = [r['auroc'] for r in fold_results]
    auprcs = [r['auprc'] for r in fold_results]
    print(f"\n   CV AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"   CV AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    print(f"   Test AUPRC: {test_auprc:.4f}")
    
    return {
        'cv_auroc_mean': float(np.mean(aurocs)),
        'cv_auroc_std': float(np.std(aurocs)),
        'cv_auprc_mean': float(np.mean(auprcs)),
        'cv_auprc_std': float(np.std(auprcs)),
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'fold_details': fold_results,
        'alpha_mean': float(alpha),
        'alpha_std': 0.0,
        'alpha_final': float(alpha)
    }


def train_late_fusion_tuned(X_ts, X_annot, y, groups, alpha_grid=None):
    """Late Fusion: 在训练折验证集内搜索 alpha，然后用均值 alpha 做测试集"""
    if alpha_grid is None:
        alpha_grid = np.linspace(0.0, 1.0, 11)
    print("\n=== Late Fusion (Tuned alpha) ===")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X_ts, y, groups=groups))
    
    X_ts_tv, X_ts_test = X_ts[train_val_idx], X_ts[test_idx]
    X_annot_tv, X_annot_test = X_annot[train_val_idx], X_annot[test_idx]
    y_tv, y_test = y[train_val_idx], y[test_idx]
    groups_tv = groups[train_val_idx]
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    best_alphas = []
    
    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_ts_tv, y_tv, groups=groups_tv)):
        # 时序模型
        scaler_ts = StandardScaler()
        X_ts_tr = scaler_ts.fit_transform(X_ts_tv[tr_idx])
        X_ts_val = scaler_ts.transform(X_ts_tv[val_idx])
        
        model_ts = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                                     use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        model_ts.fit(X_ts_tr, y_tv[tr_idx])
        pred_ts = model_ts.predict_proba(X_ts_val)[:, 1]
        
        # 标注模型
        scaler_annot = StandardScaler()
        X_annot_tr = scaler_annot.fit_transform(X_annot_tv[tr_idx])
        X_annot_val = scaler_annot.transform(X_annot_tv[val_idx])
        
        model_annot = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                                        use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
        model_annot.fit(X_annot_tr, y_tv[tr_idx])
        pred_annot = model_annot.predict_proba(X_annot_val)[:, 1]
        
        # 在该fold的val上选alpha
        best_alpha = None
        best_score = -1
        best_pred = None
        for alpha in alpha_grid:
            pred_fusion = alpha * pred_ts + (1 - alpha) * pred_annot
            try:
                score = roc_auc_score(y_tv[val_idx], pred_fusion)
            except Exception:
                score = 0.5
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_pred = pred_fusion
        
        auroc = roc_auc_score(y_tv[val_idx], best_pred)
        auprc = average_precision_score(y_tv[val_idx], best_pred)
        fold_results.append({
            'fold': fold + 1,
            'auroc': auroc,
            'auprc': auprc,
            'alpha': float(best_alpha)
        })
        best_alphas.append(best_alpha)
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}, alpha={best_alpha:.2f}")
    
    # 测试
    scaler_ts = StandardScaler()
    X_ts_tv_s = scaler_ts.fit_transform(X_ts_tv)
    X_ts_test_s = scaler_ts.transform(X_ts_test)
    
    scaler_annot = StandardScaler()
    X_annot_tv_s = scaler_annot.fit_transform(X_annot_tv)
    X_annot_test_s = scaler_annot.transform(X_annot_test)
    
    model_ts = xgb.XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE,
                                 use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model_ts.fit(X_ts_tv_s, y_tv)
    
    model_annot = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                                    use_label_encoder=False, eval_metric='logloss', n_jobs=-1)
    model_annot.fit(X_annot_tv_s, y_tv)
    
    test_pred_ts = model_ts.predict_proba(X_ts_test_s)[:, 1]
    test_pred_annot = model_annot.predict_proba(X_annot_test_s)[:, 1]
    alpha_final = float(np.mean(best_alphas)) if best_alphas else 0.5
    test_pred = alpha_final * test_pred_ts + (1 - alpha_final) * test_pred_annot
    test_auroc = roc_auc_score(y_test, test_pred)
    test_auprc = average_precision_score(y_test, test_pred)
    
    aurocs = [r['auroc'] for r in fold_results]
    auprcs = [r['auprc'] for r in fold_results]
    print(f"\n   CV AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"   CV AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    print(f"   Test AUPRC: {test_auprc:.4f}")
    print(f"   Alpha (mean±std): {np.mean(best_alphas):.2f} ± {np.std(best_alphas):.2f}")
    
    return {
        'cv_auroc_mean': float(np.mean(aurocs)),
        'cv_auroc_std': float(np.std(aurocs)),
        'cv_auprc_mean': float(np.mean(auprcs)),
        'cv_auprc_std': float(np.std(auprcs)),
        'test_auroc': float(test_auroc),
        'test_auprc': float(test_auprc),
        'fold_details': fold_results,
        'alpha_mean': float(np.mean(best_alphas)) if best_alphas else 0.5,
        'alpha_std': float(np.std(best_alphas)) if best_alphas else 0.0,
        'alpha_final': alpha_final
    }


def main():
    print("=" * 60)
    print("Fusion Baselines (Early & Late)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    ts_df = load_tabular_features()
    annot_df = load_annotation_features()
    
    # 加载 cohort
    cohort = pd.read_csv(COHORT_FILE)
    cohort['stay_id'] = cohort['stay_id'].astype(int)
    
    # 合并
    df = cohort[['stay_id', 'subject_id', 'label_mortality']].merge(
        ts_df, on='stay_id', how='inner'
    ).merge(
        annot_df, on='stay_id', how='inner'
    ).dropna()
    
    print(f"\n合并后样本数: {len(df):,}")
    
    # 准备特征
    ts_cols = [c for c in ts_df.columns if c != 'stay_id']
    annot_cols = ['n_supportive', 'n_contradictory', 'supportive_ratio', 'annotation_density']
    
    X_ts = df[ts_cols].values
    X_ts = np.nan_to_num(X_ts, nan=0.0)
    
    X_annot = df[annot_cols].values
    y = df['label_mortality'].values
    groups = df['subject_id'].values
    
    results = []
    
    # Early Fusion
    positive_rate = float(y.mean()) if len(y) > 0 else 0.0

    early = train_early_fusion(X_ts, X_annot, y, groups)
    results.append({
        'model': 'Early Fusion',
        'task': 'mortality',
        'n_samples': len(y),
        'positive_rate': positive_rate,
        **early
    })
    
    # Late Fusion (structured/text baseline predictions)
    late_xgb_rows = train_late_fusion_from_preds(task='mortality', window='24h', cohort='all')
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'fusion_results.csv', index=False)

    late_df = pd.DataFrame(late_xgb_rows)
    late_df.to_csv(OUTPUT_CSV_LATE_XGB, index=False)

    output_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seed': RANDOM_STATE,
        'input_paths': {
            'episodes_dir': str(EPISODES_DIR),
            'cohort_file': str(COHORT_FILE)
        },
        'results': results
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=True)
    print(f"Fold details saved to: {OUTPUT_JSON}")

    late_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seed': RANDOM_STATE,
        'input_paths': {
            'structured_features': str(get_features_file('24h')),
            'cohort_file': str(COHORT_FILE),
            'episodes_dir': str(EPISODES_DIR)
        },
        'results': late_xgb_rows
    }
    with open(OUTPUT_JSON_LATE_XGB, 'w') as f:
        json.dump(late_payload, f, indent=2, ensure_ascii=True)
    print(f"Late fusion results saved to: {OUTPUT_JSON_LATE_XGB}")
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("\nLate fusion (structured/text preds)")
    print(late_df.to_string(index=False))


if __name__ == "__main__":
    main()
