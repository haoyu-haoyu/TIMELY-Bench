"""
30-Day Readmission 任务训练
预测患者出院后 30 天内是否再入院

与 Mortality/LOS 任务使用相同的特征集（含 MedCAT）
"""

import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except ImportError:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置（统一使用项目 config，避免 split/cohort 口径冲突）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, COHORT_FILE, N_FOLDS, RANDOM_STATE, USE_HOLDOUT_TEST
from utils.predefined_split import resolve_predefined_partition

BASE_DIR = ROOT_DIR
EPISODES_DIR = BASE_DIR / 'episodes' / 'episodes_enhanced'
EMBEDDINGS_FILE = BASE_DIR / 'data' / 'processed' / 'text_embeddings' / 'clinical_bert_embeddings.npy'
CONCEPTS_FILE = BASE_DIR / 'data' / 'processed' / 'text_concepts' / 'spacy_concepts.csv'
MEDCAT_FILE = BASE_DIR / 'data' / 'processed' / 'medcat_full' / 'medcat_features_24h.csv'
COHORTS_FILE = Path(COHORT_FILE)
RESULTS_DIR = BASE_DIR / 'results' / 'readmission_baselines'


def load_all_data():
    """加载所有数据"""
    print("加载数据...")
    
    cohort = pd.read_csv(COHORTS_FILE)
    print(f"  队列: {len(cohort):,} 样本")
    
    embeddings = np.load(EMBEDDINGS_FILE)
    emb_ids = pd.read_csv(Path(EMBEDDINGS_FILE).parent / 'embedding_stay_ids.csv')
    emb_dict = dict(zip(emb_ids['stay_id'], range(len(emb_ids))))
    print(f"  BERT 嵌入: {embeddings.shape}")
    
    concepts = pd.read_csv(CONCEPTS_FILE)
    concepts_dict = concepts.set_index('stay_id').to_dict('index')
    print(f"  概念特征: {concepts.shape}")

    medcat_dict = None
    if MEDCAT_FILE.exists():
        medcat = pd.read_csv(MEDCAT_FILE)
        if 'window_hours' in medcat.columns:
            medcat = medcat.drop(columns=['window_hours'])
        medcat = medcat.rename(
            columns={c: f'medcat_{c}' for c in medcat.columns if c != 'stay_id'}
        )
        medcat_dict = medcat.set_index('stay_id').to_dict('index')
        print(f"  MedCAT 特征: {medcat.shape}")
    else:
        print("  未找到 MedCAT 特征，跳过")
    
    return cohort, embeddings, emb_dict, concepts_dict, medcat_dict


def get_readmission_label(stay_id):
    """从 Episode 获取 Readmission 标签"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return None
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)

        outcome = ep.get('labels', {}).get('outcome', {})
        if outcome.get('mortality', 0) == 1:
            return None

        readmission = outcome.get('readmission_30d')
        if readmission is None:
            return None
        
        return int(readmission)
    except:
        return None


def extract_episode_features(stay_id):
    """从增强 Episode 提取特征"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return None
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)
        
        features = {}
        
        vitals = ep.get('timeseries', {}).get('vitals', [])
        vital_cols = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
        for col in vital_cols:
            values = [v.get(col) for v in vitals if v.get(col) is not None]
            n_values = len(values)
            features[f'{col}_n'] = n_values
            features[f'{col}_missing'] = 0 if n_values > 0 else 1
            if values:
                features[f'{col}_mean'] = np.mean(values)
                features[f'{col}_std'] = np.std(values)
                features[f'{col}_min'] = np.min(values)
                features[f'{col}_max'] = np.max(values)
            else:
                features[f'{col}_mean'] = np.nan
                features[f'{col}_std'] = np.nan
                features[f'{col}_min'] = np.nan
                features[f'{col}_max'] = np.nan
        
        reasoning = ep.get('reasoning', {})
        features['n_supportive'] = reasoning.get('n_supportive', 0)
        features['n_contradictory'] = reasoning.get('n_contradictory', 0)
        features['n_unrelated'] = reasoning.get('n_unrelated', 0)
        features['n_patterns'] = reasoning.get('n_patterns_detected', 0)
        
        conditions = ep.get('conditions', [])
        features['n_conditions'] = len(conditions)
        features['has_sepsis'] = 1 if 'sepsis' in conditions else 0
        features['has_aki'] = 1 if 'aki' in conditions else 0
        features['has_ards'] = 1 if 'ards' in conditions else 0
        
        return features
    except:
        return None


def prepare_features(cohort, embeddings, emb_dict, concepts_dict, medcat_dict, feature_set='all'):
    """准备不同特征集"""
    features_list = []
    labels = []
    groups = []
    stay_ids = []
    
    vital_cols = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
    ts_feature_names = []
    for col in vital_cols:
        ts_feature_names.extend([
            f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max',
            f'{col}_n', f'{col}_missing'
        ])
    
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="准备特征"):
        stay_id = row['stay_id']
        subject_id = row.get('subject_id')
        
        readmission_label = get_readmission_label(stay_id)
        if readmission_label is None:
            continue
        
        ep_features = extract_episode_features(stay_id)
        if ep_features is None:
            continue
        
        final_features = {}
        
        if feature_set in ['timeseries', 'annotation', 'bert', 'concept', 'all']:
            for col in ts_feature_names:
                final_features[col] = ep_features.get(col, 0)
        
        if feature_set in ['annotation', 'bert', 'concept', 'all']:
            final_features['n_supportive'] = ep_features.get('n_supportive', 0)
            final_features['n_contradictory'] = ep_features.get('n_contradictory', 0)
            final_features['n_unrelated'] = ep_features.get('n_unrelated', 0)
            final_features['n_patterns'] = ep_features.get('n_patterns', 0)
            final_features['n_conditions'] = ep_features.get('n_conditions', 0)
            final_features['has_sepsis'] = ep_features.get('has_sepsis', 0)
            final_features['has_aki'] = ep_features.get('has_aki', 0)
            final_features['has_ards'] = ep_features.get('has_ards', 0)
        
        if feature_set in ['bert', 'all']:
            if stay_id in emb_dict:
                emb_idx = emb_dict[stay_id]
                emb = embeddings[emb_idx]
                for i in range(50):
                    final_features[f'bert_{i}'] = emb[i]
            else:
                continue
        
        if feature_set in ['concept', 'all']:
            if stay_id in concepts_dict:
                concept_row = concepts_dict[stay_id]
                for k, v in concept_row.items():
                    final_features[f'concept_{k}'] = v
            else:
                continue

            if medcat_dict is not None:
                medcat_row = medcat_dict.get(stay_id, {})
                for k, v in medcat_row.items():
                    final_features[k] = v
        
        if final_features:
            features_list.append(final_features)
            labels.append(readmission_label)
            groups.append(subject_id)
            stay_ids.append(stay_id)

    X = pd.DataFrame(features_list).fillna(0)
    y = pd.Series(labels)
    groups = pd.Series(groups)

    # 显式过滤 NaN 标签（防止死亡样本被当成 0）
    valid_mask = y.notna().values
    X = X.loc[valid_mask].reset_index(drop=True)
    y = y.loc[valid_mask].astype(int).values
    groups = groups.loc[valid_mask].values
    stay_ids = np.asarray(stay_ids)[valid_mask]

    return X, y, groups, stay_ids


def _get_cv(n_splits):
    if HAS_STRATIFIED_GROUP_KFOLD:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return GroupKFold(n_splits=n_splits)


def train_and_evaluate(X, y, groups, stay_ids, model_name='LR', n_splits=5):
    """Holdout test + group-aware CV（优先 StratifiedGroupKFold）"""
    if USE_HOLDOUT_TEST:
        train_val_idx, test_idx, fold_ids, split_info = resolve_predefined_partition(stay_ids)
        fold_train_val = fold_ids[train_val_idx]
    else:
        split_info = {"source": "runtime_groupkfold"}
        train_val_idx = np.arange(len(y))
        test_idx = np.array([], dtype=int)

    aurocs, auprcs = [], []
    if USE_HOLDOUT_TEST:
        fold_iter = []
        for fold in range(1, n_splits + 1):
            tr_idx = train_val_idx[fold_train_val != fold]
            val_idx = train_val_idx[fold_train_val == fold]
            if len(tr_idx) == 0 or len(val_idx) == 0:
                continue
            fold_iter.append((tr_idx, val_idx))
    else:
        cv = _get_cv(n_splits)
        fold_iter = []
        for tr_rel, val_rel in cv.split(X.iloc[train_val_idx], y[train_val_idx], groups[train_val_idx]):
            tr_idx = train_val_idx[tr_rel]
            val_idx = train_val_idx[val_rel]
            fold_iter.append((tr_idx, val_idx))

    for tr_idx, val_idx in fold_iter:

        X_train, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_train, y_val = y[tr_idx], y[val_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        if model_name == 'LR':
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
        
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_val_scaled)[:, 1]
        
        aurocs.append(roc_auc_score(y_val, proba))
        auprcs.append(average_precision_score(y_val, proba))

    test_auroc = None
    test_auprc = None
    if len(test_idx) > 0:
        X_train, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train, y_test = y[train_val_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        if model_name == 'LR':
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)

        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        test_auroc = float(roc_auc_score(y_test, proba))
        test_auprc = float(average_precision_score(y_test, proba))

    return (
        float(np.mean(aurocs)),
        float(np.std(aurocs)),
        float(np.mean(auprcs)),
        float(np.std(auprcs)),
        test_auroc,
        test_auprc,
        int(len(test_idx)),
        split_info["source"],
    )


def main():
    print("=" * 60)
    print("30-Day Readmission 任务训练")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    cohort, embeddings, emb_dict, concepts_dict, medcat_dict = load_all_data()
    
    feature_sets = [
        ('timeseries', '仅时序'),
        ('annotation', '时序+标注'),
        ('bert', '时序+标注+BERT'),
        ('concept', '时序+标注+概念'),
        ('all', '全特征融合')
    ]
    
    results = []
    
    for feature_set, description in feature_sets:
        print(f"\n{'='*40}")
        print(f"特征集: {description}")
        print('='*40)
        
        X, y, groups, stay_ids = prepare_features(cohort, embeddings, emb_dict, concepts_dict, medcat_dict, feature_set)
        print(f"  特征数: {X.shape[1]}")
        print(f"  样本数: {len(y):,}")
        print(f"  Readmission 率: {y.mean()*100:.1f}%")

        if not HAS_STRATIFIED_GROUP_KFOLD:
            print("  [WARN] StratifiedGroupKFold 不可用，已降级为 GroupKFold（不分层）")

        lr_auroc, lr_std, lr_auprc, lr_auprc_std, lr_test_auroc, lr_test_auprc, n_test, split_source = train_and_evaluate(
            X, y, groups, stay_ids, 'LR', n_splits=N_FOLDS
        )
        print(f"  LR CV AUROC: {lr_auroc:.4f} ± {lr_std:.4f}")
        if lr_test_auroc is not None:
            print(f"  LR TEST AUROC: {lr_test_auroc:.4f} (n_test={n_test})")
        
        results.append({
            'task': 'readmission_30d',
            'feature_set': feature_set,
            'description': description,
            'model': 'LogisticRegression',
            'cv_auroc_mean': lr_auroc,
            'cv_auroc_std': lr_std,
            'cv_auprc_mean': lr_auprc,
            'cv_auprc_std': lr_auprc_std,
            'test_auroc': lr_test_auroc,
            'test_auprc': lr_test_auprc,
            'n_test': n_test,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'positive_rate': y.mean(),
            'split_source': split_source
        })
        
        gb_auroc, gb_std, gb_auprc, gb_auprc_std, gb_test_auroc, gb_test_auprc, n_test, split_source = train_and_evaluate(
            X, y, groups, stay_ids, 'GB', n_splits=N_FOLDS
        )
        print(f"  GB CV AUROC: {gb_auroc:.4f} ± {gb_std:.4f}")
        if gb_test_auroc is not None:
            print(f"  GB TEST AUROC: {gb_test_auroc:.4f} (n_test={n_test})")
        
        results.append({
            'task': 'readmission_30d',
            'feature_set': feature_set,
            'description': description,
            'model': 'GradientBoosting',
            'cv_auroc_mean': gb_auroc,
            'cv_auroc_std': gb_std,
            'cv_auprc_mean': gb_auprc,
            'cv_auprc_std': gb_auprc_std,
            'test_auroc': gb_test_auroc,
            'test_auprc': gb_test_auprc,
            'n_test': n_test,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'positive_rate': y.mean(),
            'split_source': split_source
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'readmission_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Readmission 任务结果汇总")
    print("=" * 60)
    cols = ['feature_set', 'model', 'cv_auroc_mean', 'cv_auroc_std', 'test_auroc', 'test_auprc']
    show_cols = [c for c in cols if c in results_df.columns]
    print(results_df[show_cols].to_string(index=False))
    
    print(f"\n结果保存到: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
