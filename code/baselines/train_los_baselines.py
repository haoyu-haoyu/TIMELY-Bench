"""
Prolonged LOS (Length of Stay) 任务训练
预测 ICU 住院时间是否超过 7 天

与 Mortality 任务使用相同的特征集：
- 时序特征（vitals）
- 标注特征（patterns）
- BERT 嵌入
- 概念特征（spacy + MedCAT）
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置
BASE_DIR = Path(__file__).resolve().parents[2]
EPISODES_DIR = BASE_DIR / 'episodes' / 'episodes_enhanced'
EMBEDDINGS_FILE = BASE_DIR / 'data' / 'processed' / 'text_embeddings' / 'clinical_bert_embeddings.npy'
CONCEPTS_FILE = BASE_DIR / 'data' / 'processed' / 'text_concepts' / 'spacy_concepts.csv'
MEDCAT_FILE = BASE_DIR / 'data' / 'processed' / 'medcat_full' / 'medcat_features_24h.csv'
COHORTS_FILE = BASE_DIR / 'data' / 'processed' / 'cohorts' / 'cohort_with_conditions.csv'
RESULTS_DIR = BASE_DIR / 'results' / 'los_baselines'
RANDOM_STATE = 42
LOS_THRESHOLD_DAYS = 7  # Prolonged LOS 定义阈值


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


def get_los_label(stay_id):
    """从 Episode 获取 LOS 标签"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return None
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)

        outcome = ep.get('labels', {}).get('outcome', {})
        if outcome.get('mortality', 0) == 1:
            return None

        # 直接使用 prolonged_los 标签 (已在 Episode 中预计算)
        prolonged_los = outcome.get('prolonged_los')
        if prolonged_los is None:
            return None
        
        return int(prolonged_los)
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
        
        # 时序特征
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
        
        # 标注特征
        reasoning = ep.get('reasoning', {})
        features['n_supportive'] = reasoning.get('n_supportive', 0)
        features['n_contradictory'] = reasoning.get('n_contradictory', 0)
        features['n_unrelated'] = reasoning.get('n_unrelated', 0)
        features['n_patterns'] = reasoning.get('n_patterns_detected', 0)
        
        # 条件特征
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
    stay_ids = []
    
    # 预定义时序特征列名 (确保始终有这些特征)
    vital_cols = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
    ts_feature_names = []
    for col in vital_cols:
        ts_feature_names.extend([
            f'{col}_mean', f'{col}_std', f'{col}_min', f'{col}_max',
            f'{col}_n', f'{col}_missing'
        ])
    
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="准备特征"):
        stay_id = row['stay_id']
        
        los_label = get_los_label(stay_id)
        if los_label is None:
            continue
        
        ep_features = extract_episode_features(stay_id)
        if ep_features is None:
            continue
        
        final_features = {}
        
        # 时序特征 - 使用预定义列名，缺失值填 0
        if feature_set in ['timeseries', 'annotation', 'bert', 'concept', 'all']:
            for col in ts_feature_names:
                final_features[col] = ep_features.get(col, 0)
        
        # 标注特征
        if feature_set in ['annotation', 'bert', 'concept', 'all']:
            final_features['n_supportive'] = ep_features.get('n_supportive', 0)
            final_features['n_contradictory'] = ep_features.get('n_contradictory', 0)
            final_features['n_unrelated'] = ep_features.get('n_unrelated', 0)
            final_features['n_patterns'] = ep_features.get('n_patterns', 0)
            final_features['n_conditions'] = ep_features.get('n_conditions', 0)
            final_features['has_sepsis'] = ep_features.get('has_sepsis', 0)
            final_features['has_aki'] = ep_features.get('has_aki', 0)
            final_features['has_ards'] = ep_features.get('has_ards', 0)
        
        # BERT 嵌入 - 只有在 emb_dict 中才添加
        if feature_set in ['bert', 'all']:
            if stay_id in emb_dict:
                emb_idx = emb_dict[stay_id]
                emb = embeddings[emb_idx]
                for i in range(50):
                    final_features[f'bert_{i}'] = emb[i]
            else:
                # 如果没有 BERT 嵌入，跳过此样本（BERT 是必需特征）
                continue
        
        # 概念特征
        if feature_set in ['concept', 'all']:
            if stay_id in concepts_dict:
                concept_row = concepts_dict[stay_id]
                for k, v in concept_row.items():
                    final_features[f'concept_{k}'] = v
            else:
                # 如果没有概念特征，跳过此样本
                continue

            if medcat_dict is not None:
                medcat_row = medcat_dict.get(stay_id, {})
                for k, v in medcat_row.items():
                    final_features[k] = v
        
        if final_features:
            features_list.append(final_features)
            labels.append(los_label)
            stay_ids.append(stay_id)
    
    X = pd.DataFrame(features_list).fillna(0)
    y = np.array(labels)
    
    return X, y


def train_and_evaluate(X, y, model_name='LR', n_splits=5):
    """5-fold CV 训练和评估"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aurocs, auprcs = [], []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_name == 'LR':
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        else:
            model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
        
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        
        aurocs.append(roc_auc_score(y_test, proba))
        auprcs.append(average_precision_score(y_test, proba))
    
    return np.mean(aurocs), np.std(aurocs), np.mean(auprcs), np.std(auprcs)


def main():
    print("=" * 60)
    print("Prolonged LOS 任务训练 (LOS > 7 days)")
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
        
        X, y = prepare_features(cohort, embeddings, emb_dict, concepts_dict, medcat_dict, feature_set)
        print(f"  特征数: {X.shape[1]}")
        print(f"  样本数: {len(y):,}")
        print(f"  Prolonged LOS 率: {y.mean()*100:.1f}%")
        
        lr_auroc, lr_std, lr_auprc, lr_auprc_std = train_and_evaluate(X, y, 'LR')
        print(f"  LR AUROC: {lr_auroc:.4f} ± {lr_std:.4f}")
        
        results.append({
            'task': 'prolonged_los',
            'feature_set': feature_set,
            'description': description,
            'model': 'LogisticRegression',
            'auroc_mean': lr_auroc,
            'auroc_std': lr_std,
            'auprc_mean': lr_auprc,
            'auprc_std': lr_auprc_std,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'positive_rate': y.mean()
        })
        
        gb_auroc, gb_std, gb_auprc, gb_auprc_std = train_and_evaluate(X, y, 'GB')
        print(f"  GB AUROC: {gb_auroc:.4f} ± {gb_std:.4f}")
        
        results.append({
            'task': 'prolonged_los',
            'feature_set': feature_set,
            'description': description,
            'model': 'GradientBoosting',
            'auroc_mean': gb_auroc,
            'auroc_std': gb_std,
            'auprc_mean': gb_auprc,
            'auprc_std': gb_auprc_std,
            'n_features': X.shape[1],
            'n_samples': len(y),
            'positive_rate': y.mean()
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'los_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("LOS 任务结果汇总")
    print("=" * 60)
    print(results_df[['feature_set', 'model', 'auroc_mean', 'auroc_std']].to_string(index=False))
    
    print(f"\n结果保存到: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
