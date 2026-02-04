"""
增强版模型训练
结合 BERT 嵌入 + 概念特征 + 时序特征（含 MedCAT）

对比：
1. 基础时序模型
2. + 标注特征
3. + BERT 嵌入
4. + 概念特征
5. 全特征融合
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
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
RESULTS_DIR = BASE_DIR / 'results' / 'enhanced_training'
RANDOM_STATE = 42


def load_all_data():
    """加载所有数据"""
    print("加载数据...")
    
    # 加载队列
    cohort = pd.read_csv(COHORTS_FILE)
    print(f"  队列: {len(cohort):,} 样本")
    
    # 加载 BERT 嵌入
    embeddings = np.load(EMBEDDINGS_FILE)
    emb_ids = pd.read_csv(Path(EMBEDDINGS_FILE).parent / 'embedding_stay_ids.csv')
    emb_dict = dict(zip(emb_ids['stay_id'], range(len(emb_ids))))
    print(f"  BERT 嵌入: {embeddings.shape}")
    
    # 加载概念特征
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
    
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc="准备特征"):
        stay_id = row['stay_id']
        
        # 基础 Episode 特征
        ep_features = extract_episode_features(stay_id)
        if ep_features is None:
            continue
        
        final_features = {}
        
        # 时序特征（基础）
        if feature_set in ['timeseries', 'annotation', 'bert', 'concept', 'all']:
            ts_cols = [
                k for k in ep_features.keys()
                if any(suffix in k for suffix in ['_mean', '_std', '_min', '_max', '_n', '_missing'])
            ]
            for col in ts_cols:
                final_features[col] = ep_features.get(col, 0)
        
        # 标注特征
        if feature_set in ['annotation', 'bert', 'concept', 'all']:
            final_features['n_supportive'] = ep_features.get('n_supportive', 0)
            final_features['n_contradictory'] = ep_features.get('n_contradictory', 0)
            final_features['n_unrelated'] = ep_features.get('n_unrelated', 0)
            final_features['n_patterns'] = ep_features.get('n_patterns', 0)
        
        # BERT 嵌入（降维到前 50 维）
        if feature_set in ['bert', 'all'] and stay_id in emb_dict:
            emb_idx = emb_dict[stay_id]
            emb = embeddings[emb_idx]
            for i in range(50):
                final_features[f'bert_{i}'] = emb[i]
        
        # 概念特征
        if feature_set in ['concept', 'all'] and stay_id in concepts_dict:
            concept_row = concepts_dict[stay_id]
            for k, v in concept_row.items():
                final_features[f'concept_{k}'] = v

            if medcat_dict is not None:
                medcat_row = medcat_dict.get(stay_id, {})
                for k, v in medcat_row.items():
                    final_features[k] = v
        
        if final_features:
            features_list.append(final_features)
            labels.append(row.get('label_mortality', 0))
            stay_ids.append(stay_id)
    
    X = pd.DataFrame(features_list).fillna(0)
    y = np.array(labels)
    
    return X, y


def train_and_evaluate(X, y, model_name='LR'):
    """训练和评估模型"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_name == 'LR':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=RANDOM_STATE)
    
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)[:, 1]
    
    auroc = roc_auc_score(y_test, proba)
    auprc = average_precision_score(y_test, proba)
    
    return auroc, auprc, len(y_train), len(y_test)


def main():
    print("=" * 60)
    print("增强版模型训练")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    cohort, embeddings, emb_dict, concepts_dict, medcat_dict = load_all_data()
    
    # 特征集对比实验
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
        print(f"  死亡率: {y.mean()*100:.1f}%")
        
        # 逻辑回归
        lr_auroc, lr_auprc, n_train, n_test = train_and_evaluate(X, y, 'LR')
        print(f"  LR AUROC: {lr_auroc:.4f}, AUPRC: {lr_auprc:.4f}")
        
        results.append({
            'feature_set': feature_set,
            'description': description,
            'model': 'LogisticRegression',
            'auroc': lr_auroc,
            'auprc': lr_auprc,
            'n_features': X.shape[1],
            'n_samples': len(y)
        })
        
        # 梯度提升
        gb_auroc, gb_auprc, _, _ = train_and_evaluate(X, y, 'GB')
        print(f"  GB AUROC: {gb_auroc:.4f}, AUPRC: {gb_auprc:.4f}")
        
        results.append({
            'feature_set': feature_set,
            'description': description,
            'model': 'GradientBoosting',
            'auroc': gb_auroc,
            'auprc': gb_auprc,
            'n_features': X.shape[1],
            'n_samples': len(y)
        })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / 'feature_ablation_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 计算性能提升
    print("\n性能提升分析:")
    baseline = results_df[(results_df['feature_set']=='timeseries') & (results_df['model']=='GradientBoosting')]['auroc'].values[0]
    full = results_df[(results_df['feature_set']=='all') & (results_df['model']=='GradientBoosting')]['auroc'].values[0]
    print(f"  基线 AUROC: {baseline:.4f}")
    print(f"  全特征 AUROC: {full:.4f}")
    print(f"  提升: +{(full-baseline)*100:.2f}%")
    
    print(f"\n结果保存到: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
