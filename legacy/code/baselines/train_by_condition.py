"""
按疾病分层训练模型
对 Sepsis、AKI、ARDS 分别训练和评估模型

输出:
- results/condition_analysis/sepsis_results.csv
- results/condition_analysis/aki_results.csv  
- results/condition_analysis/ards_results.csv
- results/condition_analysis/comparison_summary.csv
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置
COHORTS_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/cohorts')
EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
EMBEDDINGS_FILE = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/text_embeddings/clinical_bert_embeddings.npy')
CONCEPTS_FILE = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/text_concepts/spacy_concepts.csv')
RESULTS_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/results/condition_analysis')
RANDOM_STATE = 42
N_FOLDS = 5


def load_episode_features(stay_id):
    """从 Episode 文件加载特征"""
    ep_file = EPISODES_DIR / f'TIMELY_v2_{stay_id}.json'
    if not ep_file.exists():
        return None
    
    try:
        with open(ep_file) as f:
            ep = json.load(f)
        
        # 提取时序特征统计
        vitals = ep.get('timeseries', {}).get('vitals', [])
        features = {}
        
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
        
        # 提取标注特征
        reasoning = ep.get('reasoning', {})
        features['n_supportive'] = reasoning.get('n_supportive', 0)
        features['n_contradictory'] = reasoning.get('n_contradictory', 0)
        features['n_unrelated'] = reasoning.get('n_unrelated', 0)
        features['n_patterns'] = reasoning.get('n_patterns_detected', 0)
        
        return features
    except:
        return None


def load_condition_data(condition_name):
    """加载指定疾病的数据"""
    cohort_file = COHORTS_DIR / f'cohort_{condition_name}.csv'
    if not cohort_file.exists():
        print(f"错误：找不到 {cohort_file}")
        return None, None, None
    
    cohort = pd.read_csv(cohort_file)
    print(f"加载 {condition_name}: {len(cohort):,} 样本")
    
    # 加载 BERT 嵌入
    embeddings = np.load(EMBEDDINGS_FILE)
    emb_stay_ids = pd.read_csv(Path(EMBEDDINGS_FILE).parent / 'embedding_stay_ids.csv')
    emb_dict = dict(zip(emb_stay_ids['stay_id'], range(len(emb_stay_ids))))
    
    # 加载概念特征
    concepts = pd.read_csv(CONCEPTS_FILE)
    
    # 合并特征
    features_list = []
    labels = []
    stay_ids = []
    subject_ids = []
    
    for _, row in tqdm(cohort.iterrows(), total=len(cohort), desc=f"加载 {condition_name} 特征"):
        stay_id = row['stay_id']
        
        # Episode 特征
        ep_features = load_episode_features(stay_id)
        if ep_features is None:
            continue
        
        # BERT 嵌入
        if stay_id in emb_dict:
            emb_idx = emb_dict[stay_id]
            emb = embeddings[emb_idx]
            # 使用前 50 维（无 PCA）
            ep_features.update({f'bert_{i}': emb[i] for i in range(min(50, len(emb)))})
        
        # 概念特征
        concept_row = concepts[concepts['stay_id'] == stay_id]
        if len(concept_row) > 0:
            for col in concept_row.columns:
                if col != 'stay_id':
                    ep_features[f'concept_{col}'] = concept_row[col].values[0]
        
        features_list.append(ep_features)
        labels.append(row.get('label_mortality', row.get('mortality', 0)))
        stay_ids.append(stay_id)
        subject_ids.append(row.get('subject_id', stay_id))
    
    if not features_list:
        return None, None, None
    
    X = pd.DataFrame(features_list).fillna(0)
    y = np.array(labels)
    groups = np.array(subject_ids)
    
    return X, y, groups


def train_and_evaluate(X, y, groups, condition_name):
    """训练和评估模型"""
    results = []
    
    gss = GroupShuffleSplit(n_splits=N_FOLDS, test_size=0.2, random_state=RANDOM_STATE)
    
    for fold, (train_idx, test_idx) in enumerate(gss.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 逻辑回归
        lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        lr.fit(X_train_scaled, y_train)
        lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
        
        lr_auroc = roc_auc_score(y_test, lr_proba)
        lr_auprc = average_precision_score(y_test, lr_proba)
        
        results.append({
            'condition': condition_name,
            'fold': fold,
            'model': 'LogisticRegression',
            'auroc': lr_auroc,
            'auprc': lr_auprc,
            'n_train': len(y_train),
            'n_test': len(y_test),
            'mortality_rate': y_test.mean()
        })
        
        # 梯度提升（仅用于样本足够的疾病）
        if len(y_train) >= 500:
            gb = GradientBoostingClassifier(
                n_estimators=100, 
                max_depth=4,
                random_state=RANDOM_STATE
            )
            gb.fit(X_train_scaled, y_train)
            gb_proba = gb.predict_proba(X_test_scaled)[:, 1]
            
            gb_auroc = roc_auc_score(y_test, gb_proba)
            gb_auprc = average_precision_score(y_test, gb_proba)
            
            results.append({
                'condition': condition_name,
                'fold': fold,
                'model': 'GradientBoosting',
                'auroc': gb_auroc,
                'auprc': gb_auprc,
                'n_train': len(y_train),
                'n_test': len(y_test),
                'mortality_rate': y_test.mean()
            })
        
        print(f"  Fold {fold}: LR AUROC={lr_auroc:.4f}")
    
    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("疾病分层训练")
    print("=" * 60)
    
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    conditions = ['sepsis', 'aki', 'ards']
    all_results = []
    
    for condition in conditions:
        print(f"\n{'='*40}")
        print(f"训练 {condition.upper()} 模型")
        print('='*40)
        
        X, y, groups = load_condition_data(condition)
        
        if X is None:
            print(f"跳过 {condition}：数据加载失败")
            continue
        
        print(f"特征数: {X.shape[1]}")
        print(f"样本数: {len(y)}")
        print(f"死亡率: {y.mean()*100:.1f}%")
        
        results = train_and_evaluate(X, y, groups, condition)
        results.to_csv(RESULTS_DIR / f'{condition}_results.csv', index=False)
        all_results.append(results)
        
        # 打印平均结果
        mean_results = results.groupby('model')[['auroc', 'auprc']].mean()
        print(f"\n{condition.upper()} 平均结果:")
        print(mean_results)
    
    # 保存汇总结果
    if all_results:
        summary = pd.concat(all_results)
        summary.to_csv(RESULTS_DIR / 'comparison_summary.csv', index=False)
        
        print("\n" + "=" * 60)
        print("汇总结果")
        print("=" * 60)
        
        summary_table = summary.groupby(['condition', 'model'])[['auroc', 'auprc']].agg(['mean', 'std'])
        print(summary_table)
        
        print(f"\n结果保存到: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
