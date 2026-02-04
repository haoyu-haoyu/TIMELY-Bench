"""
Calibration 评估 (ECE / Hosmer-Lemeshow)
计算所有模型的校准指标

ECE: Expected Calibration Error
HL: Hosmer-Lemeshow 统计量
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMESERIES_FILE, COHORT_FILE, RESULTS_DIR,
    RANDOM_STATE, TEST_SIZE, BATCH_SIZE, HIDDEN_DIM
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'calibration'


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """计算 Expected Calibration Error (ECE)"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = y_prob[in_bin].mean()
            avg_accuracy = y_true[in_bin].mean()
            ece += np.abs(avg_confidence - avg_accuracy) * prop_in_bin
    
    return ece


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """计算 Hosmer-Lemeshow 统计量和 p-value"""
    # 按预测概率分组
    sorted_indices = np.argsort(y_prob)
    y_true_sorted = y_true[sorted_indices]
    y_prob_sorted = y_prob[sorted_indices]
    
    group_size = len(y_true) // n_groups
    chi2 = 0.0
    
    for i in range(n_groups):
        start = i * group_size
        end = (i + 1) * group_size if i < n_groups - 1 else len(y_true)
        
        observed = y_true_sorted[start:end].sum()
        expected = y_prob_sorted[start:end].sum()
        n_group = end - start
        
        if expected > 0 and n_group - expected > 0:
            chi2 += ((observed - expected) ** 2) / expected
            chi2 += (((n_group - observed) - (n_group - expected)) ** 2) / (n_group - expected)
    
    # p-value (df = n_groups - 2)
    p_value = 1 - stats.chi2.cdf(chi2, n_groups - 2)
    
    return chi2, p_value


def load_data():
    """加载数据"""
    print("加载数据...")
    
    # Cohort
    cohort = pd.read_csv(COHORT_FILE)
    cohort['stay_id'] = cohort['stay_id'].astype(int)
    
    # Timeseries
    ts_df = pd.read_csv(TIMESERIES_FILE)
    ts_df['stay_id'] = pd.to_numeric(ts_df['stay_id'], errors='coerce').fillna(-1).astype(int)
    
    # Episode 标注特征
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    annotations = []
    
    for ep_file in tqdm(episode_files, desc="Loading annotations"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            stay_id = ep.get('stay_id')
            reasoning = ep.get('reasoning', {})
            
            annotations.append({
                'stay_id': stay_id,
                'n_supportive': reasoning.get('n_supportive', 0),
                'n_contradictory': reasoning.get('n_contradictory', 0)
            })
        except:
            pass
    
    annot_df = pd.DataFrame(annotations)
    
    return cohort, ts_df, annot_df


def train_and_calibrate_model(model_name, X_train, X_test, y_train, y_test):
    """训练模型并返回校准指标"""
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                  random_state=RANDOM_STATE, use_label_encoder=False,
                                  eval_metric='logloss', n_jobs=-1)
    elif model_name == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    auroc = roc_auc_score(y_test, y_prob)
    ece = expected_calibration_error(y_test, y_prob)
    hl_chi2, hl_p = hosmer_lemeshow_test(y_test, y_prob)
    
    return {
        'auroc': auroc,
        'ece': ece,
        'hl_chi2': hl_chi2,
        'hl_p_value': hl_p
    }


def main():
    print("=" * 60)
    print("Calibration 评估 (ECE / Hosmer-Lemeshow)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    cohort, ts_df, annot_df = load_data()
    
    # 准备数据
    df_keys = cohort[['stay_id', 'subject_id', 'label_mortality']].dropna()
    df_keys = df_keys.merge(annot_df, on='stay_id', how='left').fillna(0)
    
    valid_stay_ids = df_keys['stay_id'].unique()
    ts_df = ts_df[ts_df['stay_id'].isin(valid_stay_ids)]
    
    # 提取时序统计特征
    print("\n提取时序统计特征...")
    vitals_cols = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2']
    ts_features = []
    
    for stay_id in tqdm(valid_stay_ids, desc="Extracting"):
        stay_data = ts_df[ts_df['stay_id'] == stay_id]
        feat = {'stay_id': stay_id}
        
        for col in vitals_cols:
            if col in stay_data.columns:
                values = pd.to_numeric(stay_data[col], errors='coerce').dropna()
                if len(values) > 0:
                    feat[f'{col}_mean'] = values.mean()
                    feat[f'{col}_std'] = values.std() if len(values) > 1 else 0
        
        ts_features.append(feat)
    
    ts_feat_df = pd.DataFrame(ts_features)
    
    # 合并
    df = df_keys.merge(ts_feat_df, on='stay_id', how='inner').dropna()
    print(f"最终样本数: {len(df):,}")
    
    # 准备特征
    ts_cols = [c for c in ts_feat_df.columns if c != 'stay_id']
    annot_cols = ['n_supportive', 'n_contradictory']
    
    # 分割测试集
    groups = df['subject_id'].values
    y = df['label_mortality'].values
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, y, groups=groups))
    
    y_train, y_test = y[train_idx], y[test_idx]
    
    results = []
    
    # 1. Tabular (时序统计)
    print("\n=== Tabular Only ===")
    X_ts = df[ts_cols].values
    X_ts = np.nan_to_num(X_ts, nan=0.0)
    
    for model_name in ['XGBoost', 'LogisticRegression']:
        res = train_and_calibrate_model(model_name, X_ts[train_idx], X_ts[test_idx], y_train, y_test)
        res['model'] = f'Tabular_{model_name}'
        results.append(res)
        print(f"   {model_name}: AUROC={res['auroc']:.4f}, ECE={res['ece']:.4f}, HL p={res['hl_p_value']:.4f}")
    
    # 2. Text-only (标注特征)
    print("\n=== Text-only ===")
    X_annot = df[annot_cols].values
    
    for model_name in ['XGBoost', 'LogisticRegression']:
        res = train_and_calibrate_model(model_name, X_annot[train_idx], X_annot[test_idx], y_train, y_test)
        res['model'] = f'TextOnly_{model_name}'
        results.append(res)
        print(f"   {model_name}: AUROC={res['auroc']:.4f}, ECE={res['ece']:.4f}, HL p={res['hl_p_value']:.4f}")
    
    # 3. Early Fusion
    print("\n=== Early Fusion ===")
    X_fusion = np.concatenate([X_ts, X_annot], axis=1)
    
    for model_name in ['XGBoost', 'LogisticRegression']:
        res = train_and_calibrate_model(model_name, X_fusion[train_idx], X_fusion[test_idx], y_train, y_test)
        res['model'] = f'EarlyFusion_{model_name}'
        results.append(res)
        print(f"   {model_name}: AUROC={res['auroc']:.4f}, ECE={res['ece']:.4f}, HL p={res['hl_p_value']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'calibration_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 解读
    print("\n校准解读:")
    print("  - ECE 越低越好 (完美校准 = 0)")
    print("  - HL p-value > 0.05 表示校准良好")


if __name__ == "__main__":
    main()
