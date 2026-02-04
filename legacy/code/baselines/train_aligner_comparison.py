"""
不同时间窗口 Aligner 对比
测试 ±6h, ±12h, ±24h 时间窗口对模型性能的影响

根据作业要求：定义多个标准对齐窗口并进行对比
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMESERIES_FILE, COHORT_FILE,
    RESULTS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE, HIDDEN_DIM, BATCH_SIZE
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'aligner_comparison'

# 不同时间窗口
TIME_WINDOWS = [6, 12, 24]  # 小时


class SimpleGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))


class TimeWindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.y)


def load_data_with_window(window_hours: int):
    """加载指定时间窗口的数据"""
    print(f"\n加载数据 (窗口: ±{window_hours}h)...")
    
    # 加载 cohort
    cohort = pd.read_csv(COHORT_FILE)
    cohort['stay_id'] = cohort['stay_id'].astype(int)
    
    # 加载时序数据
    ts_df = pd.read_csv(TIMESERIES_FILE)
    ts_df['stay_id'] = pd.to_numeric(ts_df['stay_id'], errors='coerce').fillna(-1).astype(int)
    
    # 过滤时间窗口
    # 假设 hour 列表示从入院开始的小时数
    ts_df = ts_df[ts_df['hour'] < window_hours]
    
    feature_cols = [c for c in ts_df.columns if c not in ['stay_id', 'hour', 'subject_id', 'hadm_id', 'intime']]
    
    # 准备数据
    df_keys = cohort[['stay_id', 'subject_id']].copy()
    df_keys['label'] = cohort['label_mortality']
    df_clean = df_keys.dropna().reset_index(drop=True)
    
    valid_stay_ids = df_clean['stay_id'].unique()
    ts_df = ts_df[ts_df['stay_id'].isin(valid_stay_ids)]
    
    N = len(df_clean)
    T = window_hours
    D = len(feature_cols)
    
    id_map = {sid: i for i, sid in enumerate(df_clean['stay_id'])}
    
    # 构建张量
    mux = pd.MultiIndex.from_product([df_clean['stay_id'], range(T)], names=['stay_id', 'hour'])
    ts_df = ts_df.set_index(['stay_id', 'hour'])
    ts_df = ts_df[~ts_df.index.duplicated(keep='first')]
    ts_df = ts_df.reindex(mux)
    
    X = np.zeros((N, T, D))
    X[:, :, :] = ts_df[feature_cols].values.reshape(N, T, D)
    
    # 前向填充
    mask = np.isnan(X)
    idx_ffill = np.where(~mask, np.arange(mask.shape[1])[None, :, None], 0)
    np.maximum.accumulate(idx_ffill, axis=1, out=idx_ffill)
    X = X[np.arange(N)[:, None, None], idx_ffill, np.arange(D)[None, None, :]]
    X = np.nan_to_num(X, nan=0.0)
    
    y = df_clean['label'].values
    groups = df_clean['subject_id'].values
    
    print(f"   样本: {N}, 时间步: {T}, 特征: {D}")
    
    return X, y, groups, D


def train_with_window(X, y, groups, input_dim, window_hours):
    """训练指定时间窗口的模型"""
    print(f"\n训练模型 (窗口: ±{window_hours}h)...")
    
    device = torch.device('cpu')
    
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
        
        # 标准化
        N_tr, T, D = X_tr.shape
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr.reshape(-1, D)).reshape(N_tr, T, D)
        X_val = scaler.transform(X_val.reshape(-1, D)).reshape(-1, T, D)
        
        train_loader = DataLoader(TimeWindowDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TimeWindowDataset(X_val, y_val), batch_size=BATCH_SIZE)
        
        model = SimpleGRU(input_dim, HIDDEN_DIM).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        best_val_auc = 0
        patience = 5
        no_improve = 0
        
        for epoch in range(30):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx).squeeze(), by)
                loss.backward()
                optimizer.step()
            
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx, by in val_loader:
                    preds.extend(model(bx).squeeze().numpy())
                    targets.extend(by.numpy())
            
            val_auc = roc_auc_score(targets, preds)
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        fold_results.append(best_val_auc)
        print(f"   Fold {fold+1}: AUROC={best_val_auc:.4f}")
    
    # 测试
    N_tv, T, D = X_tv.shape
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv.reshape(-1, D)).reshape(N_tv, T, D)
    X_test_s = scaler.transform(X_test.reshape(-1, D)).reshape(-1, T, D)
    
    train_loader = DataLoader(TimeWindowDataset(X_tv_s, y_tv), batch_size=BATCH_SIZE, shuffle=True)
    
    model = SimpleGRU(input_dim, HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    for epoch in range(20):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx).squeeze(), by)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.FloatTensor(X_test_s)).squeeze().numpy()
    
    test_auroc = roc_auc_score(y_test, test_pred)
    
    print(f"\n   CV AUROC: {np.mean(fold_results):.4f} ± {np.std(fold_results):.4f}")
    print(f"   Test AUROC: {test_auroc:.4f}")
    
    return np.mean(fold_results), np.std(fold_results), test_auroc


def main():
    print("=" * 60)
    print("不同时间窗口 Aligner 对比")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for window in TIME_WINDOWS:
        print(f"\n{'='*60}")
        print(f"时间窗口: ±{window}h")
        print(f"{'='*60}")
        
        X, y, groups, input_dim = load_data_with_window(window)
        cv_auroc, cv_std, test_auroc = train_with_window(X, y, groups, input_dim, window)
        
        results.append({
            'window': f'±{window}h',
            'cv_auroc': cv_auroc,
            'cv_std': cv_std,
            'test_auroc': test_auroc
        })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'aligner_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("最终结果对比")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 找最佳窗口
    best_idx = results_df['test_auroc'].idxmax()
    print(f"\n最佳窗口: {results_df.loc[best_idx, 'window']} (Test AUROC: {results_df.loc[best_idx, 'test_auroc']:.4f})")


if __name__ == "__main__":
    main()
