"""
Temporal GRU Baselines
在多时间窗口上运行GRU模型
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import json
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_WINDOWS_DIR, COHORT_FILE, LLM_FEATURES_FILE, NOTE_TIME_FILE,
    BENCHMARK_RESULTS_DIR, WINDOWS, TASKS, COHORTS, N_FOLDS, RANDOM_STATE,
    HIDDEN_DIM, NUM_LAYERS, BATCH_SIZE, EPOCHS, LR, LLM_COLS
)

# ==========================================
# 配置
# ==========================================
DATA_DIR = DATA_WINDOWS_DIR
OUTPUT_DIR = BENCHMARK_RESULTS_DIR

# ==========================================
# 模型定义
# ==========================================
class ClinicalGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __getitem__(self, i):
        return self.X[i], self.y[i]
    
    def __len__(self):
        return len(self.y)

# ==========================================
# 数据加载
# ==========================================
class TemporalDataLoader:
    def __init__(self):
        self.cohort = pd.read_csv(COHORT_FILE)
        self.cohort['stay_id'] = self.cohort['stay_id'].astype(int)
        
        # 加载LLM特征和时间信息
        try:
            self.llm_df = pd.read_csv(LLM_FEATURES_FILE)
            self.llm_df['stay_id'] = pd.to_numeric(self.llm_df['stay_id'], errors='coerce').fillna(-1).astype(int)
            
            self.note_time_df = pd.read_csv(NOTE_TIME_FILE)
            self.note_time_df['stay_id'] = pd.to_numeric(self.note_time_df['stay_id'], errors='coerce').fillna(-1).astype(int)
            self.note_time_df['hour_offset'] = pd.to_numeric(self.note_time_df['hour_offset'], errors='coerce')
            self.note_time_df = self.note_time_df[
                (self.note_time_df['hour_offset'] >= 0) & (self.note_time_df['hour_offset'] < 24)
            ]
            
            # 合并获取hour_offset
            self.llm_with_time = self.note_time_df.merge(self.llm_df, on='stay_id', how='inner')
            print(f"Loaded LLM features with time info: {len(self.llm_with_time)} records")
        except Exception as e:
            print(f"Warning: Could not load LLM features: {e}")
            self.llm_with_time = None
    
    def load_temporal_tensor(self, window):
        """加载时序张量"""
        path = os.path.join(DATA_DIR, f'window_{window}', 'features_temporal.npy')
        X = np.load(path)

        mask_path = os.path.join(DATA_DIR, f'window_{window}', 'features_temporal_mask.npy')
        if os.path.exists(mask_path):
            X_mask = np.load(mask_path)
        else:
            raise FileNotFoundError(
                f"Missing temporal mask for {window}. Please regenerate data_windows."
            )
        
        # 加载元数据获取stay_ids顺序
        meta_path = os.path.join(DATA_DIR, f'window_{window}', 'metadata.json')
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        return X, X_mask, np.array(metadata['stay_ids']), metadata['feature_names']
    
    def inject_llm_features(self, X, X_mask, stay_ids, feature_names, window_hours):
        """将LLM特征注入到时序张量中"""
        if self.llm_with_time is None:
            return X, X_mask, feature_names
        
        N, T, D = X.shape
        D_llm = len(LLM_COLS)
        
        # 创建扩展张量
        X_new = np.zeros((N, T, D + D_llm))
        X_new[:, :, :D] = X
        mask_new = np.zeros((N, T, D + D_llm), dtype=np.float32)
        mask_new[:, :, :D] = X_mask
        
        # 创建ID映射
        id_map = {sid: i for i, sid in enumerate(stay_ids)}
        
        # 注入LLM特征
        for row in self.llm_with_time.itertuples():
            if row.stay_id in id_map:
                idx = id_map[row.stay_id]
                h = int(row.hour_offset) if hasattr(row, 'hour_offset') else 0
                if h >= window_hours:
                    continue
                
                if 0 <= h < T:
                    feats = []
                    for col in LLM_COLS:
                        val = getattr(row, col, 0) if hasattr(row, col) else 0
                        val = 0 if val == -1 else val  # -1 -> 0
                        feats.append(val)
                    
                    # 从该时间点开始填充到末尾
                    X_new[idx, h:, D:] = feats
                    mask_new[idx, h:, D:] = 1.0
        
        new_feature_names = feature_names + LLM_COLS
        return X_new, mask_new, new_feature_names
    
    def get_task_label(self, task):
        label_map = {
            'mortality': 'label_mortality',
            'prolonged_los': 'prolonged_los_7d',
        }
        return self.cohort[['stay_id', 'subject_id', label_map[task]]].rename(
            columns={label_map[task]: 'label'}
        )
    
    def filter_cohort(self, cohort_name):
        if cohort_name == 'all':
            return self.cohort['stay_id'].values
        elif cohort_name == 'sepsis':
            return self.cohort[self.cohort['has_sepsis_final'] == 1]['stay_id'].values
        elif cohort_name == 'aki':
            return self.cohort[self.cohort['has_aki_final'] == 1]['stay_id'].values
        else:
            raise ValueError(f"Unknown cohort: {cohort_name}")

# ==========================================
# 训练函数
# ==========================================
def train_gru(X_values, X_mask, y, groups, device, epochs=EPOCHS):
    """使用GroupKFold训练GRU"""
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    aurocs = []
    auprcs = []
    
    if X_mask is None:
        X_mask = np.zeros_like(X_values, dtype=np.float32)

    input_dim = X_values.shape[2] + X_mask.shape[2]
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_values, y, groups)):
        X_train, X_val = X_values[train_idx], X_values[val_idx]
        X_train_mask, X_val_mask = X_mask[train_idx], X_mask[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # 标准化 (reshape -> scale -> reshape back)
        N_train, T, D = X_train.shape
        N_val = X_val.shape[0]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, D)).reshape(N_train, T, D)
        X_val = scaler.transform(X_val.reshape(-1, D)).reshape(N_val, T, D)

        # 处理NaN
        X_train = np.nan_to_num(X_train, nan=0)
        X_val = np.nan_to_num(X_val, nan=0)

        # 拼接缺失掩码
        X_train = np.concatenate([X_train, X_train_mask], axis=2)
        X_val = np.concatenate([X_val, X_val_mask], axis=2)
        
        # DataLoader
        train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), 
                                 batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), 
                               batch_size=BATCH_SIZE)
        
        # 模型
        model = ClinicalGRU(input_dim, HIDDEN_DIM, NUM_LAYERS).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()
        
        # 训练
        model.train()
        for epoch in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx).squeeze(), by)
                loss.backward()
                optimizer.step()
        
        # 验证
        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                preds.extend(model(bx).squeeze().cpu().numpy())
                targets.extend(by.numpy())
        
        preds = np.array(preds)
        targets = np.array(targets)
        
        try:
            aurocs.append(roc_auc_score(targets, preds))
            auprcs.append(average_precision_score(targets, preds))
        except:
            aurocs.append(0.5)
            auprcs.append(targets.mean())
    
    return {
        'auroc_mean': np.mean(aurocs),
        'auroc_std': np.std(aurocs),
        'auprc_mean': np.mean(auprcs),
        'auprc_std': np.std(auprcs)
    }

# ==========================================
# 主流程
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() 
                         else 'mps' if torch.backends.mps.is_available() 
                         else 'cpu')
    print(f"Device: {device}")
    
    loader = TemporalDataLoader()
    all_results = []
    
    print("\nStarting Temporal GRU Experiments")
    print("=" * 70)
    
    for window in WINDOWS:
        print(f"\nWindow: {window}")
        
        # 加载时序数据
        X_base, X_base_mask, stay_ids, feature_names = loader.load_temporal_tensor(window)
        window_hours = int(window.replace('h', ''))
        
        # 注入LLM特征
        X_with_llm, X_with_llm_mask, feature_names_new = loader.inject_llm_features(
            X_base, X_base_mask, stay_ids, feature_names, window_hours
        )
        
        print(f"   Base tensor: {X_base.shape}")
        print(f"   With LLM: {X_with_llm.shape}")
        
        for task in TASKS:
            labels_df = loader.get_task_label(task)
            
            for cohort_name in COHORTS:
                cohort_ids = loader.filter_cohort(cohort_name)
                
                # 找到目标患者的索引
                mask = np.isin(stay_ids, cohort_ids)
                X_cohort_base = X_base[mask]
                X_cohort_base_mask = X_base_mask[mask]
                X_cohort_llm = X_with_llm[mask]
                X_cohort_llm_mask = X_with_llm_mask[mask]
                cohort_stay_ids = stay_ids[mask]
                
                # 获取标签和groups
                labels_filtered = labels_df[labels_df['stay_id'].isin(cohort_stay_ids)]
                labels_filtered = labels_filtered.set_index('stay_id').loc[cohort_stay_ids].reset_index()
                
                y = labels_filtered['label'].values
                groups = labels_filtered['subject_id'].values
                
                if len(y) < 100 or y.sum() < 10:
                    print(f"   Skipping {cohort_name}/{task}: too few samples")
                    continue
                
                print(f"\n[{window}|{task}|{cohort_name}] n={len(y)}, pos={y.sum()}")
                
                # ===== GRU (Tabular only) =====
                results_tab = train_gru(X_cohort_base, X_cohort_base_mask, y, groups, device)
                print(f"   GRU (Tab):     {results_tab['auroc_mean']:.4f} ± {results_tab['auroc_std']:.4f}")
                all_results.append({
                    'window': window, 'task': task, 'cohort': cohort_name,
                    'model': 'GRU (Tabular)', **results_tab
                })
                
                # ===== GRU (Tabular + LLM) =====
                results_fused = train_gru(X_cohort_llm, X_cohort_llm_mask, y, groups, device)
                print(f"   GRU (Tab+LLM): {results_fused['auroc_mean']:.4f} ± {results_fused['auroc_std']:.4f}")
                all_results.append({
                    'window': window, 'task': task, 'cohort': cohort_name,
                    'model': 'GRU (Tab+LLM)', **results_fused
                })
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, 'temporal_gru_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    
    # ==========================================
    # 汇总
    # ==========================================
    print("\n" + "=" * 70)
    print("TEMPORAL GRU SUMMARY")
    print("=" * 70)
    
    for task in TASKS:
        print(f"\n[Task: {task}]")
        task_df = results_df[results_df['task'] == task]
        
        pivot = task_df.pivot_table(
            index=['cohort', 'model'],
            columns='window',
            values='auroc_mean',
            aggfunc='first'
        )
        print(pivot.round(4).to_string())
    
    print("\nTemporal GRU Experiments Complete!")

if __name__ == "__main__":
    main()
