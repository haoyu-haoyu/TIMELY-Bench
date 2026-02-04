"""
增强版 Temporal GRU - 加入标注特征
在原有时序 + LLM 特征基础上，加入推理标注特征

特征：
1. 时序特征 (25维) - vitals/labs
2. LLM 特征 (5维) - severity, emotion, etc.
3. 标注特征 (4维) - n_supportive, n_contradictory, supportive_ratio, annotation_density [NEW]
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
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    TIMESERIES_FILE, NOTE_TIME_FILE, LLM_FEATURES_FILE, COHORT_FILE,
    RESULTS_DIR, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE, EPOCHS, LR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, LR_SCHEDULER_MIN_LR,
    TEST_SIZE, USE_HOLDOUT_TEST, N_FOLDS, RANDOM_STATE, LLM_COLS
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'enhanced_gru'


def load_annotation_features():
    """从 Episode 加载标注特征"""
    print("加载标注特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
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
    
    df = pd.DataFrame(annotations)
    print(f"   加载 {len(df):,} 个标注特征")
    return df


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_metric, epoch):
        score = val_metric
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        return self.early_stop


class EnhancedGRU(nn.Module):
    """增强版 GRU，包含静态标注特征"""
    def __init__(self, input_dim, hidden_dim, num_layers, static_dim, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        # 融合层：GRU输出 + 静态标注特征
        self.fc = nn.Linear(hidden_dim + static_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_seq, x_static):
        out, _ = self.gru(x_seq)
        last_hidden = out[:, -1, :]  # 最后一个时间步
        combined = torch.cat([last_hidden, x_static], dim=1)
        combined = self.dropout(combined)
        return self.sigmoid(self.fc(combined))


class EnhancedDataset(Dataset):
    def __init__(self, X_seq, X_static, y):
        self.X_seq = torch.FloatTensor(X_seq)
        self.X_static = torch.FloatTensor(X_static)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, i):
        return self.X_seq[i], self.X_static[i], self.y[i]

    def __len__(self):
        return len(self.y)


def main():
    print("=" * 60)
    print("Enhanced Temporal GRU (时序 + LLM + 标注特征)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载标注特征
    annot_df = load_annotation_features()
    
    # 加载时序数据（同 train_temporal_gru_v2.py）
    print("\n加载时序数据...")
    df_cohort = pd.read_csv(COHORT_FILE)
    df_cohort['stay_id'] = df_cohort['stay_id'].astype(int)
    
    df_keys = df_cohort[['stay_id', 'subject_id']].copy()
    df_keys['label'] = df_cohort['label_mortality']
    df_clean = df_keys.dropna(subset=['stay_id', 'subject_id', 'label']).copy().reset_index(drop=True)
    
    # 合并标注特征
    df_clean = df_clean.merge(annot_df, on='stay_id', how='left')
    df_clean = df_clean.fillna(0)
    
    valid_stay_ids = df_clean['stay_id'].unique()
    print(f"   样本数: {len(df_clean):,}")
    
    # 加载时序
    ts_df = pd.read_csv(TIMESERIES_FILE)
    ts_df['stay_id'] = pd.to_numeric(ts_df['stay_id'], errors='coerce').fillna(-1).astype(int)
    ts_df = ts_df[ts_df['stay_id'].isin(valid_stay_ids)]
    
    # 加载 LLM 特征
    note_time_df = pd.read_csv(NOTE_TIME_FILE)
    note_time_df['hour_offset'] = pd.to_numeric(note_time_df['hour_offset'], errors='coerce')
    note_time_df = note_time_df[(note_time_df['hour_offset'] >= 0) & (note_time_df['hour_offset'] < 24)]
    llm_df = pd.read_csv(LLM_FEATURES_FILE)
    llm_df['stay_id'] = pd.to_numeric(llm_df['stay_id'], errors='coerce').fillna(-1).astype(int)
    note_merged = note_time_df.merge(llm_df, on='stay_id', how='inner')
    
    # 构建张量
    print("\n构建张量...")
    feature_cols = [c for c in ts_df.columns if c not in ['stay_id', 'hour', 'subject_id', 'hadm_id', 'intime']]
    
    N = len(df_clean)
    T = 24
    D_physio = len(feature_cols)
    D_llm = len(LLM_COLS)
    D_seq_base = D_physio + D_llm
    D_static = 4  # 标注特征维度
    
    print(f"   时序维度: {D_seq_base} (physio:{D_physio} + LLM:{D_llm})")
    print(f"   静态维度: {D_static} (标注特征)")
    
    id_map = {sid: i for i, sid in enumerate(df_clean['stay_id'])}
    
    # 构建时序张量
    mux = pd.MultiIndex.from_product([df_clean['stay_id'], range(T)], names=['stay_id', 'hour'])
    ts_df = ts_df.set_index(['stay_id', 'hour'])
    ts_df = ts_df[~ts_df.index.duplicated(keep='first')]
    ts_df = ts_df.reindex(mux)
    
    X_seq_values = np.full((N, T, D_seq_base), np.nan)
    X_seq_values[:, :, :D_physio] = ts_df[feature_cols].values.reshape(N, T, D_physio)
    
    # 注入 LLM 特征
    for row in note_merged.itertuples():
        if row.stay_id in id_map:
            idx = id_map[row.stay_id]
            h = int(row.hour_offset)
            feats = [getattr(row, c, 0) for c in LLM_COLS]
            if 0 <= h < T:
                X_seq_values[idx, h:, D_physio:] = feats
    
    # 处理缺失值
    obs_mask = (~np.isnan(X_seq_values)).astype(np.float32)
    nan_mask = np.isnan(X_seq_values)
    idx_ffill = np.where(~nan_mask, np.arange(nan_mask.shape[1])[None, :, None], 0)
    np.maximum.accumulate(idx_ffill, axis=1, out=idx_ffill)
    X_seq_values = X_seq_values[np.arange(N)[:, None, None], idx_ffill, np.arange(D_seq_base)[None, None, :]]
    X_seq_values = np.nan_to_num(X_seq_values, nan=0.0)
    X_seq_mask = obs_mask

    D_seq = D_seq_base * 2
    
    # 静态标注特征
    static_cols = ['n_supportive', 'n_contradictory', 'supportive_ratio', 'annotation_density']
    X_static = df_clean[static_cols].values
    
    y = df_clean['label'].values
    subjects = df_clean['subject_id'].values
    
    print(f"\n数据准备完成: X_seq_values={X_seq_values.shape}, X_seq_mask={X_seq_mask.shape}, X_static={X_static.shape}, y={y.shape}")
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')
    print(f"使用设备: {device}")
    
    # 分离测试集
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X_seq_values, y, groups=subjects))
    
    X_seq_tv, X_seq_test = X_seq_values[train_val_idx], X_seq_values[test_idx]
    X_seq_mask_tv, X_seq_mask_test = X_seq_mask[train_val_idx], X_seq_mask[test_idx]
    X_static_tv, X_static_test = X_static[train_val_idx], X_static[test_idx]
    y_tv, y_test = y[train_val_idx], y[test_idx]
    subjects_tv = subjects[train_val_idx]
    
    print(f"训练+验证: {len(X_seq_tv)}, 测试: {len(X_seq_test)}")
    
    # 交叉验证
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    best_model_state = None
    best_scalers = None
    best_auc = -1
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_seq_tv, y_tv, groups=subjects_tv)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")
        
        X_seq_train, X_seq_val = X_seq_tv[train_idx], X_seq_tv[val_idx]
        X_seq_mask_train, X_seq_mask_val = X_seq_mask_tv[train_idx], X_seq_mask_tv[val_idx]
        X_static_train, X_static_val = X_static_tv[train_idx], X_static_tv[val_idx]
        y_train, y_val = y_tv[train_idx], y_tv[val_idx]
        
        # 标准化
        seq_scaler = StandardScaler()
        X_seq_train_2d = X_seq_train.reshape(-1, D_seq_base)
        X_seq_train_2d = seq_scaler.fit_transform(X_seq_train_2d)
        X_seq_train = X_seq_train_2d.reshape(-1, T, D_seq_base)
        
        X_seq_val_2d = X_seq_val.reshape(-1, D_seq_base)
        X_seq_val_2d = seq_scaler.transform(X_seq_val_2d)
        X_seq_val = X_seq_val_2d.reshape(-1, T, D_seq_base)

        # 拼接缺失掩码
        X_seq_train = np.concatenate([X_seq_train, X_seq_mask_train], axis=2)
        X_seq_val = np.concatenate([X_seq_val, X_seq_mask_val], axis=2)
        
        static_scaler = StandardScaler()
        X_static_train = static_scaler.fit_transform(X_static_train)
        X_static_val = static_scaler.transform(X_static_val)
        
        train_loader = DataLoader(EnhancedDataset(X_seq_train, X_static_train, y_train),
                                 batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(EnhancedDataset(X_seq_val, X_static_val, y_val),
                               batch_size=BATCH_SIZE)
        
        model = EnhancedGRU(D_seq, HIDDEN_DIM, NUM_LAYERS, D_static, DROPOUT).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=LR_SCHEDULER_FACTOR,
                                                         patience=LR_SCHEDULER_PATIENCE, min_lr=LR_SCHEDULER_MIN_LR)
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA)
        
        best_val_auc = -1
        
        for epoch in range(EPOCHS):
            # Train
            model.train()
            for bx_seq, bx_static, by in train_loader:
                bx_seq, bx_static, by = bx_seq.to(device), bx_static.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx_seq, bx_static).squeeze(), by)
                loss.backward()
                optimizer.step()
            
            # Validate
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for bx_seq, bx_static, by in val_loader:
                    bx_seq, bx_static = bx_seq.to(device), bx_static.to(device)
                    preds.extend(model(bx_seq, bx_static).squeeze().cpu().numpy())
                    targets.extend(by.numpy())
            
            val_auc = roc_auc_score(targets, preds)
            val_auprc = average_precision_score(targets, preds)
            
            if epoch % 5 == 0:
                print(f"   Epoch {epoch}: Val AUROC={val_auc:.4f}, AUPRC={val_auprc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch_model = model.state_dict().copy()
            
            scheduler.step(val_auc)
            if early_stopping(val_auc, epoch):
                print(f"   Early Stopping at epoch {epoch}")
                break
        
        fold_results.append({'fold': fold+1, 'val_auroc': best_val_auc})
        print(f"   Fold {fold+1} Best AUROC: {best_val_auc:.4f}")
        
        if best_val_auc > best_auc:
            best_auc = best_val_auc
            best_model_state = best_epoch_model
            best_scalers = (seq_scaler, static_scaler)
    
    # 测试集评估
    print(f"\n{'='*60}")
    print("测试集评估")
    print(f"{'='*60}")
    
    seq_scaler, static_scaler = best_scalers
    
    X_seq_test_2d = X_seq_test.reshape(-1, D_seq_base)
    X_seq_test_2d = seq_scaler.transform(X_seq_test_2d)
    X_seq_test_scaled = X_seq_test_2d.reshape(-1, T, D_seq_base)
    X_seq_test_scaled = np.concatenate([X_seq_test_scaled, X_seq_mask_test], axis=2)
    X_static_test_scaled = static_scaler.transform(X_static_test)
    
    test_loader = DataLoader(EnhancedDataset(X_seq_test_scaled, X_static_test_scaled, y_test),
                            batch_size=BATCH_SIZE)
    
    model = EnhancedGRU(D_seq, HIDDEN_DIM, NUM_LAYERS, D_static, DROPOUT).to(device)
    model.load_state_dict(best_model_state)
    model.eval()
    
    preds, targets = [], []
    with torch.no_grad():
        for bx_seq, bx_static, by in test_loader:
            bx_seq, bx_static = bx_seq.to(device), bx_static.to(device)
            preds.extend(model(bx_seq, bx_static).squeeze().cpu().numpy())
            targets.extend(by.numpy())
    
    test_auc = roc_auc_score(targets, preds)
    test_auprc = average_precision_score(targets, preds)
    
    print(f"Test AUROC: {test_auc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    
    # 保存结果
    results = {
        'cv_auroc_mean': np.mean([r['val_auroc'] for r in fold_results]),
        'cv_auroc_std': np.std([r['val_auroc'] for r in fold_results]),
        'test_auroc': test_auc,
        'test_auprc': test_auprc
    }
    
    results_df = pd.DataFrame([results])
    results_df.to_csv(OUTPUT_DIR / 'enhanced_gru_results.csv', index=False)
    
    print(f"\n{'='*60}")
    print("最终结果")
    print(f"{'='*60}")
    print(f"CV AUROC: {results['cv_auroc_mean']:.4f} ± {results['cv_auroc_std']:.4f}")
    print(f"Test AUROC: {results['test_auroc']:.4f}")
    print(f"Test AUPRC: {results['test_auprc']:.4f}")


if __name__ == "__main__":
    main()
