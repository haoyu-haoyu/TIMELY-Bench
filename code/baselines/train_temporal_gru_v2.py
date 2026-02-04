"""
Train temporal GRU model.
Includes early stopping, LR scheduler, and saves results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 导入配置
from config import (
    TIMESERIES_FILE, NOTE_TIME_FILE, LLM_FEATURES_FILE, COHORT_FILE,
    RESULTS_DIR, HIDDEN_DIM, NUM_LAYERS, DROPOUT, BATCH_SIZE, EPOCHS, LR,
    EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
    LR_SCHEDULER_PATIENCE, LR_SCHEDULER_FACTOR, LR_SCHEDULER_MIN_LR,
    TEST_SIZE, USE_HOLDOUT_TEST, N_FOLDS, RANDOM_STATE, LLM_COLS
)

# 输出目录
RESULT_DIR = RESULTS_DIR / 'Output_temporal_gru'
LLM_FEAT_FILE = LLM_FEATURES_FILE
LABEL_FILE = COHORT_FILE
SAVE_BEST_MODEL = True
MODEL_SAVE_DIR = RESULT_DIR / 'models'
SAVE_TRAINING_LOG = True
LOG_DIR = RESULT_DIR / 'logs'
RESULTS_CSV = RESULT_DIR / 'training_results.csv'
RESULTS_JSON = RESULT_DIR / 'training_results.json'

# Early stopping
class EarlyStopping:
    """Early Stopping 机制"""
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
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
            if self.verbose:
                print(f'      EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose and score > self.best_score:
                improvement = score - self.best_score
                print(f'      Validation improved by {improvement:.5f}')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0

        return self.early_stop

# Training logger
class TrainingLogger:
    """记录训练过程"""
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'val_auroc': [],
            'val_auprc': [],
            'learning_rate': []
        }

    def log_epoch(self, epoch, train_loss, val_loss, val_auroc, val_auprc, lr):
        """记录一个epoch的信息"""
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['val_auroc'].append(val_auroc)
        self.history['val_auprc'].append(val_auprc)
        self.history['learning_rate'].append(lr)

    def save(self, fold, filename_suffix=''):
        """保存日志到文件"""
        log_file = self.log_dir / f'training_log_fold{fold}_{filename_suffix}.json'
        with open(log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        return log_file

# File validation
def check_files():
    """验证所有必要文件是否存在"""
    print("\n检查数据文件...")

    files_to_check = {
        '时序数据': TIMESERIES_FILE,
        '笔记时间': NOTE_TIME_FILE,
        'LLM特征': LLM_FEAT_FILE,
        '标签文件': LABEL_FILE,
    }

    all_exist = True
    for name, path in files_to_check.items():
        path = Path(path)
        if path.exists():
            size = path.stat().st_size / (1024 * 1024)  # MB
            print(f"   {name}: {path.name} ({size:.1f} MB)")
        else:
            print(f"   {name}: {path} (不存在)")
            all_exist = False

    if not all_exist:
        raise FileNotFoundError("部分必要文件缺失，请检查路径配置！")

    return True

# Data loading
def load_and_process_data():
    """加载并处理数据"""
    print("\n[1/5] 加载并清洗数据...")

    try:
        # 1. 加载基础信息
        df_label = pd.read_csv(LABEL_FILE)

        # 直接使用配置中的COHORT_FILE
        df_cohort = pd.read_csv(COHORT_FILE)
        print(f"   从 {COHORT_FILE} 加载 cohort 数据")

        # 确保 ID 类型一致
        df_cohort['stay_id'] = df_cohort['stay_id'].astype(int)

        # 准备标签数据
        df_keys = df_cohort[['stay_id', 'subject_id']].copy()
        df_keys['label'] = df_cohort['label_mortality']

        # 数据质量过滤（不使用LOS避免数据泄露）
        print("   - 基于数据质量过滤（不使用LOS）...")
        df_clean = df_keys.dropna(subset=['stay_id', 'subject_id', 'label']).copy().reset_index(drop=True)

        print(f"   - 原始队列: {len(df_keys)}")
        print(f"   - 数据质量过滤后: {len(df_clean)} (移除 {len(df_keys) - len(df_clean)} 个缺失关键信息的记录)")
        print(f"   - 修复：不使用LOS过滤，避免数据泄露")

        valid_stay_ids = df_clean['stay_id'].unique()

        # 2. 加载并过滤时序数据
        print("   - 加载时序数据...")
        ts_df = pd.read_csv(TIMESERIES_FILE)
        ts_df['stay_id'] = pd.to_numeric(ts_df['stay_id'], errors='coerce').fillna(-1).astype(int)
        ts_df = ts_df[ts_df['stay_id'].isin(valid_stay_ids)]
        print(f"   - 时序数据记录数: {len(ts_df)}")

        # 3. 加载笔记和LLM特征
        print("   - 加载笔记和LLM特征...")
        note_time_df = pd.read_csv(NOTE_TIME_FILE)
        note_time_df['hour_offset'] = pd.to_numeric(note_time_df['hour_offset'], errors='coerce')
        note_time_df = note_time_df[(note_time_df['hour_offset'] >= 0) & (note_time_df['hour_offset'] < 24)]
        llm_df = pd.read_csv(LLM_FEAT_FILE)
        llm_df['stay_id'] = pd.to_numeric(llm_df['stay_id'], errors='coerce').fillna(-1).astype(int)
        note_merged = note_time_df.merge(llm_df, on='stay_id', how='inner')
        print(f"   - LLM特征记录数: {len(note_merged)}")

    except Exception as e:
        print(f"数据加载失败: {e}")
        raise

    # ==========================================
    # 构建张量
    # ==========================================
    print("\n[2/5] 构建时序张量...")

    try:
        feature_cols = [c for c in ts_df.columns
                       if c not in ['stay_id', 'hour', 'subject_id', 'hadm_id', 'intime']]

        N = len(df_clean)
        T = 24  # 24小时
        D_physio = len(feature_cols)
        D_llm = len(LLM_COLS)
        D = D_physio + D_llm

        print(f"   - 样本数量 (N): {N}")
        print(f"   - 时间步长 (T): {T}")
        print(f"   - 生理特征维度: {D_physio}")
        print(f"   - LLM特征维度: {D_llm}")
        print(f"   - 总特征维度 (D): {D}")

        # 映射 stay_id -> 索引
        id_map = {sid: i for i, sid in enumerate(df_clean['stay_id'])}

        # 构建基础张量
        mux = pd.MultiIndex.from_product([df_clean['stay_id'], range(T)],
                                        names=['stay_id', 'hour'])
        ts_df = ts_df.set_index(['stay_id', 'hour'])
        ts_df = ts_df[~ts_df.index.duplicated(keep='first')]
        ts_df = ts_df.reindex(mux)

        X_tensor = np.full((N, T, D), np.nan)
        X_tensor[:, :, :D_physio] = ts_df[feature_cols].values.reshape(N, T, D_physio)

        # 注入LLM特征
        print("   - 注入LLM特征...")
        llm_injected_count = 0
        for row in note_merged.itertuples():
            if row.stay_id in id_map:
                idx = id_map[row.stay_id]
                h = int(row.hour_offset)
                feats = [getattr(row, c, 0) for c in LLM_COLS]
                if 0 <= h < T:
                    X_tensor[idx, h:, D_physio:] = feats
                    llm_injected_count += 1

        print(f"   - LLM特征注入数量: {llm_injected_count}")

        # 缺失值处理（前向填充）+ 缺失掩码
        print("   - 处理缺失值...")
        obs_mask = (~np.isnan(X_tensor)).astype(np.float32)
        nan_mask = np.isnan(X_tensor)
        idx_ffill = np.where(~nan_mask, np.arange(nan_mask.shape[1])[None, :, None], 0)
        np.maximum.accumulate(idx_ffill, axis=1, out=idx_ffill)
        X_tensor = X_tensor[np.arange(N)[:, None, None],
                           idx_ffill,
                           np.arange(D)[None, None, :]]
        X_tensor = np.nan_to_num(X_tensor, nan=0.0)

        print(f"   张量构建完成: shape={X_tensor.shape}")

    except Exception as e:
        print(f"张量构建失败: {e}")
        raise

    return X_tensor, obs_mask, df_clean['label'].values, df_clean['subject_id'].values, D

# Model definitions
class ClinicalGRU(nn.Module):
    """临床 GRU 模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers,
                         batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.gru(x)
        return self.sigmoid(self.fc(out[:, -1, :]))

class MIMICDataset(Dataset):
    """MIMIC 数据集"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)

# Training functions
def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    n_batches = 0

    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        loss = criterion(model(bx).squeeze(), by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches

# Validation
def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    n_batches = 0
    preds, targets = [], []

    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            output = model(bx).squeeze()
            loss = criterion(output, by)
            total_loss += loss.item()
            n_batches += 1
            preds.extend(output.cpu().numpy())
            targets.extend(by.cpu().numpy())

    preds = np.array(preds)
    targets = np.array(targets)

    try:
        auroc = roc_auc_score(targets, preds)
        auprc = average_precision_score(targets, preds)
    except:
        auroc = 0.5
        auprc = targets.mean()

    avg_loss = total_loss / n_batches
    return avg_loss, auroc, auprc

# Main training
def main():
    """主函数"""
    print("=" * 60)
    print("TIMELY-Bench v2.0 - 时序 GRU 模型训练")
    print("=" * 60)

    # 创建输出目录
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_BEST_MODEL:
        MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_TRAINING_LOG:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # 检查文件
    try:
        check_files()
    except FileNotFoundError as e:
        print(f"\n{e}")
        return

    # 加载数据
    try:
        X_values, X_mask, y, subjects, base_dim = load_and_process_data()
        input_dim = base_dim * 2
        print(f"\n数据准备完成: X={X_values.shape}, mask={X_mask.shape}, y={y.shape}")
    except Exception as e:
        print(f"\n数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available()
                         else 'mps' if torch.backends.mps.is_available()
                         else 'cpu')
    print(f"\n使用设备: {device}")

    # 训练参数汇总
    print(f"\n训练配置:")
    print(f"   - 隐藏维度: {HIDDEN_DIM}")
    print(f"   - GRU层数: {NUM_LAYERS}")
    print(f"   - Dropout: {DROPOUT}")
    print(f"   - Batch大小: {BATCH_SIZE}")
    print(f"   - 最大Epoch: {EPOCHS}")
    print(f"   - 学习率: {LR}")
    print(f"   - Early Stopping耐心值: {EARLY_STOPPING_PATIENCE}")
    print(f"   - LR调度器耐心值: {LR_SCHEDULER_PATIENCE}")

    # 分离测试集
    if USE_HOLDOUT_TEST:
        print(f"\n分离独立测试集 ({TEST_SIZE*100:.0f}%)...")
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X_values, y, groups=subjects))

        X_train_val, X_test = X_values[train_val_idx], X_values[test_idx]
        X_train_val_mask, X_test_mask = X_mask[train_val_idx], X_mask[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        subjects_train_val = subjects[train_val_idx]

        print(f"   训练+验证集: {len(X_train_val)}, 测试集: {len(X_test)}")
    else:
        X_train_val, y_train_val, subjects_train_val = X_values, y, subjects
        X_train_val_mask = X_mask
        X_test, y_test, X_test_mask = None, None, None

    # 交叉验证
    print(f"\n[3/5] 开始 {N_FOLDS} 折交叉验证...")
    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_results = []
    best_model_state = None
    best_scaler_params = None
    best_auc = -1
    best_fold = -1

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val,
                                                           groups=subjects_train_val)):
        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        X_train_mask, X_val_mask = X_train_val_mask[train_idx], X_train_val_mask[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # 标准化
        N_train, T, D = X_train.shape
        N_val = X_val.shape[0]

        scaler = StandardScaler()
        X_train_2d = X_train.reshape(-1, D)
        X_train_2d = scaler.fit_transform(X_train_2d)
        X_train = X_train_2d.reshape(N_train, T, D)

        X_val_2d = X_val.reshape(-1, D)
        X_val_2d = scaler.transform(X_val_2d)
        X_val = X_val_2d.reshape(N_val, T, D)

        # 拼接缺失掩码
        X_train = np.concatenate([X_train, X_train_mask], axis=2)
        X_val = np.concatenate([X_val, X_val_mask], axis=2)

        # DataLoader
        train_loader = DataLoader(MIMICDataset(X_train, y_train),
                                 batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(MIMICDataset(X_val, y_val),
                               batch_size=BATCH_SIZE)

        # 模型初始化
        input_dim = D * 2
        model = ClinicalGRU(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        criterion = nn.BCELoss()

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=LR_SCHEDULER_FACTOR,
            patience=LR_SCHEDULER_PATIENCE, min_lr=LR_SCHEDULER_MIN_LR
        )

        # Early Stopping
        early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE,
                                      min_delta=EARLY_STOPPING_MIN_DELTA,
                                      verbose=True)

        # 训练日志
        logger = TrainingLogger(LOG_DIR) if SAVE_TRAINING_LOG else None

        # 训练循环
        best_val_auc = -1
        best_epoch_model = None

        print(f"\n开始训练...")
        for epoch in range(EPOCHS):
            # 训练
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

            # 验证
            val_loss, val_auroc, val_auprc = validate(model, val_loader, criterion, device)

            # 获取当前学习率
            current_lr = optimizer.param_groups[0]['lr']

            # 记录日志
            if logger:
                logger.log_epoch(epoch, train_loss, val_loss, val_auroc, val_auprc, current_lr)

            # 每5个epoch打印一次
            if epoch % 5 == 0 or epoch == EPOCHS - 1:
                print(f"   Epoch {epoch:3d}/{EPOCHS}: "
                      f"Train Loss={train_loss:.4f}, "
                      f"Val Loss={val_loss:.4f}, "
                      f"Val AUROC={val_auroc:.4f}, "
                      f"Val AUPRC={val_auprc:.4f}, "
                      f"LR={current_lr:.6f}")

            # 保存最佳模型
            if val_auroc > best_val_auc:
                best_val_auc = val_auroc
                best_epoch_model = {
                    'epoch': epoch,
                    'model_state': model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'val_auroc': val_auroc,
                    'val_auprc': val_auprc
                }

            # 学习率调度
            scheduler.step(val_auroc)

            # Early Stopping检查
            if early_stopping(val_auroc, epoch):
                print(f"\n   Early Stopping at epoch {epoch}")
                print(f"   Best epoch was {early_stopping.best_epoch} with AUROC={early_stopping.best_score:.4f}")
                break

        # 保存训练日志
        if logger:
            log_file = logger.save(fold, datetime.now().strftime("%Y%m%d_%H%M%S"))
            print(f"   训练日志已保存: {log_file}")

        # 记录Fold结果
        fold_result = {
            'fold': fold + 1,
            'val_auroc': best_val_auc,
            'val_auprc': best_epoch_model['val_auprc'],
            'best_epoch': best_epoch_model['epoch'],
            'total_epochs': epoch + 1
        }
        fold_results.append(fold_result)

        print(f"\n   Fold {fold+1} 完成:")
        print(f"      最佳 AUROC: {best_val_auc:.4f} (Epoch {best_epoch_model['epoch']})")
        print(f"      最佳 AUPRC: {best_epoch_model['val_auprc']:.4f}")

        # 更新全局最佳模型
        if best_val_auc > best_auc:
            best_auc = best_val_auc
            best_fold = fold + 1
            best_model_state = best_epoch_model['model_state']
            best_scaler_params = (scaler.mean_, scaler.scale_)

            # 保存最佳模型
            if SAVE_BEST_MODEL:
                model_path = MODEL_SAVE_DIR / f'best_model_fold{fold+1}.pt'
                torch.save({
                    'fold': fold + 1,
                    'epoch': best_epoch_model['epoch'],
                    'model_state_dict': best_model_state,
                    'val_auroc': best_auc,
                    'scaler_mean': scaler.mean_,
                    'scaler_scale': scaler.scale_,
                }, model_path)
                print(f"      模型已保存: {model_path}")

    # 交叉验证汇总
    print(f"\n{'='*60}")
    print(f"交叉验证结果汇总")
    print(f"{'='*60}")

    aurocs = [r['val_auroc'] for r in fold_results]
    auprcs = [r['val_auprc'] for r in fold_results]

    print(f"CV Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"CV Mean AUPRC: {np.mean(auprcs):.4f} ± {np.std(auprcs):.4f}")
    print(f"最佳Fold: {best_fold} (AUROC: {best_auc:.4f})")

    # 在独立测试集上评估
    test_result = None
    if X_test is not None and best_model_state is not None:
        print(f"\n[4/5] 在独立测试集上评估...")

        try:
            # 使用最佳模型的scaler
            N_test, T, D = X_test.shape
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_ = best_scaler_params

            X_test_2d = X_test.reshape(-1, D)
            X_test_2d = scaler.transform(X_test_2d)
            X_test_scaled = X_test_2d.reshape(N_test, T, D)
            X_test_scaled = np.concatenate([X_test_scaled, X_test_mask], axis=2)

            test_loader = DataLoader(MIMICDataset(X_test_scaled, y_test),
                                    batch_size=BATCH_SIZE)

            # 加载最佳模型
            model = ClinicalGRU(input_dim, HIDDEN_DIM, NUM_LAYERS, dropout=DROPOUT).to(device)
            model.load_state_dict(best_model_state)
            model.eval()

            # 测试
            test_preds, test_targets = [], []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(device)
                    test_preds.extend(model(bx).squeeze().cpu().numpy())
                    test_targets.extend(by.numpy())

            test_auc = roc_auc_score(test_targets, test_preds)
            test_auprc = average_precision_score(test_targets, test_preds)

            print(f"   Test AUROC: {test_auc:.4f}")
            print(f"   Test AUPRC: {test_auprc:.4f}")

            test_result = {
                'test_auroc': test_auc,
                'test_auprc': test_auprc
            }

        except Exception as e:
            print(f"   测试集评估失败: {e}")
            import traceback
            traceback.print_exc()

    # 保存结果
    print(f"\n[5/5] 保存结果...")

    # 汇总结果
    final_results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'hidden_dim': HIDDEN_DIM,
            'num_layers': NUM_LAYERS,
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'learning_rate': LR,
            'early_stopping_patience': EARLY_STOPPING_PATIENCE,
            'lr_scheduler_patience': LR_SCHEDULER_PATIENCE,
        },
        'data': {
            'total_samples': len(X_values),
            'train_val_samples': len(X_train_val) if USE_HOLDOUT_TEST else len(X_values),
            'test_samples': len(X_test) if X_test is not None else 0,
            'input_dim': input_dim,
        },
        'cross_validation': {
            'n_folds': N_FOLDS,
            'mean_auroc': float(np.mean(aurocs)),
            'std_auroc': float(np.std(aurocs)),
            'mean_auprc': float(np.mean(auprcs)),
            'std_auprc': float(np.std(auprcs)),
            'best_fold': int(best_fold),
            'best_auroc': float(best_auc),
            'fold_details': fold_results,
        }
    }

    if test_result:
        final_results['test'] = test_result

    # 保存JSON
    try:
        with open(RESULTS_JSON, 'w') as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)
        print(f"   结果已保存(JSON): {RESULTS_JSON}")
    except Exception as e:
        print(f"   JSON保存失败: {e}")

    # 保存CSV
    try:
        results_df = pd.DataFrame(fold_results)
        if test_result:
            test_row = {'fold': 'TEST', **test_result, 'best_epoch': '-', 'total_epochs': '-'}
            results_df = pd.concat([results_df, pd.DataFrame([test_row])], ignore_index=True)

        results_df.to_csv(RESULTS_CSV, index=False)
        print(f"   结果已保存(CSV): {RESULTS_CSV}")
    except Exception as e:
        print(f"   CSV保存失败: {e}")

    # 最终总结
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"{'='*60}")
    print(f"最终结果:")
    print(f"   CV AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    if test_result:
        print(f"   Test AUROC: {test_result['test_auroc']:.4f}")
        print(f"   Test AUPRC: {test_result['test_auprc']:.4f}")
    print(f"{'='*60}")

# Entry point
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断")
    except Exception as e:
        print(f"\n\n程序异常: {e}")
        import traceback
        traceback.print_exc()
