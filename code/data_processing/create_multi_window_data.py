"""
创建多时间窗口数据
根据项目要求，创建 ±6h, ±12h, ±24h 三个时间窗口的数据
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from config import (
    TIMESERIES_FILE, COHORT_FILE, DATA_WINDOWS_DIR
)

# ==========================================
# 配置路径
# ==========================================
OUTPUT_DIR = DATA_WINDOWS_DIR

# 时间窗口配置
TIME_WINDOWS = {
    '6h': 6,
    '12h': 12,
    '24h': 24
}

# 特征聚合方式
AGGREGATIONS = ['min', 'max', 'mean', 'last', 'first', 'std']

# 缺失率阈值配置
MAX_MISSING_RATE = 0.8  # 缺失率超过80%的特征将被警告
WARN_MISSING_RATE = 0.5  # 缺失率超过50%的特征将被提示

# ==========================================
# 1. 加载数据
# ==========================================
def load_data():
    print("[1] Loading data...")
    
    df_cohort = pd.read_csv(COHORT_FILE)
    df_ts = pd.read_csv(TIMESERIES_FILE)
    
    # 确保类型一致
    df_cohort['stay_id'] = df_cohort['stay_id'].astype(int)
    df_ts['stay_id'] = pd.to_numeric(df_ts['stay_id'], errors='coerce').fillna(-1).astype(int)
    
    print(f"   Cohort: {len(df_cohort)} patients")
    print(f"   Timeseries: {len(df_ts)} rows")
    
    # 获取特征列（排除ID和时间列）
    exclude_cols = ['stay_id', 'hour', 'subject_id', 'hadm_id', 'intime', 'charttime']
    feature_cols = [c for c in df_ts.columns if c not in exclude_cols]
    print(f"   Features: {len(feature_cols)}")
    
    return df_cohort, df_ts, feature_cols

# ==========================================
# 2. 为单个窗口创建聚合特征
# ==========================================
def create_window_features(df_ts, stay_ids, feature_cols, window_hours, aggregations):
    """
    创建指定时间窗口的聚合特征

    改进缺失处理
    - 添加缺失率统计
    - 添加最后测量时间特征
    - 添加测量次数特征

    Args:
        df_ts: 时序数据
        stay_ids: 患者ID列表
        feature_cols: 特征列
        window_hours: 窗口小时数
        aggregations: 聚合方式列表

    Returns:
        DataFrame with aggregated features, 缺失率统计字典
    """

    # 过滤到指定时间窗口
    df_window = df_ts[df_ts['hour'] < window_hours].copy()

    # 只保留目标患者
    df_window = df_window[df_window['stay_id'].isin(stay_ids)]

    # 保证 first/last 按时间顺序（若有charttime用于打破同小时并列）
    if 'hour' in df_window.columns:
        sort_cols = ['stay_id', 'hour']
        if 'charttime' in df_window.columns:
            df_window = df_window.copy()
            df_window['_charttime_sort'] = pd.to_datetime(df_window['charttime'], errors='coerce')
            sort_cols.append('_charttime_sort')
        df_window = df_window.sort_values(sort_cols)

    # 按患者聚合
    agg_dict = {col: aggregations for col in feature_cols}
    df_agg = df_window.groupby('stay_id').agg(agg_dict)

    # 展平多级列名
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index()
    df_agg.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 添加改进的缺失标记
    missing_stats = {}  # 记录每个特征的缺失率

    for col in feature_cols:
        # 1. 基础缺失标记：该特征在窗口内是否有任何值
        has_value = df_window.groupby('stay_id')[col].apply(lambda x: x.notna().any())
        missing_col = f'{col}_missing'
        df_agg[missing_col] = (~df_agg['stay_id'].isin(has_value[has_value].index)).astype(int)

        # 2. 新增：测量次数（数据丰富度指标）
        measurement_count = df_window.groupby('stay_id')[col].apply(lambda x: x.notna().sum())
        count_col = f'{col}_n_measurements'
        df_agg = df_agg.merge(
            measurement_count.reset_index().rename(columns={col: count_col}),
            on='stay_id', how='left'
        )
        df_agg[count_col] = df_agg[count_col].fillna(0)

        # 3. 新增：最后测量时间（相对于窗口开始）
        last_measurement_hour = df_window[df_window[col].notna()].groupby('stay_id')['hour'].max()
        last_hour_col = f'{col}_last_hour'
        df_agg = df_agg.merge(
            last_measurement_hour.reset_index().rename(columns={'hour': last_hour_col}),
            on='stay_id', how='left'
        )
        df_agg[last_hour_col] = df_agg[last_hour_col].fillna(-1)  # -1表示无测量

        # 4. 计算缺失率统计
        missing_rate = df_agg[missing_col].mean()
        missing_stats[col] = missing_rate

    # 输出缺失率警告
    print(f"\n   缺失率统计 (window={window_hours}h):")
    high_missing = []
    for col, rate in sorted(missing_stats.items(), key=lambda x: -x[1]):
        if rate >= MAX_MISSING_RATE:
            print(f"      {col}: {rate*100:.1f}% 缺失 (超过{MAX_MISSING_RATE*100:.0f}%阈值)")
            high_missing.append(col)
        elif rate >= WARN_MISSING_RATE:
            print(f"      ⚡ {col}: {rate*100:.1f}% 缺失")

    if high_missing:
        print(f"   警告: {len(high_missing)}个特征缺失率超过{MAX_MISSING_RATE*100:.0f}%")

    return df_agg, missing_stats

# ==========================================
# 3. 创建时序张量（用于GRU等模型）
# ==========================================
def create_window_tensor(df_ts, stay_ids, feature_cols, window_hours):
    """
    创建3D时序张量 (N, T, D)
    
    Args:
        df_ts: 时序数据
        stay_ids: 患者ID列表
        feature_cols: 特征列
        window_hours: 窗口小时数
    
    Returns:
        X: numpy array of shape (N, T, D)
        mask: numpy array of shape (N, T, D) with 1 where observed
    """
    
    N = len(stay_ids)
    T = window_hours
    D = len(feature_cols)
    
    # 创建ID映射
    id_map = {sid: i for i, sid in enumerate(stay_ids)}
    
    # 初始化张量
    X = np.full((N, T, D), np.nan)
    
    # 过滤数据
    df_window = df_ts[
        (df_ts['stay_id'].isin(stay_ids)) & 
        (df_ts['hour'] < window_hours)
    ].copy()
    
    # 填充张量
    for _, row in df_window.iterrows():
        sid = row['stay_id']
        hour = int(row['hour'])
        
        if sid in id_map and 0 <= hour < T:
            idx = id_map[sid]
            for j, col in enumerate(feature_cols):
                if pd.notna(row[col]):
                    X[idx, hour, j] = row[col]

    # 观测掩码（真实测量值）
    obs_mask = (~np.isnan(X)).astype(np.float32)

    # Forward fill
    nan_mask = np.isnan(X)
    idx_ffill = np.where(~nan_mask, np.arange(nan_mask.shape[1])[None, :, None], 0)
    np.maximum.accumulate(idx_ffill, axis=1, out=idx_ffill)
    X = X[np.arange(N)[:, None, None], idx_ffill, np.arange(D)[None, None, :]]
    
    # 剩余NaN填0
    X = np.nan_to_num(X, nan=0.0)
    
    return X, obs_mask

# ==========================================
# 4. 主流程
# ==========================================
def main():
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 加载数据
    df_cohort, df_ts, feature_cols = load_data()
    stay_ids = df_cohort['stay_id'].unique()
    
    print(f"\n⏱️ [2] Creating multi-window datasets...")
    print("=" * 60)
    
    for window_name, window_hours in TIME_WINDOWS.items():
        print(f"\nProcessing {window_name} window ({window_hours} hours)...")
        
        window_dir = os.path.join(OUTPUT_DIR, f'window_{window_name}')
        os.makedirs(window_dir, exist_ok=True)
        
        # ----- 3.2.1 创建聚合特征（用于XGBoost等） -----
        print(f"   Creating aggregated features...")
        df_agg, missing_stats = create_window_features(
            df_ts, stay_ids, feature_cols, window_hours, AGGREGATIONS
        )
        
        # 确保所有患者都有记录（即使全是NaN）
        all_ids = pd.DataFrame({'stay_id': stay_ids})
        df_agg = all_ids.merge(df_agg, on='stay_id', how='left')
        
        # 填充缺失值为0（对于聚合特征）
        df_agg = df_agg.fillna(0)
        
        agg_path = os.path.join(window_dir, 'features_aggregated.csv')
        df_agg.to_csv(agg_path, index=False)
        print(f"   Saved: {agg_path}")
        print(f"     Shape: {df_agg.shape} (patients × features)")
        
        # ----- 3.2.2 创建时序张量（用于GRU等） -----
        print(f"   Creating temporal tensor...")
        X_tensor, X_mask = create_window_tensor(df_ts, stay_ids, feature_cols, window_hours)
        
        tensor_path = os.path.join(window_dir, 'features_temporal.npy')
        np.save(tensor_path, X_tensor)
        print(f"   Saved: {tensor_path}")
        print(f"     Shape: {X_tensor.shape} (patients × hours × features)")

        mask_path = os.path.join(window_dir, 'features_temporal_mask.npy')
        np.save(mask_path, X_mask)
        print(f"   Saved: {mask_path}")
        print(f"     Shape: {X_mask.shape} (patients × hours × features)")
        
        # ----- 3.2.3 保存元数据 -----
        metadata = {
            'window_hours': window_hours,
            'n_patients': len(stay_ids),
            'n_features': len(feature_cols),
            'feature_names': feature_cols,
            'aggregations': AGGREGATIONS,
            'stay_ids': stay_ids.tolist(),
            'has_temporal_mask': True,
            # 添加缺失率统计到元数据
            'missing_rates': {k: float(v) for k, v in missing_stats.items()},
            'high_missing_features': [k for k, v in missing_stats.items() if v >= MAX_MISSING_RATE],
            'missing_rate_threshold': MAX_MISSING_RATE
        }
        
        metadata_path = os.path.join(window_dir, 'metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"   Saved: {metadata_path}")
    
    # ==========================================
    # 5. 统计汇总
    # ==========================================
    print("\n[3] Summary Statistics:")
    print("=" * 60)
    
    for window_name, window_hours in TIME_WINDOWS.items():
        # 统计每个窗口的数据覆盖率
        df_window = df_ts[df_ts['hour'] < window_hours]
        patients_with_data = df_window['stay_id'].nunique()
        coverage = patients_with_data / len(stay_ids) * 100
        
        # 统计平均每个患者的记录数
        records_per_patient = len(df_window) / patients_with_data if patients_with_data > 0 else 0
        
        print(f"\n[Window: {window_name}]")
        print(f"   Patients with data: {patients_with_data} ({coverage:.1f}%)")
        print(f"   Avg records/patient: {records_per_patient:.1f}")
    
    # ==========================================
    # 6. 创建统一的数据加载接口
    # ==========================================
    print("\n[4] Creating data loader utility...")
    
    loader_code = '''"""
TIMELY-Bench Data Loader
用于加载多窗口数据的统一接口
"""

import pandas as pd
import numpy as np
import json
import os

class TIMELYBenchLoader:
    """TIMELY-Bench 数据加载器"""

    def __init__(self, data_dir=None, cohort_file=None):
        from config import DATA_WINDOWS_DIR, COHORT_FILE as DEFAULT_COHORT
        self.data_dir = data_dir if data_dir else str(DATA_WINDOWS_DIR)
        cohort_path = cohort_file if cohort_file else str(DEFAULT_COHORT)
        self.cohort = pd.read_csv(cohort_path)
        self.cohort['stay_id'] = self.cohort['stay_id'].astype(int)
        
    def get_available_windows(self):
        """获取可用的时间窗口列表"""
        windows = []
        for d in os.listdir(self.data_dir):
            if d.startswith('window_'):
                windows.append(d.replace('window_', ''))
        return sorted(windows)
    
    def load_aggregated_features(self, window='24h'):
        """加载聚合特征（用于XGBoost等）"""
        path = os.path.join(self.data_dir, f'window_{window}', 'features_aggregated.csv')
        df = pd.read_csv(path)
        df['stay_id'] = df['stay_id'].astype(int)
        return df
    
    def load_temporal_features(self, window='24h'):
        """加载时序特征（用于GRU等）"""
        path = os.path.join(self.data_dir, f'window_{window}', 'features_temporal.npy')
        return np.load(path)

    def load_temporal_mask(self, window='24h'):
        """加载时序特征缺失掩码"""
        path = os.path.join(self.data_dir, f'window_{window}', 'features_temporal_mask.npy')
        return np.load(path)
    
    def load_metadata(self, window='24h'):
        """加载元数据"""
        path = os.path.join(self.data_dir, f'window_{window}', 'metadata.json')
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_labels(self, task='mortality'):
        """获取任务标签
        
        Args:
            task: 'mortality', 'prolonged_los', 'readmission'
        """
        label_map = {
            'mortality': 'label_mortality',
            'prolonged_los': 'prolonged_los_7d',
            'readmission': 'readmission_30d'
        }
        
        if task not in label_map:
            raise ValueError(f"Unknown task: {task}. Available: {list(label_map.keys())}")
        
        return self.cohort[['stay_id', label_map[task]]].copy()
    
    def get_cohort(self, disease=None):
        """获取队列信息
        
        Args:
            disease: None (全部), 'sepsis', 'aki', 'ards', 'sepsis_aki'
        """
        if disease is None:
            return self.cohort.copy()
        
        disease_map = {
            'sepsis': 'has_sepsis_final',
            'aki': 'has_aki_final',
            'ards': 'has_ards',
            'sepsis_aki': ['has_sepsis_final', 'has_aki_final']
        }
        
        if disease not in disease_map:
            raise ValueError(f"Unknown disease: {disease}")
        
        if disease == 'sepsis_aki':
            mask = (self.cohort['has_sepsis_final'] == 1) & (self.cohort['has_aki_final'] == 1)
        else:
            mask = self.cohort[disease_map[disease]] == 1
        
        return self.cohort[mask].copy()
    
    def prepare_data(self, window='24h', task='mortality', disease=None, 
                     data_type='aggregated'):
        """准备训练数据
        
        Args:
            window: '6h', '12h', '24h'
            task: 'mortality', 'prolonged_los', 'readmission'
            disease: None, 'sepsis', 'aki', etc.
            data_type: 'aggregated' or 'temporal'
        
        Returns:
            X: 特征
            y: 标签
            stay_ids: 患者ID
        """
        # 获取队列
        cohort = self.get_cohort(disease)
        stay_ids = cohort['stay_id'].values
        
        # 获取标签
        labels = self.get_labels(task)
        labels = labels[labels['stay_id'].isin(stay_ids)]
        
        # 获取特征
        if data_type == 'aggregated':
            features = self.load_aggregated_features(window)
            features = features[features['stay_id'].isin(stay_ids)]
            
            # 对齐
            merged = labels.merge(features, on='stay_id')
            X = merged.drop(columns=['stay_id', labels.columns[1]]).values
            y = merged[labels.columns[1]].values
            stay_ids = merged['stay_id'].values
            
        else:  # temporal
            X_all = self.load_temporal_features(window)
            metadata = self.load_metadata(window)
            all_stay_ids = np.array(metadata['stay_ids'])
            
            # 找到目标患者的索引
            mask = np.isin(all_stay_ids, stay_ids)
            X = X_all[mask]
            
            # 对齐标签
            filtered_ids = all_stay_ids[mask]
            labels = labels.set_index('stay_id').loc[filtered_ids].reset_index()
            y = labels[labels.columns[1]].values
            stay_ids = filtered_ids
        
        return X, y, stay_ids


# 使用示例
if __name__ == "__main__":
    loader = TIMELYBenchLoader()
    
    print("Available windows:", loader.get_available_windows())
    
    # 加载24h窗口，mortality任务，sepsis队列
    X, y, ids = loader.prepare_data(
        window='24h',
        task='mortality', 
        disease='sepsis',
        data_type='aggregated'
    )
    
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Positive rate: {y.mean()*100:.1f}%")
'''
    
    loader_path = os.path.join(OUTPUT_DIR, 'data_loader.py')
    with open(loader_path, 'w') as f:
        f.write(loader_code)
    print(f"   Saved: {loader_path}")
    
    print("\nStep 3 Complete!")
    print("=" * 60)
    print("\nOutput structure:")
    print(f"{OUTPUT_DIR}/")
    print("├── window_6h/")
    print("│   ├── features_aggregated.csv")
    print("│   ├── features_temporal.npy")
    print("│   └── metadata.json")
    print("├── window_12h/")
    print("│   └── ...")
    print("├── window_24h/")
    print("│   └── ...")
    print("└── data_loader.py")
    print("\nMulti-Window Data Creation Complete!")

if __name__ == "__main__":
    main()
