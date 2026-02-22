"""
TIMELY-Bench config
"""

import os
from pathlib import Path

# Paths
# Project root (auto-detected)
# 项目根目录 (自动计算，支持从任意位置运行)
_SCRIPT_DIR = Path(__file__).resolve().parent
if _SCRIPT_DIR.name == 'code':
    ROOT_DIR = _SCRIPT_DIR.parent
elif _SCRIPT_DIR.name in ('baselines', 'data_processing'):
    ROOT_DIR = _SCRIPT_DIR.parent.parent
else:
    ROOT_DIR = _SCRIPT_DIR

# 数据目录
DATA_DIR = ROOT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DIR = DATA_DIR / 'processed'
SPLITS_DIR = DATA_DIR / 'splits'

# 时序窗口数据
DATA_WINDOWS_DIR = PROCESSED_DIR / 'data_windows'
MERGE_OUTPUT_DIR = PROCESSED_DIR / 'merge_output'

# 模式检测和对齐
PATTERN_DETECTION_DIR = PROCESSED_DIR / 'pattern_detection'
TEMPORAL_ALIGNMENT_DIR = PROCESSED_DIR / 'temporal_alignment'

# 输出目录
RESULTS_DIR = ROOT_DIR / 'results'
BENCHMARK_RESULTS_DIR = RESULTS_DIR / 'benchmark_results'

# Data files
# 原始数据
_SORTED_TIMESERIES = RAW_DATA_DIR / "timeseries_sorted.csv"
_SORTED_TIMESERIES_EXT = RAW_DATA_DIR / "timeseries_sorted_extended.csv"
# Prefer the extended timeseries when present (adds bilirubin_total/vasopressors/rrt).
if _SORTED_TIMESERIES_EXT.exists():
    TIMESERIES_FILE = _SORTED_TIMESERIES_EXT
else:
    TIMESERIES_FILE = _SORTED_TIMESERIES if _SORTED_TIMESERIES.exists() else RAW_DATA_DIR / "timeseries.csv"
NOTE_TIME_FILE = RAW_DATA_DIR / 'note_time.csv'
CLINICAL_LABELS_FILE = RAW_DATA_DIR / 'clinical_labels.csv'

# 处理后数据
_COHORT_FINAL = MERGE_OUTPUT_DIR / "cohort_final.csv"
_COHORT_FINAL_EXT = MERGE_OUTPUT_DIR / "cohort_final_extended.csv"
# Prefer the extended cohort when present (adds external_static CKD flag).
COHORT_FILE = _COHORT_FINAL_EXT if _COHORT_FINAL_EXT.exists() else _COHORT_FINAL
LLM_FEATURES_FILE = DATA_DIR / 'llm_features' / 'llm_features_deepseek.csv'

# 数据分割
# Canonical split file (holdout test + CV fold assignment) produced by:
#   python3 code/data_processing/generate_predefined_splits.py
PREDEFINED_SPLITS_FILE = SPLITS_DIR / 'predefined_splits.csv'


# Model hyperparameters
# GRU模型
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2

# 训练配置
BATCH_SIZE = 256
EPOCHS = 50
LR = 0.001

# Early Stopping
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-4

# 学习率调度
LR_SCHEDULER_PATIENCE = 5
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_MIN_LR = 1e-6


# Data split config
TEST_SIZE = 0.2
USE_HOLDOUT_TEST = True
N_FOLDS = 5
RANDOM_STATE = 42

# Feature config
LLM_COLS = ['pneumonia', 'edema', 'pleural_effusion', 'pneumothorax', 'tubes_lines']
# Canonical windows include D0 daily aligner.
WINDOWS = ['6h', '12h', '24h', 'D0']

# 任务配置
CORE_TASKS = ['mortality', 'prolonged_los']
OPTIONAL_TASKS = ['readmission']
TASKS = CORE_TASKS

# 队列配置
COHORTS = ['all', 'sepsis', 'aki']

# Device
DEVICE = None  # 自动检测: cuda > mps > cpu

# Helpers
def get_window_dir(window: str) -> Path:
    """获取时序窗口目录"""
    return DATA_WINDOWS_DIR / f'window_{window}'

def get_features_file(window: str) -> Path:
    """获取聚合特征文件"""
    return get_window_dir(window) / 'features_aggregated.csv'

def get_temporal_file(window: str) -> Path:
    """获取时序特征文件"""
    return get_window_dir(window) / 'features_temporal.npy'

def ensure_directories():
    """创建必要目录"""
    dirs = [RESULTS_DIR, BENCHMARK_RESULTS_DIR, MERGE_OUTPUT_DIR]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return True

def validate_data():
    """验证关键数据文件"""
    files = {
        'cohort': COHORT_FILE,
        'llm_features': LLM_FEATURES_FILE,
        'predefined_splits': PREDEFINED_SPLITS_FILE,
    }
    missing = [k for k, v in files.items() if not v.exists()]
    return len(missing) == 0, missing


if __name__ == "__main__":
    print("TIMELY-Bench v2.0 Config")
    print("=" * 50)
    print(f"ROOT_DIR: {ROOT_DIR}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"RESULTS_DIR: {RESULTS_DIR}")
    print()

    ok, missing = validate_data()
    if ok:
        print("All data files found")
    else:
        print(f"Missing: {missing}")
