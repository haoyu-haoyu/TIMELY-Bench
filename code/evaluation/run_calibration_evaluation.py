"""
Run Calibration Evaluation for TIMELY-Bench
运行所有基线模型的校准评估

输出:
- results/calibration/calibration_summary.json
- results/calibration/calibration_summary.csv
- results/calibration/reliability_diagrams/

Notes:
- 作业要求.md 里明确提到 calibration (ECE/HL)，因此这里会同时输出 HL 统计量。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from calibration_metrics import (
    evaluate_calibration,
    plot_reliability_diagram,
    plot_multi_model_reliability
)

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'calibration'
WINDOWS = ['6h', '12h', '24h']
COHORTS = ['all', 'sepsis', 'aki']
RANDOM_STATE = 42
TEST_SIZE = 0.2
N_BINS = 10


def load_features(window: str) -> pd.DataFrame:
    """加载指定窗口的特征数据"""
    features_file = DATA_DIR / 'data_windows' / f'window_{window}' / 'features_aggregated.csv'
    if not features_file.exists():
        print(f"Warning: {features_file} not found")
        return None
    return pd.read_csv(features_file)


def load_cohort() -> pd.DataFrame:
    """加载队列数据（包含标签和队列信息）"""
    cohort_file = DATA_DIR / 'merge_output' / 'cohort_final.csv'
    df = pd.read_csv(cohort_file)

    # 列名映射：统一为标准名称
    if 'label_mortality' in df.columns and 'mortality' not in df.columns:
        df['mortality'] = df['label_mortality']

    # Prolonged LOS label: keep consistent with the benchmark definition (>7 days).
    if 'prolonged_los' not in df.columns:
        if 'prolonged_los_7d' in df.columns:
            df['prolonged_los'] = df['prolonged_los_7d']
        elif 'prolonged_los_3d' in df.columns:
            df['prolonged_los'] = df['prolonged_los_3d']

    # 使用has_sepsis_final替代has_sepsis（如果存在）
    if 'has_sepsis_final' in df.columns:
        df['has_sepsis'] = df['has_sepsis_final']
    if 'has_aki_final' in df.columns:
        df['has_aki'] = df['has_aki_final']

    return df


def get_cohort_mask(cohort_df: pd.DataFrame, cohort_name: str) -> np.ndarray:
    """获取队列筛选mask"""
    if cohort_name == 'all':
        return np.ones(len(cohort_df), dtype=bool)
    elif cohort_name == 'sepsis':
        return cohort_df['has_sepsis'].values == 1
    elif cohort_name == 'aki':
        return cohort_df['has_aki'].values == 1
    else:
        return np.ones(len(cohort_df), dtype=bool)


def prepare_data(features_df: pd.DataFrame, cohort_df: pd.DataFrame, cohort_name: str, task: str):
    """准备训练数据"""
    # 合并特征和标签
    merged = features_df.merge(
        cohort_df[['stay_id', 'subject_id', 'mortality', 'prolonged_los', 'has_sepsis', 'has_aki']],
        on='stay_id',
        how='inner'
    )

    # 应用队列筛选
    mask = get_cohort_mask(merged, cohort_name)
    merged = merged[mask].reset_index(drop=True)

    # 准备特征列
    exclude_cols = ['stay_id', 'subject_id', 'mortality', 'prolonged_los', 'has_sepsis', 'has_aki']
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    X = merged[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = merged[task].values
    groups = merged['subject_id'].values

    return X, y, groups


def train_and_predict(X_train, y_train, X_test, model_name: str):
    """训练模型并返回测试集预测概率"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
    else:
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)

    model.fit(X_train_scaled, y_train)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    return y_prob


def run_calibration_evaluation():
    """运行完整的校准评估"""
    print("=" * 60)
    print("TIMELY-Bench Calibration Evaluation")
    print("=" * 60)

    # 创建输出目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / 'reliability_diagrams').mkdir(exist_ok=True)

    # 加载队列数据
    cohort_df = load_cohort()
    print(f"Loaded cohort: {len(cohort_df)} samples")

    all_results = []
    predictions_store = {}  # 用于绘图

    for window in WINDOWS:
        print(f"\n--- Window: {window} ---")

        features_df = load_features(window)
        if features_df is None:
            continue

        for task in ['mortality', 'prolonged_los']:
            for cohort_name in COHORTS:
                for model_name in ['XGBoost', 'LogisticRegression']:

                    key = f"{window}_{task}_{cohort_name}_{model_name}"
                    print(f"  Evaluating: {key}")

                    try:
                        # 准备数据
                        X, y, groups = prepare_data(features_df, cohort_df, cohort_name, task)

                        if len(y) < 100:
                            print(f"    Skipping (only {len(y)} samples)")
                            continue

                        # 分割数据 (基于subject_id分组)
                        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                        train_idx, test_idx = next(gss.split(X, y, groups))

                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        # 训练并预测
                        y_prob = train_and_predict(X_train, y_train, X_test, model_name)

                        # 计算校准指标
                        metrics = evaluate_calibration(y_test, y_prob, N_BINS)

                        # 记录结果
                        result = {
                            'window': window,
                            'task': task,
                            'cohort': cohort_name,
                            'model': model_name,
                            'n_samples': len(y),
                            'n_test': len(y_test),
                            **metrics
                        }
                        all_results.append(result)

                        # 保存预测用于绘图
                        predictions_store[key] = (y_test, y_prob)

                        hl = metrics.get("hl_statistic", float("nan"))
                        print(f"    ECE={metrics['ece']:.4f}, Brier={metrics['brier_score']:.4f}, HL={hl:.3f}")

                    except Exception as e:
                        print(f"    Error: {e}")
                        continue

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(RESULTS_DIR / 'calibration_summary.csv', index=False)

    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(RESULTS_DIR / 'calibration_summary.json', 'w') as f:
        json.dump(convert_to_native({
            'generated_at': datetime.now().isoformat(),
            'n_bins': N_BINS,
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'results': all_results
        }), f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'calibration_summary.csv'}")

    # 生成可靠性图
    print("\n--- Generating Reliability Diagrams ---")

    # 为主要配置生成单独的图
    for key, (y_true, y_prob) in predictions_store.items():
        if '24h_mortality_all' in key:  # 主要配置
            fig = plot_reliability_diagram(
                y_true, y_prob, N_BINS,
                title=key.replace('_', ' ').title(),
                save_path=RESULTS_DIR / 'reliability_diagrams' / f'{key}.png'
            )
            plt.close(fig)

    # 生成对比图 (24h mortality all)
    mortality_models = {
        k.split('_')[-1]: v
        for k, v in predictions_store.items()
        if '24h_mortality_all' in k
    }

    if mortality_models:
        fig = plot_multi_model_reliability(
            mortality_models, N_BINS,
            title='Mortality Prediction Calibration (24h, All)',
            save_path=RESULTS_DIR / 'reliability_diagrams' / 'mortality_24h_all_comparison.png'
        )
        plt.close(fig)

    # 打印汇总
    print("\n" + "=" * 60)
    print("Calibration Summary (24h, mortality, all cohort)")
    print("=" * 60)
    summary = results_df[
        (results_df['window'] == '24h') &
        (results_df['task'] == 'mortality') &
        (results_df['cohort'] == 'all')
    ][['model', 'ece', 'mce', 'brier_score', 'hl_statistic', 'hl_p_value', 'n_test']]
    print(summary.to_string(index=False))

    return results_df


if __name__ == "__main__":
    run_calibration_evaluation()
