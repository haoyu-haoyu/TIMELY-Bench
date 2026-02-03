"""
Cross-window Robustness Analysis for TIMELY-Bench
分析模型在不同时间窗口下的鲁棒性

输出:
- results/robustness/robustness_analysis.json
- results/robustness/window_performance_heatmap.png
- results/robustness/window_performance_lineplot.png
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
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 配置
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'robustness'
WINDOWS = ['6h', '12h', '24h']
COHORTS = ['all', 'sepsis', 'aki']
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_features(window: str) -> pd.DataFrame:
    """加载指定窗口的特征数据"""
    features_file = DATA_DIR / 'data_windows' / f'window_{window}' / 'features_aggregated.csv'
    if not features_file.exists():
        return None
    return pd.read_csv(features_file)


def load_cohort() -> pd.DataFrame:
    """加载队列数据"""
    cohort_file = DATA_DIR / 'merge_output' / 'cohort_final.csv'
    df = pd.read_csv(cohort_file)

    if 'label_mortality' in df.columns and 'mortality' not in df.columns:
        df['mortality'] = df['label_mortality']
    if 'prolonged_los_3d' in df.columns and 'prolonged_los' not in df.columns:
        df['prolonged_los'] = df['prolonged_los_3d']
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
    return np.ones(len(cohort_df), dtype=bool)


def prepare_data(features_df: pd.DataFrame, cohort_df: pd.DataFrame, cohort_name: str, task: str):
    """准备训练数据"""
    merged = features_df.merge(
        cohort_df[['stay_id', 'subject_id', 'mortality', 'prolonged_los', 'has_sepsis', 'has_aki']],
        on='stay_id',
        how='inner'
    )

    mask = get_cohort_mask(merged, cohort_name)
    merged = merged[mask].reset_index(drop=True)

    exclude_cols = ['stay_id', 'subject_id', 'mortality', 'prolonged_los', 'has_sepsis', 'has_aki']
    feature_cols = [c for c in merged.columns if c not in exclude_cols]

    X = merged[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = merged[task].values
    groups = merged['subject_id'].values

    return X, y, groups


def train_and_evaluate(X_train, y_train, X_test, y_test, model_name: str):
    """训练模型并返回测试集指标"""
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

    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)

    return auroc, auprc


def compute_robustness_metrics(results_df: pd.DataFrame) -> dict:
    """计算跨窗口鲁棒性指标"""
    metrics = {}

    for model in results_df['model'].unique():
        for task in results_df['task'].unique():
            for cohort in results_df['cohort'].unique():
                subset = results_df[
                    (results_df['model'] == model) &
                    (results_df['task'] == task) &
                    (results_df['cohort'] == cohort)
                ]

                if len(subset) < 2:
                    continue

                auroc_values = subset['auroc'].values
                auprc_values = subset['auprc'].values

                key = f"{model}_{task}_{cohort}"
                metrics[key] = {
                    'auroc_mean': float(auroc_values.mean()),
                    'auroc_std': float(auroc_values.std()),
                    'auroc_cv': float(auroc_values.std() / auroc_values.mean()) if auroc_values.mean() > 0 else 0,
                    'auroc_range': float(auroc_values.max() - auroc_values.min()),
                    'auroc_min': float(auroc_values.min()),
                    'auroc_max': float(auroc_values.max()),
                    'auprc_mean': float(auprc_values.mean()),
                    'auprc_std': float(auprc_values.std()),
                    'auprc_cv': float(auprc_values.std() / auprc_values.mean()) if auprc_values.mean() > 0 else 0,
                    'performance_drop_6h_24h': float(
                        subset[subset['window'] == '6h']['auroc'].values[0] -
                        subset[subset['window'] == '24h']['auroc'].values[0]
                    ) if len(subset) == 3 else None
                }

    return metrics


def plot_performance_heatmap(results_df: pd.DataFrame, task: str, save_path: Path):
    """绘制性能热力图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, metric in enumerate(['auroc', 'auprc']):
        pivot = results_df[results_df['task'] == task].pivot_table(
            values=metric,
            index=['model', 'cohort'],
            columns='window',
            aggfunc='mean'
        )

        # 重新排序列
        pivot = pivot[['6h', '12h', '24h']]

        sns.heatmap(
            pivot, annot=True, fmt='.3f', cmap='RdYlGn',
            ax=axes[idx], vmin=0.5, vmax=1.0 if metric == 'auroc' else 0.6
        )
        axes[idx].set_title(f'{metric.upper()} by Window ({task})', fontsize=12)
        axes[idx].set_xlabel('Time Window')
        axes[idx].set_ylabel('')

    plt.suptitle(f'Cross-Window Performance: {task.replace("_", " ").title()}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved heatmap to {save_path}")


def plot_performance_lineplot(results_df: pd.DataFrame, task: str, save_path: Path):
    """绘制性能变化折线图"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # AUROC
    for model in results_df['model'].unique():
        for cohort in results_df['cohort'].unique():
            subset = results_df[
                (results_df['model'] == model) &
                (results_df['task'] == task) &
                (results_df['cohort'] == cohort)
            ].sort_values('window')

            if len(subset) < 2:
                continue

            window_order = ['6h', '12h', '24h']
            subset['window_idx'] = subset['window'].apply(lambda x: window_order.index(x))
            subset = subset.sort_values('window_idx')

            linestyle = '-' if model == 'XGBoost' else '--'
            marker = 'o' if cohort == 'all' else ('s' if cohort == 'sepsis' else '^')

            axes[0].plot(
                subset['window'], subset['auroc'],
                marker=marker, linestyle=linestyle,
                label=f'{model} ({cohort})'
            )

            axes[1].plot(
                subset['window'], subset['auprc'],
                marker=marker, linestyle=linestyle,
                label=f'{model} ({cohort})'
            )

    axes[0].set_xlabel('Time Window')
    axes[0].set_ylabel('AUROC')
    axes[0].set_title(f'AUROC vs Time Window ({task})')
    axes[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Time Window')
    axes[1].set_ylabel('AUPRC')
    axes[1].set_title(f'AUPRC vs Time Window ({task})')
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'Performance Stability Across Windows: {task.replace("_", " ").title()}', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved lineplot to {save_path}")


def run_robustness_analysis():
    """运行跨窗口鲁棒性分析"""
    print("=" * 60)
    print("TIMELY-Bench Cross-Window Robustness Analysis")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cohort_df = load_cohort()
    print(f"Loaded cohort: {len(cohort_df)} samples")

    all_results = []

    for window in WINDOWS:
        print(f"\n--- Window: {window} ---")

        features_df = load_features(window)
        if features_df is None:
            continue

        for task in ['mortality', 'prolonged_los']:
            for cohort_name in COHORTS:
                for model_name in ['XGBoost', 'LogisticRegression']:

                    print(f"  {window}_{task}_{cohort_name}_{model_name}...")

                    try:
                        X, y, groups = prepare_data(features_df, cohort_df, cohort_name, task)

                        if len(y) < 100:
                            continue

                        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
                        train_idx, test_idx = next(gss.split(X, y, groups))

                        X_train, X_test = X[train_idx], X[test_idx]
                        y_train, y_test = y[train_idx], y[test_idx]

                        auroc, auprc = train_and_evaluate(X_train, y_train, X_test, y_test, model_name)

                        all_results.append({
                            'window': window,
                            'task': task,
                            'cohort': cohort_name,
                            'model': model_name,
                            'auroc': auroc,
                            'auprc': auprc,
                            'n_samples': len(y),
                            'n_test': len(y_test)
                        })

                        print(f"    AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

                    except Exception as e:
                        print(f"    Error: {e}")
                        continue

    results_df = pd.DataFrame(all_results)

    # 计算鲁棒性指标
    robustness_metrics = compute_robustness_metrics(results_df)

    # 保存结果
    results_df.to_csv(RESULTS_DIR / 'window_performance.csv', index=False)

    with open(RESULTS_DIR / 'robustness_analysis.json', 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'windows': WINDOWS,
            'cohorts': COHORTS,
            'robustness_metrics': robustness_metrics,
            'raw_results': all_results
        }, f, indent=2, default=float)

    print(f"\nResults saved to {RESULTS_DIR}")

    # 生成可视化
    print("\n--- Generating Visualizations ---")

    for task in ['mortality', 'prolonged_los']:
        plot_performance_heatmap(
            results_df, task,
            RESULTS_DIR / f'heatmap_{task}.png'
        )
        plot_performance_lineplot(
            results_df, task,
            RESULTS_DIR / f'lineplot_{task}.png'
        )

    # 打印汇总
    print("\n" + "=" * 60)
    print("Robustness Summary (mortality, all cohort)")
    print("=" * 60)

    for model in ['XGBoost', 'LogisticRegression']:
        key = f"{model}_mortality_all"
        if key in robustness_metrics:
            m = robustness_metrics[key]
            print(f"\n{model}:")
            print(f"  AUROC: {m['auroc_mean']:.4f} ± {m['auroc_std']:.4f} (CV={m['auroc_cv']:.3f})")
            print(f"  AUROC Range: {m['auroc_min']:.4f} - {m['auroc_max']:.4f}")
            if m['performance_drop_6h_24h'] is not None:
                print(f"  6h→24h Drop: {m['performance_drop_6h_24h']:.4f}")

    return results_df, robustness_metrics


if __name__ == "__main__":
    run_robustness_analysis()
