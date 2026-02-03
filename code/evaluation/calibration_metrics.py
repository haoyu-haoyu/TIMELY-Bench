"""
Calibration Metrics Module for TIMELY-Bench
计算模型校准指标: ECE, MCE, Brier Score
生成可靠性图 (Reliability Diagrams)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
from pathlib import Path


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    计算Expected Calibration Error (ECE)

    ECE = sum(|accuracy(bin) - confidence(bin)| * |bin| / N)

    Args:
        y_true: 真实标签 (0 or 1)
        y_prob: 预测概率
        n_bins: 分箱数量

    Returns:
        ECE值
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        # 找到该bin中的样本
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            # 计算该bin的准确率和置信度
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()

            # 加权累加
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

    return ece


def compute_mce(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    计算Maximum Calibration Error (MCE)

    MCE = max(|accuracy(bin) - confidence(bin)|)

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量

    Returns:
        MCE值
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_ce = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)

        if in_bin.sum() > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ce = np.abs(accuracy_in_bin - avg_confidence_in_bin)
            max_ce = max(max_ce, ce)

    return max_ce


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    计算Brier Score

    Brier = mean((y_prob - y_true)^2)

    Args:
        y_true: 真实标签
        y_prob: 预测概率

    Returns:
        Brier Score
    """
    return np.mean((y_prob - y_true) ** 2)


def get_calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    获取校准曲线数据

    Returns:
        bin_centers: 各bin中心点
        bin_accuracies: 各bin的准确率
        bin_counts: 各bin的样本数
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        bin_center = (bin_lower + bin_upper) / 2

        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        count = in_bin.sum()

        if count > 0:
            accuracy = y_true[in_bin].mean()
            bin_centers.append(bin_center)
            bin_accuracies.append(accuracy)
            bin_counts.append(count)

    return np.array(bin_centers), np.array(bin_accuracies), np.array(bin_counts)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None,
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    绘制可靠性图 (Reliability Diagram)

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量
        title: 图表标题
        save_path: 保存路径
        ax: matplotlib axes (用于子图)

    Returns:
        matplotlib Figure
    """
    bin_centers, bin_accuracies, bin_counts = get_calibration_curve_data(y_true, y_prob, n_bins)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.get_figure()

    # 绘制完美校准线
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')

    # 绘制校准曲线
    ax.plot(bin_centers, bin_accuracies, 'o-', color='blue', label='Model calibration')

    # 填充误差区域
    ax.fill_between(
        bin_centers,
        bin_centers,
        bin_accuracies,
        alpha=0.2,
        color='blue'
    )

    # 计算并显示指标
    ece = compute_ece(y_true, y_prob, n_bins)
    mce = compute_mce(y_true, y_prob, n_bins)
    brier = compute_brier_score(y_true, y_prob)

    ax.text(0.05, 0.95, f'ECE: {ece:.4f}\nMCE: {mce:.4f}\nBrier: {brier:.4f}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax.set_ylabel('Fraction of Positives', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reliability diagram to {save_path}")

    return fig


def plot_multi_model_reliability(
    results_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_bins: int = 10,
    title: str = "Multi-Model Reliability Comparison",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    绘制多模型校准对比图

    Args:
        results_dict: {model_name: (y_true, y_prob)} 字典
        n_bins: 分箱数量
        title: 图表标题
        save_path: 保存路径

    Returns:
        matplotlib Figure
    """
    n_models = len(results_dict)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, (model_name, (y_true, y_prob)) in enumerate(results_dict.items()):
        plot_reliability_diagram(
            y_true, y_prob, n_bins,
            title=model_name,
            ax=axes[i]
        )

    # 隐藏空白子图
    for i in range(n_models, len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-model reliability diagram to {save_path}")

    return fig


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    综合评估校准性能

    Args:
        y_true: 真实标签
        y_prob: 预测概率
        n_bins: 分箱数量

    Returns:
        包含所有校准指标的字典
    """
    return {
        'ece': compute_ece(y_true, y_prob, n_bins),
        'mce': compute_mce(y_true, y_prob, n_bins),
        'brier_score': compute_brier_score(y_true, y_prob),
        'n_samples': len(y_true),
        'positive_rate': y_true.mean(),
        'mean_predicted_prob': y_prob.mean(),
        'n_bins': n_bins
    }


if __name__ == "__main__":
    # 测试代码
    np.random.seed(42)

    # 生成模拟数据
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)
    y_prob_good = y_true * 0.7 + (1 - y_true) * 0.2 + np.random.normal(0, 0.1, n_samples)
    y_prob_good = np.clip(y_prob_good, 0, 1)

    # 评估
    metrics = evaluate_calibration(y_true, y_prob_good)
    print("Calibration Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 绘图
    fig = plot_reliability_diagram(y_true, y_prob_good, title="Test Reliability Diagram")
    plt.show()
