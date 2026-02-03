"""
Update Robustness Analysis with All 4 Models.
Re-runs statistical tests and regenerates visualizations.

Usage:
    python code/evaluation/update_robustness_final.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from config import RESULTS_DIR

ROBUSTNESS_DIR = RESULTS_DIR / 'robustness'
INPUT_FILE = ROBUSTNESS_DIR / 'window_performance.csv'
STATS_OUTPUT = ROBUSTNESS_DIR / 'statistical_tests.json'


def run_statistical_tests(df):
    """Run Friedman and Wilcoxon tests with all 4 models."""
    print("\n" + "="*60)
    print("Running Statistical Tests (4 models)")
    print("="*60)

    results = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Statistical tests for cross-window robustness analysis',
        'n_models': len(df['model'].unique()),
        'models': df['model'].unique().tolist(),
        'methodology': {
            'friedman': 'Non-parametric test for repeated measures across 3+ conditions.',
            'wilcoxon': 'Pairwise non-parametric test between two conditions.',
            'effect_size': 'Cohen\'s d effect size. negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)'
        },
        'tests_by_task_metric': {}
    }

    for task in ['mortality', 'prolonged_los']:
        for metric in ['auroc', 'auprc']:
            key = f'{task}_{metric}'
            task_df = df[df['task'] == task]

            # Prepare data matrix: rows = (model, cohort), columns = windows
            windows = ['6h', '12h', '24h']
            models = task_df['model'].unique()
            cohorts = task_df['cohort'].unique()

            # Build data matrix
            data_matrix = []
            labels = []
            for model in models:
                for cohort in cohorts:
                    row = []
                    valid = True
                    for window in windows:
                        val = task_df[
                            (task_df['model'] == model) &
                            (task_df['cohort'] == cohort) &
                            (task_df['window'] == window)
                        ][metric]
                        if len(val) == 0:
                            valid = False
                            break
                        row.append(val.values[0])
                    if valid:
                        data_matrix.append(row)
                        labels.append(f'{model}_{cohort}')

            if len(data_matrix) < 3:
                print(f"  {key}: Insufficient data ({len(data_matrix)} subjects)")
                continue

            data_matrix = np.array(data_matrix)
            n_subjects = len(data_matrix)

            print(f"\n  {key}: {n_subjects} subjects (model×cohort combinations)")

            # Friedman test
            try:
                stat, p_value = stats.friedmanchisquare(
                    data_matrix[:, 0],
                    data_matrix[:, 1],
                    data_matrix[:, 2]
                )
                friedman_result = {
                    'test': 'Friedman',
                    'metric': metric,
                    'statistic': round(float(stat), 4),
                    'p_value': round(float(p_value), 6),
                    'n_subjects': n_subjects,
                    'n_windows': 3,
                    'significant_at_005': bool(p_value < 0.05),
                    'significant_at_001': bool(p_value < 0.01),
                    'interpretation': 'Significant difference across windows' if p_value < 0.05 else 'No significant difference'
                }
                print(f"    Friedman: χ²={stat:.2f}, p={p_value:.6f} {'*' if p_value < 0.05 else ''}")
            except Exception as e:
                friedman_result = {'error': str(e)}

            # Pairwise Wilcoxon tests
            wilcoxon_results = []
            effect_sizes = {}

            pairs = [('6h', '12h', 0, 1), ('12h', '24h', 1, 2), ('6h', '24h', 0, 2)]
            for name1, name2, i, j in pairs:
                try:
                    stat, p_value = stats.wilcoxon(
                        data_matrix[:, i],
                        data_matrix[:, j],
                        alternative='two-sided'
                    )
                    diff = data_matrix[:, j] - data_matrix[:, i]
                    mean_diff = float(np.mean(diff))
                    median_diff = float(np.median(diff))

                    wilcoxon_results.append({
                        'comparison': f'{name1} vs {name2}',
                        'metric': metric,
                        'statistic': round(float(stat), 4),
                        'p_value': round(float(p_value), 6),
                        'n_pairs': n_subjects,
                        'mean_diff': round(mean_diff, 4),
                        'median_diff': round(median_diff, 4),
                        'direction': 'improvement' if mean_diff > 0 else 'decline',
                        'significant_at_005': bool(p_value < 0.05),
                        'significant_at_001': bool(p_value < 0.01)
                    })

                    # Cohen's d
                    pooled_std = np.std(np.concatenate([data_matrix[:, i], data_matrix[:, j]]))
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    magnitude = 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small' if abs(cohens_d) > 0.2 else 'negligible'

                    effect_sizes[f'{name1}_to_{name2}'] = {
                        'mean_change': round(mean_diff, 4),
                        'cohens_d': round(float(cohens_d), 4),
                        'effect_magnitude': magnitude
                    }

                    print(f"    {name1} vs {name2}: W={stat:.1f}, p={p_value:.4f}, d={cohens_d:.2f} ({magnitude})")

                except Exception as e:
                    wilcoxon_results.append({
                        'comparison': f'{name1} vs {name2}',
                        'error': str(e)
                    })

            results['tests_by_task_metric'][key] = {
                'friedman_test': friedman_result,
                'pairwise_wilcoxon': wilcoxon_results,
                'effect_sizes': effect_sizes
            }

    return results


def generate_visualizations(df):
    """Generate heatmaps and line plots for all 4 models."""
    print("\n" + "="*60)
    print("Generating Visualizations (4 models)")
    print("="*60)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    for task in ['mortality', 'prolonged_los']:
        task_df = df[(df['task'] == task) & (df['cohort'] == 'all')]

        if task_df.empty:
            continue

        # Pivot for heatmap
        pivot_auroc = task_df.pivot(index='model', columns='window', values='auroc')

        # Reorder columns
        if set(['6h', '12h', '24h']).issubset(pivot_auroc.columns):
            pivot_auroc = pivot_auroc[['6h', '12h', '24h']]

        # Sort models by 24h performance
        if '24h' in pivot_auroc.columns:
            pivot_auroc = pivot_auroc.sort_values('24h', ascending=False)

        # 1. Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            pivot_auroc,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0.6,
            vmax=0.9,
            ax=ax,
            cbar_kws={'label': 'AUROC'}
        )
        ax.set_title(f'{task.replace("_", " ").title()} Prediction - AUROC by Window and Model', fontsize=14)
        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)

        heatmap_path = ROBUSTNESS_DIR / f'heatmap_{task}.png'
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {heatmap_path}")

        # 2. Line plot
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = {'XGBoost': '#2ecc71', 'LogisticRegression': '#3498db',
                  'ClinicalGRU': '#e74c3c', 'EarlyFusion_XGBoost': '#9b59b6'}
        markers = {'XGBoost': 'o', 'LogisticRegression': 's',
                   'ClinicalGRU': '^', 'EarlyFusion_XGBoost': 'D'}

        for model in task_df['model'].unique():
            model_df = task_df[task_df['model'] == model].sort_values('window')
            ax.plot(
                model_df['window'],
                model_df['auroc'],
                marker=markers.get(model, 'o'),
                markersize=10,
                linewidth=2,
                color=colors.get(model, '#333'),
                label=model
            )

        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'{task.replace("_", " ").title()} Prediction - AUROC Trend Across Windows', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.6, 0.9)

        lineplot_path = ROBUSTNESS_DIR / f'lineplot_{task}.png'
        plt.tight_layout()
        plt.savefig(lineplot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {lineplot_path}")

    # 3. Combined comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, task in enumerate(['mortality', 'prolonged_los']):
        task_df = df[(df['task'] == task) & (df['cohort'] == 'all')]
        ax = axes[idx]

        x = np.arange(3)
        width = 0.2
        models = ['XGBoost', 'LogisticRegression', 'ClinicalGRU', 'EarlyFusion_XGBoost']

        for i, model in enumerate(models):
            if model not in task_df['model'].values:
                continue
            model_df = task_df[task_df['model'] == model]
            aurocs = []
            for w in ['6h', '12h', '24h']:
                val = model_df[model_df['window'] == w]['auroc']
                aurocs.append(val.values[0] if len(val) > 0 else 0)

            ax.bar(x + i*width, aurocs, width, label=model,
                   color=colors.get(model, '#333'), alpha=0.8)

        ax.set_xlabel('Time Window', fontsize=12)
        ax.set_ylabel('AUROC', fontsize=12)
        ax.set_title(f'{task.replace("_", " ").title()}', fontsize=14)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(['6h', '12h', '24h'])
        ax.legend(loc='lower right', fontsize=9)
        ax.set_ylim(0.6, 0.9)
        ax.grid(True, alpha=0.3, axis='y')

    plt.suptitle('Cross-Window Robustness: All Models Comparison', fontsize=16, y=1.02)
    plt.tight_layout()

    comparison_path = ROBUSTNESS_DIR / 'model_comparison_all.png'
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_path}")


def main():
    print("="*60)
    print("Update Robustness Analysis - Final (4 Models)")
    print("="*60)

    # Load data
    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded: {INPUT_FILE}")
    print(f"  Rows: {len(df)}")
    print(f"  Models: {df['model'].unique().tolist()}")
    print(f"  Windows: {df['window'].unique().tolist()}")

    # Run statistical tests
    stats_results = run_statistical_tests(df)

    # Save stats
    with open(STATS_OUTPUT, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"\nStatistical tests saved to: {STATS_OUTPUT}")

    # Generate visualizations
    generate_visualizations(df)

    print("\n" + "="*60)
    print("Update Complete!")
    print("="*60)
    print(f"\nUpdated files:")
    print(f"  - {STATS_OUTPUT}")
    print(f"  - {ROBUSTNESS_DIR}/heatmap_mortality.png")
    print(f"  - {ROBUSTNESS_DIR}/heatmap_prolonged_los.png")
    print(f"  - {ROBUSTNESS_DIR}/lineplot_mortality.png")
    print(f"  - {ROBUSTNESS_DIR}/lineplot_prolonged_los.png")
    print(f"  - {ROBUSTNESS_DIR}/model_comparison_all.png")


if __name__ == '__main__':
    main()
