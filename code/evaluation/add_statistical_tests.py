"""
Add statistical tests to robustness analysis.
Performs Friedman test and pairwise Wilcoxon signed-rank tests
to assess whether window choice significantly affects performance.

Usage:
    python code/evaluation/add_statistical_tests.py

Outputs:
    - results/robustness/statistical_tests.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from scipy import stats
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
ROBUSTNESS_DIR = PROJECT_ROOT / 'results' / 'robustness'


def friedman_test(results_df: pd.DataFrame, metric: str = 'auroc') -> dict:
    """
    Perform Friedman test for repeated measures.
    Tests if there's a significant difference across windows.

    The Friedman test requires at least 3 "subjects" (model-cohort combos)
    measured across 3 "conditions" (windows).
    """
    windows = ['6h', '12h', '24h']

    # Each subject = unique (model, cohort) pair
    subjects = results_df.groupby(['model', 'cohort']).ngroups
    data_matrix = []

    for (model, cohort), group in results_df.groupby(['model', 'cohort']):
        values = []
        for w in windows:
            row = group[group['window'] == w]
            if len(row) > 0:
                values.append(row[metric].values[0])
        if len(values) == 3:
            data_matrix.append(values)

    if len(data_matrix) < 3:
        return {
            'test': 'Friedman',
            'metric': metric,
            'error': f'Not enough complete subjects for Friedman test (need >=3, got {len(data_matrix)})',
            'n_subjects': len(data_matrix)
        }

    # Friedman test: columns = conditions (windows), rows = subjects
    data_array = np.array(data_matrix)
    try:
        stat, p_value = stats.friedmanchisquare(
            data_array[:, 0], data_array[:, 1], data_array[:, 2]
        )
    except Exception as e:
        return {
            'test': 'Friedman',
            'metric': metric,
            'error': str(e)
        }

    return {
        'test': 'Friedman',
        'metric': metric,
        'statistic': round(float(stat), 4),
        'p_value': round(float(p_value), 6),
        'n_subjects': len(data_matrix),
        'n_windows': 3,
        'significant_at_005': bool(p_value < 0.05),
        'significant_at_001': bool(p_value < 0.01),
        'interpretation': (
            'Significant difference across windows'
            if p_value < 0.05
            else 'No significant difference across windows'
        )
    }


def wilcoxon_pairwise(results_df: pd.DataFrame, metric: str = 'auroc') -> list:
    """
    Perform pairwise Wilcoxon signed-rank tests between windows.
    """
    pairs = [('6h', '12h'), ('12h', '24h'), ('6h', '24h')]
    pairwise_results = []

    for w1, w2 in pairs:
        values_w1 = []
        values_w2 = []

        for (model, cohort), group in results_df.groupby(['model', 'cohort']):
            r1 = group[group['window'] == w1]
            r2 = group[group['window'] == w2]

            if len(r1) > 0 and len(r2) > 0:
                values_w1.append(r1[metric].values[0])
                values_w2.append(r2[metric].values[0])

        if len(values_w1) < 5:
            pairwise_results.append({
                'comparison': f'{w1} vs {w2}',
                'metric': metric,
                'error': f'Not enough pairs (need >=5, got {len(values_w1)})'
            })
            continue

        arr1 = np.array(values_w1)
        arr2 = np.array(values_w2)
        diff = arr2 - arr1

        try:
            stat, p_value = stats.wilcoxon(arr1, arr2, alternative='two-sided')
            pairwise_results.append({
                'comparison': f'{w1} vs {w2}',
                'metric': metric,
                'statistic': round(float(stat), 4),
                'p_value': round(float(p_value), 6),
                'n_pairs': len(values_w1),
                'mean_diff': round(float(diff.mean()), 4),
                'median_diff': round(float(np.median(diff)), 4),
                'direction': 'improvement' if diff.mean() > 0 else 'decline',
                'significant_at_005': bool(p_value < 0.05),
                'significant_at_001': bool(p_value < 0.01),
            })
        except Exception as e:
            pairwise_results.append({
                'comparison': f'{w1} vs {w2}',
                'metric': metric,
                'error': str(e)
            })

    return pairwise_results


def effect_size_analysis(results_df: pd.DataFrame, metric: str = 'auroc') -> dict:
    """Compute effect sizes for window transitions."""
    windows = ['6h', '12h', '24h']
    effects = {}

    for w1, w2 in [('6h', '12h'), ('12h', '24h'), ('6h', '24h')]:
        vals_w1 = results_df[results_df['window'] == w1][metric].values
        vals_w2 = results_df[results_df['window'] == w2][metric].values

        if len(vals_w1) > 0 and len(vals_w2) > 0:
            # Cohen's d
            pooled_std = np.sqrt((vals_w1.std()**2 + vals_w2.std()**2) / 2)
            cohens_d = (vals_w2.mean() - vals_w1.mean()) / pooled_std if pooled_std > 0 else 0

            effects[f'{w1}_to_{w2}'] = {
                'mean_change': round(float(vals_w2.mean() - vals_w1.mean()), 4),
                'cohens_d': round(float(cohens_d), 4),
                'effect_magnitude': (
                    'negligible' if abs(cohens_d) < 0.2 else
                    'small' if abs(cohens_d) < 0.5 else
                    'medium' if abs(cohens_d) < 0.8 else
                    'large'
                )
            }

    return effects


def main():
    print("=" * 60)
    print("Adding Statistical Tests to Robustness Analysis")
    print("=" * 60)

    # Load results
    results_file = ROBUSTNESS_DIR / 'window_performance.csv'
    if not results_file.exists():
        print(f"Results file not found: {results_file}")
        return

    results_df = pd.read_csv(results_file)
    print(f"Loaded {len(results_df)} result rows")
    print(f"Models: {results_df['model'].unique()}")
    print(f"Windows: {results_df['window'].unique()}")
    print(f"Tasks: {results_df['task'].unique()}")
    print(f"Cohorts: {results_df['cohort'].unique()}")

    # Run tests for each task
    all_tests = {}

    for task in results_df['task'].unique():
        task_df = results_df[results_df['task'] == task]
        print(f"\n--- Task: {task} ---")

        for metric in ['auroc', 'auprc']:
            print(f"\n  Metric: {metric}")

            # Friedman test
            friedman_result = friedman_test(task_df, metric)
            print(f"    Friedman: chi2={friedman_result.get('statistic', 'N/A')}, "
                  f"p={friedman_result.get('p_value', 'N/A')}")

            # Pairwise Wilcoxon
            wilcoxon_results = wilcoxon_pairwise(task_df, metric)
            for wr in wilcoxon_results:
                if 'error' not in wr:
                    print(f"    Wilcoxon {wr['comparison']}: p={wr['p_value']}, "
                          f"diff={wr['mean_diff']}, sig={wr['significant_at_005']}")

            # Effect sizes
            effects = effect_size_analysis(task_df, metric)
            for key, eff in effects.items():
                print(f"    Effect {key}: d={eff['cohens_d']} ({eff['effect_magnitude']})")

            all_tests[f'{task}_{metric}'] = {
                'friedman_test': friedman_result,
                'pairwise_wilcoxon': wilcoxon_results,
                'effect_sizes': effects
            }

    # Save results
    output = {
        'generated_at': datetime.now().isoformat(),
        'description': 'Statistical tests for cross-window robustness analysis',
        'methodology': {
            'friedman': 'Non-parametric test for repeated measures across 3+ conditions. '
                       'Tests if window choice significantly affects performance.',
            'wilcoxon': 'Pairwise non-parametric test between two conditions. '
                       'Tests if specific window pairs differ significantly.',
            'effect_size': 'Cohens d effect size for practical significance. '
                          'negligible (<0.2), small (0.2-0.5), medium (0.5-0.8), large (>0.8)'
        },
        'tests_by_task_metric': all_tests
    }

    ROBUSTNESS_DIR.mkdir(parents=True, exist_ok=True)
    output_file = ROBUSTNESS_DIR / 'statistical_tests.json'
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nStatistical tests saved to {output_file}")


if __name__ == '__main__':
    main()
