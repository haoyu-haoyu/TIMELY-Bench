"""
Merge DL Robustness Results with ML Results.
Updates window_performance.csv and re-runs statistical tests.

Usage:
    python code/evaluation/merge_dl_robustness.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from config import RESULTS_DIR

ROBUSTNESS_DIR = RESULTS_DIR / 'robustness'
ML_FILE = ROBUSTNESS_DIR / 'window_performance.csv'
DL_FILE = ROBUSTNESS_DIR / 'dl_window_performance.csv'
MERGED_FILE = ROBUSTNESS_DIR / 'window_performance_complete.csv'
STATS_FILE = ROBUSTNESS_DIR / 'statistical_tests_complete.json'
ANALYSIS_FILE = ROBUSTNESS_DIR / 'robustness_analysis_complete.json'


def load_and_merge():
    """Load ML and DL results and merge them."""
    print("Loading results files...")

    if not ML_FILE.exists():
        raise FileNotFoundError(f"ML results not found: {ML_FILE}")

    ml_df = pd.read_csv(ML_FILE)
    print(f"  ML results: {len(ml_df)} rows")

    if not DL_FILE.exists():
        print(f"  DL results not found: {DL_FILE}")
        return ml_df

    dl_df = pd.read_csv(DL_FILE)
    print(f"  DL results: {len(dl_df)} rows")

    # Standardize DL columns to match ML format
    # ML format: window,task,cohort,model,auroc,auprc,n_samples,n_test
    # DL format: window,task,cohort,model,n_samples,positive_rate,cv_auroc_mean,...

    dl_standardized = []
    for _, row in dl_df.iterrows():
        # Use test metrics if available, otherwise use CV mean
        auroc = row.get('test_auroc', row.get('cv_auroc_mean', np.nan))
        auprc = row.get('test_auprc', row.get('cv_auprc_mean', np.nan))

        dl_standardized.append({
            'window': row['window'],
            'task': row['task'],
            'cohort': row['cohort'],
            'model': row['model'],
            'auroc': auroc,
            'auprc': auprc,
            'n_samples': row['n_samples'],
            'n_test': int(row['n_samples'] * 0.2)  # Approximate test size
        })

    dl_std_df = pd.DataFrame(dl_standardized)

    # Merge, removing any duplicates from ML if DL has same model
    combined = pd.concat([ml_df, dl_std_df], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=['window', 'task', 'cohort', 'model'],
        keep='last'  # Keep DL results if there's overlap
    )

    print(f"  Combined: {len(combined)} rows")
    return combined


def compute_cross_window_cv(df):
    """Compute coefficient of variation across windows for each model-task-cohort."""
    results = []

    for task in df['task'].unique():
        for cohort in df['cohort'].unique():
            for model in df['model'].unique():
                mask = (df['task'] == task) & (df['cohort'] == cohort) & (df['model'] == model)
                subset = df[mask]

                if len(subset) < 2:
                    continue

                aurocs = subset['auroc'].values
                auprcs = subset['auprc'].values

                auroc_cv = np.std(aurocs) / np.mean(aurocs) if np.mean(aurocs) > 0 else 0
                auprc_cv = np.std(auprcs) / np.mean(auprcs) if np.mean(auprcs) > 0 else 0

                results.append({
                    'task': task,
                    'cohort': cohort,
                    'model': model,
                    'n_windows': len(subset),
                    'auroc_mean': float(np.mean(aurocs)),
                    'auroc_std': float(np.std(aurocs)),
                    'auroc_cv': float(auroc_cv),
                    'auprc_mean': float(np.mean(auprcs)),
                    'auprc_std': float(np.std(auprcs)),
                    'auprc_cv': float(auprc_cv),
                    'windows': subset['window'].tolist(),
                    'aurocs': aurocs.tolist(),
                    'auprcs': auprcs.tolist()
                })

    return results


def run_statistical_tests(df):
    """Run Friedman and Wilcoxon tests across models."""
    results = {
        'generated_at': datetime.now().isoformat(),
        'tests': []
    }

    for task in df['task'].unique():
        for cohort in df['cohort'].unique():
            task_df = df[(df['task'] == task) & (df['cohort'] == cohort)]
            models = task_df['model'].unique()

            if len(models) < 2:
                continue

            # Prepare data for Friedman test
            # Rows = windows, Columns = models
            windows = sorted(task_df['window'].unique())

            auroc_matrix = []
            model_order = []

            for model in models:
                model_aurocs = []
                valid = True
                for window in windows:
                    val = task_df[(task_df['model'] == model) & (task_df['window'] == window)]['auroc']
                    if len(val) == 0:
                        valid = False
                        break
                    model_aurocs.append(val.values[0])

                if valid:
                    auroc_matrix.append(model_aurocs)
                    model_order.append(model)

            if len(auroc_matrix) < 2 or len(auroc_matrix[0]) < 2:
                continue

            auroc_matrix = np.array(auroc_matrix).T  # windows × models

            # Friedman test
            try:
                if auroc_matrix.shape[0] >= 3:  # Need at least 3 groups
                    stat, p_value = stats.friedmanchisquare(*[auroc_matrix[:, i] for i in range(auroc_matrix.shape[1])])
                    friedman_result = {
                        'statistic': float(stat),
                        'p_value': float(p_value),
                        'significant': p_value < 0.05
                    }
                else:
                    friedman_result = {'note': 'Insufficient data for Friedman test'}
            except Exception as e:
                friedman_result = {'error': str(e)}

            # Pairwise Wilcoxon tests
            wilcoxon_results = []
            for i, model1 in enumerate(model_order):
                for j, model2 in enumerate(model_order):
                    if i >= j:
                        continue

                    try:
                        stat, p_value = stats.wilcoxon(
                            auroc_matrix[:, i],
                            auroc_matrix[:, j],
                            alternative='two-sided'
                        )
                        wilcoxon_results.append({
                            'model1': model1,
                            'model2': model2,
                            'statistic': float(stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        })
                    except Exception as e:
                        wilcoxon_results.append({
                            'model1': model1,
                            'model2': model2,
                            'error': str(e)
                        })

            results['tests'].append({
                'task': task,
                'cohort': cohort,
                'models': model_order,
                'n_windows': len(windows),
                'windows': windows,
                'friedman': friedman_result,
                'wilcoxon_pairwise': wilcoxon_results
            })

    return results


def main():
    print("="*60)
    print("Merge DL Robustness Results")
    print("="*60)

    # Load and merge
    combined_df = load_and_merge()

    # Save merged file
    combined_df.to_csv(MERGED_FILE, index=False)
    print(f"\nMerged results saved to: {MERGED_FILE}")

    # Also update the original file (backup first)
    backup_file = ROBUSTNESS_DIR / 'window_performance_backup.csv'
    if ML_FILE.exists():
        ml_df = pd.read_csv(ML_FILE)
        ml_df.to_csv(backup_file, index=False)
        print(f"Original ML results backed up to: {backup_file}")

    combined_df.to_csv(ML_FILE, index=False)
    print(f"Updated: {ML_FILE}")

    # Compute cross-window CV
    print("\nComputing cross-window coefficient of variation...")
    cv_results = compute_cross_window_cv(combined_df)

    analysis = {
        'generated_at': datetime.now().isoformat(),
        'n_rows': len(combined_df),
        'models': combined_df['model'].unique().tolist(),
        'windows': combined_df['window'].unique().tolist(),
        'tasks': combined_df['task'].unique().tolist(),
        'cohorts': combined_df['cohort'].unique().tolist(),
        'cross_window_cv': cv_results
    }

    with open(ANALYSIS_FILE, 'w') as f:
        json.dump(analysis, f, indent=2)
    print(f"Analysis saved to: {ANALYSIS_FILE}")

    # Run statistical tests
    print("\nRunning statistical tests...")
    stats_results = run_statistical_tests(combined_df)

    with open(STATS_FILE, 'w') as f:
        json.dump(stats_results, f, indent=2)
    print(f"Statistical tests saved to: {STATS_FILE}")

    # Print summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)

    print("\nModels by window:")
    for window in ['6h', '12h', '24h']:
        models = combined_df[combined_df['window'] == window]['model'].unique()
        print(f"  {window}: {', '.join(models)}")

    print("\nCross-window CV (lower = more robust):")
    cv_df = pd.DataFrame(cv_results)
    if not cv_df.empty:
        summary = cv_df[cv_df['cohort'] == 'all'][['task', 'model', 'auroc_cv']].copy()
        summary = summary.sort_values(['task', 'auroc_cv'])
        print(summary.to_string(index=False))

    print(f"\nDone at {datetime.now().isoformat()}")


if __name__ == '__main__':
    main()
