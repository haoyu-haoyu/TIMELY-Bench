"""
XGBoost & Logistic Regression Baselines
在多窗口、多任务、多队列上进行benchmark评估
"""

import sys
from pathlib import Path

# 添加项目根目录到path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_WINDOWS_DIR, MERGE_OUTPUT_DIR, BENCHMARK_RESULTS_DIR,
    WINDOWS, TASKS, COHORTS, N_FOLDS, RANDOM_STATE,
    TEST_SIZE, USE_HOLDOUT_TEST, get_features_file, PROCESSED_DIR,
    COHORT_FILE
)
from utils.predefined_split import resolve_predefined_partition

# 导入annotation特征
try:
    from data_processing.annotation_features import (
        merge_annotation_features_with_data,
        ANNOTATION_FEATURE_COLS
    )
    ANNOTATION_FEATURES_AVAILABLE = True
except ImportError:
    ANNOTATION_FEATURES_AVAILABLE = False
    ANNOTATION_FEATURE_COLS = []

# ==========================================
# 配置
# ==========================================
OUTPUT_DIR = BENCHMARK_RESULTS_DIR
MODELS = ['LogisticRegression', 'XGBoost']
USE_ANNOTATION_FEATURES = True  # 是否使用推理得分特征
# For prolonged LOS, some pipelines exclude mortality cases to avoid competing-risk censoring.
# Canonical TIMELY-Bench v2.0 results keep *all* episodes unless explicitly enabled.
EXCLUDE_MORTALITY_FOR_NON_MORTALITY_TASKS = False

# ==========================================
# 数据加载
# ==========================================
class DataLoader:
    def __init__(self):
        self.cohort = pd.read_csv(COHORT_FILE)
        self.cohort['stay_id'] = self.cohort['stay_id'].astype(int)
        self.annotation_features = None

        # 加载annotation特征
        if USE_ANNOTATION_FEATURES and ANNOTATION_FEATURES_AVAILABLE:
            annotation_path = PROCESSED_DIR / 'annotation_features.csv'
            if annotation_path.exists():
                self.annotation_features = pd.read_csv(annotation_path)
                self.annotation_features['stay_id'] = self.annotation_features['stay_id'].astype(int)
                print(f"Loaded annotation features: {len(self.annotation_features)} patients")
            else:
                print("Annotation features file not found, will compute if needed")

    def load_features(self, window):
        path = get_features_file(window)
        df = pd.read_csv(path)
        df['stay_id'] = df['stay_id'].astype(int)

        # 合并annotation特征（推理得分）
        if USE_ANNOTATION_FEATURES and self.annotation_features is not None:
            df = df.merge(self.annotation_features, on='stay_id', how='left')
            # 填充缺失值
            for col in ANNOTATION_FEATURE_COLS:
                if col in df.columns:
                    if col == 'uncertainty_score':
                        df[col] = df[col].fillna(1.0)  # 无标注=完全不确定
                    else:
                        df[col] = df[col].fillna(0.0)
            print(f"   Added annotation features: {[c for c in ANNOTATION_FEATURE_COLS if c in df.columns]}")

        return df

    def get_task_label(self, task):
        label_map = {
            'mortality': 'label_mortality',
            'prolonged_los': 'prolonged_los_7d',
            'readmission': 'readmission_30d'
        }
        return self.cohort[['stay_id', 'subject_id', label_map[task]]].rename(
            columns={label_map[task]: 'label'}
        )

    def filter_cohort(self, cohort_name):
        if cohort_name == 'all':
            return self.cohort['stay_id'].values
        elif cohort_name == 'sepsis':
            return self.cohort[self.cohort['has_sepsis_final'] == 1]['stay_id'].values
        elif cohort_name == 'aki':
            return self.cohort[self.cohort['has_aki_final'] == 1]['stay_id'].values
        elif cohort_name == 'sepsis_aki':
            mask = (self.cohort['has_sepsis_final'] == 1) & (self.cohort['has_aki_final'] == 1)
            return self.cohort[mask]['stay_id'].values
        else:
            raise ValueError(f"Unknown cohort: {cohort_name}")

# ==========================================
# 模型训练
# ==========================================
class ModelTrainer:
    def __init__(self, model_name):
        self.model_name = model_name

    def get_model(self):
        if self.model_name == 'LogisticRegression':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=RANDOM_STATE
            )
        elif self.model_name == 'XGBoost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                scale_pos_weight=1,
                random_state=RANDOM_STATE,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def train_and_evaluate(self, X, y, groups, stay_ids, use_holdout=True):
        """使用GroupKFold交叉验证，支持独立测试集"""

        if use_holdout and USE_HOLDOUT_TEST:
            train_val_idx, test_idx, fold_ids, split_info = resolve_predefined_partition(stay_ids)

            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]
            groups_train_val = groups[train_val_idx]
            fold_train_val = fold_ids[train_val_idx]
        else:
            split_info = {"source": "runtime_groupkfold"}
            X_train_val, y_train_val, groups_train_val = X, y, groups
            X_test, y_test = None, None
            gkf = GroupKFold(n_splits=N_FOLDS)

        metrics_per_fold = {'auroc': [], 'auprc': [], 'brier': []}
        best_model, best_scaler, best_auroc = None, None, -1

        if use_holdout and USE_HOLDOUT_TEST:
            fold_iter = []
            for fold in range(1, N_FOLDS + 1):
                train_idx = np.where(fold_train_val != fold)[0]
                val_idx = np.where(fold_train_val == fold)[0]
                if len(train_idx) == 0 or len(val_idx) == 0:
                    continue
                fold_iter.append((fold, train_idx, val_idx))
        else:
            fold_iter = []
            for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups_train_val), start=1):
                fold_iter.append((fold, train_idx, val_idx))

        for fold, train_idx, val_idx in fold_iter:
            X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
            y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

            # 标准化
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
            X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)

            # 训练
            model = self.get_model()
            if self.model_name == 'XGBoost':
                neg_count = (y_train == 0).sum()
                pos_count = (y_train == 1).sum()
                if pos_count > 0:
                    model.set_params(scale_pos_weight=neg_count / pos_count)

            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_val)[:, 1]

            # 计算指标
            try:
                auroc = roc_auc_score(y_val, y_pred)
            except:
                auroc = 0.5

            try:
                auprc = average_precision_score(y_val, y_pred)
            except:
                auprc = y_val.mean()

            try:
                brier = brier_score_loss(y_val, y_pred)
            except:
                brier = 1.0

            metrics_per_fold['auroc'].append(auroc)
            metrics_per_fold['auprc'].append(auprc)
            metrics_per_fold['brier'].append(brier)

            if auroc > best_auroc:
                best_auroc = auroc
                best_model = model
                best_scaler = scaler

        # 测试集评估
        test_results = {}
        if X_test is not None and best_model is not None:
            X_test_scaled = best_scaler.transform(X_test)
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
            y_test_pred = best_model.predict_proba(X_test_scaled)[:, 1]

            try:
                test_results['test_auroc'] = roc_auc_score(y_test, y_test_pred)
            except:
                test_results['test_auroc'] = 0.5

            try:
                test_results['test_auprc'] = average_precision_score(y_test, y_test_pred)
            except:
                test_results['test_auprc'] = y_test.mean()

            try:
                test_results['test_brier'] = brier_score_loss(y_test, y_test_pred)
            except:
                test_results['test_brier'] = 1.0

            y_pred_binary = (y_test_pred >= 0.5).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
            test_results['test_sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            test_results['test_specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            test_results['test_n_samples'] = len(y_test)

        return {
            'auroc_mean': np.mean(metrics_per_fold['auroc']),
            'auroc_std': np.std(metrics_per_fold['auroc']),
            'auprc_mean': np.mean(metrics_per_fold['auprc']),
            'auprc_std': np.std(metrics_per_fold['auprc']),
            'brier_mean': np.mean(metrics_per_fold['brier']),
            'brier_std': np.std(metrics_per_fold['brier']),
            'split_source': split_info['source'],
            **test_results
        }

# ==========================================
# 主流程
# ==========================================
def run_experiments():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    loader = DataLoader()
    all_results = []

    total = len(WINDOWS) * len(TASKS) * len(COHORTS) * len(MODELS)
    current = 0

    print("TIMELY-Bench Baseline Experiments")
    print("=" * 60)
    print(f"Windows: {WINDOWS}")
    print(f"Tasks: {TASKS}")
    print(f"Cohorts: {COHORTS}")
    print(f"Models: {MODELS}")
    print(f"Total: {total} experiments")
    print("=" * 60)

    for window in WINDOWS:
        print(f"\nLoading window: {window}")
        features_df = loader.load_features(window)

        for task in TASKS:
            labels_df = loader.get_task_label(task)
            if EXCLUDE_MORTALITY_FOR_NON_MORTALITY_TASKS and task != 'mortality' and 'label_mortality' in loader.cohort.columns:
                labels_df = labels_df.merge(
                    loader.cohort[['stay_id', 'label_mortality']],
                    on='stay_id',
                    how='left'
                )

            for cohort_name in COHORTS:
                cohort_ids = loader.filter_cohort(cohort_name)

                df = features_df[features_df['stay_id'].isin(cohort_ids)].merge(
                    labels_df, on='stay_id'
                )
                if EXCLUDE_MORTALITY_FOR_NON_MORTALITY_TASKS and task != 'mortality' and 'label_mortality' in df.columns:
                    df = df[df['label_mortality'] != 1]
                df = df[df['label'].notna()]

                if len(df) < 100:
                    print(f"  Skipping {cohort_name}/{task}: too few samples ({len(df)})")
                    continue

                feature_cols = [c for c in df.columns if c not in ['stay_id', 'subject_id', 'label', 'label_mortality']]
                X = df[feature_cols].values
                y = df['label'].values
                groups = df['subject_id'].values

                for model_name in MODELS:
                    current += 1

                    print(f"\n[{current}/{total}] {window} | {task} | {cohort_name} | {model_name}")
                    print(f"  n={len(df)}, pos={y.sum()} ({y.mean()*100:.1f}%)")

                    trainer = ModelTrainer(model_name)
                    results = trainer.train_and_evaluate(X, y, groups, df['stay_id'].values)

                    result_row = {
                        'window': window,
                        'task': task,
                        'cohort': cohort_name,
                        'model': model_name,
                        'n_samples': len(df),
                        'n_positive': int(y.sum()),
                        'positive_rate': y.mean(),
                        **results
                    }
                    all_results.append(result_row)

                    print(f"  AUROC: {results['auroc_mean']:.4f} +/- {results['auroc_std']:.4f}")
                    if 'test_auroc' in results:
                        print(f"  Test AUROC: {results['test_auroc']:.4f}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_path = OUTPUT_DIR / 'benchmark_results_full.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    # 汇总
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for task in TASKS:
        print(f"\n[{task}]")
        task_df = results_df[results_df['task'] == task]
        pivot = task_df.pivot_table(
            index=['cohort', 'model'],
            columns='window',
            values='auroc_mean',
            aggfunc='first'
        )
        print(pivot.round(4).to_string())

    print("\nBest per task:")
    for task in TASKS:
        task_df = results_df[results_df['task'] == task]
        if len(task_df) > 0:
            best = task_df.loc[task_df['auroc_mean'].idxmax()]
            print(f"  {task}: AUROC={best['auroc_mean']:.4f} ({best['model']}, {best['window']})")

    print("\nDone.")
    return results_df


def generate_report(results_df):
    """生成benchmark报告"""

    report = f"""# Benchmark Results

## Overview

Experiments across:
- Time Windows: {', '.join(WINDOWS)}
- Tasks: Mortality, Prolonged LOS
- Cohorts: All, Sepsis, AKI
- Models: Logistic Regression, XGBoost

## Results

"""

    for task in TASKS:
        task_df = results_df[results_df['task'] == task]
        if len(task_df) == 0:
            continue

        report += f"\n### {task.replace('_', ' ').title()}\n\n"
        header_cols = " | ".join(WINDOWS)
        report += f"| Cohort | Model | {header_cols} |\n"
        report += f"|--------|-------|{'|'.join(['-----'] * len(WINDOWS))}|\n"

        for cohort in COHORTS:
            for model in MODELS:
                row_data = task_df[(task_df['cohort'] == cohort) & (task_df['model'] == model)]

                aurocs = {}
                for window in WINDOWS:
                    w_data = row_data[row_data['window'] == window]
                    if len(w_data) > 0:
                        aurocs[window] = f"{w_data['auroc_mean'].values[0]:.4f}"
                    else:
                        aurocs[window] = "-"

                values = " | ".join([aurocs.get(w, '-') for w in WINDOWS])
                report += f"| {cohort} | {model} | {values} |\n"

    report_path = OUTPUT_DIR / 'benchmark_report.md'
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"Report: {report_path}")


if __name__ == "__main__":
    results_df = run_experiments()
    generate_report(results_df)
