"""
Text-only and Fusion Baselines
LLM特征 + Early/Late Fusion实验
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb

from config import (
    DATA_WINDOWS_DIR, MERGE_OUTPUT_DIR, BENCHMARK_RESULTS_DIR,
    LLM_FEATURES_FILE, WINDOWS, TASKS, COHORTS, N_FOLDS, RANDOM_STATE,
    TEST_SIZE, USE_HOLDOUT_TEST, LLM_COLS, get_features_file, PROCESSED_DIR
)

# 导入annotation特征（推理得分）
try:
    from data_processing.annotation_features import ANNOTATION_FEATURE_COLS
    ANNOTATION_FEATURES_AVAILABLE = True
except ImportError:
    ANNOTATION_FEATURES_AVAILABLE = False
    ANNOTATION_FEATURE_COLS = []


# ==========================================
# 校准度评估
# ==========================================
def compute_ece(y_true, y_prob, n_bins=10):
    """计算Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])

        if bin_mask.sum() > 0:
            bin_acc = y_true[bin_mask].mean()  # 真实阳性率
            bin_conf = y_prob[bin_mask].mean()  # 平均预测概率
            bin_count = bin_mask.sum()
        else:
            bin_acc = 0
            bin_conf = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            bin_count = 0

        bin_accuracies.append(bin_acc)
        bin_confidences.append(bin_conf)
        bin_counts.append(bin_count)

    # 计算ECE
    total_samples = np.sum(bin_counts)
    if total_samples == 0:
        return 1.0, bin_accuracies, bin_confidences, bin_counts

    ece = sum(
        (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
        for i in range(n_bins)
    )

    return ece, bin_accuracies, bin_confidences, bin_counts


def compute_mce(y_true, y_prob, n_bins=10):
    """计算Maximum Calibration Error"""
    _, bin_accuracies, bin_confidences, bin_counts = compute_ece(y_true, y_prob, n_bins)

    mce = max(
        abs(bin_accuracies[i] - bin_confidences[i])
        for i in range(n_bins) if bin_counts[i] > 0
    ) if any(c > 0 for c in bin_counts) else 1.0

    return mce


import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 配置
# ==========================================
DATA_DIR = DATA_WINDOWS_DIR
COHORT_FILE = MERGE_OUTPUT_DIR / 'cohort_final.csv'
OUTPUT_DIR = BENCHMARK_RESULTS_DIR

# 实验配置
WINDOWS = ['6h', '12h', '24h']
TASKS = ['mortality', 'prolonged_los']  # readmission效果太差，暂时跳过
COHORTS = ['all', 'sepsis', 'aki']

N_FOLDS = 5
RANDOM_STATE = 42

# 独立测试集配置
TEST_SIZE = 0.2  # 20%作为独立测试集
USE_HOLDOUT_TEST = True  # 启用独立测试集

# LLM特征列
LLM_COLS = ['pneumonia', 'edema', 'pleural_effusion', 'pneumothorax', 'tubes_lines']

# 是否使用推理得分特征
USE_ANNOTATION_FEATURES = True

# ==========================================
# 数据加载
# ==========================================
class FusionDataLoader:
    def __init__(self):
        self.cohort = pd.read_csv(COHORT_FILE)
        self.cohort['stay_id'] = self.cohort['stay_id'].astype(int)

        # 加载LLM特征
        self.llm_features = pd.read_csv(LLM_FEATURES_FILE)
        self.llm_features['stay_id'] = pd.to_numeric(
            self.llm_features['stay_id'], errors='coerce'
        ).fillna(-1).astype(int)

        print(f"Loaded LLM features for {len(self.llm_features)} patients")
        print(f"LLM columns: {LLM_COLS}")

        # 加载annotation特征（推理得分）
        self.annotation_features = None
        if USE_ANNOTATION_FEATURES and ANNOTATION_FEATURES_AVAILABLE:
            annotation_path = PROCESSED_DIR / 'annotation_features.csv'
            if annotation_path.exists():
                self.annotation_features = pd.read_csv(annotation_path)
                self.annotation_features['stay_id'] = self.annotation_features['stay_id'].astype(int)
                print(f"Loaded annotation features: {len(self.annotation_features)} patients")
                print(f"Reasoning features: {ANNOTATION_FEATURE_COLS}")

    def load_tabular_features(self, window):
        """加载结构化特征"""
        path = get_features_file(window)
        df = pd.read_csv(path)
        df['stay_id'] = df['stay_id'].astype(int)

        # 合并annotation特征（推理得分）
        if USE_ANNOTATION_FEATURES and self.annotation_features is not None:
            df = df.merge(self.annotation_features, on='stay_id', how='left')
            for col in ANNOTATION_FEATURE_COLS:
                if col in df.columns:
                    if col == 'uncertainty_score':
                        df[col] = df[col].fillna(1.0)
                    else:
                        df[col] = df[col].fillna(0.0)
            print(f"   Added reasoning features to tabular data")

        return df
    
    def get_llm_features(self, stay_ids):
        """获取LLM特征"""
        df = self.llm_features[self.llm_features['stay_id'].isin(stay_ids)].copy()
        
        # 只保留需要的列
        available_cols = ['stay_id'] + [c for c in LLM_COLS if c in df.columns]
        df = df[available_cols].copy()
        
        # 处理缺失值：-1 (unknown) 替换为 0
        for col in LLM_COLS:
            if col in df.columns:
                df[col] = df[col].replace(-1, 0)
        
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
        else:
            raise ValueError(f"Unknown cohort: {cohort_name}")

# ==========================================
# 模型训练
# ==========================================
def train_and_evaluate(X, y, groups, model_type='xgboost'):
    """
    训练和评估模型

    添加独立测试集支持
    添加校准度评估 (ECE, Brier Score)
    """

    if USE_HOLDOUT_TEST:
        # 分离独立测试集
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X, y, groups))

        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        groups_train_val = groups[train_val_idx]
    else:
        X_train_val, y_train_val, groups_train_val = X, y, groups
        X_test, y_test = None, None

    gkf = GroupKFold(n_splits=N_FOLDS)
    aurocs = []
    auprcs = []
    briers = []  # 添加Brier score
    eces = []    # 添加ECE
    best_model = None
    best_scaler = None
    best_auroc = -1

    for train_idx, val_idx in gkf.split(X_train_val, y_train_val, groups_train_val):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # 处理NaN
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_val = np.nan_to_num(X_val, nan=0, posinf=0, neginf=0)

        # 模型
        if model_type == 'xgboost':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'
            )
        else:
            model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)

        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]

        try:
            auroc = roc_auc_score(y_val, y_pred)
            aurocs.append(auroc)
            auprcs.append(average_precision_score(y_val, y_pred))

            # 计算校准度指标
            brier = brier_score_loss(y_val, y_pred)
            briers.append(brier)

            ece, _, _, _ = compute_ece(y_val, y_pred)
            eces.append(ece)

            if auroc > best_auroc:
                best_auroc = auroc
                best_model = model
                best_scaler = scaler
        except:
            aurocs.append(0.5)
            auprcs.append(y_val.mean())
            briers.append(1.0)
            eces.append(1.0)

    # 独立测试集评估
    test_results = {}
    if X_test is not None and best_model is not None:
        X_test_scaled = best_scaler.transform(X_test)
        X_test_scaled = np.nan_to_num(X_test_scaled, nan=0, posinf=0, neginf=0)
        y_test_pred = best_model.predict_proba(X_test_scaled)[:, 1]

        try:
            test_results['test_auroc'] = roc_auc_score(y_test, y_test_pred)
            test_results['test_auprc'] = average_precision_score(y_test, y_test_pred)
            test_results['test_brier'] = brier_score_loss(y_test, y_test_pred)
            test_results['test_ece'], _, _, _ = compute_ece(y_test, y_test_pred)
        except:
            test_results['test_auroc'] = 0.5
            test_results['test_auprc'] = y_test.mean()
            test_results['test_brier'] = 1.0
            test_results['test_ece'] = 1.0

    return {
        'auroc_mean': np.mean(aurocs),
        'auroc_std': np.std(aurocs),
        'auprc_mean': np.mean(auprcs),
        'auprc_std': np.std(auprcs),
        'brier_mean': np.mean(briers),
        'brier_std': np.std(briers),
        'ece_mean': np.mean(eces),
        'ece_std': np.std(eces),
        **test_results
    }

# ==========================================
# Late Fusion
# ==========================================
def late_fusion_evaluate(X_tab, X_text, y, groups, fusion_method='average'):
    """Late fusion: 分别训练两个模型，然后融合预测概率

    添加校准度评估 (ECE, Brier Score)
    """

    gkf = GroupKFold(n_splits=N_FOLDS)
    aurocs = []
    briers = []  # 添加Brier score
    eces = []    # 添加ECE

    for train_idx, val_idx in gkf.split(X_tab, y, groups):
        # 分割数据
        X_tab_train, X_tab_val = X_tab[train_idx], X_tab[val_idx]
        X_text_train, X_text_val = X_text[train_idx], X_text[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 标准化
        scaler_tab = StandardScaler()
        X_tab_train = scaler_tab.fit_transform(X_tab_train)
        X_tab_val = scaler_tab.transform(X_tab_val)

        scaler_text = StandardScaler()
        X_text_train = scaler_text.fit_transform(X_text_train)
        X_text_val = scaler_text.transform(X_text_val)

        # 处理NaN
        X_tab_train = np.nan_to_num(X_tab_train, nan=0)
        X_tab_val = np.nan_to_num(X_tab_val, nan=0)
        X_text_train = np.nan_to_num(X_text_train, nan=0)
        X_text_val = np.nan_to_num(X_text_val, nan=0)

        # 训练Tabular模型
        model_tab = xgb.XGBClassifier(
            n_estimators=100, max_depth=5, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss'
        )
        model_tab.fit(X_tab_train, y_train)
        pred_tab = model_tab.predict_proba(X_tab_val)[:, 1]

        # 训练Text模型
        model_text = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_STATE)
        model_text.fit(X_text_train, y_train)
        pred_text = model_text.predict_proba(X_text_val)[:, 1]

        # 融合
        if fusion_method == 'average':
            pred_fused = (pred_tab + pred_text) / 2
        elif fusion_method == 'weighted':
            # 给tabular更高权重（因为它通常更强）
            pred_fused = 0.7 * pred_tab + 0.3 * pred_text
        else:
            pred_fused = np.maximum(pred_tab, pred_text)

        try:
            aurocs.append(roc_auc_score(y_val, pred_fused))

            # 计算校准度指标
            brier = brier_score_loss(y_val, pred_fused)
            briers.append(brier)

            ece, _, _, _ = compute_ece(y_val, pred_fused)
            eces.append(ece)
        except:
            aurocs.append(0.5)
            briers.append(1.0)
            eces.append(1.0)

    return {
        'auroc_mean': np.mean(aurocs),
        'auroc_std': np.std(aurocs),
        'brier_mean': np.mean(briers),
        'brier_std': np.std(briers),
        'ece_mean': np.mean(eces),
        'ece_std': np.std(eces)
    }

# ==========================================
# 主流程
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    loader = FusionDataLoader()
    all_results = []
    
    print("Starting Fusion Experiments")
    print("=" * 70)
    
    for window in WINDOWS:
        print(f"\nWindow: {window}")
        tabular_df = loader.load_tabular_features(window)
        
        for task in TASKS:
            labels_df = loader.get_task_label(task)
            
            for cohort_name in COHORTS:
                cohort_ids = loader.filter_cohort(cohort_name)
                
                # 合并数据
                df = tabular_df[tabular_df['stay_id'].isin(cohort_ids)].merge(
                    labels_df, on='stay_id'
                )
                
                # 获取LLM特征
                llm_df = loader.get_llm_features(df['stay_id'].values)
                df = df.merge(llm_df, on='stay_id', how='left')
                
                # 填充缺失的LLM特征
                for col in LLM_COLS:
                    if col in df.columns:
                        df[col] = df[col].fillna(0)
                    else:
                        df[col] = 0
                
                if len(df) < 100:
                    continue
                
                # 准备特征
                tab_cols = [c for c in df.columns if c not in 
                           ['stay_id', 'subject_id', 'label'] + LLM_COLS]
                
                X_tab = df[tab_cols].values
                X_text = df[LLM_COLS].values
                X_early = np.concatenate([X_tab, X_text], axis=1)  # Early fusion
                y = df['label'].values
                groups = df['subject_id'].values
                
                print(f"\n[{window}|{task}|{cohort_name}] n={len(df)}, pos={y.sum()}")

                # ===== 1. Text-only =====
                results_text = train_and_evaluate(X_text, y, groups, 'lr')
                print(f"   Text-only (LR): AUROC={results_text['auroc_mean']:.4f} ECE={results_text['ece_mean']:.4f}")
                all_results.append({
                    'window': window, 'task': task, 'cohort': cohort_name,
                    'model': 'Text-only (LR)', **results_text
                })

                # ===== 2. Early Fusion =====
                results_early = train_and_evaluate(X_early, y, groups, 'xgboost')
                print(f"   Early Fusion (XGB): AUROC={results_early['auroc_mean']:.4f} ECE={results_early['ece_mean']:.4f}")
                all_results.append({
                    'window': window, 'task': task, 'cohort': cohort_name,
                    'model': 'Early Fusion (XGB)', **results_early
                })

                # ===== 3. Late Fusion (Average) =====
                results_late_avg = late_fusion_evaluate(X_tab, X_text, y, groups, 'average')
                print(f"   Late Fusion (Avg): AUROC={results_late_avg['auroc_mean']:.4f} ECE={results_late_avg['ece_mean']:.4f}")
                all_results.append({
                    'window': window, 'task': task, 'cohort': cohort_name,
                    'model': 'Late Fusion (Avg)', **results_late_avg
                })

                # ===== 4. Late Fusion (Weighted) =====
                results_late_wt = late_fusion_evaluate(X_tab, X_text, y, groups, 'weighted')
                print(f"   Late Fusion (Wt): AUROC={results_late_wt['auroc_mean']:.4f} ECE={results_late_wt['ece_mean']:.4f}")
                all_results.append({
                    'window': window, 'task': task, 'cohort': cohort_name,
                    'model': 'Late Fusion (Wt)', **results_late_wt
                })
    
    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(OUTPUT_DIR, 'fusion_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")
    
    # ==========================================
    # 汇总对比
    # ==========================================
    print("\n" + "=" * 70)
    print("FUSION vs TABULAR COMPARISON")
    print("=" * 70)
    
    # 加载之前的tabular结果
    tabular_path = os.path.join(OUTPUT_DIR, 'benchmark_results_full.csv')
    if os.path.exists(tabular_path):
        tabular_df = pd.read_csv(tabular_path)
        
        print("\n[Mortality - 24h - All Cohort]")
        print("-" * 50)
        
        # Tabular baseline
        tab_result = tabular_df[
            (tabular_df['window'] == '24h') & 
            (tabular_df['task'] == 'mortality') & 
            (tabular_df['cohort'] == 'all') &
            (tabular_df['model'] == 'XGBoost')
        ]
        if len(tab_result) > 0:
            print(f"   Tabular (XGBoost):     {tab_result['auroc_mean'].values[0]:.4f}")
        
        # Fusion results
        fusion_24h = results_df[
            (results_df['window'] == '24h') & 
            (results_df['task'] == 'mortality') & 
            (results_df['cohort'] == 'all')
        ]
        for _, row in fusion_24h.iterrows():
            print(f"   {row['model']:25s} {row['auroc_mean']:.4f}")
    
    print("\nFusion Experiments Complete!")

if __name__ == "__main__":
    main()