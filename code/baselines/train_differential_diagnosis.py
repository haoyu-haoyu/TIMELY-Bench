"""
鉴别诊断模型训练
使用隐性特征区分相似症状的疾病 (Sepsis vs AKI vs ARDS)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# 配置
BASE_DIR = Path(__file__).resolve().parents[2]
EPISODES_DIR = BASE_DIR / 'episodes' / 'episodes_enhanced'
HIDDEN_FEATURES_FILE = BASE_DIR / 'data' / 'processed' / 'hidden_features' / 'hidden_features.csv'
MEDCAT_FILE = BASE_DIR / 'data' / 'processed' / 'medcat_full' / 'medcat_features_24h.csv'
OUTPUT_DIR = BASE_DIR / 'results' / 'differential_diagnosis'


def load_labels():
    """加载疾病标签"""
    print("加载疾病标签...")
    labels = []
    
    for ep_file in EPISODES_DIR.glob('*.json'):
        try:
            ep = json.load(open(ep_file))
            stay_id = ep.get('stay_id')
            conditions = set(ep.get('conditions', []))
            
            labels.append({
                'stay_id': stay_id,
                'has_sepsis': 1 if 'sepsis' in conditions else 0,
                'has_aki': 1 if 'aki' in conditions else 0,
                'has_ards': 1 if 'ards' in conditions else 0,
                'n_conditions': len(conditions)
            })
        except:
            continue
    
    return pd.DataFrame(labels)


def create_multi_class_label(row):
    """创建多分类标签"""
    if row['has_sepsis'] and row['has_aki']:
        return 'sepsis_aki'
    elif row['has_sepsis']:
        return 'sepsis_only'
    elif row['has_aki']:
        return 'aki_only'
    elif row['has_ards']:
        return 'ards_only'
    else:
        return 'none'


def train_binary_classifier(X, y, name, model):
    """训练二分类器"""
    if y.sum() < 10 or (len(y) - y.sum()) < 10:
        return None
    
    # 5 折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 计算 AUC
    try:
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    except:
        return None
    
    return {
        'name': name,
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std(),
        'f1_mean': f1_scores.mean(),
        'f1_std': f1_scores.std()
    }


def analyze_feature_importance(X, y, feature_names):
    """分析特征重要性"""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance


def main():
    print("=" * 70)
    print("鉴别诊断模型训练 - 使用隐性特征")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    hidden_features = pd.read_csv(HIDDEN_FEATURES_FILE)
    print(f"隐性特征: {len(hidden_features):,} 样本, {len(hidden_features.columns)-1} 特征")
    
    # 加载 MedCAT 特征
    if MEDCAT_FILE.exists():
        medcat = pd.read_csv(MEDCAT_FILE)
        # 重命名可能冲突的列（除 stay_id 以外全部加前缀）
        medcat_cols = {c: f'medcat_{c}' for c in medcat.columns if c != 'stay_id'}
        medcat = medcat.rename(columns=medcat_cols)
        print(f"MedCAT 特征: {len(medcat):,} 样本, {len(medcat.columns)-1} 特征")
    else:
        medcat = None
    
    # 加载标签
    labels = load_labels()
    print(f"标签: {len(labels):,} 样本")
    
    # 合并数据
    data = hidden_features.merge(labels, on='stay_id', how='inner')
    if medcat is not None:
        data = data.merge(medcat, on='stay_id', how='left')
    
    print(f"合并后: {len(data):,} 样本")
    
    # 特征列
    feature_cols = [c for c in data.columns if c not in ['stay_id', 'has_sepsis', 'has_aki', 'has_ards', 'n_conditions']]
    X = data[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"特征维度: {X.shape[1]}")
    
    # ============================================================
    # 1. 二分类任务：各疾病预测
    # ============================================================
    print("\n" + "=" * 70)
    print("【1】二分类任务 - 各疾病预测")
    print("=" * 70)
    
    results = []
    
    for disease in ['sepsis', 'aki', 'ards']:
        y = data[f'has_{disease}'].values
        print(f"\n{disease.upper()}: 正例 {y.sum():,} ({y.mean()*100:.1f}%)")
        
        for model_name, model in [
            ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42)),
            ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
            ('GradientBoosting', GradientBoostingClassifier(n_estimators=100, random_state=42))
        ]:
            result = train_binary_classifier(X_scaled, y, f"{disease}_{model_name}", model)
            if result:
                result['disease'] = disease
                result['model'] = model_name
                results.append(result)
                print(f"  {model_name}: AUC={result['auc_mean']:.3f}±{result['auc_std']:.3f}, F1={result['f1_mean']:.3f}±{result['f1_std']:.3f}")
    
    # ============================================================
    # 2. 多分类任务：疾病鉴别
    # ============================================================
    print("\n" + "=" * 70)
    print("【2】多分类任务 - 疾病鉴别诊断")
    print("=" * 70)
    
    # 创建多分类标签
    data['multi_label'] = data.apply(create_multi_class_label, axis=1)
    label_counts = data['multi_label'].value_counts()
    print("\n类别分布:")
    for label, count in label_counts.items():
        print(f"  {label}: {count:,} ({count/len(data)*100:.1f}%)")
    
    # 只保留样本数足够的类别
    valid_labels = label_counts[label_counts >= 100].index.tolist()
    data_valid = data[data['multi_label'].isin(valid_labels)]
    
    y_multi = pd.Categorical(data_valid['multi_label']).codes
    X_multi = data_valid[feature_cols].fillna(0).values
    X_multi_scaled = scaler.fit_transform(X_multi)
    
    print(f"\n有效样本: {len(data_valid):,}, 类别数: {len(valid_labels)}")
    
    # 训练多分类模型
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for model_name, model in [
        ('LogisticRegression', LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')),
        ('RandomForest', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ]:
        try:
            acc_scores = cross_val_score(model, X_multi_scaled, y_multi, cv=cv, scoring='accuracy')
            f1_scores = cross_val_score(model, X_multi_scaled, y_multi, cv=cv, scoring='f1_macro')
            print(f"  {model_name}: Accuracy={acc_scores.mean():.3f}±{acc_scores.std():.3f}, F1_macro={f1_scores.mean():.3f}±{f1_scores.std():.3f}")
        except Exception as e:
            print(f"  {model_name}: 错误 - {e}")
    
    # ============================================================
    # 3. 特征重要性分析
    # ============================================================
    print("\n" + "=" * 70)
    print("【3】特征重要性分析")
    print("=" * 70)
    
    for disease in ['sepsis', 'aki']:
        y = data[f'has_{disease}'].values
        importance = analyze_feature_importance(X_scaled, y, feature_cols)
        
        print(f"\n{disease.upper()} Top 10 特征:")
        for i, row in importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # 保存
        importance.to_csv(OUTPUT_DIR / f'{disease}_feature_importance.csv', index=False)
    
    # ============================================================
    # 4. 隐性因素区分能力分析
    # ============================================================
    print("\n" + "=" * 70)
    print("【4】隐性因素区分能力分析 (Sepsis vs AKI)")
    print("=" * 70)
    
    # 只比较 Sepsis-only vs AKI-only
    sep_only = data[data['multi_label'] == 'sepsis_only']
    aki_only = data[data['multi_label'] == 'aki_only']
    
    print(f"\nSepsis-only: {len(sep_only):,}, AKI-only: {len(aki_only):,}")
    
    # 关键隐性特征对比
    hidden_cols = ['hrv_sdnn', 'hrv_cv', 'creatinine_slope', 'lactate_rate', 
                   'text_support_ratio', 'has_negative_finding', 'pattern_recurrence']
    
    print("\n隐性特征对比:")
    print(f"{'Feature':<25} {'Sepsis':<12} {'AKI':<12} {'Diff%':<10} {'判断'}")
    print("-" * 70)
    
    for col in hidden_cols:
        if col in sep_only.columns and col in aki_only.columns:
            sep_mean = sep_only[col].mean()
            aki_mean = aki_only[col].mean()
            diff_pct = (sep_mean - aki_mean) / aki_mean * 100 if aki_mean != 0 else 0
            
            if abs(diff_pct) > 20:
                judgment = "⭐ 显著差异"
            elif abs(diff_pct) > 10:
                judgment = "↗ 有差异"
            else:
                judgment = "≈ 相似"
            
            print(f"{col:<25} {sep_mean:<12.3f} {aki_mean:<12.3f} {diff_pct:+.1f}%     {judgment}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'classification_results.csv', index=False)
    
    print(f"\n结果已保存到: {OUTPUT_DIR}")
    print("\n" + "=" * 70)
    print("训练完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
