"""
Ablation by Note Category
按临床笔记类别进行消融实验

分析不同类型的笔记对预测性能的贡献：
- Radiology（放射学报告）
- Nursing（护理记录）
- Physician（医生记录）
- Discharge Summary（出院总结）
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    COHORT_FILE, RESULTS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'note_ablation'

# 笔记类别
NOTE_CATEGORIES = ['Radiology', 'Nursing', 'Physician', 'Discharge summary', 'ECG', 'Echo']


def extract_features_by_note_type(episode_path: Path) -> dict:
    """从 Episode 提取按笔记类型分类的特征"""
    with open(episode_path) as f:
        ep = json.load(f)
    
    features = {
        'stay_id': ep.get('stay_id'),
        'subject_id': ep.get('patient', {}).get('subject_id')
    }
    
    # 初始化各类别计数
    for cat in NOTE_CATEGORIES:
        features[f'{cat}_supportive'] = 0
        features[f'{cat}_contradictory'] = 0
        features[f'{cat}_count'] = 0
    
    # 从 pattern_annotations 中提取
    reasoning = ep.get('reasoning', {})
    annotations = reasoning.get('pattern_annotations', [])
    
    for annot in annotations:
        note_type = annot.get('note_type', 'Unknown')
        category = annot.get('annotation_category', 'UNRELATED')
        
        # 匹配笔记类别
        matched_cat = None
        for cat in NOTE_CATEGORIES:
            if cat.lower() in note_type.lower():
                matched_cat = cat
                break
        
        if matched_cat:
            features[f'{matched_cat}_count'] += 1
            if category == 'SUPPORTIVE':
                features[f'{matched_cat}_supportive'] += 1
            elif category == 'CONTRADICTORY':
                features[f'{matched_cat}_contradictory'] += 1
    
    # 总体标注
    features['total_supportive'] = reasoning.get('n_supportive', 0)
    features['total_contradictory'] = reasoning.get('n_contradictory', 0)
    
    # 标签
    labels = ep.get('labels', {})
    outcome = labels.get('outcome', {})
    features['mortality'] = outcome.get('mortality', 0)
    
    return features


def load_all_features():
    """加载所有 Episode 的笔记类别特征"""
    print("加载 Episode 笔记类别特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
    features_list = []
    for ep_file in tqdm(episode_files, desc="Extracting features"):
        try:
            features = extract_features_by_note_type(ep_file)
            features_list.append(features)
        except Exception as e:
            pass
    
    df = pd.DataFrame(features_list)
    print(f"   提取特征: {len(df):,} 个样本")
    
    return df


def train_ablation(df, feature_cols, y, groups, ablation_name):
    """训练消融实验"""
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    if X.shape[1] == 0:
        return None
    
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                              random_state=RANDOM_STATE, use_label_encoder=False,
                              eval_metric='logloss', n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    
    auroc = roc_auc_score(y_test, y_pred)
    auprc = average_precision_score(y_test, y_pred)
    
    return {'ablation': ablation_name, 'test_auroc': auroc, 'test_auprc': auprc, 'n_features': len(feature_cols)}


def main():
    print("=" * 60)
    print("Ablation by Note Category")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_all_features()
    
    y = df['mortality'].values
    groups = df['subject_id'].values
    
    results = []
    
    # 1. 全部笔记类别
    print("\n=== 全部笔记类别 ===")
    all_cols = []
    for cat in NOTE_CATEGORIES:
        all_cols.extend([f'{cat}_supportive', f'{cat}_contradictory', f'{cat}_count'])
    
    res = train_ablation(df, all_cols, y, groups, 'All Categories')
    if res:
        results.append(res)
        print(f"   AUROC: {res['test_auroc']:.4f}")
    
    # 2. 仅总体标注
    print("\n=== 仅总体标注 ===")
    res = train_ablation(df, ['total_supportive', 'total_contradictory'], y, groups, 'Total Only')
    if res:
        results.append(res)
        print(f"   AUROC: {res['test_auroc']:.4f}")
    
    # 3. 按单个笔记类别
    print("\n=== 按笔记类别消融 ===")
    for cat in NOTE_CATEGORIES:
        cat_cols = [f'{cat}_supportive', f'{cat}_contradictory', f'{cat}_count']
        res = train_ablation(df, cat_cols, y, groups, f'{cat} Only')
        if res:
            results.append(res)
            print(f"   {cat}: AUROC={res['test_auroc']:.4f}")
    
    # 4. 排除单个类别
    print("\n=== 排除单个类别 ===")
    for exclude_cat in NOTE_CATEGORIES:
        include_cols = []
        for cat in NOTE_CATEGORIES:
            if cat != exclude_cat:
                include_cols.extend([f'{cat}_supportive', f'{cat}_contradictory', f'{cat}_count'])
        
        res = train_ablation(df, include_cols, y, groups, f'Exclude {exclude_cat}')
        if res:
            results.append(res)
            print(f"   Exclude {exclude_cat}: AUROC={res['test_auroc']:.4f}")
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'note_ablation_results.csv', index=False)
    
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # 分析最重要的笔记类别
    print("\n分析:")
    category_results = results_df[results_df['ablation'].str.contains('Only') & ~results_df['ablation'].str.contains('Total')]
    if len(category_results) > 0:
        best_cat = category_results.loc[category_results['test_auroc'].idxmax(), 'ablation']
        print(f"  - 最重要的笔记类别: {best_cat}")


if __name__ == "__main__":
    main()
