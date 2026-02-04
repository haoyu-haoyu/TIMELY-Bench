"""
Text-only Baseline (使用 Episode 中的文本特征)
使用预提取的 LLM 特征作为文本表示

特征：
1. LLM 提取的 5 维特征 (severity, emotion, etc.)
2. 标注统计特征 (n_supportive, n_contradictory)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    PROCESSED_DIR, RESULTS_DIR, N_FOLDS, RANDOM_STATE,
    TEST_SIZE, USE_HOLDOUT_TEST, LLM_FEATURES_FILE
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
OUTPUT_DIR = RESULTS_DIR / 'text_only_baselines'
OUTPUT_JSON = OUTPUT_DIR / 'text_only_results_folds.json'


def extract_text_features(episode_path: Path) -> dict:
    """从 Episode 提取文本相关特征"""
    with open(episode_path) as f:
        ep = json.load(f)
    
    features = {
        'stay_id': ep.get('stay_id'),
        'subject_id': ep.get('patient', {}).get('subject_id')
    }
    
    # 临床文本统计
    clinical = ep.get('clinical_text', {})
    features['n_notes'] = clinical.get('n_notes', 0)
    
    notes = clinical.get('notes', [])
    if notes:
        features['total_text_length'] = sum(len(n.get('text', '')) for n in notes)
        features['avg_text_length'] = features['total_text_length'] / len(notes)
    else:
        features['total_text_length'] = 0
        features['avg_text_length'] = 0
    
    # 标注特征（这是核心的推理特征）
    reasoning = ep.get('reasoning', {})
    features['n_patterns'] = len(reasoning.get('detected_patterns', []))
    features['n_supportive'] = reasoning.get('n_supportive', 0)
    features['n_contradictory'] = reasoning.get('n_contradictory', 0)
    features['n_alignments'] = reasoning.get('n_alignments', 0)
    
    # 计算推理得分
    total_annot = features['n_supportive'] + features['n_contradictory']
    if total_annot > 0:
        features['supportive_ratio'] = features['n_supportive'] / total_annot
        features['contradictory_ratio'] = features['n_contradictory'] / total_annot
    else:
        features['supportive_ratio'] = 0.5
        features['contradictory_ratio'] = 0.5
    
    # 置信度分数
    if features['n_alignments'] > 0:
        features['annotation_density'] = total_annot / features['n_alignments']
    else:
        features['annotation_density'] = 0
    
    # 标签
    labels = ep.get('labels', {})
    outcome = labels.get('outcome', {})
    features['mortality'] = outcome.get('mortality', 0)
    features['prolonged_los'] = outcome.get('prolonged_los', 0)
    
    return features


def load_all_features():
    """加载所有 Episode 的文本特征"""
    print("加载 Episode 文本特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
    features_list = []
    for ep_file in tqdm(episode_files, desc="Extracting text features"):
        try:
            features = extract_text_features(ep_file)
            features_list.append(features)
        except Exception as e:
            pass
    
    df = pd.DataFrame(features_list)
    print(f"   提取特征: {len(df):,} 个样本, {len(df.columns)} 个特征")
    
    return df


def train_and_evaluate(X, y, groups, model_name='XGBoost'):
    """训练和评估"""
    
    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        groups_train_val = groups[train_val_idx]
    else:
        X_train_val, y_train_val, groups_train_val = X, y, groups
        X_test, y_test = None, None
    
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups=groups_train_val)):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric='logloss', n_jobs=-1
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_val)[:, 1]
        
        auroc = roc_auc_score(y_val, y_pred)
        auprc = average_precision_score(y_val, y_pred)
        
        fold_results.append({'fold': fold + 1, 'auroc': auroc, 'auprc': auprc})
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    
    test_result = None
    if X_test is not None:
        scaler = StandardScaler()
        X_train_all = scaler.fit_transform(X_train_val)
        X_test_scaled = scaler.transform(X_test)
        
        if model_name == 'XGBoost':
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=RANDOM_STATE, use_label_encoder=False,
                eval_metric='logloss', n_jobs=-1
            )
        else:
            model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        
        model.fit(X_train_all, y_train_val)
        y_test_pred = model.predict_proba(X_test_scaled)[:, 1]
        
        test_auroc = roc_auc_score(y_test, y_test_pred)
        test_auprc = average_precision_score(y_test, y_test_pred)
        test_result = {'auroc': test_auroc, 'auprc': test_auprc}
    
    return fold_results, test_result


def main():
    print("=" * 60)
    print("Text-only Baseline (标注特征)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_all_features()
    
    feature_cols = [c for c in df.columns if c not in ['stay_id', 'subject_id', 'mortality', 'prolonged_los']]
    print(f"\n使用特征: {feature_cols}")
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    groups = df['subject_id'].values
    
    results = []

    for task in ['mortality', 'prolonged_los']:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        
        y = df[task].values
        n_samples = len(y)
        positive_rate = float(y.mean()) if n_samples > 0 else 0.0
        
        for model_name in ['XGBoost', 'LogisticRegression']:
            print(f"\n{model_name}:")
            
            fold_results, test_result = train_and_evaluate(X, y, groups, model_name)
            
            mean_auroc = np.mean([r['auroc'] for r in fold_results])
            std_auroc = np.std([r['auroc'] for r in fold_results])
            mean_auprc = np.mean([r['auprc'] for r in fold_results])
            std_auprc = np.std([r['auprc'] for r in fold_results])
            
            print(f"\n   CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
            print(f"   CV AUPRC: {mean_auprc:.4f} ± {std_auprc:.4f}")
            
            if test_result:
                print(f"   Test AUROC: {test_result['auroc']:.4f}")
                print(f"   Test AUPRC: {test_result['auprc']:.4f}")
            
            results.append({
                'task': task,
                'model': model_name,
                'n_samples': n_samples,
                'positive_rate': positive_rate,
                'cv_auroc_mean': mean_auroc,
                'cv_auroc_std': std_auroc,
                'cv_auprc_mean': mean_auprc,
                'cv_auprc_std': std_auprc,
                'test_auroc': test_result['auroc'] if test_result else None,
                'test_auprc': test_result['auprc'] if test_result else None,
                'fold_details': fold_results
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'text_only_results.csv', index=False)
    print(f"\n结果保存到: {OUTPUT_DIR / 'text_only_results.csv'}")

    # Save fold-level details
    output_payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'seed': RANDOM_STATE,
        'input_paths': {
            'episodes_dir': str(EPISODES_DIR),
        },
        'results': results
    }
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=True)
    print(f"Fold details saved to: {OUTPUT_JSON}")
    
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
