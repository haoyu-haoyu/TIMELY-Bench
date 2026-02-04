"""
Tabular Baselines (XGBoost & Logistic Regression)
使用 Episode 格式的时序数据进行训练

特征：
1. 24小时时序数据的统计特征 (min, max, mean, last)
2. 标注统计特征 (n_supportive, n_contradictory)
3. MedCAT 概念统计特征 (medcat_full, 24h)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
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
    TEST_SIZE, USE_HOLDOUT_TEST, COHORT_FILE
)

# 配置
EPISODES_DIR = Path(__file__).parent.parent.parent / 'episodes' / 'episodes_enhanced'
SPLITS_DIR = PROCESSED_DIR.parent / 'splits'
OUTPUT_DIR = RESULTS_DIR / 'tabular_baselines'
MEDCAT_FILE = PROCESSED_DIR / 'medcat_full' / 'medcat_features_24h.csv'

# 时序特征列
VITALS_COLS = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2']


def extract_features_from_episode(episode_path: Path) -> dict:
    """从单个 Episode 提取表格特征"""
    with open(episode_path) as f:
        ep = json.load(f)
    
    features = {
        'stay_id': ep.get('stay_id'),
        'subject_id': ep.get('patient', {}).get('subject_id')
    }
    
    # 时序特征
    ts = ep.get('timeseries', {})
    vitals = ts.get('vitals', [])
    
    if vitals:
        vitals_df = pd.DataFrame(vitals)
        if 'hour' in vitals_df.columns:
            vitals_df = vitals_df.sort_values('hour')
        for col in VITALS_COLS:
            if col in vitals_df.columns:
                col_values = pd.to_numeric(vitals_df[col], errors='coerce')
                values = col_values.dropna()
                n_values = len(values)
                features[f'{col}_n'] = n_values
                features[f'{col}_missing'] = 0 if n_values > 0 else 1
                if n_values > 0:
                    features[f'{col}_mean'] = values.mean()
                    features[f'{col}_min'] = values.min()
                    features[f'{col}_max'] = values.max()
                    features[f'{col}_last'] = values.iloc[-1]
                    features[f'{col}_std'] = values.std() if len(values) > 1 else 0
                else:
                    for suffix in ['mean', 'min', 'max', 'last', 'std']:
                        features[f'{col}_{suffix}'] = np.nan
            else:
                features[f'{col}_n'] = 0
                features[f'{col}_missing'] = 1
                for suffix in ['mean', 'min', 'max', 'last', 'std']:
                    features[f'{col}_{suffix}'] = np.nan
    else:
        for col in VITALS_COLS:
            features[f'{col}_n'] = 0
            features[f'{col}_missing'] = 1
            for suffix in ['mean', 'min', 'max', 'last', 'std']:
                features[f'{col}_{suffix}'] = np.nan
    
    # Labs 特征 (如果存在)
    labs = ts.get('labs', [])
    if labs:
        features['n_labs'] = len(labs)
    else:
        features['n_labs'] = 0
    
    # 标注特征
    reasoning = ep.get('reasoning', {})
    features['n_patterns'] = len(reasoning.get('detected_patterns', []))
    features['n_supportive'] = reasoning.get('n_supportive', 0)
    features['n_contradictory'] = reasoning.get('n_contradictory', 0)
    features['n_alignments'] = reasoning.get('n_alignments', 0)
    
    # 标签
    labels = ep.get('labels', {})
    outcome = labels.get('outcome', {})
    features['mortality'] = outcome.get('mortality', 0)
    features['prolonged_los'] = outcome.get('prolonged_los', 0)
    
    return features


def load_all_features():
    """加载所有 Episode 的特征"""
    print("加载 Episode 特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
    features_list = []
    for ep_file in tqdm(episode_files, desc="Extracting features"):
        try:
            features = extract_features_from_episode(ep_file)
            features_list.append(features)
        except Exception as e:
            pass
    
    df = pd.DataFrame(features_list)
    print(f"   提取特征: {len(df):,} 个样本, {len(df.columns)} 个特征")
    
    return df


def load_medcat_features():
    """加载 MedCAT 概念特征"""
    if not MEDCAT_FILE.exists():
        print("未找到 MedCAT 特征文件，跳过合并")
        return None

    medcat = pd.read_csv(MEDCAT_FILE)
    if 'window_hours' in medcat.columns:
        medcat = medcat.drop(columns=['window_hours'])
    medcat = medcat.rename(
        columns={c: f'medcat_{c}' for c in medcat.columns if c != 'stay_id'}
    )
    print(f"   MedCAT 特征: {len(medcat):,} 个样本, {len(medcat.columns)-1} 个特征")
    return medcat


def train_and_evaluate(X, y, groups, model_name='XGBoost'):
    """训练和评估模型"""
    
    # 分离测试集
    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
        
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        groups_train_val = groups[train_val_idx]
    else:
        X_train_val, y_train_val, groups_train_val = X, y, groups
        X_test, y_test = None, None
    
    # 交叉验证
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train_val, y_train_val, groups=groups_train_val)):
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]
        
        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        
        # 训练
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
        
        fold_results.append({
            'fold': fold + 1,
            'auroc': auroc,
            'auprc': auprc
        })
        
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")
    
    # 测试集评估
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
        
        test_result = {
            'auroc': test_auroc,
            'auprc': test_auprc
        }
    
    return fold_results, test_result


def main():
    print("=" * 60)
    print("Tabular Baselines (XGBoost & Logistic Regression)")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载特征
    df = load_all_features()
    medcat = load_medcat_features()
    if medcat is not None:
        df = df.merge(medcat, on='stay_id', how='left')
        df = df.fillna(0)
    
    # 准备特征和标签
    feature_cols = [c for c in df.columns if c not in ['stay_id', 'subject_id', 'mortality', 'prolonged_los']]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    
    groups = df['subject_id'].values
    
    results = []
    
    for task in ['mortality', 'prolonged_los']:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")
        
        y = df[task].values
        
        for model_name in ['XGBoost', 'LogisticRegression']:
            print(f"\n{model_name}:")
            
            fold_results, test_result = train_and_evaluate(X, y, groups, model_name)
            
            # 汇总
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
                'cv_auroc_mean': mean_auroc,
                'cv_auroc_std': std_auroc,
                'cv_auprc_mean': mean_auprc,
                'cv_auprc_std': std_auprc,
                'test_auroc': test_result['auroc'] if test_result else None,
                'test_auprc': test_result['auprc'] if test_result else None
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'tabular_results.csv', index=False)
    print(f"\n结果保存到: {OUTPUT_DIR / 'tabular_results.csv'}")
    
    # 打印最终汇总
    print("\n" + "=" * 60)
    print("最终结果汇总")
    print("=" * 60)
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
