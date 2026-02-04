"""
Retrain models with enhanced features.
Uses syndrome_detection, reasoning_chain, disease_timeline.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 配置
EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/results/enhanced_reasoning')
N_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.15
USE_DISEASE_TIMELINE = False


def extract_enhanced_features(episode_path: Path) -> dict:
    """提取增强特征，包括新的 reasoning 字段"""
    with open(episode_path) as f:
        ep = json.load(f)
    
    features = {
        'stay_id': ep.get('stay_id'),
        'subject_id': ep.get('patient', {}).get('subject_id')
    }
    
    # 1. 基础时序特征
    vitals = ep.get('timeseries', {}).get('vitals', [])
    vital_cols = ['heart_rate', 'sbp', 'dbp', 'resp_rate', 'spo2', 'temperature']
    
    for col in vital_cols:
        values = [v.get(col) for v in vitals if v.get(col) is not None]
        n_values = len(values)
        features[f'{col}_n'] = n_values
        features[f'{col}_missing'] = 0 if n_values > 0 else 1
        if values:
            features[f'{col}_mean'] = np.mean(values)
            features[f'{col}_std'] = np.std(values)
            features[f'{col}_min'] = np.min(values)
            features[f'{col}_max'] = np.max(values)
        else:
            features[f'{col}_mean'] = np.nan
            features[f'{col}_std'] = np.nan
            features[f'{col}_min'] = np.nan
            features[f'{col}_max'] = np.nan
    
    # 2. 原有标注特征
    reasoning = ep.get('reasoning', {})
    features['n_supportive'] = reasoning.get('n_supportive', 0)
    features['n_contradictory'] = reasoning.get('n_contradictory', 0)
    features['n_patterns'] = reasoning.get('n_patterns_detected', 0)
    
    # 3. 新增：syndrome_detection 特征
    sd = reasoning.get('syndrome_detection', {})
    
    # Sepsis 检测
    sepsis = sd.get('sepsis', {})
    features['sepsis_detected'] = 1 if sepsis.get('detected') else 0
    features['sepsis_sirs_count'] = sepsis.get('sirs_count', 0)
    features['sepsis_infection'] = 1 if sepsis.get('infection_evidence') else 0
    
    # AKI 检测
    aki = sd.get('aki', {})
    features['aki_detected'] = 1 if aki.get('detected') else 0
    features['aki_stage'] = aki.get('stage', 0)
    features['aki_creatinine_ratio'] = aki.get('creatinine_ratio', 1.0)
    
    # ARDS 检测
    ards = sd.get('ards', {})
    features['ards_detected'] = 1 if ards.get('detected') else 0
    
    # 4. 新增：reasoning_chain 特征
    rc = reasoning.get('reasoning_chain', {})
    features['rc_evidence_count'] = len(rc.get('evidence', []))
    features['rc_confidence'] = rc.get('confidence', 0.5)
    
    # 5. 新增：disease_timeline 特征（默认关闭以避免标签泄漏）
    if USE_DISEASE_TIMELINE:
        dt = reasoning.get('disease_timeline', {})

        # 疾病类型编码
        disease = dt.get('primary_disease', 'none')
        features['dt_has_sepsis'] = 1 if 'sepsis' in disease else 0
        features['dt_has_aki'] = 1 if 'aki' in disease else 0
        features['dt_has_none'] = 1 if disease == 'none' else 0

        # onset hour
        onset = dt.get('onset_hour')
        features['dt_onset_hour'] = onset if isinstance(onset, (int, float)) else -1

        # prognosis 编码
        prognosis = dt.get('prognosis', 'unknown')
        features['dt_deteriorating'] = 1 if prognosis == 'deteriorating' else 0
        features['dt_stable'] = 1 if prognosis == 'stable' else 0
        features['dt_improving'] = 1 if prognosis == 'improving' else 0

        # phases 数量
        phases = dt.get('phases', [])
        features['dt_n_phases'] = len(phases) if phases else 0
    
    # 6. patient_state_space 特征
    pss = ep.get('patient_state_space', [])
    if pss:
        features['pss_hours'] = len(pss)
        # 最后状态的严重程度
        last_state = pss[-1] if pss else {}
        features['pss_last_severity'] = last_state.get('severity', 0)
    else:
        features['pss_hours'] = 0
        features['pss_last_severity'] = 0
    
    # 7. 标签
    labels = ep.get('labels', {})
    outcome = labels.get('outcome', {})
    features['mortality'] = outcome.get('mortality', 0)
    features['prolonged_los'] = outcome.get('prolonged_los', 0)
    
    return features


def load_all_features():
    """加载所有增强特征"""
    print("加载增强 Episode 特征...")
    
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
    features_list = []
    for ep_file in tqdm(episode_files, desc="提取增强特征"):
        try:
            features = extract_enhanced_features(ep_file)
            features_list.append(features)
        except Exception as e:
            pass
    
    df = pd.DataFrame(features_list)
    print(f"   提取特征: {len(df):,} 个样本, {len(df.columns)} 个特征")
    
    return df


def train_and_evaluate(X, y, groups, model_name='XGBoost'):
    """训练和评估"""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train_val, X_test = X[train_val_idx], X[test_idx]
    y_train_val, y_test = y[train_val_idx], y[test_idx]
    groups_train_val = groups[train_val_idx]
    
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
        print(f"   Fold {fold+1}: AUROC={auroc:.4f}")
    
    # 测试集评估
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
    
    return fold_results, {'auroc': test_auroc, 'auprc': test_auprc}


def main():
    print("=" * 70)
    print("增强 Reasoning 特征训练")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = load_all_features()
    
    # 排除非特征列
    exclude_cols = ['stay_id', 'subject_id', 'mortality', 'prolonged_los']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"\n使用 {len(feature_cols)} 个特征:")
    if USE_DISEASE_TIMELINE:
        print("  新增 reasoning 特征: sepsis_detected, aki_detected, dt_has_sepsis, ...")
    else:
        print("  新增 reasoning 特征: sepsis_detected, aki_detected, (timeline 特征已关闭)")
    
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    groups = df['subject_id'].values
    
    results = []
    
    for task in ['mortality', 'prolonged_los']:
        print(f"\n{'='*70}")
        print(f"Task: {task}")
        print(f"{'='*70}")
        
        y = df[task].values
        
        for model_name in ['XGBoost', 'LogisticRegression']:
            print(f"\n{model_name}:")
            
            fold_results, test_result = train_and_evaluate(X, y, groups, model_name)
            
            mean_auroc = np.mean([r['auroc'] for r in fold_results])
            std_auroc = np.std([r['auroc'] for r in fold_results])
            
            print(f"\n   CV AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
            print(f"   Test AUROC: {test_result['auroc']:.4f}")
            
            results.append({
                'task': task,
                'model': model_name,
                'cv_auroc_mean': mean_auroc,
                'cv_auroc_std': std_auroc,
                'test_auroc': test_result['auroc'],
                'test_auprc': test_result['auprc']
            })
    
    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'enhanced_reasoning_results.csv', index=False)
    
    print("\n" + "=" * 70)
    print("最终结果")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    print(f"\n结果保存到: {OUTPUT_DIR / 'enhanced_reasoning_results.csv'}")


if __name__ == "__main__":
    main()
