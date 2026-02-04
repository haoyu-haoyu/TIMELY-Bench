"""
隐性诊断特征提取器
用于提取区分相似症状疾病的隐性因素
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm
from scipy import stats

class HiddenFeatureExtractor:
    """隐性特征提取器"""
    
    def __init__(self):
        pass
    
    def extract_hrv_features(self, heart_rates: List[float]) -> Dict:
        """提取心率变异性 (HRV) 特征"""
        if len(heart_rates) < 5:
            return {'hrv_sdnn': 0, 'hrv_rmssd': 0, 'hrv_cv': 0}
        
        hr = np.array(heart_rates)
        
        # 时域 HRV 特征
        sdnn = np.std(hr)  # 标准差
        
        # 相邻差值的均方根
        diff = np.diff(hr)
        rmssd = np.sqrt(np.mean(diff ** 2)) if len(diff) > 0 else 0
        
        # 变异系数
        cv = sdnn / np.mean(hr) if np.mean(hr) > 0 else 0
        
        return {
            'hrv_sdnn': round(sdnn, 3),
            'hrv_rmssd': round(rmssd, 3),
            'hrv_cv': round(cv, 4)
        }
    
    def extract_trend_features(self, values: List[tuple], feature_name: str) -> Dict:
        """提取趋势特征 (变化率、斜率)"""
        if len(values) < 3:
            return {
                f'{feature_name}_slope': 0,
                f'{feature_name}_rate': 0,
                f'{feature_name}_volatility': 0
            }
        
        hours = np.array([v[0] for v in values])
        vals = np.array([v[1] for v in values])
        
        # 线性回归斜率
        if len(set(hours)) > 1:
            slope, _, r_value, _, _ = stats.linregress(hours, vals)
        else:
            slope, r_value = 0, 0
        
        # 变化率 (每小时)
        total_change = vals[-1] - vals[0]
        time_span = hours[-1] - hours[0]
        rate = total_change / time_span if time_span > 0 else 0
        
        # 波动性 (高阶变化)
        volatility = np.std(np.diff(vals)) if len(vals) > 2 else 0
        
        return {
            f'{feature_name}_slope': round(slope, 4),
            f'{feature_name}_rate': round(rate, 4),
            f'{feature_name}_volatility': round(volatility, 4),
            f'{feature_name}_r2': round(r_value ** 2, 4)
        }
    
    def extract_correlation_features(self, vitals: List[dict]) -> Dict:
        """提取多指标相关性特征"""
        if len(vitals) < 5:
            return {
                'corr_hr_temp': 0,
                'corr_rr_spo2': 0,
                'corr_hr_sbp': 0
            }
        
        data = pd.DataFrame(vitals)
        correlations = {}
        
        # HR vs Temperature (感染活动性)
        if 'heart_rate' in data.columns and 'temperature' in data.columns:
            valid = data[['heart_rate', 'temperature']].dropna()
            if len(valid) >= 3:
                correlations['corr_hr_temp'] = round(valid['heart_rate'].corr(valid['temperature']), 3)
        
        # RR vs SpO2 (呼吸代偿)
        if 'resp_rate' in data.columns and 'spo2' in data.columns:
            valid = data[['resp_rate', 'spo2']].dropna()
            if len(valid) >= 3:
                correlations['corr_rr_spo2'] = round(valid['resp_rate'].corr(valid['spo2']), 3)
        
        # HR vs SBP (血流动力学)
        if 'heart_rate' in data.columns and 'sbp' in data.columns:
            valid = data[['heart_rate', 'sbp']].dropna()
            if len(valid) >= 3:
                correlations['corr_hr_sbp'] = round(valid['heart_rate'].corr(valid['sbp']), 3)
        
        return {k: v if not pd.isna(v) else 0 for k, v in correlations.items()}
    
    def extract_text_evidence_features(self, annotations: List[dict]) -> Dict:
        """提取文本证据特征"""
        if not annotations:
            return {
                'text_support_ratio': 0,
                'text_contradict_ratio': 0,
                'text_evidence_density': 0,
                'has_negative_finding': 0
            }
        
        categories = [a.get('annotation_category', '') for a in annotations]
        n_total = len(categories)
        
        n_supportive = categories.count('SUPPORTIVE')
        n_contradictory = categories.count('CONTRADICTORY')
        
        # 提取阴性发现
        negative_keywords = ['no infection', 'culture negative', 'no evidence', 'normal', 'negative']
        has_negative = 0
        for ann in annotations:
            text = str(ann.get('aligned_text', '')).lower()
            if any(kw in text for kw in negative_keywords):
                has_negative = 1
                break
        
        return {
            'text_support_ratio': round(n_supportive / n_total, 3) if n_total > 0 else 0,
            'text_contradict_ratio': round(n_contradictory / n_total, 3) if n_total > 0 else 0,
            'text_evidence_density': n_supportive + n_contradictory,
            'has_negative_finding': has_negative
        }
    
    def extract_temporal_pattern_features(self, patterns: List[dict]) -> Dict:
        """提取时序模式特征"""
        if not patterns:
            return {
                'pattern_diversity': 0,
                'pattern_concentration': 0,
                'pattern_recurrence': 0
            }
        
        pattern_names = [p.get('pattern_name') for p in patterns if p.get('pattern_name')]
        pattern_hours = [p.get('hour', 0) for p in patterns if p.get('hour') is not None]
        
        if not pattern_names:
            return {'pattern_diversity': 0, 'pattern_concentration': 0, 'pattern_recurrence': 0}
        
        # 多样性 (unique patterns / total)
        diversity = len(set(pattern_names)) / len(pattern_names)
        
        # 时间集中度 (patterns per hour variance)
        if pattern_hours:
            hour_counts = pd.Series(pattern_hours).value_counts()
            concentration = hour_counts.std() / hour_counts.mean() if hour_counts.mean() > 0 else 0
        else:
            concentration = 0
        
        # 复发率 (same pattern appearing multiple times)
        from collections import Counter
        pattern_counts = Counter(pattern_names)
        recurrence = sum(1 for c in pattern_counts.values() if c > 1) / len(pattern_counts)
        
        return {
            'pattern_diversity': round(diversity, 3),
            'pattern_concentration': round(concentration, 3) if not pd.isna(concentration) else 0,
            'pattern_recurrence': round(recurrence, 3)
        }
    
    def extract_all_hidden_features(self, episode: dict) -> Dict:
        """提取所有隐性特征"""
        vitals = episode.get('timeseries', {}).get('vitals', [])
        labs = episode.get('timeseries', {}).get('labs', [])
        patterns = episode.get('reasoning', {}).get('detected_patterns', [])
        annotations = episode.get('reasoning', {}).get('pattern_annotations', [])
        
        features = {'stay_id': episode.get('stay_id')}
        
        # 1. HRV 特征
        heart_rates = [v.get('heart_rate') for v in vitals if v.get('heart_rate')]
        features.update(self.extract_hrv_features(heart_rates))
        
        # 2. 肌酐趋势特征
        creatinine = [(l.get('hour', 0), l.get('creatinine')) for l in labs if l.get('creatinine')]
        features.update(self.extract_trend_features(creatinine, 'creatinine'))
        
        # 3. 乳酸趋势特征
        lactate = [(l.get('hour', 0), l.get('lactate')) for l in labs if l.get('lactate')]
        features.update(self.extract_trend_features(lactate, 'lactate'))
        
        # 4. 多指标相关性
        features.update(self.extract_correlation_features(vitals))
        
        # 5. 文本证据特征
        features.update(self.extract_text_evidence_features(annotations))
        
        # 6. 时序模式特征
        features.update(self.extract_temporal_pattern_features(patterns))
        
        return features


def main():
    print("=" * 60)
    print("隐性诊断特征提取")
    print("=" * 60)
    
    EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
    OUTPUT_FILE = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/hidden_features/hidden_features.csv')
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    extractor = HiddenFeatureExtractor()
    
    episode_files = list(EPISODES_DIR.glob('*.json'))
    print(f"待处理 Episodes: {len(episode_files):,}")
    
    all_features = []
    for ep_file in tqdm(episode_files, desc="提取隐性特征"):
        try:
            ep = json.load(open(ep_file))
            features = extractor.extract_all_hidden_features(ep)
            all_features.append(features)
        except Exception as e:
            continue
    
    df = pd.DataFrame(all_features).fillna(0)
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"\n完成！")
    print(f"  样本数: {len(df):,}")
    print(f"  特征维度: {len(df.columns)}")
    print(f"  保存到: {OUTPUT_FILE}")
    
    print(f"\n特征列表:")
    for col in df.columns:
        if col != 'stay_id':
            print(f"  - {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}")


if __name__ == "__main__":
    main()
