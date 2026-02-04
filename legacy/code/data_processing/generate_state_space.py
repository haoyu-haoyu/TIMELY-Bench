"""
重建患者状态空间 - 每小时完整状态向量
纯代码实现，无需 LLM
"""

import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')


def interpolate_vitals(vitals, target_hour):
    """插值获取目标小时的生命体征"""
    if not vitals:
        return {}
    
    # 找到最近的数据点
    closest = None
    min_diff = float('inf')
    
    for v in vitals:
        hour = v.get('hour', 0)
        diff = abs(hour - target_hour)
        if diff < min_diff:
            min_diff = diff
            closest = v
    
    if closest and min_diff <= 2:  # 2小时内的数据
        return {
            'heart_rate': closest.get('heart_rate'),
            'temperature': closest.get('temperature'),
            'sbp': closest.get('sbp'),
            'dbp': closest.get('dbp'),
            'resp_rate': closest.get('resp_rate'),
            'spo2': closest.get('spo2')
        }
    return {}


def forward_fill_labs(labs, target_hour):
    """前向填充实验室检查值"""
    if not labs:
        return {}
    
    result = {}
    for l in labs:
        hour = l.get('hour', 0)
        if hour <= target_hour:
            # 更新为最新值
            if l.get('creatinine') is not None:
                result['creatinine'] = l['creatinine']
            if l.get('lactate') is not None:
                result['lactate'] = l['lactate']
            if l.get('wbc') is not None:
                result['wbc'] = l['wbc']
    
    return result


def get_active_patterns(episode, target_hour, window=2):
    """获取目标小时附近的活动模式"""
    patterns = episode.get('reasoning', {}).get('detected_patterns', [])
    active = []
    
    for p in patterns:
        hour = p.get('hour', 0)
        if abs(hour - target_hour) <= window:
            active.append(p.get('pattern_name', 'unknown'))
    
    return list(set(active))


def calculate_severity(episode, hour, vitals_state, labs_state):
    """计算当前严重程度得分 (0-1)"""
    score = 0.0
    count = 0
    
    # 心率异常
    hr = vitals_state.get('heart_rate')
    if hr:
        if hr > 100:
            score += min((hr - 100) / 50, 1.0) * 0.2
        elif hr < 60:
            score += min((60 - hr) / 30, 1.0) * 0.2
        count += 1
    
    # 血氧异常
    spo2 = vitals_state.get('spo2')
    if spo2:
        if spo2 < 95:
            score += min((95 - spo2) / 15, 1.0) * 0.3
        count += 1
    
    # 肌酐异常
    cr = labs_state.get('creatinine')
    if cr:
        if cr > 1.2:
            score += min((cr - 1.2) / 2.0, 1.0) * 0.25
        count += 1
    
    # 乳酸异常
    lactate = labs_state.get('lactate')
    if lactate:
        if lactate > 2.0:
            score += min((lactate - 2.0) / 4.0, 1.0) * 0.25
        count += 1
    
    return round(min(score, 1.0), 3)


def reconstruct_state_space(episode):
    """重建完整状态空间"""
    vitals = episode.get('timeseries', {}).get('vitals', [])
    labs = episode.get('timeseries', {}).get('labs', [])
    
    state_space = []
    max_hour = 48
    
    for hour in range(max_hour):
        vitals_state = interpolate_vitals(vitals, hour)
        labs_state = forward_fill_labs(labs, hour)
        active_patterns = get_active_patterns(episode, hour)
        severity = calculate_severity(episode, hour, vitals_state, labs_state)
        
        state = {
            'hour': hour,
            'vitals': vitals_state,
            'labs': labs_state,
            'active_patterns': active_patterns,
            'severity_score': severity,
            'data_available': bool(vitals_state or labs_state)
        }
        state_space.append(state)
    
    return state_space


def process_episode(ep_file):
    """处理单个 Episode"""
    ep = json.load(open(ep_file))
    
    # 重建状态空间
    state_space = reconstruct_state_space(ep)
    
    # 添加到 Episode
    ep['patient_state_space'] = state_space
    
    # 保存
    with open(ep_file, 'w') as f:
        json.dump(ep, f, indent=2, ensure_ascii=False)
    
    return True


def main():
    print("=" * 70)
    print("Patient State-Space 重建")
    print("=" * 70)
    
    episode_files = list(EPISODES_DIR.glob('*.json'))
    print(f"待处理 Episodes: {len(episode_files):,}")
    
    success = 0
    for ep_file in tqdm(episode_files, desc="重建状态空间"):
        try:
            if process_episode(ep_file):
                success += 1
        except Exception as e:
            continue
    
    print(f"\n完成！成功: {success:,}")


if __name__ == "__main__":
    main()
