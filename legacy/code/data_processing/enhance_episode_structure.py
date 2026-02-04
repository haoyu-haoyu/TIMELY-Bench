"""
Episode 结构增强脚本
添加缺失字段：conditions, condition_graph_nodes, notes_spans, physiology_templates
"""

import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 配置
EPISODES_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')
COHORTS_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/data/processed/cohorts')
OUTPUT_DIR = Path('/home/ubuntu/TIMELY-Bench_Final/episodes/episodes_enhanced')

# ICD 到 UMLS CUI 映射
CONDITION_CUI_MAP = {
    'sepsis': {'cui': 'C0243026', 'name': 'Sepsis'},
    'aki': {'cui': 'C0022660', 'name': 'Acute Kidney Injury'},
    'ards': {'cui': 'C0035222', 'name': 'Respiratory Distress Syndrome, Adult'},
    'shock': {'cui': 'C0036974', 'name': 'Shock'},
    'pneumonia': {'cui': 'C0032285', 'name': 'Pneumonia'},
    'heart_failure': {'cui': 'C0018801', 'name': 'Heart Failure'},
    'respiratory_failure': {'cui': 'C1145670', 'name': 'Respiratory Failure'}
}

# 生理学模板定义
PHYSIOLOGY_TEMPLATES = {
    'tachycardia': {
        'pattern_type': 'vital_sign',
        'threshold': 'HR > 100',
        'clinical_significance': 'stress, infection, hypovolemia',
        'related_conditions': ['sepsis', 'fever', 'shock']
    },
    'bradycardia': {
        'pattern_type': 'vital_sign',
        'threshold': 'HR < 60',
        'clinical_significance': 'medication effect, heart block',
        'related_conditions': ['heart_failure']
    },
    'hypotension': {
        'pattern_type': 'vital_sign',
        'threshold': 'SBP < 90 or MAP < 65',
        'clinical_significance': 'shock, sepsis, hypovolemia',
        'related_conditions': ['sepsis', 'shock']
    },
    'hypertension': {
        'pattern_type': 'vital_sign',
        'threshold': 'SBP > 180 or DBP > 120',
        'clinical_significance': 'hypertensive emergency',
        'related_conditions': ['stroke', 'heart_failure']
    },
    'fever': {
        'pattern_type': 'vital_sign',
        'threshold': 'Temp > 38.3°C',
        'clinical_significance': 'infection, inflammation',
        'related_conditions': ['sepsis', 'pneumonia']
    },
    'hypothermia': {
        'pattern_type': 'vital_sign',
        'threshold': 'Temp < 36°C',
        'clinical_significance': 'severe sepsis, exposure',
        'related_conditions': ['sepsis', 'shock']
    },
    'hypoxemia': {
        'pattern_type': 'vital_sign',
        'threshold': 'SpO2 < 90%',
        'clinical_significance': 'respiratory failure, ARDS',
        'related_conditions': ['ards', 'pneumonia', 'respiratory_failure']
    },
    'tachypnea': {
        'pattern_type': 'vital_sign',
        'threshold': 'RR > 22',
        'clinical_significance': 'respiratory distress, metabolic acidosis',
        'related_conditions': ['ards', 'sepsis', 'respiratory_failure']
    }
}


def load_condition_mapping():
    """加载疾病映射"""
    cohort = pd.read_csv(COHORTS_DIR / 'cohort_with_conditions.csv')
    return cohort.set_index('stay_id').to_dict('index')


def enhance_episode(ep, condition_info):
    """增强 Episode 结构"""
    stay_id = ep.get('stay_id')
    
    # 1. 添加 conditions 列表
    conditions = []
    condition_graph_nodes = []
    
    if condition_info:
        condition_cols = ['has_sepsis_final', 'has_aki_final', 'has_ards', 
                          'has_shock', 'has_pneumonia', 'has_heart_failure', 
                          'has_respiratory_failure']
        
        for col in condition_cols:
            if condition_info.get(col, 0) == 1:
                cond_name = col.replace('has_', '').replace('_final', '')
                conditions.append(cond_name)
                
                if cond_name in CONDITION_CUI_MAP:
                    cui_info = CONDITION_CUI_MAP[cond_name]
                    condition_graph_nodes.append({
                        'cui': cui_info['cui'],
                        'name': cui_info['name'],
                        'source': 'ICD/Clinical'
                    })
    
    ep['conditions'] = conditions
    ep['condition_graph_nodes'] = condition_graph_nodes
    
    # 2. 添加 notes_spans（从现有 notes 重构）
    notes_spans = []
    clinical_text = ep.get('clinical_text', {})
    notes = clinical_text.get('notes', [])
    
    for note in notes:
        if isinstance(note, dict):
            notes_spans.append({
                'note_id': note.get('note_id', ''),
                'note_type': note.get('note_type', ''),
                'chart_time': note.get('chart_time', ''),
                'chart_hour': note.get('chart_hour', 0),
                'text_length': note.get('text_length', len(note.get('text_full', '')))
            })
    
    ep['notes_spans'] = notes_spans
    
    # 3. 填充 physiology_templates
    reasoning = ep.get('reasoning', {})
    detected_patterns = reasoning.get('detected_patterns', [])
    
    matched_templates = {}
    for pattern in detected_patterns:
        pattern_name = pattern.get('pattern_name', '').lower()
        for template_name, template in PHYSIOLOGY_TEMPLATES.items():
            if template_name in pattern_name:
                matched_templates[template_name] = template
    
    reasoning['physiology_templates'] = matched_templates
    ep['reasoning'] = reasoning
    
    return ep


def main():
    print("=" * 60)
    print("Episode 结构增强")
    print("=" * 60)
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 加载疾病映射
    print("加载疾病映射...")
    condition_map = load_condition_mapping()
    print(f"加载了 {len(condition_map):,} 条映射")
    
    # 获取所有 Episode 文件
    episode_files = list(EPISODES_DIR.glob('TIMELY_v2_*.json'))
    print(f"找到 {len(episode_files):,} 个 Episode 文件")
    
    # 统计
    enhanced_count = 0
    conditions_stats = {}
    
    for ep_file in tqdm(episode_files, desc="增强 Episode"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            stay_id = ep.get('stay_id')
            condition_info = condition_map.get(stay_id, {})
            
            # 增强
            enhanced_ep = enhance_episode(ep, condition_info)
            
            # 保存到增强目录
            output_file = OUTPUT_DIR / ep_file.name
            with open(output_file, 'w') as f:
                json.dump(enhanced_ep, f, indent=2)
            
            enhanced_count += 1
            
            # 统计
            for cond in enhanced_ep.get('conditions', []):
                conditions_stats[cond] = conditions_stats.get(cond, 0) + 1
                
        except Exception as e:
            print(f"错误处理 {ep_file.name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"完成！增强了 {enhanced_count:,} 个 Episode")
    print(f"保存到: {OUTPUT_DIR}")
    print("\n疾病分布统计:")
    for cond, count in sorted(conditions_stats.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {count:,}")


if __name__ == "__main__":
    main()
