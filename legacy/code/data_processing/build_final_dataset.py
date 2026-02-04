"""
Build Final TIMELY-Bench Dataset
为每个ICU episode创建一个完整的结构化记录

包含内容：
1. 患者基本信息和conditions
2. 时序数据片段（带时间戳）
3. 临床笔记片段（带时间戳）
4. 检测到的pattern及其对应的时序/文本证据
5. Pattern annotations (SUPPORTIVE/CONTRADICTORY/AMBIGUOUS)
6. 预测标签
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime

from config import (
    COHORT_FILE, TIMESERIES_FILE, NOTE_TIME_FILE,
    PATTERN_DETECTION_DIR, TEMPORAL_ALIGNMENT_DIR, ROOT_DIR
)

# ==========================================
# 配置
# ==========================================
PATTERN_DETECTION_FILE = PATTERN_DETECTION_DIR / 'detected_patterns_24h.csv'
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'
PATTERN_TEMPLATES_FILE = ROOT_DIR / 'documentation' / 'pattern_templates.json'

OUTPUT_DIR = ROOT_DIR / 'TIMELY_Bench_Dataset'

# ==========================================
# 1. 加载所有数据
# ==========================================

def load_all_data():
    """加载所有必要的数据"""
    
    print("Loading all data sources...")
    
    data = {}
    
    # 1. Cohort信息
    print("   Loading cohort...")
    data['cohort'] = pd.read_csv(COHORT_FILE)
    data['cohort']['stay_id'] = data['cohort']['stay_id'].astype(int)
    print(f"      {len(data['cohort'])} patients")
    
    # 2. 时序数据
    print("   Loading timeseries...")
    data['timeseries'] = pd.read_csv(TIMESERIES_FILE)
    data['timeseries']['stay_id'] = data['timeseries']['stay_id'].astype(int)
    if TIMESERIES_FILE.name == 'timeseries.csv':
        data['timeseries'] = data['timeseries'].sort_values(['stay_id', 'hour'])
    print(f"      {len(data['timeseries'])} records")
    
    # 3. 临床笔记
    print("   Loading notes...")
    data['notes'] = pd.read_csv(NOTE_FILE)
    data['notes']['stay_id'] = pd.to_numeric(data['notes']['stay_id'], errors='coerce').fillna(-1).astype(int)
    # 处理列名
    if 'radiology_text' in data['notes'].columns:
        data['notes']['text'] = data['notes']['radiology_text']
    print(f"      {len(data['notes'])} notes")
    
    # 4. Pattern检测结果
    print("   Loading pattern detections...")
    if os.path.exists(PATTERN_DETECTION_FILE):
        data['patterns'] = pd.read_csv(PATTERN_DETECTION_FILE)
        print(f"      {len(data['patterns'])} detections")
    else:
        data['patterns'] = pd.DataFrame()
        print("      No pattern detection file found")
    
    # 5. Alignment结果
    print("   Loading alignments...")
    if os.path.exists(ALIGNMENT_FILE):
        data['alignments'] = pd.read_csv(ALIGNMENT_FILE)
        print(f"      {len(data['alignments'])} alignments")
    else:
        data['alignments'] = pd.DataFrame()
        print("      No alignment file found")
    
    # 6. Pattern模板
    print("   Loading pattern templates...")
    if os.path.exists(PATTERN_TEMPLATES_FILE):
        with open(PATTERN_TEMPLATES_FILE, 'r') as f:
            data['templates'] = json.load(f)
        print(f"      {sum(len(v['patterns']) for v in data['templates'].values())} templates")
    else:
        data['templates'] = {}
        print("      No template file found")
    
    return data

# ==========================================
# 2. 构建Condition Graph
# ==========================================

def build_condition_graph():
    """构建疾病关系图"""
    
    # 定义疾病之间的临床关系
    condition_relationships = {
        "sepsis_to_aki": {
            "source": "sepsis",
            "target": "aki",
            "relationship": "can_cause",
            "mechanism": "Sepsis-induced hypoperfusion and nephrotoxicity"
        },
        "sepsis_to_ards": {
            "source": "sepsis",
            "target": "ards",
            "relationship": "can_cause",
            "mechanism": "Inflammatory cascade causing lung injury"
        },
        "aki_to_ards": {
            "source": "aki",
            "target": "ards",
            "relationship": "bidirectional",
            "mechanism": "Fluid overload and inflammatory mediators"
        },
        "shock_to_aki": {
            "source": "shock",
            "target": "aki",
            "relationship": "can_cause",
            "mechanism": "Renal hypoperfusion"
        },
        "shock_to_ards": {
            "source": "shock",
            "target": "ards",
            "relationship": "can_cause",
            "mechanism": "Ischemia-reperfusion injury"
        }
    }
    
    return condition_relationships

def get_patient_condition_graph(conditions: List[str], graph: Dict) -> List[Dict]:
    """获取特定患者的疾病关系子图"""
    
    nodes = []
    for cond in conditions:
        nodes.append({
            "id": cond,
            "type": "condition",
            "present": True
        })
    
    edges = []
    for edge_id, edge_data in graph.items():
        if edge_data['source'] in conditions and edge_data['target'] in conditions:
            edges.append({
                "source": edge_data['source'],
                "target": edge_data['target'],
                "relationship": edge_data['relationship'],
                "mechanism": edge_data['mechanism']
            })
    
    return {"nodes": nodes, "edges": edges}

# ==========================================
# 3. 提取时序片段
# ==========================================

def extract_timeseries_for_episode(
    stay_id: int, 
    timeseries_df: pd.DataFrame,
    window_hours: int = 24
) -> Dict[str, Any]:
    """提取episode的时序数据"""
    
    patient_ts = timeseries_df[
        (timeseries_df['stay_id'] == stay_id) & 
        (timeseries_df['hour'] < window_hours)
    ].copy()
    
    if len(patient_ts) == 0:
        return {"hours": [], "features": {}}
    
    # 按小时组织
    hours = sorted(patient_ts['hour'].unique().tolist())
    
    # 提取各特征
    feature_cols = [c for c in patient_ts.columns if c not in ['stay_id', 'hour']]
    
    features = {}
    for col in feature_cols:
        # 提取时间序列
        series = []
        for h in hours:
            hour_data = patient_ts[patient_ts['hour'] == h]
            if len(hour_data) > 0:
                val = hour_data[col].values[0]
                if pd.notna(val):
                    series.append({"hour": int(h), "value": float(val)})
        
        if series:
            features[col] = series
    
    return {
        "window_hours": window_hours,
        "hours_available": hours,
        "n_hours": len(hours),
        "features": features
    }

# ==========================================
# 4. 提取笔记片段
# ==========================================

def extract_notes_for_episode(
    stay_id: int,
    notes_df: pd.DataFrame,
    window_hours: int = 24
) -> List[Dict]:
    """提取episode的临床笔记"""
    
    patient_notes = notes_df[
        (notes_df['stay_id'] == stay_id) &
        (notes_df['hour_offset'] >= 0) &
        (notes_df['hour_offset'] < window_hours)
    ].copy()
    
    if len(patient_notes) == 0:
        return []
    
    notes_list = []
    for _, row in patient_notes.iterrows():
        note_text = str(row.get('text', ''))[:1000]  # 截断长文本
        
        notes_list.append({
            "note_id": str(row.get('note_id', '')),
            "hour_offset": float(row['hour_offset']),
            "category": row.get('category', 'Radiology'),
            "text": note_text,
            "text_length": len(note_text)
        })
    
    return notes_list

# ==========================================
# 5. 提取Pattern和Annotations
# ==========================================

def extract_patterns_for_episode(
    stay_id: int,
    patterns_df: pd.DataFrame,
    alignments_df: pd.DataFrame
) -> Dict[str, Any]:
    """提取episode的pattern检测和对齐结果"""
    
    patient_patterns = patterns_df[patterns_df['stay_id'] == stay_id]
    patient_alignments = alignments_df[alignments_df['stay_id'] == stay_id] if len(alignments_df) > 0 else pd.DataFrame()
    
    # 按pattern分组
    pattern_summary = {}
    
    for pattern_name in patient_patterns['pattern_name'].unique():
        pattern_data = patient_patterns[patient_patterns['pattern_name'] == pattern_name]
        
        # 检测事件
        detections = []
        for _, row in pattern_data.iterrows():
            detections.append({
                "hour": int(row['hour']),
                "value": float(row['value']),
                "severity": row['severity'],
                "disease": row['disease']
            })
        
        # 对齐的文本
        text_alignments = []
        if len(patient_alignments) > 0:
            pattern_aligns = patient_alignments[patient_alignments['pattern_name'] == pattern_name]
            for _, row in pattern_aligns.iterrows():
                relevant_text = row.get('note_text_relevant', '')
                if relevant_text and len(str(relevant_text)) > 5:
                    text_alignments.append({
                        "pattern_hour": int(row['pattern_hour']),
                        "note_hour": float(row['note_hour']),
                        "time_delta": float(row['time_delta_hours']),
                        "relevant_text": str(relevant_text)[:500],
                        # 预留标注字段
                        "annotation": None  # SUPPORTIVE/CONTRADICTORY/AMBIGUOUS/UNRELATED
                    })
        
        pattern_summary[pattern_name] = {
            "n_detections": len(detections),
            "detections": detections[:20],  # 限制数量
            "n_text_alignments": len(text_alignments),
            "text_alignments": text_alignments[:10],  # 限制数量
            "first_detection_hour": min(d['hour'] for d in detections) if detections else None,
            "severity_max": max(d['severity'] for d in detections) if detections else None
        }
    
    return pattern_summary

def get_referenced_templates(
    patterns: Dict[str, Any],
    templates: Dict
) -> Dict[str, Any]:
    """获取被引用的pattern模板元数据"""
    
    referenced = {}
    pattern_names = set(patterns.keys())
    
    for disease_key, disease_data in templates.items():
        for template in disease_data.get('patterns', []):
            if template['name'] in pattern_names:
                referenced[template['name']] = {
                    "disease": disease_data['disease'],
                    "clinical_standard": disease_data['clinical_standard'],
                    "type": template['type'],
                    "feature": template['feature'],
                    "threshold": template.get('threshold'),
                    "direction": template.get('direction'),
                    "description": template['description'],
                    "severity": template['severity'],
                    "unit": template.get('unit', '')
                }
    
    return referenced

# ==========================================
# 6. 构建单个Episode
# ==========================================

def build_episode_record(
    stay_id: int,
    cohort_row: pd.Series,
    timeseries_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    patterns_df: pd.DataFrame,
    alignments_df: pd.DataFrame,
    templates: Dict,
    condition_graph: Dict,
    window_hours: int = 24
) -> Dict[str, Any]:
    """构建单个episode的完整记录"""
    
    # 1. 基本信息
    episode = {
        "episode_id": f"mimic4_{stay_id}",
        "stay_id": int(stay_id),
        "patient_id": int(cohort_row.get('subject_id', 0)),
        "hadm_id": int(cohort_row.get('hadm_id', 0)),
    }
    
    # 2. Conditions
    conditions = []
    if cohort_row.get('has_sepsis_final', 0) == 1:
        conditions.append("sepsis")
    if cohort_row.get('has_aki_final', 0) == 1:
        conditions.append("aki")
    if cohort_row.get('has_ards', 0) == 1:
        conditions.append("ards")
    if cohort_row.get('has_shock', 0) == 1:
        conditions.append("shock")
    
    episode["conditions"] = conditions
    episode["n_conditions"] = len(conditions)
    
    # 3. Condition Graph
    episode["condition_graph"] = get_patient_condition_graph(conditions, condition_graph)
    
    # 4. 时序数据
    episode["timeseries"] = extract_timeseries_for_episode(stay_id, timeseries_df, window_hours)
    
    # 5. 笔记片段
    episode["notes_spans"] = extract_notes_for_episode(stay_id, notes_df, window_hours)
    episode["n_notes"] = len(episode["notes_spans"])
    
    # 6. Pattern检测和对齐
    patterns = extract_patterns_for_episode(stay_id, patterns_df, alignments_df)
    episode["pattern_detections"] = patterns
    episode["n_patterns_detected"] = len(patterns)
    
    # 7. 引用的模板元数据
    episode["physiology_templates"] = get_referenced_templates(patterns, templates)
    
    # 8. Pattern Annotations摘要
    episode["pattern_annotations"] = {
        "total_alignments": sum(p.get('n_text_alignments', 0) for p in patterns.values()),
        "annotated": 0,  # 待标注
        "supportive": 0,
        "contradictory": 0,
        "ambiguous": 0,
        "unrelated": 0
    }
    
    # 9. Labels
    episode["labels"] = {
        "mortality": int(cohort_row.get('label_mortality', 0)),
        "prolonged_los_7d": int(cohort_row.get('prolonged_los_7d', 0)),
        "readmission_30d": int(cohort_row.get('readmission_30d', 0))
        if pd.notna(cohort_row.get('readmission_30d')) else None,
        "los_hours": float(cohort_row.get('los_hours', 0)) if pd.notna(cohort_row.get('los_hours')) else None,
        "aki_stage_max": int(cohort_row.get('aki_stage_max', 0)) if pd.notna(cohort_row.get('aki_stage_max')) else None,
    }
    
    # 10. 元数据
    episode["metadata"] = {
        "window_hours": window_hours,
        "data_source": "MIMIC-IV v3.1",
        "created_at": datetime.now().isoformat()
    }
    
    return episode

# ==========================================
# 7. 批量构建数据集
# ==========================================

def build_dataset(
    data: Dict,
    output_dir: str,
    max_episodes: Optional[int] = None,
    window_hours: int = 24
) -> Dict[str, Any]:
    """构建完整的TIMELY-Bench数据集"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔨 Building TIMELY-Bench Dataset...")
    print("=" * 60)
    
    # 构建condition graph
    condition_graph = build_condition_graph()
    
    # 获取所有stay_ids
    stay_ids = data['cohort']['stay_id'].unique()
    if max_episodes:
        stay_ids = stay_ids[:max_episodes]
    
    print(f"   Processing {len(stay_ids)} episodes...")
    
    # 构建每个episode
    episodes = []
    stats = defaultdict(int)
    
    for i, stay_id in enumerate(stay_ids):
        if (i + 1) % 5000 == 0:
            print(f"   Processed {i+1}/{len(stay_ids)} episodes...")
        
        # 获取cohort行
        cohort_row = data['cohort'][data['cohort']['stay_id'] == stay_id].iloc[0]
        
        # 构建episode
        episode = build_episode_record(
            stay_id=stay_id,
            cohort_row=cohort_row,
            timeseries_df=data['timeseries'],
            notes_df=data['notes'],
            patterns_df=data['patterns'],
            alignments_df=data['alignments'],
            templates=data['templates'],
            condition_graph=condition_graph,
            window_hours=window_hours
        )
        
        episodes.append(episode)
        
        # 统计
        stats['total_episodes'] += 1
        stats['with_notes'] += 1 if episode['n_notes'] > 0 else 0
        stats['with_patterns'] += 1 if episode['n_patterns_detected'] > 0 else 0
        stats['with_alignments'] += 1 if episode['pattern_annotations']['total_alignments'] > 0 else 0
        for cond in episode['conditions']:
            stats[f'condition_{cond}'] += 1
        stats['mortality_positive'] += episode['labels']['mortality']
    
    print(f"\n   Built {len(episodes)} episodes")
    
    # ==========================================
    # 保存数据集
    # ==========================================
    
    # 1. 完整数据集 (JSONL格式，每行一个episode)
    jsonl_path = os.path.join(output_dir, 'timely_bench_episodes.jsonl')
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for episode in episodes:
            f.write(json.dumps(episode, ensure_ascii=False) + '\n')
    print(f"\nSaved: {jsonl_path}")
    
    # 2. 采样数据集 (用于检查和展示)
    sample_path = os.path.join(output_dir, 'sample_episodes.json')
    sample_episodes = episodes[:100]  # 前100个
    with open(sample_path, 'w', encoding='utf-8') as f:
        json.dump(sample_episodes, f, indent=2, ensure_ascii=False)
    print(f"Saved: {sample_path}")
    
    # 3. 数据集统计
    stats_dict = dict(stats)
    stats_dict['total_patterns'] = sum(e['n_patterns_detected'] for e in episodes)
    stats_dict['total_notes'] = sum(e['n_notes'] for e in episodes)
    stats_dict['total_alignments'] = sum(e['pattern_annotations']['total_alignments'] for e in episodes)
    
    stats_path = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    print(f"Saved: {stats_path}")
    
    # 4. Condition Graph
    graph_path = os.path.join(output_dir, 'condition_graph.json')
    with open(graph_path, 'w') as f:
        json.dump(condition_graph, f, indent=2)
    print(f"Saved: {graph_path}")
    
    # ==========================================
    # 打印统计
    # ==========================================
    
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    print(f"\n[Episodes]")
    print(f"   Total: {stats['total_episodes']}")
    print(f"   With notes: {stats['with_notes']} ({stats['with_notes']/stats['total_episodes']*100:.1f}%)")
    print(f"   With patterns: {stats['with_patterns']} ({stats['with_patterns']/stats['total_episodes']*100:.1f}%)")
    print(f"   With alignments: {stats['with_alignments']} ({stats['with_alignments']/stats['total_episodes']*100:.1f}%)")
    
    print(f"\n[Conditions]")
    for key, val in stats_dict.items():
        if key.startswith('condition_'):
            cond_name = key.replace('condition_', '')
            print(f"   {cond_name}: {val} ({val/stats['total_episodes']*100:.1f}%)")
    
    print(f"\n[Labels]")
    print(f"   Mortality positive: {stats['mortality_positive']} ({stats['mortality_positive']/stats['total_episodes']*100:.1f}%)")
    
    print(f"\n[Totals]")
    print(f"   Patterns: {stats_dict['total_patterns']}")
    print(f"   Notes: {stats_dict['total_notes']}")
    print(f"   Alignments: {stats_dict['total_alignments']}")
    
    return stats_dict

# ==========================================
# 8. 生成数据集文档
# ==========================================

def generate_dataset_readme(output_dir: str, stats: Dict):
    """生成数据集README"""
    
    readme = f"""# TIMELY-Bench Dataset

## Overview

TIMELY-Bench is a benchmark dataset for time-aligned fusion of clinical time-series and notes in MIMIC-IV.

## Dataset Structure

Each episode is a JSON object with the following structure:

```json
{{
  "episode_id": "mimic4_12345",
  "patient_id": 10001,
  "stay_id": 12345,
  "hadm_id": 20001,
  
  "conditions": ["sepsis", "aki"],
  "n_conditions": 2,
  
  "condition_graph": {{
    "nodes": [...],
    "edges": [...]
  }},
  
  "timeseries": {{
    "window_hours": 24,
    "hours_available": [0, 1, 2, ...],
    "features": {{
      "heart_rate": [{{"hour": 0, "value": 85}}, ...],
      "creatinine": [{{"hour": 2, "value": 1.2}}, ...],
      ...
    }}
  }},
  
  "notes_spans": [
    {{
      "note_id": "note_001",
      "hour_offset": 3.5,
      "category": "Radiology",
      "text": "Chest X-ray shows..."
    }},
    ...
  ],
  
  "pattern_detections": {{
    "tachycardia": {{
      "n_detections": 5,
      "detections": [{{"hour": 2, "value": 105, "severity": "mild"}}, ...],
      "text_alignments": [{{"relevant_text": "Patient tachycardic...", "annotation": null}}, ...]
    }},
    ...
  }},
  
  "physiology_templates": {{
    "tachycardia": {{
      "disease": "Sepsis",
      "clinical_standard": "Sepsis-3",
      "threshold": 90,
      "direction": "above",
      ...
    }},
    ...
  }},
  
  "pattern_annotations": {{
    "total_alignments": 15,
    "annotated": 0,
    "supportive": 0,
    "contradictory": 0,
    "ambiguous": 0,
    "unrelated": 0
  }},
  
  "labels": {{
    "mortality": 0,
    "prolonged_los_7d": 1,
    "readmission_30d": 0,
    "los_hours": 168.5,
    "aki_stage_max": 2
  }},
  
  "metadata": {{
    "window_hours": 24,
    "data_source": "MIMIC-IV v3.1",
    "created_at": "2025-01-15T10:30:00"
  }}
}}
```

## Statistics

| Metric | Value |
|--------|-------|
| Total Episodes | {stats.get('total_episodes', 'N/A')} |
| With Notes | {stats.get('with_notes', 'N/A')} |
| With Patterns | {stats.get('with_patterns', 'N/A')} |
| With Alignments | {stats.get('with_alignments', 'N/A')} |
| Total Patterns | {stats.get('total_patterns', 'N/A')} |
| Total Alignments | {stats.get('total_alignments', 'N/A')} |

## Files

- `timely_bench_episodes.jsonl` - Complete dataset (one episode per line)
- `sample_episodes.json` - Sample of 100 episodes for inspection
- `dataset_stats.json` - Dataset statistics
- `condition_graph.json` - Disease relationship graph

## Usage

```python
import json

# Load episodes
episodes = []
with open('timely_bench_episodes.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.loads(line))

# Filter by condition
sepsis_episodes = [e for e in episodes if 'sepsis' in e['conditions']]

# Get patterns for an episode
for pattern_name, pattern_data in episodes[0]['pattern_detections'].items():
    print(f"{{pattern_name}}: {{pattern_data['n_detections']}} detections")
```

## License

This dataset is derived from MIMIC-IV and requires PhysioNet credentialing.

## Citation

```bibtex
@misc{{timely-bench,
  title={{TIMELY-Bench: A Benchmark for Time-Aligned Fusion of Clinical Time-Series and Notes}},
  author={{Wang, Haoyu}},
  year={{2025}},
  institution={{King's College London}}
}}
```
"""
    
    readme_path = os.path.join(output_dir, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme)
    print(f"Saved: {readme_path}")

# ==========================================
# Main
# ==========================================

def main():
    print("Building TIMELY-Bench Dataset")
    print("=" * 60)
    
    # 加载数据
    data = load_all_data()
    
    # 构建数据集
    stats = build_dataset(
        data=data,
        output_dir=OUTPUT_DIR,
        max_episodes=None,  # 处理所有episode
        window_hours=24
    )
    
    # 生成README
    generate_dataset_readme(OUTPUT_DIR, stats)
    
    print("\n" + "=" * 60)
    print("TIMELY-Bench Dataset Complete!")
    print("=" * 60)
    print(f"\nOutput directory: {OUTPUT_DIR}/")
    print("   - timely_bench_episodes.jsonl (complete dataset)")
    print("   - sample_episodes.json (100 samples)")
    print("   - dataset_stats.json")
    print("   - condition_graph.json")
    print("   - README.md")

if __name__ == "__main__":
    main()
