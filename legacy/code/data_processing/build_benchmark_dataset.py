import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
import uuid

from config import TIMESERIES_FILE, NOTE_TIME_FILE, COHORT_FILE, ROOT_DIR

# 路径配置
OUTPUT_DIR = ROOT_DIR / 'TIMELY_Bench_Dataset'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 扩展后的医学知识库 (基于 SOFA 和 Sepsis-3 指南)
TEMPLATES = {
    # 1. 循环系统 (Cardiovascular)
    "shock_hypotension": {
        "condition": "Shock (Hypotension)",
        "feature": "mbp",
        "threshold": 65.0,
        "operator": "<",
        "description": "MAP < 65 mmHg indicating circulatory failure"
    },
    "shock_lactate": {
        "condition": "Septic Shock (Metabolic)",
        "feature": "lactate",
        "threshold": 2.0,
        "operator": ">",
        "description": "Lactate > 2.0 indicating cellular hypoxia"
    },

    # 2. 呼吸系统 (Respiratory)
    "respiratory_failure_hypoxia": {
        "condition": "Respiratory Failure",
        "feature": "spo2",
        "threshold": 90.0,
        "operator": "<",
        "description": "SpO2 < 90% indicating hypoxemia"
    },

    # 3. 肾脏系统 (Renal) - KDIGO标准
    "aki_creatinine": {
        "condition": "Acute Kidney Injury (AKI)",
        "feature": "creatinine",
        "threshold": 1.2,
        "operator": ">",
        "description": "Elevated Creatinine indicating renal dysfunction"
    },
    "aki_oliguria": {
        "condition": "Acute Kidney Injury (AKI)",
        "feature": "urineoutput",
        "threshold": 30.0,
        "operator": "<",
        "description": "Low urine output (Oliguria)"
    },

    # 4. 肝脏系统 (Liver)
    "liver_injury": {
        "condition": "Liver Injury",
        "feature": "bilirubin_total",
        "threshold": 1.2,
        "operator": ">",
        "description": "Elevated Bilirubin indicating liver dysfunction"
    },

    # 5. 凝血系统 (Coagulation)
    "coagulopathy": {
        "condition": "Coagulopathy",
        "feature": "platelet",
        "threshold": 150.0,
        "operator": "<",
        "description": "Thrombocytopenia (Low Platelets)"
    },

    # 6. 神经系统 (CNS)
    "altered_mental_status": {
        "condition": "CNS Dysfunction",
        "feature": "gcs_min",
        "threshold": 13.0,
        "operator": "<",
        "description": "Low GCS indicating altered mental status"
    }
}

def check_pattern(row, template_key):
    """检查单行数据是否符合某种病理模式"""
    tmpl = TEMPLATES[template_key]
    feat_name = tmpl['feature']
    
    # 检查特征是否存在
    if feat_name not in row:
        return False
    
    val = row.get(feat_name, np.nan)
    if pd.isna(val):
        return False
    
    if tmpl['operator'] == '<':
        return val < tmpl['threshold']
    if tmpl['operator'] == '>':
        return val > tmpl['threshold']
    
    return False

def build_dataset():
    print("Starting TIMELY-Bench Dataset Construction...")
    
    # 1. 加载数据
    print("   - Loading raw data...")
    df_ts = pd.read_csv(TIMESERIES_FILE)
    df_ts['stay_id'] = pd.to_numeric(df_ts['stay_id'], errors='coerce').fillna(-1).astype(int)
    
    df_notes = pd.read_csv(NOTE_TIME_FILE)
    df_notes['stay_id'] = pd.to_numeric(df_notes['stay_id'], errors='coerce').fillna(-1).astype(int)
    
    df_cohort = pd.read_csv(COHORT_FILE)
    df_cohort['stay_id'] = pd.to_numeric(df_cohort['stay_id'], errors='coerce').fillna(-1).astype(int)

    # 2. 预先分组 (性能优化)
    print("   - Grouping data by stay_id...")
    ts_groups = df_ts.groupby('stay_id')
    note_groups = df_notes.groupby('stay_id')
    
    # 3. 打开文件准备流式写入
    output_path = os.path.join(OUTPUT_DIR, 'timely_bench_samples.jsonl')
    unique_ids = df_cohort['stay_id'].unique()
    
    print(f"   - Processing {len(unique_ids)} patients...")

    with open(output_path, 'w') as f:
        for stay_id in tqdm(unique_ids):
            if stay_id not in ts_groups.groups:
                continue
                
            # 获取分组数据
            patient_ts = ts_groups.get_group(stay_id).sort_values('hour')
            patient_notes = note_groups.get_group(stay_id).sort_values('hour_offset') if stay_id in note_groups.groups else pd.DataFrame()
            patient_info = df_cohort[df_cohort['stay_id'] == stay_id].iloc[0]

            # A. 捕捉生理异常模式 (Pattern Matching)
            detected_patterns = []
            conditions_found = set()
            
            for _, row in patient_ts.iterrows():
                hour = int(row['hour'])
                current_hour_conditions = []
                
                for pattern_name, tmpl in TEMPLATES.items():
                    feat_name = tmpl['feature']
                    if feat_name not in row:
                        continue
                    
                    if check_pattern(row, pattern_name):
                        val = float(row[feat_name])
                        detected_patterns.append({
                            "pattern_id": pattern_name,
                            "condition": tmpl['condition'],
                            "hour": hour,
                            "evidence": {feat_name: val},
                            "description": tmpl['description']
                        })
                        current_hour_conditions.append(tmpl['condition'])
                        conditions_found.add(tmpl['condition'])
                
                # 识别交叉 (Multi-organ dysfunction)
                if "Acute Kidney Injury (AKI)" in current_hour_conditions and \
                   "Respiratory Failure" in current_hour_conditions:
                    pass  # 高危交叉时刻

            # B. 关联文本片段
            aligned_notes = []
            for _, note in patient_notes.iterrows():
                note_hour = int(note['hour_offset'])
                relevant_patterns = [p['pattern_id'] for p in detected_patterns if abs(p['hour'] - note_hour) <= 2]
                
                aligned_notes.append({
                    "note_id": str(note['note_id']),
                    "hour_offset": note_hour,
                    "text_snippet": str(note['radiology_text'])[:500],
                    "potentially_related_patterns": list(set(relevant_patterns))
                })

            # C. 构建 JSON 对象
            episode_obj = {
                "episode_id": str(uuid.uuid4()),
                "stay_id": int(stay_id),
                "subject_id": int(patient_info['subject_id']),
                "conditions": list(conditions_found),
                "is_multimorbidity": len(conditions_found) > 1,
                "condition_count": len(conditions_found),
                "timeseries_snapshot": {
                    "hours": [int(x) for x in patient_ts['hour'].tolist()],
                    "mbp": [float(x) for x in patient_ts['mbp'].fillna(-1).tolist()],
                    "lactate": [float(x) for x in patient_ts['lactate'].fillna(-1).tolist()],
                    "creatinine": [float(x) for x in patient_ts['creatinine'].fillna(-1).tolist()],
                    "spo2": [float(x) for x in patient_ts['spo2'].fillna(-1).tolist()]
                },
                "notes_spans": aligned_notes,
                "pattern_annotations": detected_patterns,
                "labels": {
                    "mortality": int(patient_info['label_mortality']),
                    "los": float(patient_info.get('los', -1))
                }
            }
            
            f.write(json.dumps(episode_obj) + '\n')

    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    build_dataset()