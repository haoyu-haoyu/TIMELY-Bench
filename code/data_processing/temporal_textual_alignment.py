"""
Temporal-Textual Alignment
将检测到的生理模式与临床笔记进行时间对齐

支持笔记类型: Discharge, Nursing, Lab Comments, Radiology

功能:
1. 匹配模式事件与时间窗口内的临床笔记
2. 提取相关文本片段
3. 可选LLM标注 (SUPPORTIVE/CONTRADICTORY/AMBIGUOUS/UNRELATED)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import json
import re
import os

from config import (
    PATTERN_DETECTION_DIR, NOTE_TIME_FILE, COHORT_FILE, TEMPORAL_ALIGNMENT_DIR, RAW_DATA_DIR
)

# 导入多类型笔记加载器
from load_multi_notes import (
    load_all_notes,
    get_note_types_for_pattern,
    get_keywords_for_pattern_and_note_type,
    PATTERN_NOTE_MAPPING,
    print_notes_summary
)

# ==========================================
# 配置
# ==========================================
OUTPUT_DIR = TEMPORAL_ALIGNMENT_DIR

# 时间对齐窗口（严格因果：只看过去）
ALIGNMENT_WINDOW_BEFORE = 6  # 模式发生前6小时的笔记
ALIGNMENT_WINDOW_AFTER = 0   # 模式发生后0小时的笔记（禁用lookahead）

# 是否使用多类型笔记
USE_MULTI_NOTES = True  # 设为True使用新的多类型笔记

# 预测任务默认不使用出院小结
INCLUDE_DISCHARGE_NOTES = False

# 是否允许同一条 note 匹配多个 pattern 事件
ALLOW_NOTE_MULTI_MATCH = False

# ==========================================
# 1. 数据加载
# ==========================================

def load_notes(
    note_path: str = None,
    use_multi_notes: bool = USE_MULTI_NOTES,
    include_discharge: bool = INCLUDE_DISCHARGE_NOTES,
) -> pd.DataFrame:
    """
    加载临床笔记数据

    Args:
        note_path: 单一笔记文件路径 (仅当use_multi_notes=False时使用)
        use_multi_notes: 是否使用多类型笔记

    Returns:
        笔记DataFrame，包含统一的列
    """
    if use_multi_notes:
        print("Using Multi-Type Notes (Discharge, Nursing, Lab, Radiology)")
        notes = load_all_notes(include_discharge=include_discharge)
        return notes

    # 原有单文件加载逻辑 (向后兼容)
    print("Using single note file (legacy mode)")
    notes = pd.read_csv(note_path)
    notes['stay_id'] = pd.to_numeric(notes['stay_id'], errors='coerce').fillna(-1).astype(int)

    # 列名映射（适应不同的数据格式）
    column_mapping = {
        'radiology_text': 'text',
        'note_text': 'text',
        'content': 'text',
    }

    for old_col, new_col in column_mapping.items():
        if old_col in notes.columns and new_col not in notes.columns:
            notes[new_col] = notes[old_col]
            print(f"   Mapped column: {old_col} -> {new_col}")

    # 如果没有category列，添加默认值
    if 'category' not in notes.columns:
        notes['category'] = 'Radiology'
        print("   Added default category: 'Radiology'")

    # 如果没有note_type列，添加默认值
    if 'note_type' not in notes.columns:
        notes['note_type'] = 'radiology'

    # 如果没有note_id，生成一个
    if 'note_id' not in notes.columns:
        notes['note_id'] = notes.index.astype(str)
        print("   Generated note_id from index")

    # 确保有text列
    if 'text' not in notes.columns:
        raise ValueError("Missing 'text' column. Available columns: " + str(notes.columns.tolist()))

    # 获取小时偏移
    if 'hour_offset' not in notes.columns:
        if 'charttime' in notes.columns and 'intime' in notes.columns:
            notes['charttime'] = pd.to_datetime(notes['charttime'])
            notes['intime'] = pd.to_datetime(notes['intime'])
            notes['hour_offset'] = (notes['charttime'] - notes['intime']).dt.total_seconds() / 3600
        else:
            # 如果没有时间信息，假设在第0小时
            notes['hour_offset'] = 0

    print(f"Loaded {len(notes)} notes for {notes['stay_id'].nunique()} patients")
    print(f"   Columns: {notes.columns.tolist()}")
    return notes

def load_pattern_detections(detection_path: str) -> pd.DataFrame:
    """加载模式检测结果"""
    detections = pd.read_csv(detection_path)
    print(f"Loaded {len(detections)} pattern detections")
    return detections

# ==========================================
# 2. 文本预处理
# ==========================================

def clean_note_text(text: str) -> str:
    """清理临床笔记文本"""
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    
    # 移除de-identification标记
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', '[REDACTED]', text)
    
    return text.strip()

def extract_relevant_sentences(text: str, keywords: List[str], max_sentences: int = 5) -> str:
    """从文本中提取包含关键词的句子"""
    if not text:
        return ""
    
    # 分句
    sentences = re.split(r'[.!?]\s+', text)
    
    # 找到相关句子
    relevant = []
    for sent in sentences:
        sent_lower = sent.lower()
        for kw in keywords:
            if kw.lower() in sent_lower:
                relevant.append(sent.strip())
                break
    
    # 限制数量
    if len(relevant) > max_sentences:
        relevant = relevant[:max_sentences]
    
    return ' '.join(relevant)

# ==========================================
# 3. 模式到关键词的映射
# ==========================================

PATTERN_KEYWORDS = {
    # Sepsis patterns
    'fever': ['fever', 'febrile', 'temperature', 'temp', 'hyperthermia', '°C', '°F'],
    'hypothermia': ['hypothermia', 'hypothermic', 'cold', 'temperature'],
    'tachycardia': ['tachycardia', 'tachycardic', 'heart rate', 'HR', 'pulse'],
    'tachypnea': ['tachypnea', 'tachypneic', 'respiratory rate', 'RR', 'breathing'],
    'hypotension': ['hypotension', 'hypotensive', 'blood pressure', 'BP', 'SBP', 'shock'],
    'map_low': ['MAP', 'mean arterial', 'perfusion'],
    'hypoxemia': ['hypoxia', 'hypoxemic', 'oxygen', 'O2', 'saturation', 'SpO2', 'desaturation'],
    'lactate_elevated': ['lactate', 'lactic', 'acidosis'],
    'thrombocytopenia': ['thrombocytopenia', 'platelets', 'PLT', 'low platelets'],
    'hyperbilirubinemia': ['bilirubin', 'jaundice', 'icterus', 'liver'],
    'leukocytosis': ['leukocytosis', 'WBC', 'white blood cell', 'elevated WBC'],
    'leukopenia': ['leukopenia', 'WBC', 'white blood cell', 'low WBC'],
    
    # AKI patterns
    'creatinine_elevated': ['creatinine', 'Cr', 'renal', 'kidney'],
    'creatinine_severe': ['creatinine', 'renal failure', 'kidney injury', 'AKI'],
    'creatinine_rise_acute': ['creatinine', 'rising', 'acute kidney', 'AKI'],
    'bun_elevated': ['BUN', 'urea', 'azotemia'],
    'bun_severe': ['BUN', 'uremia', 'renal failure'],
    'oliguria': ['oliguria', 'urine output', 'UOP', 'anuria'],
    'hyperkalemia': ['hyperkalemia', 'potassium', 'K+', 'elevated K'],
    'metabolic_acidosis': ['acidosis', 'bicarbonate', 'HCO3', 'pH'],
    
    # ARDS patterns
    'hypoxemia_mild': ['hypoxia', 'P/F ratio', 'PaO2', 'FiO2', 'ARDS'],
    'hypoxemia_moderate': ['hypoxia', 'respiratory failure', 'ARDS', 'intubation'],
    'hypoxemia_severe': ['severe hypoxia', 'refractory', 'ARDS', 'ECMO'],
    'spo2_low': ['desaturation', 'SpO2', 'oxygen'],
    'respiratory_distress': ['respiratory distress', 'dyspnea', 'tachypnea', 'work of breathing'],
    
    # Critical patterns
    'bradycardia': ['bradycardia', 'bradycardic', 'slow heart rate'],
    'severe_tachycardia': ['tachycardia', 'SVT', 'rapid heart rate'],
    'hypertensive_crisis': ['hypertensive', 'hypertension', 'elevated BP', 'blood pressure'],
    'anemia': ['anemia', 'hemoglobin', 'Hgb', 'Hb', 'transfusion'],
    'severe_anemia': ['severe anemia', 'transfusion', 'hemorrhage', 'bleeding'],
    'altered_consciousness': ['altered mental status', 'AMS', 'confusion', 'lethargy', 'GCS'],
    'coma': ['coma', 'unresponsive', 'GCS', 'unconscious'],
}

def get_keywords_for_pattern(pattern_name: str, note_type: str = None) -> List[str]:
    """
    获取模式对应的关键词

    改进：支持根据笔记类型返回优化的关键词
    """
    if note_type and USE_MULTI_NOTES:
        return get_keywords_for_pattern_and_note_type(pattern_name, note_type)
    return PATTERN_KEYWORDS.get(pattern_name, [pattern_name])

# ==========================================
# 4. Temporal-Textual Alignment
# ==========================================

@dataclass
class AlignedEvent:
    """对齐后的模式-文本事件"""
    stay_id: int
    pattern_hour: int
    pattern_name: str
    pattern_value: float
    pattern_severity: str
    pattern_disease: str
    note_id: str
    note_hour: float
    note_category: str
    note_type: str  # 新增：笔记类型 (discharge, nursing, lab_comment, radiology)
    note_text_full: str
    note_text_relevant: str
    time_delta_hours: float  # 笔记时间 - 模式时间
    alignment_quality: str = 'unknown'  # 新增：对齐质量预估 (high, medium, low)

    def to_dict(self):
        return {
            'stay_id': self.stay_id,
            'pattern_hour': self.pattern_hour,
            'pattern_name': self.pattern_name,
            'pattern_value': self.pattern_value,
            'pattern_severity': self.pattern_severity,
            'pattern_disease': self.pattern_disease,
            'note_id': self.note_id,
            'note_hour': self.note_hour,
            'note_category': self.note_category,
            'note_type': self.note_type,  # 新增
            'note_text_full': self.note_text_full[:500] if self.note_text_full else '',  # 截断
            'note_text_relevant': self.note_text_relevant,
            'time_delta_hours': self.time_delta_hours,
            'alignment_quality': self.alignment_quality,  # 新增
        }

def align_patterns_with_notes(
    patterns_df: pd.DataFrame,
    notes_df: pd.DataFrame,
    window_before: int = ALIGNMENT_WINDOW_BEFORE,
    window_after: int = ALIGNMENT_WINDOW_AFTER,
    max_patterns_per_patient: int = 50,  # 每个患者最多处理的模式数
    sample_severe_first: bool = True,     # 优先处理严重模式
    use_pattern_note_mapping: bool = USE_MULTI_NOTES  # 使用Pattern-Note映射
) -> List[AlignedEvent]:
    """
    将模式事件与临床笔记对齐

    改进：
    - 根据Pattern类型选择最合适的笔记类型
    - 添加对齐质量预估
    - 优先使用匹配度高的笔记类型

    对于每个模式事件，找到时间窗口内的所有笔记

    优化：
    - 每个患者最多处理 max_patterns_per_patient 个模式
    - 优先处理 severe 模式
    """

    aligned_events = []

    # 按患者分组处理
    patient_ids = patterns_df['stay_id'].unique()
    patients_with_notes = notes_df['stay_id'].unique()

    # 只处理有笔记的患者
    patient_ids = [pid for pid in patient_ids if pid in patients_with_notes]
    print(f"   Processing {len(patient_ids)} patients with notes...")

    # 统计各笔记类型的使用情况
    note_type_stats = {nt: 0 for nt in notes_df['note_type'].unique()} if 'note_type' in notes_df.columns else {}

    for i, stay_id in enumerate(patient_ids):
        if (i + 1) % 2000 == 0:
            print(f"   Processed {i+1}/{len(patient_ids)} patients... ({len(aligned_events)} alignments)")

        patient_patterns = patterns_df[patterns_df['stay_id'] == stay_id].copy()
        patient_notes = notes_df[notes_df['stay_id'] == stay_id]
        used_note_ids = set()

        if len(patient_notes) == 0:
            continue

        # 采样模式：优先severe，然后moderate，最后mild
        if len(patient_patterns) > max_patterns_per_patient:
            if sample_severe_first:
                severity_order = {'severe': 0, 'moderate': 1, 'mild': 2}
                patient_patterns['_severity_order'] = patient_patterns['severity'].map(severity_order)
                patient_patterns = patient_patterns.sort_values('_severity_order')
            patient_patterns = patient_patterns.head(max_patterns_per_patient)

        for _, pattern_row in patient_patterns.iterrows():
            pattern_hour = pattern_row['hour']
            pattern_name = pattern_row['pattern_name']

            # 获取该Pattern应该使用的笔记类型
            if use_pattern_note_mapping and 'note_type' in patient_notes.columns:
                preferred_note_types = get_note_types_for_pattern(pattern_name)
                if preferred_note_types and not INCLUDE_DISCHARGE_NOTES:
                    preferred_note_types = [nt for nt in preferred_note_types if nt != 'discharge']
                    if not preferred_note_types:
                        preferred_note_types = None
            else:
                preferred_note_types = None

            # 找到时间窗口内的笔记
            # 对于discharge notes，使用更宽松的时间窗口（出院小结通常在出院时写）
            if 'note_type' in patient_notes.columns:
                # 分别处理不同类型的笔记
                matching_notes_list = []

                for note_type in patient_notes['note_type'].unique():
                    type_notes = patient_notes[patient_notes['note_type'] == note_type]

                    if note_type == 'discharge' and not INCLUDE_DISCHARGE_NOTES:
                        continue
                    # 所有笔记统一按时间窗口过滤（严格因果）
                    note_mask = (
                        (type_notes['hour_offset'] >= pattern_hour - window_before) &
                        (type_notes['hour_offset'] <= pattern_hour + window_after)
                    )
                    matching = type_notes[note_mask]

                    if len(matching) > 0:
                        matching_notes_list.append(matching)

                if matching_notes_list:
                    matching_notes = pd.concat(matching_notes_list, ignore_index=True)
                else:
                    continue
            else:
                # 原有逻辑
                note_mask = (
                    (patient_notes['hour_offset'] >= pattern_hour - window_before) &
                    (patient_notes['hour_offset'] <= pattern_hour + window_after)
                )
                matching_notes = patient_notes[note_mask]

            if len(matching_notes) == 0:
                continue

            # 按笔记类型优先级排序
            if preferred_note_types and 'note_type' in matching_notes.columns:
                # 创建优先级
                type_priority = {nt: idx for idx, nt in enumerate(preferred_note_types)}
                matching_notes = matching_notes.copy()
                matching_notes['_priority'] = matching_notes['note_type'].map(
                    lambda x: type_priority.get(x, 100)
                )
                matching_notes = matching_notes.sort_values('_priority')

            for _, note_row in matching_notes.iterrows():
                note_type = note_row.get('note_type', 'unknown')
                note_id = str(note_row.get('note_id', ''))
                if not ALLOW_NOTE_MULTI_MATCH and note_id in used_note_ids:
                    continue
                used_note_ids.add(note_id)
                note_text = clean_note_text(note_row.get('text', ''))

                # 获取针对该笔记类型的关键词
                keywords = get_keywords_for_pattern(pattern_name, note_type)
                
                # === 修复：对于短文本直接使用整个内容 ===
                # nursing笔记平均17字符，lab_comment平均71字符
                # 只有长文本才进行关键词句子提取
                SHORT_TEXT_THRESHOLD = 200  # 短于200字符直接使用
                
                if len(note_text) < SHORT_TEXT_THRESHOLD:
                    # 短文本：直接使用整个内容（如nursing观察项）
                    relevant_text = note_text
                else:
                    # 长文本：按关键词提取相关句子
                    relevant_text = extract_relevant_sentences(note_text, keywords)
                    # 如果关键词提取返回空，但文本本身存在，使用截取的摘要
                    if not relevant_text and note_text:
                        relevant_text = note_text[:500]  # 取前500字符作为摘要

                # 计算对齐质量
                alignment_quality = 'low'
                if preferred_note_types and note_type in preferred_note_types:
                    if len(relevant_text) > 50:
                        alignment_quality = 'high'
                    elif len(relevant_text) > 0:
                        alignment_quality = 'medium'
                elif len(relevant_text) > 0:
                    alignment_quality = 'medium'

                # 统计
                if note_type in note_type_stats:
                    note_type_stats[note_type] += 1

                aligned_events.append(AlignedEvent(
                    stay_id=stay_id,
                    pattern_hour=pattern_hour,
                    pattern_name=pattern_name,
                    pattern_value=pattern_row['value'],
                    pattern_severity=pattern_row['severity'],
                    pattern_disease=pattern_row['disease'],
                    note_id=str(note_row.get('note_id', '')),
                    note_hour=note_row['hour_offset'],
                    note_category=note_row.get('category', 'Unknown'),
                    note_type=note_type,
                    note_text_full=note_text,
                    note_text_relevant=relevant_text,
                    time_delta_hours=note_row['hour_offset'] - pattern_hour,
                    alignment_quality=alignment_quality,
                ))

    # 打印笔记类型使用统计
    if note_type_stats:
        print(f"\n   [Note Type Usage Statistics]")
        for nt, count in sorted(note_type_stats.items(), key=lambda x: -x[1]):
            print(f"      {nt}: {count} alignments")

    return aligned_events

# ==========================================
# 5. 生成对齐数据集
# ==========================================

def create_alignment_dataset(
    patterns_path: str,
    notes_path: str = None,
    output_dir: str = OUTPUT_DIR,
    use_multi_notes: bool = USE_MULTI_NOTES
) -> pd.DataFrame:
    """
    创建完整的对齐数据集

    改进：
    - 支持多笔记类型
    - 流式写入CSV避免内存溢出
    """
    import csv
    
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    patterns_df = load_pattern_detections(patterns_path)
    notes_df = load_notes(
        notes_path,
        use_multi_notes=use_multi_notes,
        include_discharge=INCLUDE_DISCHARGE_NOTES,
    )

    # 打印笔记摘要
    if use_multi_notes and len(notes_df) > 0:
        print_notes_summary(notes_df)

    print(f"\nAligning patterns with notes...")
    print(f"   Window: [-{ALIGNMENT_WINDOW_BEFORE}h, +{ALIGNMENT_WINDOW_AFTER}h]")
    print(f"   Multi-Note Mode: {'ON' if use_multi_notes else 'OFF'}")
    
    # === 修复OOM：流式写入CSV ===
    output_path = os.path.join(output_dir, 'temporal_textual_alignment.csv')
    
    # 按患者分组处理
    patient_ids = patterns_df['stay_id'].unique()
    patients_with_notes = notes_df['stay_id'].unique()
    patient_ids = [pid for pid in patient_ids if pid in patients_with_notes]
    
    print(f"   Processing {len(patient_ids)} patients with notes...")
    print(f"   Streaming to: {output_path}")
    
    # CSV字段名
    fieldnames = [
        'stay_id', 'pattern_hour', 'pattern_name', 'pattern_value', 
        'pattern_severity', 'pattern_disease', 'note_id', 'note_hour',
        'note_category', 'note_type', 'note_text_full', 'note_text_relevant',
        'time_delta_hours', 'alignment_quality'
    ]
    
    total_alignments = 0
    note_type_stats = {nt: 0 for nt in notes_df['note_type'].unique()} if 'note_type' in notes_df.columns else {}
    
    # 流式写入CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        BATCH_SIZE = 2000  # 每2000患者输出进度
        
        for i, stay_id in enumerate(patient_ids):
            if (i + 1) % BATCH_SIZE == 0:
                print(f"   Processed {i+1}/{len(patient_ids)} patients... ({total_alignments} alignments)")
            
            patient_patterns = patterns_df[patterns_df['stay_id'] == stay_id].copy()
            patient_notes = notes_df[notes_df['stay_id'] == stay_id]
            used_note_ids = set()
            
            if len(patient_notes) == 0:
                continue
            
            # 采样模式：优先severe
            max_patterns_per_patient = 50
            if len(patient_patterns) > max_patterns_per_patient:
                severity_order = {'severe': 0, 'moderate': 1, 'mild': 2}
                patient_patterns['_severity_order'] = patient_patterns['severity'].map(severity_order)
                patient_patterns = patient_patterns.sort_values('_severity_order').head(max_patterns_per_patient)
            
            for _, pattern_row in patient_patterns.iterrows():
                pattern_hour = pattern_row['hour']
                pattern_name = pattern_row['pattern_name']
                
                # 获取该Pattern应该使用的笔记类型
                if use_multi_notes and 'note_type' in patient_notes.columns:
                    preferred_note_types = get_note_types_for_pattern(pattern_name)
                    if preferred_note_types and not INCLUDE_DISCHARGE_NOTES:
                        preferred_note_types = [nt for nt in preferred_note_types if nt != 'discharge']
                        if not preferred_note_types:
                            preferred_note_types = None
                else:
                    preferred_note_types = None
                
                # 找到时间窗口内的笔记
                if 'note_type' in patient_notes.columns:
                    matching_notes_list = []
                    for note_type in patient_notes['note_type'].unique():
                        type_notes = patient_notes[patient_notes['note_type'] == note_type]
                        if note_type == 'discharge' and not INCLUDE_DISCHARGE_NOTES:
                            continue
                        note_mask = (
                            (type_notes['hour_offset'] >= pattern_hour - ALIGNMENT_WINDOW_BEFORE) &
                            (type_notes['hour_offset'] <= pattern_hour)
                        )
                        matching = type_notes[note_mask]
                        if len(matching) > 0:
                            matching_notes_list.append(matching)
                    
                    if matching_notes_list:
                        matching_notes = pd.concat(matching_notes_list, ignore_index=True)
                    else:
                        continue
                else:
                    note_mask = (
                        (patient_notes['hour_offset'] >= pattern_hour - ALIGNMENT_WINDOW_BEFORE) &
                        (patient_notes['hour_offset'] <= pattern_hour)
                    )
                    matching_notes = patient_notes[note_mask]
                
                if len(matching_notes) == 0:
                    continue
                
                # 按笔记类型优先级排序
                if preferred_note_types and 'note_type' in matching_notes.columns:
                    type_priority = {nt: idx for idx, nt in enumerate(preferred_note_types)}
                    matching_notes = matching_notes.copy()
                    matching_notes['_priority'] = matching_notes['note_type'].map(lambda x: type_priority.get(x, 100))
                    matching_notes = matching_notes.sort_values('_priority')
                
                for _, note_row in matching_notes.iterrows():
                    note_type = note_row.get('note_type', 'unknown')
                    note_id = str(note_row.get('note_id', ''))
                    if not ALLOW_NOTE_MULTI_MATCH and note_id in used_note_ids:
                        continue
                    used_note_ids.add(note_id)
                    note_text = clean_note_text(note_row.get('text', ''))
                    
                    # 获取针对该笔记类型的关键词
                    keywords = get_keywords_for_pattern(pattern_name, note_type)
                    
                    # 短文本直接使用，长文本提取关键词
                    SHORT_TEXT_THRESHOLD = 200
                    if len(note_text) < SHORT_TEXT_THRESHOLD:
                        relevant_text = note_text
                    else:
                        relevant_text = extract_relevant_sentences(note_text, keywords)
                        if not relevant_text and note_text:
                            relevant_text = note_text[:500]
                    
                    # 计算对齐质量
                    alignment_quality = 'low'
                    if preferred_note_types and note_type in preferred_note_types:
                        if len(relevant_text) > 50:
                            alignment_quality = 'high'
                        elif len(relevant_text) > 0:
                            alignment_quality = 'medium'
                    elif len(relevant_text) > 0:
                        alignment_quality = 'medium'
                    
                    # 统计
                    if note_type in note_type_stats:
                        note_type_stats[note_type] += 1
                    
                    # 直接写入CSV行
                    writer.writerow({
                        'stay_id': stay_id,
                        'pattern_hour': pattern_hour,
                        'pattern_name': pattern_name,
                        'pattern_value': pattern_row['value'],
                        'pattern_severity': pattern_row['severity'],
                        'pattern_disease': pattern_row['disease'],
                        'note_id': str(note_row.get('note_id', '')),
                        'note_hour': note_row['hour_offset'],
                        'note_category': note_row.get('category', 'Unknown'),
                        'note_type': note_type,
                        'note_text_full': note_text,
                        'note_text_relevant': relevant_text,
                        'time_delta_hours': note_row['hour_offset'] - pattern_hour,
                        'alignment_quality': alignment_quality,
                    })
                    total_alignments += 1
    
    # 打印笔记类型使用统计
    if note_type_stats:
        print(f"\n   [Note Type Usage Statistics]")
        for nt, count in sorted(note_type_stats.items(), key=lambda x: -x[1]):
            print(f"      {nt}: {count} alignments")
    
    print(f"\n   Total alignments written: {total_alignments}")
    print(f"Saved: {output_path}")

    # ==========================================
    # 统计摘要 (简化版 - 避免重新加载大文件)
    # ==========================================
    print("\n" + "=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)

    print(f"\nTotal alignments: {total_alignments}")
    print(f"Unique patients processed: {len(patient_ids)}")
    
    # 使用已收集的note_type_stats
    if note_type_stats:
        print("\n[Alignments by Note Type]")
        for note_type, count in sorted(note_type_stats.items(), key=lambda x: -x[1]):
            pct = count / total_alignments * 100 if total_alignments > 0 else 0
            print(f"   {note_type}: {count} ({pct:.1f}%)")

    # 保存简化统计
    stats = {
        'total_alignments': total_alignments,
        'unique_patients': len(patient_ids),
        'alignments_by_note_type': note_type_stats,
        'multi_note_mode': use_multi_notes,
    }

    stats_path = os.path.join(output_dir, 'alignment_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nStats saved: {stats_path}")
    
    # 返回空DataFrame以保持接口兼容
    return pd.DataFrame()

# ==========================================
# 6. 生成LLM标注样本
# ==========================================

def create_llm_annotation_samples(
    alignment_df: pd.DataFrame,
    output_dir: str,
    n_samples: int = None,  # None表示使用全部样本
    prioritize_high_quality: bool = True  # 优先采样高质量对齐
) -> pd.DataFrame:
    """
    创建用于LLM标注的样本

    Args:
        n_samples: 采样数量，None表示不限制（全量）

    改进：
    - 优先采样高质量对齐
    - 包含笔记类型信息
    - 改进提示词模板
    """

    # 筛选有相关文本的样本
    has_relevant = alignment_df[alignment_df['note_text_relevant'].str.len() > 10]
    if 'note_type' in has_relevant.columns:
        has_relevant = has_relevant[
            ~has_relevant['note_type'].astype(str).str.lower().str.contains('discharge', na=False)
        ]

    if len(has_relevant) == 0:
        print("No samples with relevant text found!")
        return pd.DataFrame()

    # 优先采样高质量对齐
    if prioritize_high_quality and 'alignment_quality' in has_relevant.columns:
        high_quality = has_relevant[has_relevant['alignment_quality'] == 'high']
        medium_quality = has_relevant[has_relevant['alignment_quality'] == 'medium']

        print(f"   High quality samples: {len(high_quality)}")
        print(f"   Medium quality samples: {len(medium_quality)}")

        # 优先使用高质量样本
        # 如果n_samples为None（全量模式），使用所有高质量+中等质量样本
        if n_samples is None:
            sampling_pool = pd.concat([high_quality, medium_quality], ignore_index=True)
        elif len(high_quality) >= n_samples:
            sampling_pool = high_quality
        else:
            sampling_pool = pd.concat([high_quality, medium_quality], ignore_index=True)
    else:
        sampling_pool = has_relevant

    # 分层采样：确保各种模式都有代表
    samples = []
    patterns = sampling_pool['pattern_name'].unique()

    # 如果n_samples为None，使用全部样本
    if n_samples is None:
        sample_df = sampling_pool.copy()
    else:
        samples_per_pattern = max(1, n_samples // len(patterns))

        for pattern in patterns:
            pattern_data = sampling_pool[sampling_pool['pattern_name'] == pattern]
            n = min(samples_per_pattern, len(pattern_data))
            if n > 0:
                samples.append(pattern_data.sample(n=n, random_state=42))

        sample_df = pd.concat(samples, ignore_index=True)

        # 限制总数
        if len(sample_df) > n_samples:
            sample_df = sample_df.sample(n=n_samples, random_state=42)

    # 添加标注提示
    sample_df['llm_prompt'] = sample_df.apply(
        lambda row: create_annotation_prompt(row), axis=1
    )

    # 保存
    output_path = os.path.join(output_dir, 'llm_annotation_samples.csv')
    sample_df.to_csv(output_path, index=False)
    print(f"Saved {len(sample_df)} annotation samples: {output_path}")

    # 打印采样统计
    print(f"\n   [Sample Statistics]")
    print(f"   Total samples: {len(sample_df)}")
    print(f"   Unique patterns: {sample_df['pattern_name'].nunique()}")

    if 'note_type' in sample_df.columns:
        print(f"   By note type:")
        for nt, count in sample_df['note_type'].value_counts().items():
            print(f"      {nt}: {count}")

    if 'alignment_quality' in sample_df.columns:
        print(f"   By quality:")
        for q, count in sample_df['alignment_quality'].value_counts().items():
            print(f"      {q}: {count}")

    # 改进的prompt模板
    prompt_template = """You are a clinical expert annotator. Given a detected physiological pattern and a clinical note excerpt, classify the alignment relationship.

PATTERN:
- Name: {pattern_name}
- Value: {pattern_value}
- Severity: {pattern_severity}
- Disease Context: {pattern_disease}

NOTE TYPE: {note_type}  # 新增笔记类型

CLINICAL NOTE:
{note_text_relevant}

CLASSIFICATION OPTIONS:
1. SUPPORTIVE - The note explicitly mentions or confirms the pattern (e.g., "patient febrile" for fever pattern)
2. CONTRADICTORY - The note contradicts the pattern (e.g., "afebrile" for fever pattern)
3. AMBIGUOUS - The note mentions related concepts but doesn't clearly confirm/deny
4. UNRELATED - The note doesn't discuss anything related to this pattern

Respond with only one word: SUPPORTIVE, CONTRADICTORY, AMBIGUOUS, or UNRELATED
"""

    prompt_path = os.path.join(output_dir, 'annotation_prompt_template.txt')
    with open(prompt_path, 'w') as f:
        f.write(prompt_template)

    return sample_df


def create_annotation_prompt(row) -> str:
    """为单行数据创建标注提示"""
    note_type = row.get('note_type', 'unknown')
    return f"""PATTERN: {row['pattern_name']} ({row['pattern_severity']})
VALUE: {row['pattern_value']}
DISEASE: {row['pattern_disease']}
NOTE_TYPE: {note_type}
NOTE_TEXT: {row['note_text_relevant'][:300]}"""

# ==========================================
# Main
# ==========================================

def main():
    print("Starting Temporal-Textual Alignment Pipeline (Multi-Note Version)")
    print("=" * 70)
    print(f"Multi-Note Mode: {'ON' if USE_MULTI_NOTES else 'OFF'}")

    # 检查输入文件
    patterns_path = os.path.join(PATTERN_DETECTION_DIR, 'detected_patterns_24h.csv')

    if not os.path.exists(patterns_path):
        print(f"Pattern detection file not found: {patterns_path}")
        print("Please run pattern_detector.py first!")
        return

    # 检查笔记文件
    if USE_MULTI_NOTES:
        print("\nChecking multi-note files...")
        note_files_exist = {
            'discharge': (RAW_DATA_DIR / 'discharge_notes.csv').exists(),
            'nursing': (RAW_DATA_DIR / 'nursing_notes.csv').exists(),
            'lab_comment': (RAW_DATA_DIR / 'lab_comments.csv').exists(),
            'radiology': NOTE_TIME_FILE.exists(),
        }
        for note_type, exists in note_files_exist.items():
            status = 'OK' if exists else 'MISSING'
            print(f"   {status} {note_type}_notes")

        if not any(note_files_exist.values()):
            print("No note files found! Please download the required data.")
            return
    else:
        if not NOTE_TIME_FILE.exists():
            print(f"Notes file not found: {NOTE_TIME_FILE}")
            return

    # 创建对齐数据集
    alignment_df = create_alignment_dataset(
        patterns_path=patterns_path,
        notes_path=NOTE_TIME_FILE if not USE_MULTI_NOTES else None,
        output_dir=OUTPUT_DIR,
        use_multi_notes=USE_MULTI_NOTES
    )

    if len(alignment_df) > 0:
        # 创建LLM标注样本 (n_samples=None表示全量)
        print("\n" + "=" * 70)
        print("Creating LLM annotation samples...")
        samples = create_llm_annotation_samples(
            alignment_df=alignment_df,
            output_dir=OUTPUT_DIR,
            n_samples=None,  # 全量标注，不再限制500条
            prioritize_high_quality=True
        )

    print("\nTemporal-Textual Alignment Complete!")
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("   - temporal_textual_alignment.csv")
    print("   - alignment_stats.json")
    print("   - llm_annotation_samples.csv")
    print("   - annotation_prompt_template.txt")

    # 打印改进摘要
    if USE_MULTI_NOTES and len(alignment_df) > 0:
        print("\n" + "=" * 70)
        print("MULTI-NOTE IMPROVEMENT SUMMARY")
        print("=" * 70)

        if 'note_type' in alignment_df.columns:
            print("\n[Note Type Distribution]")
            for nt, count in alignment_df['note_type'].value_counts().items():
                pct = count / len(alignment_df) * 100
                print(f"   {nt}: {count} ({pct:.1f}%)")

        if 'alignment_quality' in alignment_df.columns:
            high_quality = (alignment_df['alignment_quality'] == 'high').sum()
            medium_quality = (alignment_df['alignment_quality'] == 'medium').sum()
            total = len(alignment_df)

            print(f"\n[Quality Improvement]")
            print(f"   High quality alignments: {high_quality} ({high_quality/total*100:.1f}%)")
            print(f"   Medium quality alignments: {medium_quality} ({medium_quality/total*100:.1f}%)")
            print(f"   Usable alignments: {high_quality + medium_quality} ({(high_quality + medium_quality)/total*100:.1f}%)")

if __name__ == "__main__":
    main()
