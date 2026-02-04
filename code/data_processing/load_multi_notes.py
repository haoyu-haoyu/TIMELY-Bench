"""
Multi-Type Clinical Notes Loader
加载多种类型的临床笔记数据：Discharge, Nursing, Lab Comments, Radiology

核心功能：
1. 加载四种笔记类型
2. 统一数据格式
3. 提供Pattern-Note类型映射
4. 支持时间窗口过滤
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import os
import re

from config import RAW_DATA_DIR

# ==========================================
# 配置
# ==========================================
DATA_DIR = RAW_DATA_DIR

# Strict prediction window defaults (hours)
DEFAULT_WINDOW_START = 0
DEFAULT_WINDOW_END = 24

# Prediction tasks should not use discharge notes.
INCLUDE_DISCHARGE_NOTES = False

# 笔记文件路径
NOTE_FILES = {
    'discharge': 'discharge_notes.csv',
    'nursing': 'nursing_notes.csv',
    'lab_comment': 'lab_comments.csv',
    'radiology': 'note_time.csv'
}

# Pattern到Note类型的映射 (基于pattern_note_matrix.csv分析)
PATTERN_NOTE_MAPPING = {
    # Sepsis - 生命体征类 -> Discharge + Nursing
    "fever": ["discharge", "nursing"],
    "hypothermia": ["discharge", "nursing"],
    "tachycardia": ["discharge", "nursing"],
    "tachypnea": ["discharge", "nursing"],
    "hypotension": ["discharge", "nursing"],
    "map_low": ["discharge", "nursing"],
    "hypoxemia": ["radiology", "discharge", "nursing"],

    # Sepsis - 实验室类 -> Discharge + Lab_Comment
    "lactate_elevated": ["discharge", "lab_comment"],
    "thrombocytopenia": ["discharge", "lab_comment"],
    "hyperbilirubinemia": ["discharge", "lab_comment"],
    "leukocytosis": ["discharge", "lab_comment"],
    "leukopenia": ["discharge", "lab_comment"],

    # AKI - 肾功能类 -> Discharge + Lab_Comment
    "creatinine_elevated": ["discharge", "lab_comment"],
    "creatinine_severe": ["discharge", "lab_comment"],
    "creatinine_rise_acute": ["discharge", "lab_comment"],
    "bun_elevated": ["discharge", "lab_comment"],
    "bun_severe": ["discharge", "lab_comment"],
    "oliguria": ["discharge", "nursing"],
    "hyperkalemia": ["discharge", "lab_comment"],
    "metabolic_acidosis": ["discharge", "lab_comment"],

    # ARDS - 呼吸类 -> Radiology + Discharge + Nursing
    "hypoxemia_mild": ["radiology", "discharge", "nursing"],
    "hypoxemia_moderate": ["radiology", "discharge", "nursing"],
    "hypoxemia_severe": ["radiology", "discharge", "nursing"],
    "spo2_low": ["radiology", "nursing"],
    "respiratory_distress": ["radiology", "discharge", "nursing"],

    # Critical - 神经/血液类 -> Discharge + Nursing
    "bradycardia": ["discharge", "nursing"],
    "severe_tachycardia": ["discharge", "nursing"],
    "hypertensive_crisis": ["discharge", "nursing"],
    "anemia": ["discharge", "lab_comment"],
    "severe_anemia": ["discharge", "lab_comment"],
    "altered_consciousness": ["discharge", "nursing"],
    "coma": ["discharge", "nursing"],
}

# 每种笔记类型的关键词增强
NOTE_TYPE_KEYWORDS = {
    'discharge': {
        # Discharge notes contain comprehensive summaries
        'fever': ['fever', 'febrile', 'temperature', 'hyperthermia', 'infectious'],
        'tachycardia': ['tachycardia', 'heart rate', 'pulse', 'cardiac'],
        'hypotension': ['hypotension', 'hypotensive', 'shock', 'pressors', 'vasopressors'],
        'creatinine_elevated': ['creatinine', 'renal', 'kidney', 'AKI', 'acute kidney'],
        'lactate_elevated': ['lactate', 'lactic acidosis', 'sepsis', 'shock'],
    },
    'nursing': {
        # Nursing notes contain real-time assessments
        'fever': ['temp', 'temperature', 'febrile', 'chills'],
        'tachycardia': ['HR', 'heart rate', 'pulse', 'tachycardic'],
        'hypotension': ['BP', 'blood pressure', 'SBP', 'MAP', 'hypotensive'],
        'altered_consciousness': ['confused', 'oriented', 'GCS', 'responsive', 'alert'],
        'oliguria': ['urine output', 'UOP', 'foley', 'void'],
    },
    'lab_comment': {
        # Lab comments contain abnormal value notes
        'creatinine_elevated': ['creatinine', 'elevated', 'abnormal', 'critical'],
        'hyperkalemia': ['potassium', 'K+', 'elevated', 'hemolysis'],
        'lactate_elevated': ['lactate', 'elevated', 'critical'],
        'anemia': ['hemoglobin', 'hematocrit', 'low', 'critical'],
    },
    'radiology': {
        # Radiology notes contain imaging findings
        'hypoxemia_mild': ['infiltrate', 'opacity', 'consolidation', 'edema'],
        'hypoxemia_moderate': ['bilateral', 'ARDS', 'infiltrates', 'diffuse'],
        'hypoxemia_severe': ['ARDS', 'severe', 'diffuse', 'bilateral opacity'],
    }
}

# ==========================================
# 1. 加载各类型笔记
# ==========================================

def clean_text(text: str) -> str:
    """清理笔记文本"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 移除de-identification标记
    text = re.sub(r'\[\*\*[^\]]*\*\*\]', '[REDACTED]', text)
    # 移除多余空白
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_discharge_notes(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    加载出院小结 (Discharge Notes)

    列: stay_id, subject_id, hadm_id, icu_intime, note_id, note_time,
        hour_offset, discharge_text, text_length
    """
    path = os.path.join(data_dir, NOTE_FILES['discharge'])
    if not os.path.exists(path):
        print(f"   Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['stay_id'] = pd.to_numeric(df['stay_id'], errors='coerce').fillna(-1).astype(int)

    # 标准化列名
    df = df.rename(columns={
        'discharge_text': 'text',
        'note_time': 'charttime'
    })

    # 添加笔记类型
    df['note_type'] = 'discharge'
    df['category'] = 'Discharge Summary'

    # 清理文本
    df['text'] = df['text'].apply(clean_text)

    # 确保有note_id
    if 'note_id' not in df.columns:
        df['note_id'] = df.index.astype(str) + '_discharge'

    print(f"   Loaded {len(df)} discharge notes for {df['stay_id'].nunique()} patients")
    return df


def load_nursing_notes(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    加载护理评估笔记 (Nursing Notes)

    列: stay_id, subject_id, hadm_id, charttime, hour_offset,
        item_label, category, chart_text, valuenum
    """
    path = os.path.join(data_dir, NOTE_FILES['nursing'])
    if not os.path.exists(path):
        print(f"   Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['stay_id'] = pd.to_numeric(df['stay_id'], errors='coerce').fillna(-1).astype(int)

    # 标准化列名
    df = df.rename(columns={
        'chart_text': 'text'
    })

    # 添加笔记类型
    df['note_type'] = 'nursing'

    # 合并item_label到category以获得更详细的类别
    if 'item_label' in df.columns:
        df['category'] = df['category'].fillna('') + ': ' + df['item_label'].fillna('')
        df['category'] = df['category'].str.strip(': ')

    # 清理文本
    df['text'] = df['text'].apply(clean_text)

    # 生成note_id
    df['note_id'] = df.index.astype(str) + '_nursing'

    # 过滤观察窗口内的数据 (0-24h)
    if 'hour_offset' in df.columns:
        df = df[(df['hour_offset'] >= 0) & (df['hour_offset'] < 24)]

    print(f"   Loaded {len(df)} nursing notes for {df['stay_id'].nunique()} patients")
    return df


def load_lab_comments(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    加载实验室注释 (Lab Comments)

    列: stay_id, subject_id, hadm_id, charttime, hour_offset, lab_name,
        value, valuenum, valueuom, flag, ref_range_lower, ref_range_upper, lab_comment
    """
    path = os.path.join(data_dir, NOTE_FILES['lab_comment'])
    if not os.path.exists(path):
        print(f"   Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['stay_id'] = pd.to_numeric(df['stay_id'], errors='coerce').fillna(-1).astype(int)

    # 创建富文本：结合实验室名称、值和注释
    def create_lab_text(row):
        parts = []
        if pd.notna(row.get('lab_name')):
            parts.append(f"Lab: {row['lab_name']}")
        if pd.notna(row.get('valuenum')):
            parts.append(f"Value: {row['valuenum']} {row.get('valueuom', '')}")
        if pd.notna(row.get('flag')) and row['flag'] == 'abnormal':
            parts.append("FLAG: ABNORMAL")
        if pd.notna(row.get('ref_range_lower')) and pd.notna(row.get('ref_range_upper')):
            parts.append(f"Ref Range: {row['ref_range_lower']}-{row['ref_range_upper']}")
        if pd.notna(row.get('lab_comment')):
            parts.append(f"Comment: {row['lab_comment']}")
        return ' | '.join(parts)

    df['text'] = df.apply(create_lab_text, axis=1)

    # 添加笔记类型
    df['note_type'] = 'lab_comment'
    df['category'] = 'Lab Comment: ' + df['lab_name'].fillna('Unknown')

    # 生成note_id
    df['note_id'] = df.index.astype(str) + '_lab'

    # 过滤观察窗口内的数据 (0-24h)
    if 'hour_offset' in df.columns:
        df = df[(df['hour_offset'] >= 0) & (df['hour_offset'] < 24)]

    print(f"   Loaded {len(df)} lab comments for {df['stay_id'].nunique()} patients")
    return df


def load_radiology_notes(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """
    加载放射科报告 (Radiology Notes) - 原有数据
    """
    path = os.path.join(data_dir, NOTE_FILES['radiology'])
    if not os.path.exists(path):
        print(f"   Warning: {path} not found")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df['stay_id'] = pd.to_numeric(df['stay_id'], errors='coerce').fillna(-1).astype(int)

    # 标准化列名
    column_mapping = {
        'radiology_text': 'text',
        'note_text': 'text',
        'content': 'text',
    }
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]

    # 添加笔记类型
    df['note_type'] = 'radiology'
    if 'category' not in df.columns:
        df['category'] = 'Radiology'

    # 清理文本
    if 'text' in df.columns:
        df['text'] = df['text'].apply(clean_text)

    # 确保有note_id
    if 'note_id' not in df.columns:
        df['note_id'] = df.index.astype(str) + '_radiology'

    # 获取hour_offset
    if 'hour_offset' not in df.columns:
        df['hour_offset'] = 0  # 默认值

    print(f"   Loaded {len(df)} radiology notes for {df['stay_id'].nunique()} patients")
    return df


# ==========================================
# 2. 合并所有笔记类型
# ==========================================

def load_all_notes(
    data_dir: str = DATA_DIR,
    stay_ids: Optional[List[int]] = None,
    window_hours: int = DEFAULT_WINDOW_END,
    window_start: int = DEFAULT_WINDOW_START,
    include_discharge: bool = INCLUDE_DISCHARGE_NOTES,
) -> pd.DataFrame:
    """
    加载所有类型的笔记并合并

    Args:
        data_dir: 数据目录
        stay_ids: 过滤的患者ID列表
        window_hours: 时间窗口（小时）

    Returns:
        合并后的DataFrame，包含统一的列：
        - stay_id, note_id, note_type, category, hour_offset, text
    """
    print("\nLoading all note types...")

    all_notes = []

    # 加载各类型笔记
    if include_discharge:
        discharge_df = load_discharge_notes(data_dir)
        if len(discharge_df) > 0:
            all_notes.append(discharge_df)

    nursing_df = load_nursing_notes(data_dir)
    if len(nursing_df) > 0:
        all_notes.append(nursing_df)

    lab_df = load_lab_comments(data_dir)
    if len(lab_df) > 0:
        all_notes.append(lab_df)

    radiology_df = load_radiology_notes(data_dir)
    if len(radiology_df) > 0:
        all_notes.append(radiology_df)

    if not all_notes:
        print("   Warning: No notes loaded!")
        return pd.DataFrame()

    # 合并
    merged = pd.concat(all_notes, ignore_index=True)

    # 确保必要列存在
    required_cols = ['stay_id', 'note_id', 'note_type', 'category', 'hour_offset', 'text']
    for col in required_cols:
        if col not in merged.columns:
            merged[col] = ''

    # 过滤患者ID
    if stay_ids is not None:
        merged = merged[merged['stay_id'].isin(stay_ids)]

    # 过滤时间窗口（严格 0-24h 或用户指定窗口）
    merged['hour_offset'] = pd.to_numeric(merged['hour_offset'], errors='coerce')
    merged = merged[merged['hour_offset'].notna()]
    mask = (merged['hour_offset'] >= window_start) & (merged['hour_offset'] <= window_hours)
    merged = merged[mask]

    print(f"\n   Total merged notes: {len(merged)}")
    print(f"   Patients with notes: {merged['stay_id'].nunique()}")
    print(f"\n   [By Note Type]")
    for note_type, count in merged['note_type'].value_counts().items():
        print(f"      {note_type}: {count}")

    return merged[required_cols + [c for c in merged.columns if c not in required_cols]]


# ==========================================
# 3. 获取Pattern对应的最佳笔记类型
# ==========================================

def get_note_types_for_pattern(pattern_name: str) -> List[str]:
    """获取Pattern应该使用的笔记类型列表"""
    note_types = PATTERN_NOTE_MAPPING.get(pattern_name, ['discharge', 'radiology'])
    if not INCLUDE_DISCHARGE_NOTES:
        note_types = [nt for nt in note_types if nt != 'discharge']
    return note_types


def get_keywords_for_pattern_and_note_type(
    pattern_name: str,
    note_type: str
) -> List[str]:
    """获取特定Pattern和笔记类型的关键词"""
    # 首先检查笔记类型特定的关键词
    if note_type in NOTE_TYPE_KEYWORDS:
        type_keywords = NOTE_TYPE_KEYWORDS[note_type].get(pattern_name, [])
        if type_keywords:
            return type_keywords

    # 回退到默认关键词（从原始mapping）
    default_keywords = {
        'fever': ['fever', 'febrile', 'temperature', 'temp', 'hyperthermia'],
        'hypothermia': ['hypothermia', 'hypothermic', 'cold', 'temperature'],
        'tachycardia': ['tachycardia', 'tachycardic', 'heart rate', 'HR', 'pulse'],
        'tachypnea': ['tachypnea', 'tachypneic', 'respiratory rate', 'RR', 'breathing'],
        'hypotension': ['hypotension', 'hypotensive', 'blood pressure', 'BP', 'SBP', 'shock'],
        'map_low': ['MAP', 'mean arterial', 'perfusion'],
        'hypoxemia': ['hypoxia', 'hypoxemic', 'oxygen', 'O2', 'saturation', 'SpO2'],
        'lactate_elevated': ['lactate', 'lactic', 'acidosis'],
        'thrombocytopenia': ['thrombocytopenia', 'platelets', 'PLT', 'low platelets'],
        'hyperbilirubinemia': ['bilirubin', 'jaundice', 'icterus', 'liver'],
        'leukocytosis': ['leukocytosis', 'WBC', 'white blood cell', 'elevated WBC'],
        'leukopenia': ['leukopenia', 'WBC', 'white blood cell', 'low WBC'],
        'creatinine_elevated': ['creatinine', 'Cr', 'renal', 'kidney'],
        'creatinine_severe': ['creatinine', 'renal failure', 'kidney injury', 'AKI'],
        'creatinine_rise_acute': ['creatinine', 'rising', 'acute kidney', 'AKI'],
        'bun_elevated': ['BUN', 'urea', 'azotemia'],
        'bun_severe': ['BUN', 'uremia', 'renal failure'],
        'oliguria': ['oliguria', 'urine output', 'UOP', 'anuria'],
        'hyperkalemia': ['hyperkalemia', 'potassium', 'K+', 'elevated K'],
        'metabolic_acidosis': ['acidosis', 'bicarbonate', 'HCO3', 'pH'],
        'hypoxemia_mild': ['hypoxia', 'P/F ratio', 'PaO2', 'FiO2', 'ARDS'],
        'hypoxemia_moderate': ['hypoxia', 'respiratory failure', 'ARDS', 'intubation'],
        'hypoxemia_severe': ['severe hypoxia', 'refractory', 'ARDS', 'ECMO'],
        'spo2_low': ['desaturation', 'SpO2', 'oxygen'],
        'respiratory_distress': ['respiratory distress', 'dyspnea', 'tachypnea', 'work of breathing'],
        'bradycardia': ['bradycardia', 'bradycardic', 'slow heart rate'],
        'severe_tachycardia': ['tachycardia', 'SVT', 'rapid heart rate'],
        'hypertensive_crisis': ['hypertensive', 'hypertension', 'elevated BP', 'blood pressure'],
        'anemia': ['anemia', 'hemoglobin', 'Hgb', 'Hb', 'transfusion'],
        'severe_anemia': ['severe anemia', 'transfusion', 'hemorrhage', 'bleeding'],
        'altered_consciousness': ['altered mental status', 'AMS', 'confusion', 'lethargy', 'GCS'],
        'coma': ['coma', 'unresponsive', 'GCS', 'unconscious'],
    }

    return default_keywords.get(pattern_name, [pattern_name])


# ==========================================
# 4. 笔记数据统计
# ==========================================

def print_notes_summary(notes_df: pd.DataFrame):
    """打印笔记数据统计摘要"""
    print("\n" + "=" * 70)
    print("MULTI-TYPE NOTES SUMMARY")
    print("=" * 70)

    print(f"\nTotal notes: {len(notes_df)}")
    print(f"Unique patients: {notes_df['stay_id'].nunique()}")

    print("\n[By Note Type]")
    for note_type, group in notes_df.groupby('note_type'):
        print(f"   {note_type}:")
        print(f"      Count: {len(group)}")
        print(f"      Patients: {group['stay_id'].nunique()}")
        if 'text' in group.columns:
            avg_len = group['text'].str.len().mean()
            print(f"      Avg text length: {avg_len:.0f} chars")

    print("\n[Pattern Coverage Estimation]")
    note_types_available = set(notes_df['note_type'].unique())
    covered = 0
    total = len(PATTERN_NOTE_MAPPING)

    for pattern, required_types in PATTERN_NOTE_MAPPING.items():
        if any(t in note_types_available for t in required_types):
            covered += 1

    print(f"   Patterns with matching notes: {covered}/{total} ({covered/total*100:.1f}%)")


# ==========================================
# Main (测试用)
# ==========================================

if __name__ == "__main__":
    print("Testing Multi-Type Notes Loader")
    print("=" * 70)

    # 加载所有笔记
    all_notes = load_all_notes()

    if len(all_notes) > 0:
        print_notes_summary(all_notes)

        # 测试Pattern-Note映射
        print("\n[Pattern-Note Mapping Test]")
        test_patterns = ['fever', 'creatinine_elevated', 'hypoxemia_moderate']
        for pattern in test_patterns:
            note_types = get_note_types_for_pattern(pattern)
            print(f"   {pattern} -> {note_types}")

    print("\nMulti-Type Notes Loader Test Complete!")
