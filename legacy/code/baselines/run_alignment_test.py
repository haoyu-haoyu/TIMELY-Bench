"""
轻量级测试脚本：运行时序-文本对齐 pipeline
限制患者数量以避免内存问题
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
import json
import gc

from config import PATTERN_DETECTION_DIR, TEMPORAL_ALIGNMENT_DIR, RAW_DATA_DIR

# 配置
MAX_PATIENTS = 5000
MAX_PATTERNS_PER_PATIENT = 30
SAMPLE_SIZE = 500

OUTPUT_DIR = TEMPORAL_ALIGNMENT_DIR

def main():
    print("Starting Lightweight Temporal-Textual Alignment Test")
    print("=" * 70)
    print(f"   Max patients: {MAX_PATIENTS}")
    print(f"   Max patterns/patient: {MAX_PATTERNS_PER_PATIENT}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载模式检测结果（采样）
    print("\nLoading pattern detections...")
    patterns_path = os.path.join(PATTERN_DETECTION_DIR, 'detected_patterns_24h.csv')
    patterns_df = pd.read_csv(patterns_path)

    # 获取唯一患者ID并采样
    unique_patients = patterns_df['stay_id'].unique()
    print(f"   Total patients with patterns: {len(unique_patients)}")

    if len(unique_patients) > MAX_PATIENTS:
        np.random.seed(42)
        sampled_patients = np.random.choice(unique_patients, MAX_PATIENTS, replace=False)
        patterns_df = patterns_df[patterns_df['stay_id'].isin(sampled_patients)]
        print(f"   Sampled to {MAX_PATIENTS} patients")

    print(f"   Total patterns to process: {len(patterns_df)}")

    # 2. 加载笔记数据（分批加载）
    print("\nLoading clinical notes...")

    all_notes = []
    stay_ids_needed = set(patterns_df['stay_id'].unique())

    # 加载 Discharge Notes
    discharge_path = RAW_DATA_DIR / 'discharge_notes.csv'
    if discharge_path.exists():
        print("   Loading discharge notes...")
        discharge_df = pd.read_csv(discharge_path)
        discharge_df['stay_id'] = pd.to_numeric(discharge_df['stay_id'], errors='coerce').fillna(-1).astype(int)
        discharge_df = discharge_df[discharge_df['stay_id'].isin(stay_ids_needed)]
        discharge_df = discharge_df.rename(columns={'discharge_text': 'text', 'note_time': 'charttime'})
        discharge_df['note_type'] = 'discharge'
        discharge_df['category'] = 'Discharge Summary'
        if 'note_id' not in discharge_df.columns:
            discharge_df['note_id'] = discharge_df.index.astype(str) + '_discharge'
        all_notes.append(discharge_df[['stay_id', 'note_id', 'note_type', 'category', 'hour_offset', 'text']].copy())
        print(f"      Loaded {len(discharge_df)} discharge notes")
        del discharge_df
        gc.collect()

    # 加载 Nursing Notes
    nursing_path = RAW_DATA_DIR / 'nursing_notes.csv'
    if nursing_path.exists():
        print("   Loading nursing notes...")
        nursing_df = pd.read_csv(nursing_path)
        nursing_df['stay_id'] = pd.to_numeric(nursing_df['stay_id'], errors='coerce').fillna(-1).astype(int)
        nursing_df = nursing_df[nursing_df['stay_id'].isin(stay_ids_needed)]
        nursing_df = nursing_df.rename(columns={'chart_text': 'text'})
        nursing_df['note_type'] = 'nursing'
        if 'note_id' not in nursing_df.columns:
            nursing_df['note_id'] = nursing_df.index.astype(str) + '_nursing'
        # 过滤24小时窗口
        if 'hour_offset' in nursing_df.columns:
            nursing_df = nursing_df[(nursing_df['hour_offset'] >= 0) & (nursing_df['hour_offset'] <= 24)]
        all_notes.append(nursing_df[['stay_id', 'note_id', 'note_type', 'category', 'hour_offset', 'text']].copy())
        print(f"      Loaded {len(nursing_df)} nursing notes")
        del nursing_df
        gc.collect()

    # 加载 Lab Comments
    lab_path = RAW_DATA_DIR / 'lab_comments.csv'
    if lab_path.exists():
        print("   Loading lab comments...")
        lab_df = pd.read_csv(lab_path)
        lab_df['stay_id'] = pd.to_numeric(lab_df['stay_id'], errors='coerce').fillna(-1).astype(int)
        lab_df = lab_df[lab_df['stay_id'].isin(stay_ids_needed)]
        # 创建富文本
        lab_df['text'] = lab_df.apply(
            lambda r: f"Lab: {r.get('lab_name', 'N/A')} | Value: {r.get('valuenum', 'N/A')} | Comment: {r.get('lab_comment', 'N/A')}",
            axis=1
        )
        lab_df['note_type'] = 'lab_comment'
        lab_df['category'] = 'Lab Comment'
        if 'note_id' not in lab_df.columns:
            lab_df['note_id'] = lab_df.index.astype(str) + '_lab'
        if 'hour_offset' in lab_df.columns:
            lab_df = lab_df[(lab_df['hour_offset'] >= 0) & (lab_df['hour_offset'] <= 24)]
        all_notes.append(lab_df[['stay_id', 'note_id', 'note_type', 'category', 'hour_offset', 'text']].copy())
        print(f"      Loaded {len(lab_df)} lab comments")
        del lab_df
        gc.collect()

    # 合并所有笔记
    if not all_notes:
        print("   ERROR: No notes loaded!")
        return

    notes_df = pd.concat(all_notes, ignore_index=True)
    del all_notes
    gc.collect()

    print(f"\n   Total merged notes: {len(notes_df)}")
    print(f"   Patients with notes: {notes_df['stay_id'].nunique()}")
    print(f"\n   [By Note Type]")
    for note_type, count in notes_df['note_type'].value_counts().items():
        print(f"      {note_type}: {count}")

    # 3. 执行对齐
    print("\nAligning patterns with notes...")

    from load_multi_notes import get_note_types_for_pattern, get_keywords_for_pattern_and_note_type
    import re

    def clean_text(text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r'\[\*\*[^\]]*\*\*\]', '[REDACTED]', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def extract_relevant_sentences(text, keywords, max_sentences=5):
        if not text:
            return ""
        sentences = re.split(r'[.!?]\s+', text)
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            for kw in keywords:
                if kw.lower() in sent_lower:
                    relevant.append(sent.strip())
                    break
        return ' '.join(relevant[:max_sentences])

    # Pattern-Note类型映射
    PATTERN_NOTE_MAPPING = {
        "fever": ["discharge", "nursing"],
        "hypothermia": ["discharge", "nursing"],
        "tachycardia": ["discharge", "nursing"],
        "tachypnea": ["discharge", "nursing"],
        "hypotension": ["discharge", "nursing"],
        "map_low": ["discharge", "nursing"],
        "hypoxemia": ["radiology", "discharge", "nursing"],
        "lactate_elevated": ["discharge", "lab_comment"],
        "thrombocytopenia": ["discharge", "lab_comment"],
        "hyperbilirubinemia": ["discharge", "lab_comment"],
        "leukocytosis": ["discharge", "lab_comment"],
        "leukopenia": ["discharge", "lab_comment"],
        "creatinine_elevated": ["discharge", "lab_comment"],
        "creatinine_severe": ["discharge", "lab_comment"],
        "creatinine_rise_acute": ["discharge", "lab_comment"],
        "bun_elevated": ["discharge", "lab_comment"],
        "bun_severe": ["discharge", "lab_comment"],
        "oliguria": ["discharge", "nursing"],
        "hyperkalemia": ["discharge", "lab_comment"],
        "metabolic_acidosis": ["discharge", "lab_comment"],
        "hypoxemia_mild": ["radiology", "discharge", "nursing"],
        "hypoxemia_moderate": ["radiology", "discharge", "nursing"],
        "hypoxemia_severe": ["radiology", "discharge", "nursing"],
        "spo2_low": ["radiology", "nursing"],
        "respiratory_distress": ["radiology", "discharge", "nursing"],
        "bradycardia": ["discharge", "nursing"],
        "severe_tachycardia": ["discharge", "nursing"],
        "hypertensive_crisis": ["discharge", "nursing"],
        "anemia": ["discharge", "lab_comment"],
        "severe_anemia": ["discharge", "lab_comment"],
        "altered_consciousness": ["discharge", "nursing"],
        "coma": ["discharge", "nursing"],
    }

    PATTERN_KEYWORDS = {
        'fever': ['fever', 'febrile', 'temperature', 'temp', 'hyperthermia'],
        'tachycardia': ['tachycardia', 'tachycardic', 'heart rate', 'HR', 'pulse'],
        'hypotension': ['hypotension', 'hypotensive', 'blood pressure', 'BP', 'SBP', 'shock'],
        'creatinine_elevated': ['creatinine', 'Cr', 'renal', 'kidney', 'AKI'],
        'creatinine_severe': ['creatinine', 'renal failure', 'kidney injury', 'AKI'],
        'creatinine_rise_acute': ['creatinine', 'rising', 'acute kidney', 'AKI'],
        'lactate_elevated': ['lactate', 'lactic', 'acidosis'],
        'hyperkalemia': ['hyperkalemia', 'potassium', 'K+', 'elevated K'],
        'anemia': ['anemia', 'hemoglobin', 'Hgb', 'Hb', 'transfusion'],
        'hypoxemia': ['hypoxia', 'hypoxemic', 'oxygen', 'O2', 'saturation'],
        'altered_consciousness': ['altered mental', 'AMS', 'confusion', 'lethargy', 'GCS'],
    }

    aligned_events = []
    patient_ids = patterns_df['stay_id'].unique()
    patients_with_notes = set(notes_df['stay_id'].unique())
    patient_ids = [pid for pid in patient_ids if pid in patients_with_notes]

    print(f"   Processing {len(patient_ids)} patients...")

    window_before = 6
    window_after = 2

    for i, stay_id in enumerate(patient_ids):
        if (i + 1) % 500 == 0:
            print(f"   Processed {i+1}/{len(patient_ids)} patients... ({len(aligned_events)} alignments)")

        patient_patterns = patterns_df[patterns_df['stay_id'] == stay_id].copy()
        patient_notes = notes_df[notes_df['stay_id'] == stay_id]

        if len(patient_notes) == 0:
            continue

        # 限制每个患者的模式数
        if len(patient_patterns) > MAX_PATTERNS_PER_PATIENT:
            severity_order = {'severe': 0, 'moderate': 1, 'mild': 2}
            patient_patterns['_severity_order'] = patient_patterns['severity'].map(severity_order).fillna(2)
            patient_patterns = patient_patterns.sort_values('_severity_order').head(MAX_PATTERNS_PER_PATIENT)

        for _, pattern_row in patient_patterns.iterrows():
            pattern_hour = pattern_row['hour']
            pattern_name = pattern_row['pattern_name']

            # 获取该Pattern应该使用的笔记类型
            preferred_note_types = PATTERN_NOTE_MAPPING.get(pattern_name, ['discharge'])

            # 找到时间窗口内的笔记
            matching_notes_list = []
            for note_type in patient_notes['note_type'].unique():
                type_notes = patient_notes[patient_notes['note_type'] == note_type]

                if note_type == 'discharge':
                    matching = type_notes  # discharge notes不限时间
                else:
                    note_mask = (
                        (type_notes['hour_offset'] >= pattern_hour - window_before) &
                        (type_notes['hour_offset'] <= pattern_hour + window_after)
                    )
                    matching = type_notes[note_mask]

                if len(matching) > 0:
                    matching_notes_list.append(matching)

            if not matching_notes_list:
                continue

            matching_notes = pd.concat(matching_notes_list, ignore_index=True)

            # 按笔记类型优先级排序
            type_priority = {nt: idx for idx, nt in enumerate(preferred_note_types)}
            matching_notes = matching_notes.copy()
            matching_notes['_priority'] = matching_notes['note_type'].map(lambda x: type_priority.get(x, 100))
            matching_notes = matching_notes.sort_values('_priority')

            for _, note_row in matching_notes.head(3).iterrows():  # 每个pattern最多3个notes
                note_type = note_row.get('note_type', 'unknown')
                note_text = clean_text(note_row.get('text', ''))

                keywords = PATTERN_KEYWORDS.get(pattern_name, [pattern_name])
                relevant_text = extract_relevant_sentences(note_text, keywords)

                # 计算对齐质量
                alignment_quality = 'low'
                if note_type in preferred_note_types:
                    if len(relevant_text) > 50:
                        alignment_quality = 'high'
                    elif len(relevant_text) > 0:
                        alignment_quality = 'medium'
                elif len(relevant_text) > 0:
                    alignment_quality = 'medium'

                aligned_events.append({
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
                    'note_text_full': note_text[:500] if note_text else '',
                    'note_text_relevant': relevant_text,
                    'time_delta_hours': note_row['hour_offset'] - pattern_hour,
                    'alignment_quality': alignment_quality,
                })

    print(f"\n   Found {len(aligned_events)} pattern-note alignments")

    # 4. 保存结果
    alignment_df = pd.DataFrame(aligned_events)

    if len(alignment_df) == 0:
        print("   ERROR: No alignments found!")
        return

    output_path = os.path.join(OUTPUT_DIR, 'temporal_textual_alignment.csv')
    alignment_df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

    # 5. 统计摘要
    print("\n" + "=" * 70)
    print("ALIGNMENT SUMMARY")
    print("=" * 70)

    print(f"\nTotal alignments: {len(alignment_df)}")
    print(f"Unique patients: {alignment_df['stay_id'].nunique()}")
    print(f"Unique patterns: {alignment_df['pattern_name'].nunique()}")

    print("\n[Alignments by Note Type]")
    type_counts = alignment_df['note_type'].value_counts()
    for note_type, count in type_counts.items():
        pct = count / len(alignment_df) * 100
        print(f"   {note_type}: {count} ({pct:.1f}%)")

    print("\n[Alignments by Quality]")
    quality_counts = alignment_df['alignment_quality'].value_counts()
    for quality, count in quality_counts.items():
        pct = count / len(alignment_df) * 100
        print(f"   {quality}: {count} ({pct:.1f}%)")

    print("\n[Has Relevant Text]")
    has_relevant = (alignment_df['note_text_relevant'].str.len() > 0).sum()
    print(f"   With relevant keywords: {has_relevant} ({has_relevant/len(alignment_df)*100:.1f}%)")

    # 6. 创建LLM标注样本
    print("\nCreating LLM annotation samples...")

    has_relevant_df = alignment_df[alignment_df['note_text_relevant'].str.len() > 10]

    if len(has_relevant_df) > 0:
        # 优先高质量样本
        high_quality = has_relevant_df[has_relevant_df['alignment_quality'] == 'high']
        medium_quality = has_relevant_df[has_relevant_df['alignment_quality'] == 'medium']

        print(f"   High quality samples: {len(high_quality)}")
        print(f"   Medium quality samples: {len(medium_quality)}")

        if len(high_quality) >= SAMPLE_SIZE:
            sample_df = high_quality.sample(n=SAMPLE_SIZE, random_state=42)
        else:
            sampling_pool = pd.concat([high_quality, medium_quality], ignore_index=True)
            n = min(SAMPLE_SIZE, len(sampling_pool))
            sample_df = sampling_pool.sample(n=n, random_state=42)

        samples_path = os.path.join(OUTPUT_DIR, 'llm_annotation_samples.csv')
        sample_df.to_csv(samples_path, index=False)
        print(f"Saved {len(sample_df)} samples: {samples_path}")

    # 7. 保存统计
    stats = {
        'total_alignments': len(alignment_df),
        'unique_patients': int(alignment_df['stay_id'].nunique()),
        'unique_patterns': int(alignment_df['pattern_name'].nunique()),
        'alignments_by_note_type': type_counts.to_dict(),
        'alignments_by_quality': quality_counts.to_dict(),
        'alignments_with_relevant_text': int(has_relevant),
        'multi_note_mode': True,
    }

    stats_path = os.path.join(OUTPUT_DIR, 'alignment_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats: {stats_path}")

    # 8. 改进摘要
    print("\n" + "=" * 70)
    print("MULTI-NOTE IMPROVEMENT SUMMARY")
    print("=" * 70)

    high_q = (alignment_df['alignment_quality'] == 'high').sum()
    medium_q = (alignment_df['alignment_quality'] == 'medium').sum()
    total = len(alignment_df)

    print(f"\n[Quality Improvement]")
    print(f"   High quality alignments: {high_q} ({high_q/total*100:.1f}%)")
    print(f"   Medium quality alignments: {medium_q} ({medium_q/total*100:.1f}%)")
    print(f"   Usable alignments (high+medium): {high_q + medium_q} ({(high_q + medium_q)/total*100:.1f}%)")

    print("\nTemporal-Textual Alignment Complete!")

if __name__ == "__main__":
    main()
