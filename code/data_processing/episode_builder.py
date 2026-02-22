"""
Episode Builder - combines CSV data into Episode JSON format.

Data sources: timeseries.csv, cohort_final.csv, note_time.csv,
temporal_textual_alignment.csv, detected_patterns_24h.csv, llm_features_deepseek.csv

Output: episodes/ directory with one JSON per patient
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import asdict


# JSON serialization helper

class NumpyEncoder(json.JSONEncoder):
    """处理numpy类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if pd.isna(obj):
            return None
        return super().default(obj)


def to_python_type(value):
    """将numpy类型转换为Python原生类型"""
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value):
        return None
    return value

# 导入配置
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ROOT_DIR, RAW_DATA_DIR, PROCESSED_DIR, RESULTS_DIR,
    TIMESERIES_FILE, NOTE_TIME_FILE, LLM_FEATURES_FILE, COHORT_FILE,
    PATTERN_DETECTION_DIR, TEMPORAL_ALIGNMENT_DIR
)

# 导入Episode Schema
from episode_schema import (
    Episode, EpisodeMetadata, PatientDemographics,
    TimeSeriesData, VitalSign, LabValue, Intervention,
    ClinicalText, NoteSpan, LLMExtractedFeatures,
    ReasoningArtefacts, DetectedPattern, PatternTextAlignment,
    ConditionGraph, ConditionGraphNode, ConditionGraphEdge,
    Labels, OutcomeLabels, ProcessLabels
)

# 导入对齐数据索引器（用于处理 47GB 大文件）
from alignment_index import AlignmentIndexer

# Config

# 输入文件
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'
ALIGNMENT_FILE_CORE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment_core3000.csv'  # 筛选后的核心数据（2.23GB）
PATTERNS_FILE = PATTERN_DETECTION_DIR / 'detected_patterns_24h.csv'
ANNOTATIONS_DIR = PROCESSED_DIR / 'pattern_annotations'

# 输出目录
OUTPUT_DIR = PROCESSED_DIR / 'episodes'

# 观察窗口
OBSERVATION_WINDOW_HOURS = 24


# Episode Builder class

class EpisodeBuilder:
    """从CSV数据构建Episode对象"""

    def __init__(self, use_alignment_index: bool = True, use_llm_features: bool = False):
        self.data_loaded = False
        self.timeseries_df = None
        self.timeseries_grouped = None  # 按 stay_id 分组
        self.cohort_df = None
        self.notes_df = None
        self.notes_grouped = None  # 按 stay_id 分组
        self.alignment_df = None  # 保留以兼容小文件直接加载
        self.alignment_grouped = None  # 按 stay_id 分组的字典，O(1)查询
        self.alignment_indexer: AlignmentIndexer = None  # 大文件索引器
        self.use_alignment_index = use_alignment_index  # 是否使用索引模式
        self.use_llm_features = use_llm_features  # opt-in: 是否加载 LLM features
        self.patterns_df = None
        self.patterns_grouped = None  # 按 stay_id 分组
        self.llm_features_df = None
        self.llm_features_grouped = None  # 按 stay_id 分组
        self.annotated_df = None
        self.annotated_grouped = None  # 按 stay_id 分组
        self.pattern_templates = {}  # 生理学模板映射

    def load_all_data(self):
        """加载所有数据文件"""
        print("Loading data files...")

        # 时序数据
        if TIMESERIES_FILE.exists():
            self.timeseries_df = pd.read_csv(TIMESERIES_FILE)
            self.timeseries_df['stay_id'] = self.timeseries_df['stay_id'].astype(int)
            self.timeseries_grouped = self.timeseries_df.groupby('stay_id')
            print(f"   timeseries.csv: {len(self.timeseries_df)} records ({self.timeseries_grouped.ngroups} stay_ids)")
        else:
            print(f"   timeseries.csv not found")

        # 队列数据
        if COHORT_FILE.exists():
            self.cohort_df = pd.read_csv(COHORT_FILE)
            self.cohort_df['stay_id'] = self.cohort_df['stay_id'].astype(int)
            print(f"   cohort_final.csv: {len(self.cohort_df)} patients")
        else:
            print(f"   cohort_final.csv not found")

        # 笔记时间
        if NOTE_TIME_FILE.exists():
            self.notes_df = pd.read_csv(NOTE_TIME_FILE)
            if 'stay_id' in self.notes_df.columns:
                self.notes_df['stay_id'] = self.notes_df['stay_id'].astype(int)
                self.notes_grouped = self.notes_df.groupby('stay_id')
            print(f"   note_time.csv: {len(self.notes_df)} notes")
        else:
            print(f"   note_time.csv not found")

        # 对齐数据 - 优先使用筛选后的核心文件（2.23GB），否则使用索引模式
        alignment_file_to_use = None
        if ALIGNMENT_FILE_CORE.exists():
            # 优先使用筛选后的核心文件（小、快）
            alignment_file_to_use = ALIGNMENT_FILE_CORE
            file_size_gb = ALIGNMENT_FILE_CORE.stat().st_size / 1e9
            print(f"   temporal_textual_alignment_core3000.csv: {file_size_gb:.2f} GB (核心数据)")
        elif ALIGNMENT_FILE.exists():
            alignment_file_to_use = ALIGNMENT_FILE
            file_size_gb = ALIGNMENT_FILE.stat().st_size / 1e9
            print(f"   temporal_textual_alignment.csv: {file_size_gb:.1f} GB")
        
        if alignment_file_to_use is not None:
            file_size_gb = alignment_file_to_use.stat().st_size / 1e9
            if self.use_alignment_index and file_size_gb > 10.0:  # 大于 10GB 才使用索引
                print(f"   文件过大，使用索引模式...")
                self.alignment_indexer = AlignmentIndexer(alignment_file_to_use)
                if not self.alignment_indexer.load_index():
                    print("   索引不存在，开始构建（首次需要 10-20 分钟）...")
                    self.alignment_indexer.build_index()
                print(f"   已索引 {len(self.alignment_indexer.get_stay_ids())} 个 stay_id")
            else:
                # 直接加载（核心筛选文件 2.23GB，可以直接加载）
                print(f"   直接加载到内存...")
                self.alignment_df = pd.read_csv(alignment_file_to_use, low_memory=False)
                if 'stay_id' in self.alignment_df.columns:
                    self.alignment_df['stay_id'] = self.alignment_df['stay_id'].astype(int)
                print(f"   加载完成: {len(self.alignment_df):,} 条对齐记录")
                # 创建 groupby 分组，实现 O(1) 查询
                print(f"   创建 stay_id 分组索引...")
                self.alignment_grouped = self.alignment_df.groupby('stay_id')
                print(f"   分组完成: {self.alignment_grouped.ngroups:,} 个 stay_id")
        else:
            print(f"   temporal_textual_alignment.csv not found")

        # 检测到的模式
        if PATTERNS_FILE.exists():
            self.patterns_df = pd.read_csv(PATTERNS_FILE)
            self.patterns_df['stay_id'] = self.patterns_df['stay_id'].astype(int)
            self.patterns_grouped = self.patterns_df.groupby('stay_id')
            print(f"   detected_patterns_24h.csv: {len(self.patterns_df)} patterns ({self.patterns_grouped.ngroups} stay_ids)")
        else:
            print(f"   detected_patterns_24h.csv not found")

        # LLM特征 (opt-in: 需要显式启用 use_llm_features=True)
        if self.use_llm_features and LLM_FEATURES_FILE.exists():
            self.llm_features_df = pd.read_csv(LLM_FEATURES_FILE)
            if 'stay_id' in self.llm_features_df.columns:
                self.llm_features_df['stay_id'] = self.llm_features_df['stay_id'].astype(int)
                self.llm_features_grouped = self.llm_features_df.groupby('stay_id')
            print(f"   llm_features_deepseek.csv: {len(self.llm_features_df)} features")
        else:
            print(f"   llm_features_deepseek.csv: SKIPPED (use_llm_features=False or not found)")

        # 尝试加载标注样本（从pattern_annotations目录）
        annotated_files = list(ANNOTATIONS_DIR.glob('annotated_samples_*.csv'))
        if annotated_files:
            dfs = [pd.read_csv(f) for f in annotated_files]
            self.annotated_df = pd.concat(dfs, ignore_index=True)
            if 'stay_id' in self.annotated_df.columns:
                self.annotated_df['stay_id'] = self.annotated_df['stay_id'].astype(int)
                self.annotated_grouped = self.annotated_df.groupby('stay_id')
            print(f"   annotated_samples: {len(self.annotated_df)} annotations ({self.annotated_grouped.ngroups if self.annotated_grouped else 0} stay_ids)")
        else:
            print(f"   No annotated_samples files found in {ANNOTATIONS_DIR}")

        # 加载生理学模板
        templates_file = ROOT_DIR / 'documentation' / 'pattern_templates.json'
        if templates_file.exists():
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            # 构建pattern_name到template的映射
            for disease_key, disease_data in templates_data.items():
                for pattern in disease_data.get('patterns', []):
                    self.pattern_templates[pattern['name']] = {
                        'disease': disease_data['disease'],
                        'clinical_standard': disease_data['clinical_standard'],
                        **pattern
                    }
            print(f"   pattern_templates: {len(self.pattern_templates)} templates loaded")
        else:
            print(f"   pattern_templates.json not found")

        self.data_loaded = True
        print("All data loaded")

    def load_all_data_full(self):
        """加载所有数据文件（使用完整47GB对齐数据）"""
        print("Loading data files (FULL mode - 47GB alignment data)...")

        # 时序数据
        if TIMESERIES_FILE.exists():
            self.timeseries_df = pd.read_csv(TIMESERIES_FILE)
            self.timeseries_df['stay_id'] = self.timeseries_df['stay_id'].astype(int)
            self.timeseries_grouped = self.timeseries_df.groupby('stay_id')
            print(f"   timeseries.csv: {len(self.timeseries_df)} records ({self.timeseries_grouped.ngroups} stay_ids)")
        else:
            print(f"   timeseries.csv not found")

        # 队列数据
        if COHORT_FILE.exists():
            self.cohort_df = pd.read_csv(COHORT_FILE)
            self.cohort_df['stay_id'] = self.cohort_df['stay_id'].astype(int)
            print(f"   cohort_final.csv: {len(self.cohort_df)} patients")
        else:
            print(f"   cohort_final.csv not found")

        # 笔记时间
        if NOTE_TIME_FILE.exists():
            self.notes_df = pd.read_csv(NOTE_TIME_FILE)
            if 'stay_id' in self.notes_df.columns:
                self.notes_df['stay_id'] = self.notes_df['stay_id'].astype(int)
                self.notes_grouped = self.notes_df.groupby('stay_id')
            print(f"   note_time.csv: {len(self.notes_df)} notes")
        else:
            print(f"   note_time.csv not found")

        # 强制使用完整对齐数据（47GB）
        if ALIGNMENT_FILE.exists():
            file_size_gb = ALIGNMENT_FILE.stat().st_size / 1e9
            print(f"   temporal_textual_alignment.csv: {file_size_gb:.1f} GB (FULL)")
            print(f"   直接加载到内存（需要 ~6 分钟）...")
            import time
            start = time.time()
            self.alignment_df = pd.read_csv(ALIGNMENT_FILE, low_memory=True)
            if 'stay_id' in self.alignment_df.columns:
                self.alignment_df['stay_id'] = self.alignment_df['stay_id'].astype(int)
            load_time = time.time() - start
            print(f"   加载完成: {len(self.alignment_df):,} 条对齐记录, 耗时 {load_time:.1f}秒")
            print(f"   创建 stay_id 分组索引...")
            self.alignment_grouped = self.alignment_df.groupby('stay_id')
            print(f"   分组完成: {self.alignment_grouped.ngroups:,} 个 stay_id")
        else:
            print(f"   temporal_textual_alignment.csv not found")

        # 模式检测
        if PATTERNS_FILE.exists():
            self.patterns_df = pd.read_csv(PATTERNS_FILE)
            if 'stay_id' in self.patterns_df.columns:
                self.patterns_df['stay_id'] = self.patterns_df['stay_id'].astype(int)
                self.patterns_grouped = self.patterns_df.groupby('stay_id')
            print(f"   detected_patterns_24h.csv: {len(self.patterns_df)} patterns ({self.patterns_grouped.ngroups} stay_ids)")
        else:
            print(f"   detected_patterns_24h.csv not found")

        # LLM特征 (opt-in: 需要显式启用 use_llm_features=True)
        if self.use_llm_features and LLM_FEATURES_FILE.exists():
            self.llm_features_df = pd.read_csv(LLM_FEATURES_FILE)
            if 'stay_id' in self.llm_features_df.columns:
                self.llm_features_df['stay_id'] = self.llm_features_df['stay_id'].astype(int)
                self.llm_features_grouped = self.llm_features_df.groupby('stay_id')
            print(f"   llm_features_deepseek.csv: {len(self.llm_features_df)} features")
        else:
            print(f"   llm_features_deepseek.csv: SKIPPED (use_llm_features=False or not found)")

        # 标注样本
        annotated_files = list(ANNOTATIONS_DIR.glob('annotated_samples_*.csv'))
        if annotated_files:
            dfs = [pd.read_csv(f) for f in annotated_files]
            self.annotated_df = pd.concat(dfs, ignore_index=True)
            if 'stay_id' in self.annotated_df.columns:
                self.annotated_df['stay_id'] = self.annotated_df['stay_id'].astype(int)
                self.annotated_grouped = self.annotated_df.groupby('stay_id')
            print(f"   annotated_samples: {len(self.annotated_df)} annotations ({self.annotated_grouped.ngroups if self.annotated_grouped else 0} stay_ids)")
        else:
            print(f"   No annotated_samples files found")

        # 加载生理学模板
        templates_file = ROOT_DIR / 'documentation' / 'pattern_templates.json'
        if templates_file.exists():
            with open(templates_file, 'r', encoding='utf-8') as f:
                templates_data = json.load(f)
            for disease_key, disease_data in templates_data.items():
                for pattern in disease_data.get('patterns', []):
                    self.pattern_templates[pattern['name']] = {
                        'disease': disease_data['disease'],
                        'clinical_standard': disease_data['clinical_standard'],
                        **pattern
                    }
            print(f"   pattern_templates: {len(self.pattern_templates)} templates loaded")

        self.data_loaded = True
        print("All data loaded (FULL mode)")

    def build_timeseries(self, stay_id: int) -> TimeSeriesData:
        """构建时序数据"""
        ts_data = TimeSeriesData()

        if self.timeseries_grouped is None:
            return ts_data

        # 使用分组查询 O(1)
        try:
            patient_ts_all = self.timeseries_grouped.get_group(stay_id)
        except KeyError:
            return ts_data
        
        # 过滤观察窗口
        patient_ts = patient_ts_all[patient_ts_all['hour'] < OBSERVATION_WINDOW_HOURS].copy()
        sort_cols = ['hour']
        if 'charttime' in patient_ts.columns:
            patient_ts['_charttime_sort'] = pd.to_datetime(patient_ts['charttime'], errors='coerce')
            sort_cols.append('_charttime_sort')
        patient_ts = patient_ts.sort_values(sort_cols)

        if len(patient_ts) == 0:
            return ts_data

        # 生命体征列
        vital_cols = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate',
                      'temperature', 'spo2', 'gcs_min', 'urineoutput']

        # 化验列
        lab_cols = ['creatinine', 'bun', 'sodium', 'potassium', 'bicarbonate',
                    'chloride', 'ph', 'lactate', 'wbc', 'hemoglobin',
                    'hematocrit', 'platelet', 'glucose_lab', 'albumin',
                    'bilirubin_total']

        # 干预/治疗列（可选：扩展时序文件才会包含）
        intervention_cols = ['vasopressors', 'rrt']
        has_interventions = any(c in patient_ts.columns for c in intervention_cols)

        # 处理每个时间点
        for _, row in patient_ts.iterrows():
            hour = to_python_type(row['hour'])

            # 生命体征
            vital = VitalSign(hour=hour)
            if 'charttime' in row and pd.notna(row['charttime']):
                vital.timestamp = str(row['charttime'])
            for col in vital_cols:
                if col in row and pd.notna(row[col]):
                    # 处理列名映射
                    attr_name = 'gcs' if col == 'gcs_min' else col
                    if hasattr(vital, attr_name):
                        setattr(vital, attr_name, to_python_type(row[col]))
            ts_data.vitals.append(vital)

            # 化验值（检查是否有化验数据）
            has_lab = any(col in row and pd.notna(row[col]) for col in lab_cols)
            if has_lab:
                lab = LabValue(hour=hour)
                if 'charttime' in row and pd.notna(row['charttime']):
                    lab.timestamp = str(row['charttime'])
                for col in lab_cols:
                    if col in row and pd.notna(row[col]):
                        attr_name = 'glucose' if col == 'glucose_lab' else col
                        if hasattr(lab, attr_name):
                            setattr(lab, attr_name, to_python_type(row[col]))
                ts_data.labs.append(lab)

            # 干预/治疗（binary indicators）
            if has_interventions:
                interv = Intervention(hour=hour)
                if 'charttime' in row and pd.notna(row['charttime']):
                    interv.timestamp = str(row['charttime'])
                for col in intervention_cols:
                    if col in row and pd.notna(row[col]) and hasattr(interv, col):
                        v = to_python_type(row[col])
                        try:
                            v = int(v)
                        except Exception:
                            pass
                        setattr(interv, col, v)
                ts_data.interventions.append(interv)

        ts_data.n_timepoints = len(patient_ts)
        ts_data.start_hour = 0
        ts_data.end_hour = OBSERVATION_WINDOW_HOURS

        # 计算缺失率
        for col in vital_cols + lab_cols + (intervention_cols if has_interventions else []):
            if col in patient_ts.columns:
                missing_rate = patient_ts[col].isna().mean()
                ts_data.missing_rate[col] = round(float(missing_rate), 3)

        return ts_data

    def _get_alignment_data(self, stay_id: int) -> Optional[pd.DataFrame]:
        """获取指定 stay_id 的对齐数据（支持索引模式和普通模式）"""
        if self.alignment_indexer is not None:
            # 索引模式：按需加载
            return self.alignment_indexer.get_alignment_data(stay_id)
        elif self.alignment_grouped is not None:
            # 分组模式：O(1)字典查找（快）
            try:
                return self.alignment_grouped.get_group(stay_id)
            except KeyError:
                return None
        elif self.alignment_df is not None:
            # 普通模式：直接过滤（慢）
            return self.alignment_df[self.alignment_df['stay_id'] == stay_id]
        return None

    def build_clinical_text(self, stay_id: int) -> ClinicalText:
        """构建临床文本数据"""
        text_data = ClinicalText()
        seen_notes = set()

        # 1. 从note_time.csv获取radiology笔记（完整文本）
        if self.notes_grouped is not None:
            try:
                patient_notes = self.notes_grouped.get_group(stay_id)
            except KeyError:
                patient_notes = None
            
            if patient_notes is not None:
                for _, row in patient_notes.iterrows():
                    note_id = str(row.get('note_id', ''))
                    if note_id in seen_notes:
                        continue
                    seen_notes.add(note_id)

                    # radiology_text是完整的放射学报告文本
                    radiology_text = str(row.get('radiology_text', '')) if pd.notna(row.get('radiology_text')) else ''
                    if radiology_text:
                        hour_offset = to_python_type(row.get('hour_offset', 0))
                        if hour_offset is None or hour_offset < 0 or hour_offset >= OBSERVATION_WINDOW_HOURS:
                            continue
                        note = NoteSpan(
                            note_id=note_id,
                            note_type='radiology',
                            note_category='Radiology',
                            chart_hour=hour_offset if hour_offset else 0,
                            text_full=radiology_text,
                            text_relevant=radiology_text[:500] if len(radiology_text) > 500 else radiology_text
                        )
                        note.text_length = len(note.text_full)
                        text_data.notes.append(note)

        # 2. 从对齐数据中获取其他笔记（nursing, lab_comment等）
        patient_align = self._get_alignment_data(stay_id)
        if patient_align is not None and len(patient_align) > 0:

            for _, row in patient_align.iterrows():
                note_id = str(row.get('note_id', ''))
                if note_id in seen_notes:
                    continue
                seen_notes.add(note_id)

                note_type = str(row.get('note_type', ''))
                if note_type == 'discharge':
                    continue
                chart_hour = to_python_type(row.get('note_hour', 0))
                if chart_hour is None or chart_hour < 0 or chart_hour >= OBSERVATION_WINDOW_HOURS:
                    continue

                note = NoteSpan(
                    note_id=note_id,
                    note_type=note_type,
                    note_category=str(row.get('note_category', '')),
                    chart_hour=chart_hour,
                    text_full=str(row.get('note_text_full', '')),
                    text_relevant=str(row.get('note_text_relevant', ''))
                )
                note.text_length = len(note.text_full)
                text_data.notes.append(note)

        # 添加LLM特征
        if self.llm_features_df is not None:
            patient_llm = self.llm_features_df[self.llm_features_df['stay_id'] == stay_id]

            for _, row in patient_llm.iterrows():
                llm_feat = LLMExtractedFeatures(
                    note_id=str(row.get('note_id', '')),
                    pneumonia=to_python_type(row.get('pneumonia')) if pd.notna(row.get('pneumonia')) else None,
                    edema=to_python_type(row.get('edema')) if pd.notna(row.get('edema')) else None,
                    pleural_effusion=to_python_type(row.get('pleural_effusion')) if pd.notna(row.get('pleural_effusion')) else None,
                    pneumothorax=to_python_type(row.get('pneumothorax')) if pd.notna(row.get('pneumothorax')) else None,
                    tubes_lines=to_python_type(row.get('tubes_lines')) if pd.notna(row.get('tubes_lines')) else None,
                    model_version="deepseek"
                )
                text_data.llm_features.append(llm_feat)

        text_data.n_notes = len(text_data.notes)
        text_data.note_types = list(set(n.note_type for n in text_data.notes))
        text_data.coverage_hours = list(set(n.chart_hour for n in text_data.notes))

        return text_data

    def build_reasoning_artefacts(self, stay_id: int, cohort_row: pd.Series) -> ReasoningArtefacts:
        """构建推理构件"""
        reasoning = ReasoningArtefacts()

        # 1. 检测到的模式 - 使用分组查询 O(1)
        if self.patterns_grouped is not None:
            try:
                patient_patterns = self.patterns_grouped.get_group(stay_id)
            except KeyError:
                patient_patterns = None
            
            if patient_patterns is not None:
                for _, row in patient_patterns.iterrows():
                    pattern_name = str(row.get('pattern_name', ''))

                    # 获取模板信息
                    template = self.pattern_templates.get(pattern_name, {})

                    pattern = DetectedPattern(
                        pattern_name=pattern_name,
                        detection_hour=to_python_type(row.get('hour', 0)),
                        value=to_python_type(row.get('value', 0)),
                        threshold=to_python_type(row.get('threshold')) if pd.notna(row.get('threshold')) else None,
                        disease=str(row.get('disease', '')),
                        feature=str(row.get('feature', '')),
                        severity=str(row.get('severity', 'moderate')),
                        description=str(row.get('description', '')),
                        # 添加临床模板信息
                        clinical_source=template.get('clinical_source'),
                        reference_pmid=template.get('reference_pmid'),
                        evidence_level=template.get('evidence_level'),
                        unit=template.get('unit')
                    )
                    reasoning.detected_patterns.append(pattern)

        reasoning.n_patterns_detected = len(reasoning.detected_patterns)

        # 2. 模式-文本对齐
        patient_align = self._get_alignment_data(stay_id)
        if patient_align is not None and len(patient_align) > 0:

            for _, row in patient_align.iterrows():
                # 修复Bug: 处理NaN值，避免转成字符串"nan"
                note_text = row.get('note_text_relevant', '')
                if pd.isna(note_text) or str(note_text).lower() == 'nan':
                    note_text = ''  # 使用空字符串替代
                else:
                    note_text = str(note_text)
                
                annotation = PatternTextAlignment(
                    pattern_name=str(row.get('pattern_name', '')),
                    pattern_hour=to_python_type(row.get('pattern_hour', 0)),
                    note_id=str(row.get('note_id', '')),
                    note_hour=to_python_type(row.get('note_hour', 0)),
                    note_type=str(row.get('note_type', '')),
                    time_delta_hours=to_python_type(row.get('time_delta_hours', 0)),
                    alignment_quality=str(row.get('alignment_quality', '')),
                    aligned_text=note_text
                )

                # === Bug4修复：三层模糊匹配策略 ===
                # 添加LLM标注（如果有）- 使用字典索引优化
                if self.annotated_df is not None and self.annotated_grouped is not None:
                    match = None
                    pattern_hour = annotation.pattern_hour
                    
                    # 使用 groupby 索引进行 O(1) 查找
                    try:
                        # 先获取该 stay_id 的所有标注
                        stay_annots = self.annotated_grouped.get_group(stay_id)
                        
                        # 在小数据集上进行过滤
                        # 第1层：精确匹配（pattern_name, pattern_hour, note_type）
                        match = stay_annots[
                            (stay_annots['pattern_name'] == annotation.pattern_name) &
                            (stay_annots['pattern_hour'] == pattern_hour) &
                            (stay_annots['note_type'] == annotation.note_type)
                        ]
                        
                        # 第2层：±0.5h容差匹配
                        if len(match) == 0:
                            match = stay_annots[
                                (stay_annots['pattern_name'] == annotation.pattern_name) &
                                (stay_annots['pattern_hour'] >= pattern_hour - 0.5) &
                                (stay_annots['pattern_hour'] <= pattern_hour + 0.5) &
                                (stay_annots['note_type'] == annotation.note_type)
                            ]
                        
                        # 第3层：±1h容差 + 放宽note_type
                        if len(match) == 0:
                            match = stay_annots[
                                (stay_annots['pattern_name'] == annotation.pattern_name) &
                                (stay_annots['pattern_hour'] >= pattern_hour - 1) &
                                (stay_annots['pattern_hour'] <= pattern_hour + 1)
                            ]
                        
                        # 第4层：仅基于pattern_name的最近邻匹配
                        if len(match) == 0:
                            pattern_annots = stay_annots[
                                stay_annots['pattern_name'] == annotation.pattern_name
                            ]
                            if len(pattern_annots) > 0:
                                pattern_annots = pattern_annots.copy()
                                pattern_annots['hour_diff'] = abs(pattern_annots['pattern_hour'] - pattern_hour)
                                match = pattern_annots.nsmallest(1, 'hour_diff')
                    except KeyError:
                        # 该 stay_id 没有标注数据
                        match = None
                    
                    if match is not None and len(match) > 0:
                        # 如果有多个匹配，选择置信度最高的
                        if len(match) > 1 and 'annotation_confidence' in match.columns:
                            match = match.sort_values('annotation_confidence', ascending=False)
                        annotation.annotation_category = str(match.iloc[0].get('annotation_category', ''))
                        annotation.annotation_confidence = to_python_type(match.iloc[0].get('annotation_confidence', 0))
                        annotation.annotation_reasoning = str(match.iloc[0].get('annotation_reasoning', ''))

                reasoning.pattern_annotations.append(annotation)

        reasoning.n_alignments = len(reasoning.pattern_annotations)
        reasoning.n_supportive = sum(1 for a in reasoning.pattern_annotations
                                     if a.annotation_category == 'SUPPORTIVE')
        reasoning.n_contradictory = sum(1 for a in reasoning.pattern_annotations
                                        if a.annotation_category == 'CONTRADICTORY')

        # 3. 构建疾病依赖图
        reasoning.condition_graph = self._build_condition_graph(stay_id, cohort_row, reasoning)

        return reasoning

    def _build_condition_graph(
        self,
        stay_id: int,
        cohort_row: pd.Series,
        reasoning: ReasoningArtefacts
    ) -> ConditionGraph:
        """构建疾病依赖图"""
        graph = ConditionGraph()

        # 从诊断标签获取疾病节点
        conditions = []

        # Sepsis
        if cohort_row.get('has_sepsis_final', 0) == 1:
            # 从模式检测中估计发病时间
            sepsis_patterns = [p for p in reasoning.detected_patterns
                               if p.disease == 'Sepsis']
            onset_hour = min(p.detection_hour for p in sepsis_patterns) if sepsis_patterns else None

            conditions.append(ConditionGraphNode(
                condition="Sepsis",
                is_present=True,
                onset_hour=onset_hour,
                confidence=1.0,
                source="diagnosis"
            ))

        # AKI
        if cohort_row.get('has_aki_final', 0) == 1:
            aki_patterns = [p for p in reasoning.detected_patterns
                           if p.disease == 'AKI']
            onset_hour = min(p.detection_hour for p in aki_patterns) if aki_patterns else None

            conditions.append(ConditionGraphNode(
                condition="AKI",
                is_present=True,
                onset_hour=onset_hour,
                confidence=1.0,
                source="diagnosis"
            ))

        # ARDS
        if cohort_row.get('has_ards', 0) == 1:
            ards_patterns = [p for p in reasoning.detected_patterns
                            if p.disease == 'ARDS']
            onset_hour = min(p.detection_hour for p in ards_patterns) if ards_patterns else None

            conditions.append(ConditionGraphNode(
                condition="ARDS",
                is_present=True,
                onset_hour=onset_hour,
                confidence=1.0,
                source="diagnosis"
            ))

        graph.nodes = conditions

        # 构建边（基于临床知识）
        edges = []

        # Sepsis → AKI (已知因果关系)
        sepsis_node = next((n for n in conditions if n.condition == "Sepsis"), None)
        aki_node = next((n for n in conditions if n.condition == "AKI"), None)

        if sepsis_node and aki_node:
            time_delta = None
            if sepsis_node.onset_hour is not None and aki_node.onset_hour is not None:
                time_delta = aki_node.onset_hour - sepsis_node.onset_hour

            edges.append(ConditionGraphEdge(
                source="Sepsis",
                target="AKI",
                relationship="causes",
                temporal_order="before" if time_delta and time_delta > 0 else "concurrent",
                time_delta_hours=time_delta,
                evidence_type="clinical_guideline"
            ))

        # Sepsis → ARDS
        ards_node = next((n for n in conditions if n.condition == "ARDS"), None)
        if sepsis_node and ards_node:
            time_delta = None
            if sepsis_node.onset_hour is not None and ards_node.onset_hour is not None:
                time_delta = ards_node.onset_hour - sepsis_node.onset_hour

            edges.append(ConditionGraphEdge(
                source="Sepsis",
                target="ARDS",
                relationship="causes",
                temporal_order="before" if time_delta and time_delta > 0 else "concurrent",
                time_delta_hours=time_delta,
                evidence_type="clinical_guideline"
            ))

        graph.edges = edges

        # 确定主要疾病
        if sepsis_node:
            graph.primary_condition = "Sepsis"
        elif aki_node:
            graph.primary_condition = "AKI"
        elif ards_node:
            graph.primary_condition = "ARDS"

        # 计算复杂度分数
        graph.complexity_score = len(conditions) + len(edges) * 0.5

        return graph

    def build_labels(self, cohort_row: pd.Series, reasoning: ReasoningArtefacts) -> Labels:
        """构建标签"""
        labels = Labels()

        # 结局标签
        labels.outcome = OutcomeLabels(
            mortality=to_python_type(cohort_row.get('label_mortality', 0)),
            prolonged_los=to_python_type(cohort_row.get('prolonged_los_7d', 0)),
            readmission_30d=to_python_type(cohort_row.get('readmission_30d', 0)) if pd.notna(cohort_row.get('readmission_30d')) else None,
            los_days=to_python_type(cohort_row.get('los', 0)) if pd.notna(cohort_row.get('los')) else None
        )

        # 过程标签（从检测模式推断）
        process = ProcessLabels()

        # Sepsis发病时间
        sepsis_patterns = [p for p in reasoning.detected_patterns if p.disease == 'Sepsis']
        if sepsis_patterns:
            process.sepsis_onset_hour = min(p.detection_hour for p in sepsis_patterns)

        # AKI发病时间
        aki_patterns = [p for p in reasoning.detected_patterns if p.disease == 'AKI']
        if aki_patterns:
            process.aki_onset_hour = min(p.detection_hour for p in aki_patterns)
            process.aki_stage_max = to_python_type(cohort_row.get('aki_stage_max', 0)) if pd.notna(cohort_row.get('aki_stage_max')) else None

        # ARDS发病时间
        ards_patterns = [p for p in reasoning.detected_patterns if p.disease == 'ARDS']
        if ards_patterns:
            process.ards_onset_hour = min(p.detection_hour for p in ards_patterns)

        # 首次严重模式时间
        severe_patterns = [p for p in reasoning.detected_patterns if p.severity == 'severe']
        if severe_patterns:
            process.first_severe_pattern_hour = min(p.detection_hour for p in severe_patterns)

        labels.process = process

        # 疾病标志
        labels.has_sepsis = bool(cohort_row.get('has_sepsis_final', 0))
        labels.has_aki = bool(cohort_row.get('has_aki_final', 0))
        labels.has_ards = bool(cohort_row.get('has_ards', 0))

        # ICD诊断码
        icd_str = str(cohort_row.get('icd_codes', ''))
        labels.icd_codes = icd_str.split(',') if icd_str else []
        labels.diagnoses_text = str(cohort_row.get('diagnoses_text', ''))

        return labels

    def build_episode(self, stay_id: int) -> Optional[Episode]:
        """构建单个Episode"""
        if not self.data_loaded:
            self.load_all_data()

        # 获取患者队列信息
        if self.cohort_df is None:
            return None

        cohort_row = self.cohort_df[self.cohort_df['stay_id'] == stay_id]
        if len(cohort_row) == 0:
            return None
        cohort_row = cohort_row.iloc[0]

        # 构建各组件
        episode = Episode(
            episode_id=f"TIMELY_v2_{stay_id}",
            stay_id=stay_id,
            patient=PatientDemographics(
                age=to_python_type(cohort_row.get('anchor_age', 0)) if pd.notna(cohort_row.get('anchor_age')) else None,
                gender=str(cohort_row.get('gender', '')) if pd.notna(cohort_row.get('gender')) else None,
                subject_id=to_python_type(cohort_row.get('subject_id', 0)) if pd.notna(cohort_row.get('subject_id')) else None,
                hadm_id=to_python_type(cohort_row.get('hadm_id', 0)) if pd.notna(cohort_row.get('hadm_id')) else None
            )
        )

        # 时序数据
        episode.timeseries = self.build_timeseries(stay_id)

        # 临床文本
        episode.clinical_text = self.build_clinical_text(stay_id)

        # 推理构件
        episode.reasoning = self.build_reasoning_artefacts(stay_id, cohort_row)

        # 标签
        episode.labels = self.build_labels(cohort_row, episode.reasoning)

        # 元数据
        episode.metadata = EpisodeMetadata(
            schema_version="2.0",
            created_at=datetime.now().isoformat(),
            source_database="MIMIC-IV",
            source_version="3.1",
            observation_window_hours=OBSERVATION_WINDOW_HOURS,
            data_quality_score=self._calculate_quality_score(episode)
        )

        return episode

    def _calculate_quality_score(self, episode: Episode) -> float:
        """计算数据质量分数"""
        score = 0.0
        components = 0

        # 时序数据完整性
        if episode.timeseries.n_timepoints > 0:
            # 平均缺失率
            if episode.timeseries.missing_rate:
                avg_missing = np.mean(list(episode.timeseries.missing_rate.values()))
                score += (1 - avg_missing) * 0.3
            else:
                score += 0.15
            components += 0.3

        # 临床文本覆盖
        if episode.clinical_text.n_notes > 0:
            score += min(episode.clinical_text.n_notes / 5, 1.0) * 0.2
            components += 0.2

        # 模式检测
        if episode.reasoning.n_patterns_detected > 0:
            score += 0.2
            components += 0.2

        # 对齐数据
        if episode.reasoning.n_alignments > 0:
            score += 0.15
            components += 0.15

        # LLM标注
        if episode.reasoning.n_supportive + episode.reasoning.n_contradictory > 0:
            score += 0.15
            components += 0.15

        return round(score / max(components, 0.01), 3)


# ==========================================
# 批量处理
# ==========================================

def build_all_episodes(
    max_episodes: Optional[int] = None,
    output_dir: Path = OUTPUT_DIR,
    save_individual: bool = True,
    core_stay_ids: Optional[List[int]] = None
) -> Tuple[List[Episode], pd.DataFrame]:
    """
    构建所有Episode

    Args:
        max_episodes: 最大Episode数量（用于测试）
        output_dir: 输出目录
        save_individual: 是否保存单独的JSON文件
        core_stay_ids: 核心数据集中的stay_ids（如果提供，只处理这些）

    Returns:
        (episodes列表, 摘要DataFrame)
    """
    print("=" * 70)
    print("TIMELY-Bench Episode Builder")
    print("=" * 70)

    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化Builder
    builder = EpisodeBuilder()
    builder.load_all_data()

    # 获取所有stay_ids
    if builder.cohort_df is None:
        print("No cohort data available")
        return [], pd.DataFrame()

    # 如果提供了核心stay_ids，只处理这些
    if core_stay_ids is not None:
        stay_ids = [sid for sid in core_stay_ids if sid in builder.cohort_df['stay_id'].values]
        print(f"Using core dataset: {len(stay_ids)} patients")
    else:
        stay_ids = builder.cohort_df['stay_id'].unique()

    if max_episodes:
        stay_ids = stay_ids[:max_episodes]

    print(f"\nBuilding {len(stay_ids)} episodes...")

    episodes = []
    summaries = []

    for i, stay_id in enumerate(stay_ids):
        if (i + 1) % 1000 == 0:
            print(f"   Processed {i+1}/{len(stay_ids)} episodes...")

        episode = builder.build_episode(stay_id)
        if episode is None:
            continue

        episodes.append(episode)

        # 收集摘要信息
        summary = {
            'episode_id': episode.episode_id,
            'stay_id': episode.stay_id,
            'age': episode.patient.age,
            'gender': episode.patient.gender,
            'n_vitals': len(episode.timeseries.vitals),
            'n_labs': len(episode.timeseries.labs),
            'n_notes': episode.clinical_text.n_notes,
            'n_patterns': episode.reasoning.n_patterns_detected,
            'n_alignments': episode.reasoning.n_alignments,
            'n_supportive': episode.reasoning.n_supportive,
            'n_contradictory': episode.reasoning.n_contradictory,
            'mortality': episode.labels.outcome.mortality,
            'prolonged_los': episode.labels.outcome.prolonged_los,
            'has_sepsis': episode.labels.has_sepsis,
            'has_aki': episode.labels.has_aki,
            'has_ards': episode.labels.has_ards,
            'quality_score': episode.metadata.data_quality_score
        }
        summaries.append(summary)

        # 保存单独的JSON文件
        if save_individual:
            json_path = output_dir / f"{episode.episode_id}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(episode.to_dict(), f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    # 创建摘要DataFrame
    summary_df = pd.DataFrame(summaries)

    # 保存摘要
    summary_path = output_dir / 'episodes_summary.csv'
    summary_df.to_csv(summary_path, index=False)

    # 打印统计
    print("\n" + "=" * 70)
    print("BUILD SUMMARY")
    print("=" * 70)
    print(f"Total episodes: {len(episodes)}")
    print(f"With clinical text: {(summary_df['n_notes'] > 0).sum()}")
    print(f"With patterns: {(summary_df['n_patterns'] > 0).sum()}")
    print(f"With alignments: {(summary_df['n_alignments'] > 0).sum()}")
    print(f"With LLM annotations: {(summary_df['n_supportive'] + summary_df['n_contradictory'] > 0).sum()}")

    print("\n[Labels Distribution]")
    print(f"Mortality: {summary_df['mortality'].sum()} ({summary_df['mortality'].mean()*100:.1f}%)")
    print(f"Prolonged LOS: {summary_df['prolonged_los'].sum()} ({summary_df['prolonged_los'].mean()*100:.1f}%)")
    print(f"Sepsis: {summary_df['has_sepsis'].sum()}")
    print(f"AKI: {summary_df['has_aki'].sum()}")
    print(f"ARDS: {summary_df['has_ards'].sum()}")

    print(f"\n[Quality Scores]")
    print(f"Mean: {summary_df['quality_score'].mean():.3f}")
    print(f"Median: {summary_df['quality_score'].median():.3f}")
    print(f"Min: {summary_df['quality_score'].min():.3f}")
    print(f"Max: {summary_df['quality_score'].max():.3f}")

    print(f"\nSaved to: {output_dir}/")
    print(f"   - {len(episodes)} individual JSON files")
    print(f"   - episodes_summary.csv")

    return episodes, summary_df


# ==========================================
# 生成样例数据集
# ==========================================

def generate_sample_dataset(n_samples: int = 100, output_dir: Optional[Path] = None):
    """生成样例Episode数据集"""
    print("\n" + "=" * 70)
    print(f"Generating Sample Dataset ({n_samples} episodes)")
    print("=" * 70)

    if output_dir is None:
        output_dir = ROOT_DIR / 'episodes_sample'

    episodes, summary = build_all_episodes(
        max_episodes=n_samples,
        output_dir=output_dir,
        save_individual=True
    )

    # 创建合并的JSON文件（方便查看）
    all_episodes_path = output_dir / 'all_episodes.json'
    with open(all_episodes_path, 'w', encoding='utf-8') as f:
        json.dump([e.to_dict() for e in episodes], f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nSample dataset generated: {output_dir}/")
    print(f"   - all_episodes.json (combined)")
    print(f"   - Individual JSON files")
    print(f"   - episodes_summary.csv")


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Build TIMELY-Bench Episodes')
    parser.add_argument('--mode', choices=['sample', 'full', 'core'], default='sample',
                        help='Build mode: sample (100 episodes), full (all episodes), or core (core dataset only)')
    parser.add_argument('--n', type=int, default=100,
                        help='Number of episodes for sample mode')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')

    args = parser.parse_args()

    if args.mode == 'sample':
        output_dir = Path(args.output) if args.output else ROOT_DIR / 'episodes_sample'
        generate_sample_dataset(n_samples=args.n, output_dir=output_dir)
    elif args.mode == 'core':
        # 核心模式：只处理core_episode_selection.csv中的患者
        core_file = ROOT_DIR / 'episodes_core' / 'core_episode_selection.csv'
        if core_file.exists():
            core_df = pd.read_csv(core_file)
            core_stay_ids = core_df['stay_id'].astype(int).tolist()
            print(f"Loading {len(core_stay_ids)} core episodes from {core_file}")
            output_dir = Path(args.output) if args.output else ROOT_DIR / 'episodes_core'
            build_all_episodes(output_dir=output_dir, core_stay_ids=core_stay_ids)
        else:
            print(f"Core episode selection file not found: {core_file}")
            print("   Please run build_core_dataset.py first")
    else:
        output_dir = Path(args.output) if args.output else OUTPUT_DIR
        build_all_episodes(output_dir=output_dir)

    print("\nEpisode building complete!")
