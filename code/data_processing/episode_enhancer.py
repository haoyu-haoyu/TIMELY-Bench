"""
Episode Enhancer - 深度增强Episode数据包

整合三个核心任务：
1. 深度时序-文本对齐 (Temporal-Textual Alignment)
2. 条件图构建 (Condition Graph Implementation)
3. LLM辅助增强 (Reasoning Artefacts Enhancement)

设计原则：
- 体现"对齐"的深度，而非简单拼接
- 基于临床指南构建知识约束
- 验证数字与文本的一致性
"""

import pandas as pd
import numpy as np
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict

# ==========================================
# 配置 - 修复路径问题，使用config.py中的正确路径
# ==========================================

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加code目录到路径

from config import (
    ROOT_DIR, RAW_DATA_DIR, PROCESSED_DIR,
    TIMESERIES_FILE, NOTE_TIME_FILE, COHORT_FILE,
    PATTERN_DETECTION_DIR
)

# 模式检测文件
PATTERNS_FILE = PATTERN_DETECTION_DIR / 'detected_patterns_24h.csv'

# 笔记类型优先级（越小越优先）
NOTE_TYPE_PRIORITY = {
    'Nursing': 1,           # 实时体征变化
    'Nursing/other': 2,
    'Physician': 3,         # 医生评估
    'Respiratory': 4,       # 呼吸治疗
    'Radiology': 5,         # 影像报告
    'ECG': 6,
    'Discharge summary': 10,  # 出院总结（对齐到最后）
}

# ==========================================
# 任务一：深度时序-文本对齐
# ==========================================

@dataclass
class NoteSpan:
    """笔记片段 - 对应特定时间窗口"""
    note_id: str
    note_type: str
    chart_hour: float           # 相对ICU入院的小时数
    span_start_hour: float      # 片段对应的开始时间
    span_end_hour: float        # 片段对应的结束时间
    text: str
    keywords: List[str] = field(default_factory=list)
    relevance_score: float = 0.0


class TemporalTextualAligner:
    """深度时序-文本对齐器"""

    # 时间相关关键词模式
    TIME_PATTERNS = [
        (r'this morning', -12, 0),
        (r'overnight', -12, 0),
        (r'last night', -12, -6),
        (r'earlier today', -6, 0),
        (r'currently', -1, 1),
        (r'at this time', -1, 1),
        (r'now', -1, 1),
        (r'past (\d+) hours?', None, None),  # 动态计算
        (r'since admission', 0, None),
    ]

    # 临床事件关键词（用于切片）
    CLINICAL_KEYWORDS = {
        'vitals': ['blood pressure', 'bp', 'heart rate', 'hr', 'temperature',
                   'temp', 'respiratory rate', 'rr', 'oxygen', 'spo2', 'o2 sat'],
        'labs': ['creatinine', 'lactate', 'potassium', 'sodium', 'wbc',
                 'hemoglobin', 'platelet', 'bilirubin', 'glucose'],
        'interventions': ['intubated', 'extubated', 'vasopressor', 'dialysis',
                         'transfusion', 'antibiotics', 'sedation'],
        'assessments': ['stable', 'unstable', 'improved', 'deteriorating',
                       'critical', 'sepsis', 'shock', 'aki', 'ards'],
    }

    def __init__(self):
        self.notes_df = None
        self.cohort_df = None

    def load_data(self):
        """加载笔记和队列数据"""
        if NOTE_TIME_FILE.exists():
            self.notes_df = pd.read_csv(NOTE_TIME_FILE)
            # 检测并标准化列名
            if 'stay_id' in self.notes_df.columns:
                self.notes_df['stay_id'] = self.notes_df['stay_id'].astype(int)

            # 标准化笔记文本列
            if 'radiology_text' in self.notes_df.columns:
                self.notes_df['text'] = self.notes_df['radiology_text']
                self.notes_df['note_type'] = 'Radiology'  # 添加默认类型
            elif 'text' not in self.notes_df.columns and 'note_text' in self.notes_df.columns:
                self.notes_df['text'] = self.notes_df['note_text']

            # 标准化时间列
            if 'hour_offset' in self.notes_df.columns and 'chart_hour' not in self.notes_df.columns:
                self.notes_df['chart_hour'] = self.notes_df['hour_offset']

            # 确保有note_type列
            if 'note_type' not in self.notes_df.columns:
                if 'category' in self.notes_df.columns:
                    self.notes_df['note_type'] = self.notes_df['category']
                else:
                    self.notes_df['note_type'] = 'Unknown'

            print(f"   Loaded {len(self.notes_df)} notes")
            print(f"   Columns: {list(self.notes_df.columns)}")
        else:
            print(f"   note_time.csv not found")
            return False

        if COHORT_FILE.exists():
            self.cohort_df = pd.read_csv(COHORT_FILE)
            self.cohort_df['stay_id'] = self.cohort_df['stay_id'].astype(int)
            # 转换时间列
            if 'intime' in self.cohort_df.columns:
                self.cohort_df['intime'] = pd.to_datetime(self.cohort_df['intime'])

        return True

    def calculate_relative_hours(self, stay_id: int, charttime: str) -> Optional[float]:
        """
        动作A：计算相对偏移量
        将笔记的charttime转换为相对于ICU入院时间的小时数
        """
        if self.cohort_df is None:
            return None

        patient = self.cohort_df[self.cohort_df['stay_id'] == stay_id]
        if len(patient) == 0:
            return None

        intime = patient.iloc[0]['intime']
        if pd.isna(intime):
            return None

        try:
            chart_dt = pd.to_datetime(charttime)
            delta = chart_dt - intime
            return delta.total_seconds() / 3600  # 转换为小时
        except:
            return None

    def filter_notes_by_type(self, notes: pd.DataFrame,
                             max_notes: int = 20) -> pd.DataFrame:
        """
        动作B：按类型过滤笔记
        优先选择Nursing Notes和Physician Notes
        """
        if len(notes) == 0:
            return notes

        # 添加优先级列
        notes = notes.copy()
        notes['priority'] = notes['note_type'].map(
            lambda x: NOTE_TYPE_PRIORITY.get(x, 99)
        )

        # 按优先级和时间排序
        notes = notes.sort_values(['priority', 'chart_hour'])

        # 确保包含多种类型
        selected = []
        type_counts = defaultdict(int)
        max_per_type = max(3, max_notes // 4)

        for _, row in notes.iterrows():
            note_type = row['note_type']
            if type_counts[note_type] < max_per_type:
                selected.append(row)
                type_counts[note_type] += 1

            if len(selected) >= max_notes:
                break

        return pd.DataFrame(selected)

    def slice_note_into_spans(self, note_text: str, chart_hour: float,
                              note_id: str, note_type: str) -> List[NoteSpan]:
        """
        动作C：文本切片 (Note Spans)
        将长笔记切分为小片段，对应特定时间窗口
        """
        spans = []

        if not note_text or len(note_text) < 50:
            # 短文本不需要切片
            return [NoteSpan(
                note_id=note_id,
                note_type=note_type,
                chart_hour=chart_hour,
                span_start_hour=max(0, chart_hour - 2),
                span_end_hour=chart_hour + 1,
                text=note_text,
                keywords=self._extract_keywords(note_text)
            )]

        # 按段落或句子切分
        paragraphs = self._split_into_paragraphs(note_text)

        for i, para in enumerate(paragraphs):
            if len(para.strip()) < 20:
                continue

            # 估计这段话对应的时间范围
            time_range = self._estimate_time_range(para, chart_hour, i, len(paragraphs))
            keywords = self._extract_keywords(para)

            # 计算相关性分数
            relevance = self._calculate_relevance(para, keywords)

            spans.append(NoteSpan(
                note_id=f"{note_id}_span{i}",
                note_type=note_type,
                chart_hour=chart_hour,
                span_start_hour=time_range[0],
                span_end_hour=time_range[1],
                text=para,
                keywords=keywords,
                relevance_score=relevance
            ))

        return spans

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """将文本分割为段落"""
        # 按双换行或特定标记分割
        paragraphs = re.split(r'\n\n|\n(?=[A-Z])|(?<=\.)\s+(?=[A-Z])', text)

        # 合并过短的段落
        merged = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) < 300:
                current += " " + para
            else:
                if current:
                    merged.append(current.strip())
                current = para
        if current:
            merged.append(current.strip())

        return merged

    def _estimate_time_range(self, text: str, chart_hour: float,
                             para_index: int, total_paras: int) -> Tuple[float, float]:
        """估计段落对应的时间范围"""
        text_lower = text.lower()

        # 检查时间相关词
        for pattern, start_offset, end_offset in self.TIME_PATTERNS:
            if re.search(pattern, text_lower):
                if start_offset is not None:
                    return (max(0, chart_hour + start_offset),
                            chart_hour + (end_offset or 0))

        # 根据段落位置估计
        # 假设笔记按时间顺序撰写
        time_span = 6  # 假设笔记覆盖最近6小时
        para_duration = time_span / max(total_paras, 1)

        start = max(0, chart_hour - time_span + para_index * para_duration)
        end = start + para_duration

        return (start, end)

    def _extract_keywords(self, text: str) -> List[str]:
        """提取临床关键词"""
        text_lower = text.lower()
        found = []

        for category, keywords in self.CLINICAL_KEYWORDS.items():
            for kw in keywords:
                if kw in text_lower:
                    found.append(kw)

        return found

    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """计算临床相关性分数"""
        score = 0.0

        # 关键词加分
        score += len(keywords) * 0.1

        # 数值加分（包含具体数字通常更有价值）
        numbers = re.findall(r'\d+\.?\d*', text)
        score += min(len(numbers) * 0.05, 0.3)

        # 长度适中加分
        if 100 < len(text) < 500:
            score += 0.2

        return min(score, 1.0)

    def align_notes_for_patient(self, stay_id: int,
                                 observation_window: int = 24) -> List[NoteSpan]:
        """为单个患者生成对齐的笔记片段"""
        if self.notes_df is None:
            return []

        # 获取患者的笔记
        patient_notes = self.notes_df[self.notes_df['stay_id'] == stay_id].copy()

        if len(patient_notes) == 0:
            return []

        # 计算相对时间
        if 'charttime' in patient_notes.columns:
            patient_notes['chart_hour'] = patient_notes['charttime'].apply(
                lambda x: self.calculate_relative_hours(stay_id, x)
            )
        elif 'chart_hour' in patient_notes.columns:
            pass  # 已有
        else:
            return []

        # 过滤观察窗口内的笔记
        patient_notes = patient_notes[
            (patient_notes['chart_hour'] >= 0) &
            (patient_notes['chart_hour'] < observation_window)
        ]

        # 按类型过滤
        patient_notes = self.filter_notes_by_type(patient_notes)

        # 生成切片
        all_spans = []
        for _, row in patient_notes.iterrows():
            text = row.get('text', row.get('note_text', ''))
            note_id = str(row.get('note_id', ''))
            note_type = str(row.get('note_type', row.get('category', '')))
            chart_hour = float(row.get('chart_hour', 0))

            spans = self.slice_note_into_spans(text, chart_hour, note_id, note_type)
            all_spans.extend(spans)

        # 按时间排序
        all_spans.sort(key=lambda x: x.span_start_hour)

        return all_spans


# ==========================================
# 任务二：条件图构建 (Condition Graphs)
# ==========================================

@dataclass
class ClinicalRule:
    """临床规则定义"""
    name: str
    description: str
    source: str  # 临床指南来源
    pmid: Optional[str] = None

    # 触发条件
    required_patterns: List[str] = field(default_factory=list)  # 必须全部存在
    optional_patterns: List[str] = field(default_factory=list)  # 至少一个存在
    time_window_hours: int = 6  # 模式必须在此时间窗内

    # 产生的节点
    output_condition: str = ""
    severity: str = "moderate"


# Sepsis-3 诊断规则
SEPSIS3_RULES = [
    ClinicalRule(
        name="tissue_hypoperfusion",
        description="组织低灌注：低血压 + 高乳酸",
        source="Sepsis-3 (Singer et al., JAMA 2016)",
        pmid="26903338",
        required_patterns=["hypotension", "lactate_elevated"],
        time_window_hours=6,
        output_condition="Tissue Hypoperfusion",
        severity="severe"
    ),
    ClinicalRule(
        name="septic_shock_criteria",
        description="感染性休克标准：低血压 + 乳酸>2 + 需要血管活性药物",
        source="Sepsis-3 (Singer et al., JAMA 2016)",
        pmid="26903338",
        required_patterns=["map_low", "lactate_elevated"],
        optional_patterns=["hypotension"],
        time_window_hours=6,
        output_condition="Septic Shock",
        severity="critical"
    ),
    ClinicalRule(
        name="sirs_criteria",
        description="SIRS标准：≥2项（发热/低温、心动过速、呼吸急促、白细胞异常）",
        source="SIRS Criteria (Bone et al., Chest 1992)",
        pmid="1303622",
        required_patterns=[],  # 需要特殊处理 - 4选2
        optional_patterns=["fever", "hypothermia", "tachycardia", "tachypnea",
                          "leukocytosis", "leukopenia"],
        time_window_hours=24,
        output_condition="SIRS",
        severity="moderate"
    ),
    ClinicalRule(
        name="respiratory_failure",
        description="呼吸衰竭：严重低氧 + 呼吸窘迫",
        source="Berlin ARDS Definition",
        pmid="22797452",
        required_patterns=["spo2_low"],
        optional_patterns=["respiratory_distress", "hypoxemia"],
        time_window_hours=6,
        output_condition="Respiratory Failure",
        severity="severe"
    ),
]

# KDIGO AKI规则
KDIGO_RULES = [
    ClinicalRule(
        name="aki_with_hyperkalemia",
        description="AKI伴高钾血症（危险组合）",
        source="KDIGO AKI Guidelines 2012",
        pmid="25018915",
        required_patterns=["creatinine_elevated", "hyperkalemia"],
        time_window_hours=12,
        output_condition="AKI with Hyperkalemia",
        severity="critical"
    ),
    ClinicalRule(
        name="aki_metabolic_derangement",
        description="AKI伴代谢紊乱",
        source="KDIGO AKI Guidelines 2012",
        pmid="25018915",
        required_patterns=["creatinine_elevated", "metabolic_acidosis"],
        time_window_hours=12,
        output_condition="AKI with Metabolic Acidosis",
        severity="severe"
    ),
]

# 多器官功能障碍规则
MODS_RULES = [
    ClinicalRule(
        name="cardiovascular_respiratory",
        description="心血管+呼吸系统同时受累",
        source="SOFA Score Components",
        pmid="26903338",
        required_patterns=["hypotension", "hypoxemia"],
        time_window_hours=6,
        output_condition="Cardiorespiratory Failure",
        severity="critical"
    ),
    ClinicalRule(
        name="sepsis_aki_cascade",
        description="Sepsis导致AKI的级联反应",
        source="Clinical Knowledge",
        required_patterns=["lactate_elevated", "creatinine_elevated"],
        optional_patterns=["hypotension", "oliguria"],
        time_window_hours=12,
        output_condition="Sepsis-induced AKI",
        severity="severe"
    ),
]

# ==========================================
# 疾病级联规则 (Disease Cascade Rules)
# ==========================================

CASCADE_RULES = [
    # === Sepsis诱发的器官损伤级联 ===
    ClinicalRule(
        name="sepsis_induced_aki_hemodynamic",
        description="Sepsis诱发AKI - 血流动力学机制：脓毒症休克导致肾灌注不足",
        source="KDIGO 2021 Clinical Practice Guideline for AKI",
        pmid="34217646",
        required_patterns=["map_low", "lactate_elevated", "creatinine_elevated"],
        optional_patterns=["hypotension", "oliguria", "bun_elevated"],
        time_window_hours=12,
        output_condition="Sepsis-induced AKI (Hemodynamic)",
        severity="critical"
    ),
    ClinicalRule(
        name="sepsis_induced_ards",
        description="Sepsis诱发ARDS：感染导致肺血管内皮损伤和低氧血症",
        source="Berlin ARDS Definition & Sepsis-3 Guidelines",
        pmid="22797452",
        required_patterns=["lactate_elevated", "hypoxemia"],
        optional_patterns=["respiratory_distress", "spo2_low", "tachycardia"],
        time_window_hours=24,
        output_condition="Sepsis-induced ARDS",
        severity="critical"
    ),

    # === 心血管级联 ===
    ClinicalRule(
        name="sepsis_cardiovascular_cascade",
        description="脓毒症心血管级联：血管扩张 → 低灌注 → 代谢性酸中毒",
        source="Surviving Sepsis Campaign 2021",
        pmid="33131646",
        required_patterns=["hypotension", "lactate_elevated"],
        optional_patterns=["map_low", "severe_tachycardia", "metabolic_acidosis"],
        time_window_hours=6,
        output_condition="Septic Cardiovascular Failure",
        severity="critical"
    ),
    ClinicalRule(
        name="cardiogenic_shock_cascade",
        description="心源性休克级联：心输出量下降 → 低灌注 → 乳酸酸中毒",
        source="ESC Guidelines on Acute Heart Failure 2021",
        pmid="34447992",
        required_patterns=["hypotension", "lactate_elevated", "tachycardia"],
        optional_patterns=["spo2_low", "oliguria"],
        time_window_hours=6,
        output_condition="Cardiogenic Shock Cascade",
        severity="critical"
    ),

    # === AKI级联并发症 ===
    ClinicalRule(
        name="aki_uremic_cascade",
        description="AKI尿毒症级联：肾功能下降 → 电解质紊乱 → 代谢性酸中毒",
        source="KDIGO AKI Guidelines 2012",
        pmid="25018915",
        required_patterns=["creatinine_elevated", "bun_elevated"],
        optional_patterns=["hyperkalemia", "metabolic_acidosis", "oliguria"],
        time_window_hours=24,
        output_condition="AKI with Uremic Complications",
        severity="severe"
    ),
    ClinicalRule(
        name="aki_fluid_overload",
        description="AKI液体过载：少尿 → 水钠潴留 → 呼吸功能受累",
        source="KDIGO AKI Guidelines 2012",
        pmid="25018915",
        required_patterns=["oliguria", "spo2_low"],
        optional_patterns=["creatinine_elevated", "respiratory_distress"],
        time_window_hours=12,
        output_condition="AKI with Fluid Overload",
        severity="severe"
    ),

    # === 呼吸-代谢级联 ===
    ClinicalRule(
        name="respiratory_metabolic_cascade",
        description="呼吸衰竭代谢级联：低氧 → 组织缺氧 → 乳酸升高",
        source="Berlin ARDS Definition",
        pmid="22797452",
        required_patterns=["hypoxemia", "lactate_elevated"],
        optional_patterns=["respiratory_distress", "tachycardia", "metabolic_acidosis"],
        time_window_hours=12,
        output_condition="Hypoxic-Metabolic Cascade",
        severity="severe"
    ),

    # === 多器官功能障碍综合征 (MODS) ===
    ClinicalRule(
        name="mods_two_organ",
        description="MODS双器官：至少两个器官系统同时受累",
        source="SOFA Score Criteria",
        pmid="26903338",
        required_patterns=["lactate_elevated"],
        optional_patterns=["creatinine_elevated", "hypoxemia", "hypotension",
                          "thrombocytopenia", "hyperbilirubinemia"],
        time_window_hours=24,
        output_condition="MODS - Two Organ Dysfunction",
        severity="critical"
    ),
    ClinicalRule(
        name="mods_cardiovascular_renal",
        description="MODS心肾综合征：心血管衰竭 + 肾功能不全",
        source="Cardiorenal Syndrome Classification (Ronco et al.)",
        pmid="18802884",
        required_patterns=["hypotension", "creatinine_elevated"],
        optional_patterns=["oliguria", "lactate_elevated", "tachycardia"],
        time_window_hours=12,
        output_condition="Cardiorenal Syndrome",
        severity="critical"
    ),
]

# 合并所有临床规则
ALL_CLINICAL_RULES = SEPSIS3_RULES + KDIGO_RULES + MODS_RULES + CASCADE_RULES


@dataclass
class GraphNode:
    """图节点"""
    id: str
    name: str
    level: str  # 'raw', 'pattern', 'condition'
    onset_hour: Optional[int] = None
    value: Optional[float] = None
    severity: str = "moderate"
    source: str = ""


@dataclass
class GraphEdge:
    """图边"""
    source_id: str
    target_id: str
    relationship: str  # 'triggers', 'indicates', 'causes', 'concurrent_with'
    confidence: float = 1.0
    time_delta_hours: Optional[int] = None
    clinical_rule: Optional[str] = None


class ConditionGraphBuilder:
    """条件图构建器 - 基于临床规则"""

    def __init__(self):
        self.rules = ALL_CLINICAL_RULES

    def build_graph(self, detected_patterns: List[Dict],
                    observation_window: int = 24) -> Tuple[List[GraphNode], List[GraphEdge]]:
        """
        构建三层条件图：
        - 底层：原始体征 (Raw)
        - 中层：临床模式 (Patterns)
        - 顶层：诊断条件 (Conditions)
        """
        nodes = []
        edges = []

        # === 底层 + 中层：Pattern节点 ===
        pattern_by_name = defaultdict(list)
        for p in detected_patterns:
            pattern_name = p.get('pattern_name', '')
            pattern_by_name[pattern_name].append(p)

            # 创建Pattern节点
            node_id = f"pattern_{pattern_name}_{p.get('detection_hour', 0)}"
            nodes.append(GraphNode(
                id=node_id,
                name=pattern_name,
                level='pattern',
                onset_hour=p.get('detection_hour'),
                value=p.get('value'),
                severity=p.get('severity', 'moderate'),
                source=p.get('disease', '')
            ))

        # === 顶层：应用临床规则生成Condition节点 ===
        for rule in self.rules:
            triggered, trigger_patterns = self._check_rule(
                rule, pattern_by_name, observation_window
            )

            if triggered:
                # 创建Condition节点
                condition_id = f"condition_{rule.name}"
                onset_hour = self._get_earliest_onset(trigger_patterns)

                nodes.append(GraphNode(
                    id=condition_id,
                    name=rule.output_condition,
                    level='condition',
                    onset_hour=onset_hour,
                    severity=rule.severity,
                    source=rule.source
                ))

                # 创建边：Pattern -> Condition
                for p in trigger_patterns:
                    pattern_node_id = f"pattern_{p.get('pattern_name')}_{p.get('detection_hour')}"
                    edges.append(GraphEdge(
                        source_id=pattern_node_id,
                        target_id=condition_id,
                        relationship='indicates',
                        confidence=0.9,
                        clinical_rule=rule.name
                    ))

        # === 添加Condition之间的因果边 ===
        edges.extend(self._add_condition_relationships(nodes))

        return nodes, edges

    def _check_rule(self, rule: ClinicalRule,
                    pattern_by_name: Dict[str, List],
                    observation_window: int) -> Tuple[bool, List[Dict]]:
        """检查规则是否被触发"""

        # 特殊处理SIRS（4选2）
        if rule.name == "sirs_criteria":
            return self._check_sirs_rule(rule, pattern_by_name)

        trigger_patterns = []

        # 检查必需模式
        for pattern_name in rule.required_patterns:
            if pattern_name not in pattern_by_name:
                return False, []
            trigger_patterns.extend(pattern_by_name[pattern_name])

        # 检查可选模式（至少一个）
        if rule.optional_patterns:
            has_optional = False
            for pattern_name in rule.optional_patterns:
                if pattern_name in pattern_by_name:
                    has_optional = True
                    trigger_patterns.extend(pattern_by_name[pattern_name])

            if rule.required_patterns and not has_optional:
                pass  # 有必需模式时，可选模式不是必须的
            elif not rule.required_patterns and not has_optional:
                return False, []  # 没有必需模式时，必须有可选模式

        # 检查时间窗口
        if trigger_patterns and rule.time_window_hours:
            hours = [p.get('detection_hour', 0) for p in trigger_patterns]
            if max(hours) - min(hours) > rule.time_window_hours:
                return False, []

        return len(trigger_patterns) > 0, trigger_patterns

    def _check_sirs_rule(self, rule: ClinicalRule,
                         pattern_by_name: Dict[str, List]) -> Tuple[bool, List[Dict]]:
        """检查SIRS规则（4选2）"""
        sirs_patterns = ['fever', 'hypothermia', 'tachycardia', 'tachypnea',
                        'leukocytosis', 'leukopenia']

        found_patterns = []
        found_categories = set()

        for pattern_name in sirs_patterns:
            if pattern_name in pattern_by_name:
                # 体温异常算一类
                if pattern_name in ['fever', 'hypothermia']:
                    if 'temperature' not in found_categories:
                        found_categories.add('temperature')
                        found_patterns.extend(pattern_by_name[pattern_name])
                # WBC异常算一类
                elif pattern_name in ['leukocytosis', 'leukopenia']:
                    if 'wbc' not in found_categories:
                        found_categories.add('wbc')
                        found_patterns.extend(pattern_by_name[pattern_name])
                else:
                    found_categories.add(pattern_name)
                    found_patterns.extend(pattern_by_name[pattern_name])

        # SIRS需要≥2个类别
        return len(found_categories) >= 2, found_patterns

    def _get_earliest_onset(self, patterns: List[Dict]) -> Optional[int]:
        """获取最早发病时间"""
        hours = [p.get('detection_hour') for p in patterns if p.get('detection_hour') is not None]
        return min(hours) if hours else None

    def _add_condition_relationships(self, nodes: List[GraphNode]) -> List[GraphEdge]:
        """添加Condition之间的因果关系边"""
        edges = []
        condition_nodes = [n for n in nodes if n.level == 'condition']

        # 已知的因果关系（扩展版：包含完整级联关系）
        causal_relations = [
            # === 基础脓毒症进展链 ===
            ('SIRS', 'Tissue Hypoperfusion', 'progresses_to'),
            ('Tissue Hypoperfusion', 'Septic Shock', 'progresses_to'),

            # === Sepsis诱发的器官损伤级联 ===
            ('Septic Shock', 'Sepsis-induced AKI', 'cascade_causes'),
            ('Septic Shock', 'Sepsis-induced AKI (Hemodynamic)', 'cascade_causes'),
            ('Septic Shock', 'Sepsis-induced ARDS', 'cascade_causes'),
            ('Septic Shock', 'Septic Cardiovascular Failure', 'cascade_causes'),

            # === 心血管级联 ===
            ('Septic Cardiovascular Failure', 'Sepsis-induced AKI (Hemodynamic)', 'cascade_causes'),
            ('Septic Cardiovascular Failure', 'Cardiogenic Shock Cascade', 'contributes_to'),
            ('Cardiogenic Shock Cascade', 'Cardiorenal Syndrome', 'cascade_causes'),

            # === 呼吸系统级联 ===
            ('Respiratory Failure', 'Cardiorespiratory Failure', 'contributes_to'),
            ('Sepsis-induced ARDS', 'Cardiorespiratory Failure', 'contributes_to'),
            ('Respiratory Failure', 'Hypoxic-Metabolic Cascade', 'cascade_causes'),

            # === AKI级联并发症 ===
            ('Sepsis-induced AKI', 'AKI with Hyperkalemia', 'complicates'),
            ('Sepsis-induced AKI', 'AKI with Metabolic Acidosis', 'complicates'),
            ('Sepsis-induced AKI (Hemodynamic)', 'AKI with Uremic Complications', 'progresses_to'),
            ('AKI with Uremic Complications', 'AKI with Fluid Overload', 'contributes_to'),
            ('AKI with Fluid Overload', 'Cardiorespiratory Failure', 'contributes_to'),

            # === 多器官功能障碍综合征 (MODS) 级联 ===
            ('Septic Shock', 'MODS - Two Organ Dysfunction', 'cascade_causes'),
            ('Cardiorenal Syndrome', 'MODS - Two Organ Dysfunction', 'contributes_to'),
            ('Cardiorespiratory Failure', 'MODS - Two Organ Dysfunction', 'contributes_to'),

            # === 代谢级联 ===
            ('Hypoxic-Metabolic Cascade', 'MODS - Two Organ Dysfunction', 'contributes_to'),
        ]

        name_to_node = {n.name: n for n in condition_nodes}

        for source_name, target_name, relation in causal_relations:
            if source_name in name_to_node and target_name in name_to_node:
                source_node = name_to_node[source_name]
                target_node = name_to_node[target_name]

                time_delta = None
                if source_node.onset_hour and target_node.onset_hour:
                    time_delta = target_node.onset_hour - source_node.onset_hour

                # 级联关系使用更高置信度
                confidence = 0.9 if relation == 'cascade_causes' else 0.8

                edges.append(GraphEdge(
                    source_id=source_node.id,
                    target_id=target_node.id,
                    relationship=relation,
                    confidence=confidence,
                    time_delta_hours=time_delta
                ))

        return edges


# ==========================================
# 任务三：LLM辅助增强
# ==========================================

@dataclass
class ConsistencyCheck:
    """一致性检查结果"""
    pattern_name: str
    pattern_hour: int
    pattern_value: float

    note_span_id: str
    note_text: str

    consistency: str  # 'SUPPORTIVE', 'CONTRADICTORY', 'NEUTRAL'
    confidence: float
    reasoning: str

    # 新增：时间加权相关字段
    temporal_weight: float = 1.0
    relevance_weight: float = 1.0
    final_score: float = 0.0


def temporal_weight(pattern_hour: float, span_start: float, span_end: float) -> float:
    """
    计算时间接近度加权

    核心逻辑：
    - 如果pattern时间在span时间窗口内，权重为1.0
    - 否则，根据距离衰减（每小时衰减15%）
    - 最低权重为0.1（避免完全忽略）

    这解决了用户指出的问题：Hour 0的笔记不应该判断Hour 23的pattern
    """
    if span_start <= pattern_hour <= span_end:
        # 完全匹配时间窗口
        return 1.0

    # 计算与时间窗口的距离
    distance = min(abs(pattern_hour - span_start), abs(pattern_hour - span_end))

    # 距离衰减：每小时衰减15%，最低0.1
    weight = max(0.1, 1.0 - distance * 0.15)

    return round(weight, 3)


class LLMEnhancer:
    """LLM辅助增强器"""

    # 文本中的数值模式
    VALUE_PATTERNS = {
        'heart_rate': [r'hr\s*(?:of\s*)?(\d+)', r'heart rate\s*(?:of\s*)?(\d+)'],
        'blood_pressure': [r'bp\s*(\d+)/(\d+)', r'blood pressure\s*(\d+)/(\d+)'],
        'temperature': [r'temp\s*(\d+\.?\d*)', r'temperature\s*(\d+\.?\d*)'],
        'spo2': [r'spo2\s*(\d+)', r'o2\s*sat\s*(\d+)', r'saturation\s*(\d+)'],
        'creatinine': [r'creatinine\s*(\d+\.?\d*)', r'cr\s*(\d+\.?\d*)'],
    }

    # 描述性词语映射（扩展版）
    DESCRIPTIVE_PATTERNS = {
        # 血流动力学
        'stable': {'vital_stability': 'stable'},
        'unstable': {'vital_stability': 'unstable'},
        'hemodynamically stable': {'hemodynamic': 'stable'},
        'hemodynamically unstable': {'hemodynamic': 'unstable'},
        # 趋势
        'elevated': {'trend': 'high'},
        'low': {'trend': 'low'},
        'normal': {'trend': 'normal'},
        'improved': {'trend': 'improving'},
        'worsening': {'trend': 'worsening'},
        'increasing': {'trend': 'increasing'},
        'decreasing': {'trend': 'decreasing'},
        # 血压
        'hypotensive': {'blood_pressure': 'low'},
        'hypertensive': {'blood_pressure': 'high'},
        'normotensive': {'blood_pressure': 'normal'},
        # 心率
        'tachycardic': {'heart_rate': 'high'},
        'bradycardic': {'heart_rate': 'low'},
        # 体温
        'febrile': {'temperature': 'high'},
        'afebrile': {'temperature': 'normal'},
        # 呼吸（放射学报告常用）
        'intubated': {'respiratory_support': 'intubated'},
        'extubated': {'respiratory_support': 'extubated'},
        'respiratory distress': {'respiratory_status': 'distress'},
        'respiratory failure': {'respiratory_status': 'failure'},
        'pulmonary edema': {'lung_findings': 'edema'},
        'pleural effusion': {'lung_findings': 'effusion'},
        'pneumonia': {'lung_findings': 'pneumonia'},
        'consolidation': {'lung_findings': 'consolidation'},
        'infiltrate': {'lung_findings': 'infiltrate'},
        'atelectasis': {'lung_findings': 'atelectasis'},
        # 肾功能
        'oliguria': {'urine_output': 'low'},
        'anuria': {'urine_output': 'absent'},
        'polyuria': {'urine_output': 'high'},
        # 意识
        'alert': {'mental_status': 'alert'},
        'oriented': {'mental_status': 'oriented'},
        'confused': {'mental_status': 'confused'},
        'sedated': {'mental_status': 'sedated'},
    }

    def extract_clinical_events(self, text: str) -> List[Dict]:
        """
        动作A：从文本中提取临床事件
        """
        events = []
        text_lower = text.lower()

        # 提取数值
        for feature, patterns in self.VALUE_PATTERNS.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower)
                for match in matches:
                    if isinstance(match, tuple):
                        value = match[0]
                    else:
                        value = match
                    try:
                        events.append({
                            'type': 'numeric',
                            'feature': feature,
                            'value': float(value),
                            'source': 'text_extraction'
                        })
                    except:
                        pass

        # 提取描述性词语
        for keyword, attributes in self.DESCRIPTIVE_PATTERNS.items():
            if keyword in text_lower:
                for attr_name, attr_value in attributes.items():
                    events.append({
                        'type': 'descriptive',
                        'feature': attr_name,
                        'value': attr_value,
                        'source': 'text_extraction'
                    })

        return events

    def check_consistency(self, pattern: Dict, note_span: NoteSpan) -> ConsistencyCheck:
        """
        动作B：验证数字和文本的一致性（集成时间加权）

        改进：使用temporal_weight函数计算时间接近度，
        避免用Hour 0的笔记判断Hour 23的pattern这类误判
        """
        pattern_name = pattern.get('pattern_name', '')
        pattern_value = pattern.get('value', 0)
        pattern_hour = pattern.get('detection_hour', 0)

        # 计算时间加权
        t_weight = temporal_weight(
            pattern_hour,
            note_span.span_start_hour,
            note_span.span_end_hour
        )

        # 如果时间距离太远（权重<0.3），直接返回NEUTRAL
        if t_weight < 0.3:
            return ConsistencyCheck(
                pattern_name=pattern_name,
                pattern_hour=pattern_hour,
                pattern_value=pattern_value,
                note_span_id=note_span.note_id,
                note_text=note_span.text[:200],
                consistency='NEUTRAL',
                confidence=0.3,
                reasoning=f"Time distance too large (weight={t_weight})",
                temporal_weight=t_weight,
                relevance_weight=note_span.relevance_score,
                final_score=0.0
            )

        # 提取文本中的临床事件
        text_events = self.extract_clinical_events(note_span.text)

        # 检查一致性
        consistency, raw_confidence, reasoning = self._evaluate_consistency(
            pattern_name, pattern_value, text_events, note_span.text
        )

        # 计算最终分数：raw_confidence * temporal_weight * relevance_weight
        relevance_weight = note_span.relevance_score if note_span.relevance_score > 0 else 0.5
        final_score = raw_confidence * t_weight * relevance_weight

        # 如果最终分数太低，降级为NEUTRAL
        if consistency != 'NEUTRAL' and final_score < 0.4:
            consistency = 'NEUTRAL'
            reasoning += f" (downgraded: final_score={final_score:.2f} < 0.4)"

        return ConsistencyCheck(
            pattern_name=pattern_name,
            pattern_hour=pattern_hour,
            pattern_value=pattern_value,
            note_span_id=note_span.note_id,
            note_text=note_span.text[:200],
            consistency=consistency,
            confidence=raw_confidence,
            reasoning=reasoning,
            temporal_weight=t_weight,
            relevance_weight=relevance_weight,
            final_score=round(final_score, 3)
        )

    def _evaluate_consistency(self, pattern_name: str, pattern_value: float,
                              text_events: List[Dict], text: str) -> Tuple[str, float, str]:
        """评估一致性"""
        text_lower = text.lower()

        # 扩展的pattern到feature映射
        pattern_to_feature = {
            # 心率相关
            'tachycardia': 'heart_rate',
            'severe_tachycardia': 'heart_rate',
            'bradycardia': 'heart_rate',
            # 血压相关
            'hypotension': 'blood_pressure',
            'map_low': 'blood_pressure',
            'hypertensive_crisis': 'blood_pressure',
            # 体温相关
            'fever': 'temperature',
            'hypothermia': 'temperature',
            # 氧饱和度相关
            'hypoxemia': 'spo2',
            'spo2_low': 'spo2',
            # 肾功能相关
            'creatinine_elevated': 'creatinine',
            'oliguria': 'urineoutput',
            # 代谢相关
            'metabolic_acidosis': 'bicarbonate',
            'hyperkalemia': 'potassium',
            'hypokalemia': 'potassium',
            # 血液相关
            'anemia': 'hemoglobin',
            'severe_anemia': 'hemoglobin',
            'leukocytosis': 'wbc',
            'leukopenia': 'wbc',
            # 呼吸相关
            'tachypnea': 'resp_rate',
            # 乳酸相关
            'lactate_elevated': 'lactate',
        }

        feature = pattern_to_feature.get(pattern_name)
        if not feature:
            return 'NEUTRAL', 0.5, f"No mapping for pattern {pattern_name}"

        # 检查文本中提取的数值
        for event in text_events:
            if event['feature'] == feature and event['type'] == 'numeric':
                text_value = event['value']
                # 比较数值
                diff_pct = abs(text_value - pattern_value) / max(pattern_value, 1) * 100

                if diff_pct < 10:
                    return 'SUPPORTIVE', 0.9, f"Text value {text_value} matches pattern value {pattern_value}"
                elif diff_pct > 30:
                    return 'CONTRADICTORY', 0.8, f"Text value {text_value} differs significantly from {pattern_value}"

        # 扩展的描述性词语检查
        descriptive_checks = [
            # 心率
            ('tachycardia', ['tachycardic', 'rapid heart rate', 'elevated hr', 'hr elevated'], 'SUPPORTIVE'),
            ('tachycardia', ['bradycardic', 'slow heart rate', 'hr normal', 'heart rate normal'], 'CONTRADICTORY'),
            ('severe_tachycardia', ['tachycardic', 'severe tachycardia', 'critically elevated hr'], 'SUPPORTIVE'),
            # 血压
            ('hypotension', ['hypotensive', 'low bp', 'bp low', 'blood pressure low'], 'SUPPORTIVE'),
            ('hypotension', ['stable bp', 'normotensive', 'bp stable', 'blood pressure stable'], 'CONTRADICTORY'),
            ('map_low', ['hypotensive', 'low map', 'mean arterial pressure low'], 'SUPPORTIVE'),
            # 氧饱和度
            ('hypoxemia', ['hypoxic', 'desaturating', 'low oxygen', 'o2 low', 'respiratory distress'], 'SUPPORTIVE'),
            ('hypoxemia', ['saturating well', 'o2 stable', 'spo2 normal'], 'CONTRADICTORY'),
            ('spo2_low', ['hypoxic', 'desaturating', 'oxygen desaturation', 'respiratory failure'], 'SUPPORTIVE'),
            ('spo2_low', ['saturating well', 'room air', 'adequate oxygenation'], 'CONTRADICTORY'),
            # 体温
            ('fever', ['febrile', 'fever', 'elevated temp', 'temp elevated'], 'SUPPORTIVE'),
            ('fever', ['afebrile', 'normothermic', 'temp normal'], 'CONTRADICTORY'),
            # 肾功能
            ('oliguria', ['oliguria', 'low urine output', 'decreased urine', 'uo low', 'anuria'], 'SUPPORTIVE'),
            ('oliguria', ['adequate urine', 'good urine output', 'uo adequate'], 'CONTRADICTORY'),
            # 代谢
            ('metabolic_acidosis', ['acidotic', 'acidosis', 'low bicarbonate', 'metabolic acidosis'], 'SUPPORTIVE'),
            ('metabolic_acidosis', ['normal ph', 'ph normal', 'bicarb normal'], 'CONTRADICTORY'),
            # 贫血
            ('anemia', ['anemic', 'low hemoglobin', 'hgb low', 'transfusion'], 'SUPPORTIVE'),
            ('anemia', ['hemoglobin stable', 'hgb stable'], 'CONTRADICTORY'),
            # 呼吸
            ('tachypnea', ['tachypneic', 'rapid breathing', 'rr elevated', 'respiratory distress'], 'SUPPORTIVE'),
            ('tachypnea', ['breathing comfortably', 'rr normal', 'resp rate normal'], 'CONTRADICTORY'),
            # 呼吸支持相关（放射学报告）
            ('tachypnea', ['intubated', 'on ventilator', 'mechanical ventilation'], 'SUPPORTIVE'),
            ('hypoxemia', ['intubated', 'pulmonary edema', 'pleural effusion'], 'SUPPORTIVE'),
            ('spo2_low', ['pulmonary edema', 'consolidation', 'infiltrate', 'pneumonia'], 'SUPPORTIVE'),
        ]

        for check_pattern, keywords, result in descriptive_checks:
            if pattern_name == check_pattern:
                for keyword in keywords:
                    if keyword in text_lower:
                        confidence = 0.85 if result == 'SUPPORTIVE' else 0.75
                        reasoning = f"Text contains '{keyword}' which {'supports' if result == 'SUPPORTIVE' else 'contradicts'} {pattern_name}"
                        return result, confidence, reasoning

        return 'NEUTRAL', 0.5, "No clear consistency signal found"


# ==========================================
# 主增强流程
# ==========================================

class EpisodeEnhancer:
    """Episode增强器 - 整合三个任务"""

    def __init__(self):
        self.aligner = TemporalTextualAligner()
        self.graph_builder = ConditionGraphBuilder()
        self.llm_enhancer = LLMEnhancer()

    def enhance_episode(self, episode_dict: Dict) -> Dict:
        """增强单个Episode"""
        stay_id = episode_dict.get('stay_id')
        observation_window = 24  # 观察窗口（小时）

        # 任务一：深度时序-文本对齐
        note_spans = self.aligner.align_notes_for_patient(stay_id)
        
        # === Bug3修复：时间轴隔离 ===
        # 严格过滤：仅保留chart_hour < 24的笔记作为特征
        aligned_spans = []
        evidence_context = []  # 存放出院小结等未来信息
        
        for span in note_spans:
            span_dict = {
                'note_id': span.note_id,
                'note_type': span.note_type,
                'chart_hour': span.chart_hour,
                'span_start_hour': span.span_start_hour,
                'span_end_hour': span.span_end_hour,
                'text': span.text,
                'keywords': span.keywords,
                'relevance_score': span.relevance_score
            }
            
            # 时间轴隔离：根据chart_hour分流
            if span.chart_hour < observation_window:
                # 24小时内的笔记 -> 作为模型特征
                aligned_spans.append(span_dict)
            else:
                # 24小时后的笔记（主要是出院小结）-> 作为结局验证证据
                span_dict['is_future_note'] = True
                evidence_context.append(span_dict)
        
        episode_dict['clinical_text']['aligned_spans'] = aligned_spans
        
        # 将出院小结移到labels.outcome.evidence_context
        if evidence_context:
            if 'labels' not in episode_dict:
                episode_dict['labels'] = {}
            if 'outcome' not in episode_dict['labels']:
                episode_dict['labels']['outcome'] = {}
            episode_dict['labels']['outcome']['evidence_context'] = evidence_context

        # 任务二：构建条件图
        patterns = episode_dict.get('reasoning', {}).get('detected_patterns', [])
        nodes, edges = self.graph_builder.build_graph(patterns)

        episode_dict['reasoning']['condition_graph'] = {
            'nodes': [
                {
                    'id': n.id,
                    'name': n.name,
                    'level': n.level,
                    'onset_hour': n.onset_hour,
                    'value': n.value,
                    'severity': n.severity,
                    'source': n.source
                }
                for n in nodes
            ],
            'edges': [
                {
                    'source_id': e.source_id,
                    'target_id': e.target_id,
                    'relationship': e.relationship,
                    'confidence': e.confidence,
                    'time_delta_hours': e.time_delta_hours,
                    'clinical_rule': e.clinical_rule
                }
                for e in edges
            ],
            'n_pattern_nodes': sum(1 for n in nodes if n.level == 'pattern'),
            'n_condition_nodes': sum(1 for n in nodes if n.level == 'condition'),
            'n_edges': len(edges)
        }

        # 任务三：一致性检查（改进版：使用时间加权）
        consistency_checks = []
        all_checks = []  # 保存所有检查用于统计

        for pattern in patterns[:15]:  # 增加检查数量
            for span in note_spans[:8]:
                check = self.llm_enhancer.check_consistency(pattern, span)
                all_checks.append(check)

                # 只保留非NEUTRAL且final_score >= 0.4的检查
                if check.consistency != 'NEUTRAL' and check.final_score >= 0.4:
                    consistency_checks.append({
                        'pattern_name': check.pattern_name,
                        'pattern_hour': check.pattern_hour,
                        'note_span_id': check.note_span_id,
                        'consistency': check.consistency,
                        'confidence': check.confidence,
                        'reasoning': check.reasoning,
                        # 新增字段
                        'temporal_weight': check.temporal_weight,
                        'relevance_weight': check.relevance_weight,
                        'final_score': check.final_score
                    })

        episode_dict['reasoning']['consistency_checks'] = consistency_checks
        
        # 统计信息 - 同时统计consistency_checks和pattern_annotations
        # 修复Bug: 之前只统计consistency_checks，现在改为统计pattern_annotations
        pattern_annotations = episode_dict.get('reasoning', {}).get('pattern_annotations', [])
        episode_dict['reasoning']['n_supportive'] = sum(
            1 for a in pattern_annotations 
            if a.get('annotation_category') == 'SUPPORTIVE'
        )
        episode_dict['reasoning']['n_contradictory'] = sum(
            1 for a in pattern_annotations 
            if a.get('annotation_category') == 'CONTRADICTORY'
        )
        
        # 额外统计consistency_checks的结果
        episode_dict['reasoning']['n_consistency_supportive'] = sum(
            1 for c in consistency_checks if c['consistency'] == 'SUPPORTIVE'
        )
        episode_dict['reasoning']['n_consistency_contradictory'] = sum(
            1 for c in consistency_checks if c['consistency'] == 'CONTRADICTORY'
        )

        # 新增：统计信息
        episode_dict['reasoning']['consistency_stats'] = {
            'total_checks': len(all_checks),
            'filtered_by_time': sum(1 for c in all_checks if c.temporal_weight < 0.3),
            'filtered_by_score': sum(1 for c in all_checks if c.consistency == 'NEUTRAL'),
            'avg_temporal_weight': round(sum(c.temporal_weight for c in all_checks) / max(len(all_checks), 1), 3),
            'avg_final_score': round(sum(c.final_score for c in all_checks if c.consistency != 'NEUTRAL') / max(sum(1 for c in all_checks if c.consistency != 'NEUTRAL'), 1), 3)
        }

        return episode_dict


# ==========================================
# 运行入口
# ==========================================

def enhance_sample_episodes(input_dir: Path, output_dir: Path, max_episodes: int = 10):
    """增强样例Episode"""
    print("=" * 70)
    print("Episode Enhancer - 深度增强")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化增强器
    enhancer = EpisodeEnhancer()

    # 加载对齐器数据
    print("\nLoading alignment data...")
    enhancer.aligner.load_data()

    # 处理Episode文件
    episode_files = list(input_dir.glob('TIMELY_v2_*.json'))[:max_episodes]
    print(f"\nEnhancing {len(episode_files)} episodes...")

    stats = {
        'total': 0,
        'with_spans': 0,
        'with_conditions': 0,
        'with_consistency': 0
    }

    for ep_file in episode_files:
        with open(ep_file, 'r', encoding='utf-8') as f:
            episode = json.load(f)

        # 增强
        enhanced = enhancer.enhance_episode(episode)

        # 统计
        stats['total'] += 1
        if enhanced['clinical_text'].get('aligned_spans'):
            stats['with_spans'] += 1
        if enhanced['reasoning']['condition_graph'].get('n_condition_nodes', 0) > 0:
            stats['with_conditions'] += 1
        if enhanced['reasoning'].get('consistency_checks'):
            stats['with_consistency'] += 1

        # 保存
        output_file = output_dir / ep_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("Enhancement Summary")
    print("=" * 70)
    print(f"Total episodes: {stats['total']}")
    print(f"With aligned spans: {stats['with_spans']}")
    print(f"With condition nodes: {stats['with_conditions']}")
    print(f"With consistency checks: {stats['with_consistency']}")
    print(f"\nEnhanced episodes saved to: {output_dir}/")


if __name__ == "__main__":
    input_dir = ROOT_DIR / 'episodes_sample'
    output_dir = ROOT_DIR / 'episodes_enhanced'

    enhance_sample_episodes(input_dir, output_dir, max_episodes=20)
