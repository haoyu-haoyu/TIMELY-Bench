"""
TIMELY-Bench Episode Schema
Episode数据结构定义: 时序数据 + 临床文本 + 推理构件 + 标签
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json


# ==========================================
# 1. 基础枚举类型
# ==========================================

class AlignmentQuality(Enum):
    """时序-文本对齐质量"""
    EXACT = "exact"           # 完全匹配（时间差 < 1小时）
    CLOSE = "close"           # 近似匹配（时间差 1-6小时）
    MODERATE = "moderate"     # 中等匹配（时间差 6-12小时）
    DISTANT = "distant"       # 远距离匹配（时间差 > 12小时）


class AnnotationCategory(Enum):
    """LLM标注类别"""
    SUPPORTIVE = "SUPPORTIVE"           # 文本支持生理模式
    CONTRADICTORY = "CONTRADICTORY"     # 文本与模式矛盾
    UNRELATED = "UNRELATED"             # 文本与模式无关
    AMBIGUOUS = "AMBIGUOUS"             # 无法确定关系


class Severity(Enum):
    """严重程度"""
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class EvidenceLevel(Enum):
    """证据等级"""
    GUIDELINE = "guideline"           # 临床指南
    META_ANALYSIS = "meta-analysis"   # Meta分析
    RCT = "rct"                       # 随机对照试验
    OBSERVATIONAL = "observational"   # 观察性研究
    EXPERT = "expert"                 # 专家意见


# ==========================================
# 2. 时序数据结构
# ==========================================

@dataclass
class VitalSign:
    """单个时间点的生命体征"""
    hour: int                            # 相对ICU入院的小时数
    timestamp: Optional[str] = None      # 绝对时间戳 (ISO 8601)

    # 生命体征
    heart_rate: Optional[float] = None        # bpm
    sbp: Optional[float] = None               # mmHg (收缩压)
    dbp: Optional[float] = None               # mmHg (舒张压)
    mbp: Optional[float] = None               # mmHg (平均动脉压)
    resp_rate: Optional[float] = None         # breaths/min
    temperature: Optional[float] = None        # °C
    spo2: Optional[float] = None              # %

    # 意识状态
    gcs: Optional[float] = None               # Glasgow Coma Scale (3-15)

    # 尿量
    urineoutput: Optional[float] = None       # mL


@dataclass
class LabValue:
    """单个时间点的实验室检验"""
    hour: int
    timestamp: Optional[str] = None

    # 肾功能
    creatinine: Optional[float] = None        # mg/dL
    bun: Optional[float] = None               # mg/dL

    # 电解质
    sodium: Optional[float] = None            # mEq/L
    potassium: Optional[float] = None         # mEq/L
    bicarbonate: Optional[float] = None       # mEq/L
    chloride: Optional[float] = None          # mEq/L

    # 血气
    ph: Optional[float] = None
    lactate: Optional[float] = None           # mmol/L

    # 血常规
    wbc: Optional[float] = None               # ×10³/µL
    hemoglobin: Optional[float] = None        # g/dL
    hematocrit: Optional[float] = None        # %
    platelet: Optional[float] = None          # ×10³/µL

    # 代谢
    glucose: Optional[float] = None           # mg/dL
    albumin: Optional[float] = None           # g/dL
    bilirubin_total: Optional[float] = None   # mg/dL


@dataclass
class Intervention:
    """单个时间点的干预/治疗信号（结构化）"""
    hour: int
    timestamp: Optional[str] = None

    # Binary indicators (0/1). These are derived from medication/procedure records.
    vasopressors: Optional[int] = None
    rrt: Optional[int] = None


@dataclass
class TimeSeriesData:
    """完整的时序数据"""
    vitals: List[VitalSign] = field(default_factory=list)
    labs: List[LabValue] = field(default_factory=list)
    interventions: List[Intervention] = field(default_factory=list)

    # 元数据
    start_hour: int = 0
    end_hour: int = 24
    resolution_hours: int = 1    # 时间分辨率（小时）
    n_timepoints: int = 0

    # 数据质量
    missing_rate: Dict[str, float] = field(default_factory=dict)  # 各特征缺失率


# ==========================================
# 3. 临床文本结构
# ==========================================

@dataclass
class NoteSpan:
    """临床笔记片段"""
    note_id: str
    note_type: str                        # Radiology, Nursing, Physician, etc.
    note_category: str                    # 细分类别

    # 时间信息
    chart_hour: int                       # 记录时间（相对ICU入院）
    chart_time: Optional[str] = None      # 绝对时间戳

    # 文本内容
    text_full: str = ""                   # 完整笔记文本
    text_relevant: str = ""               # 相关片段（如有）

    # 元数据
    text_length: int = 0
    has_llm_features: bool = False        # 是否有LLM提取的特征


@dataclass
class LLMExtractedFeatures:
    """LLM从文本中提取的临床特征"""
    note_id: str

    # 放射学发现（from DeepSeek）
    pneumonia: Optional[int] = None           # 0/1
    edema: Optional[int] = None               # 肺水肿
    pleural_effusion: Optional[int] = None    # 胸腔积液
    pneumothorax: Optional[int] = None        # 气胸
    tubes_lines: Optional[int] = None         # 管线位置异常

    # 扩展特征（29维临床特征）
    extended_features: Dict[str, Any] = field(default_factory=dict)

    # 置信度
    extraction_confidence: float = 0.0
    model_version: str = ""


@dataclass
class ClinicalText:
    """完整的临床文本数据"""
    notes: List[NoteSpan] = field(default_factory=list)
    llm_features: List[LLMExtractedFeatures] = field(default_factory=list)

    # 统计
    n_notes: int = 0
    note_types: List[str] = field(default_factory=list)
    coverage_hours: List[int] = field(default_factory=list)  # 有笔记的小时


# ==========================================
# 4. 推理构件 (Reasoning Artefacts)
# ==========================================

@dataclass
class PhysiologyTemplate:
    """生理模式模板定义"""
    name: str                             # 模式名称
    pattern_type: str                     # threshold, delta, trend, etc.
    disease: str                          # 关联疾病

    # 检测参数
    feature: str                          # 监测特征
    threshold: Optional[float] = None
    direction: Optional[str] = None       # above/below
    delta_threshold: Optional[float] = None
    delta_window_hours: Optional[int] = None

    # 临床描述
    description: str = ""
    unit: str = ""
    severity: str = "moderate"

    # 证据来源
    clinical_source: str = ""             # 临床标准来源
    reference_pmid: Optional[str] = None
    evidence_level: str = "guideline"


@dataclass
class DetectedPattern:
    """检测到的生理模式实例"""
    pattern_name: str
    detection_hour: int                   # 检测时间点

    # 检测值
    value: float
    threshold: Optional[float] = None

    # 模式信息
    disease: str = ""
    feature: str = ""
    severity: str = "moderate"
    description: str = ""

    # 持续信息
    start_hour: Optional[int] = None
    end_hour: Optional[int] = None
    duration_hours: Optional[int] = None

    # 临床模板信息（生理学参考）
    clinical_source: Optional[str] = None      # 临床标准来源
    reference_pmid: Optional[str] = None       # PubMed ID
    evidence_level: Optional[str] = None       # 证据级别
    unit: Optional[str] = None                 # 单位


@dataclass
class PatternTextAlignment:
    """模式-文本对齐关系"""
    pattern_name: str
    pattern_hour: int

    # 对齐的笔记
    note_id: str
    note_hour: int
    note_type: str

    # 对齐质量
    time_delta_hours: float
    alignment_quality: str               # exact/close/moderate/distant

    # 文本内容
    aligned_text: str = ""

    # LLM标注（如有）
    annotation_category: Optional[str] = None    # SUPPORTIVE/CONTRADICTORY/...
    annotation_confidence: Optional[float] = None
    annotation_reasoning: Optional[str] = None


@dataclass
class ConditionGraphNode:
    """疾病依赖图节点"""
    condition: str                        # 疾病/状态名称
    icd_codes: List[str] = field(default_factory=list)

    # 检测信息
    is_present: bool = False
    onset_hour: Optional[int] = None
    confidence: float = 0.0

    # 来源
    source: str = ""                      # diagnosis/pattern/text


@dataclass
class ConditionGraphEdge:
    """疾病依赖图边"""
    source: str                           # 源疾病
    target: str                           # 目标疾病
    relationship: str                     # causes/precedes/complicates/...

    # 时间关系
    temporal_order: str = ""              # before/after/concurrent
    time_delta_hours: Optional[int] = None

    # 证据
    evidence_type: str = ""               # clinical_guideline/observed/inferred
    evidence_pmid: Optional[str] = None


@dataclass
class ConditionGraph:
    """疾病依赖图"""
    nodes: List[ConditionGraphNode] = field(default_factory=list)
    edges: List[ConditionGraphEdge] = field(default_factory=list)

    # 图属性
    primary_condition: Optional[str] = None
    complexity_score: float = 0.0         # 复杂度评分


@dataclass
class ReasoningArtefacts:
    """完整的推理构件"""
    # 疾病依赖图
    condition_graph: Optional[ConditionGraph] = None

    # 生理模式模板
    physiology_templates: List[PhysiologyTemplate] = field(default_factory=list)

    # 检测到的模式
    detected_patterns: List[DetectedPattern] = field(default_factory=list)

    # 模式-文本对齐
    pattern_annotations: List[PatternTextAlignment] = field(default_factory=list)

    # 统计
    n_patterns_detected: int = 0
    n_alignments: int = 0
    n_supportive: int = 0
    n_contradictory: int = 0


# ==========================================
# 5. 标签结构
# ==========================================

@dataclass
class OutcomeLabels:
    """结局标签"""
    # 核心任务标签
    mortality: int = 0                    # In-hospital mortality (0/1)
    prolonged_los: int = 0                # Prolonged LOS > 7 days (0/1)

    # 可选任务标签
    readmission_30d: Optional[int] = None

    # 原始值
    los_days: Optional[float] = None      # 实际住院天数
    death_time: Optional[str] = None      # 死亡时间（如有）


@dataclass
class ProcessLabels:
    """过程标签（疾病发生时间点）"""
    # Sepsis相关
    sepsis_onset_hour: Optional[int] = None
    sepsis_sofa_time: Optional[int] = None

    # AKI相关
    aki_onset_hour: Optional[int] = None
    aki_stage_max: Optional[int] = None   # KDIGO分期 (1-3)

    # ARDS相关
    ards_onset_hour: Optional[int] = None

    # 通用
    first_severe_pattern_hour: Optional[int] = None


@dataclass
class Labels:
    """完整标签"""
    outcome: OutcomeLabels = field(default_factory=OutcomeLabels)
    process: ProcessLabels = field(default_factory=ProcessLabels)

    # 疾病诊断标志
    has_sepsis: bool = False
    has_aki: bool = False
    has_ards: bool = False

    # ICD诊断码
    icd_codes: List[str] = field(default_factory=list)
    diagnoses_text: str = ""


# ==========================================
# 6. Episode主结构
# ==========================================

@dataclass
class PatientDemographics:
    """患者人口学信息"""
    age: Optional[int] = None
    gender: Optional[str] = None

    # 脱敏的标识符
    subject_id: Optional[int] = None
    hadm_id: Optional[int] = None


@dataclass
class EpisodeMetadata:
    """Episode元数据"""
    # 版本信息
    schema_version: str = "2.0"
    created_at: str = ""

    # 数据来源
    source_database: str = "MIMIC-IV"
    source_version: str = "3.1"

    # 数据窗口
    observation_window_hours: int = 24

    # 质量标注
    data_quality_score: float = 0.0
    completeness: Dict[str, float] = field(default_factory=dict)

    # 处理信息
    preprocessing_version: str = ""
    llm_annotation_model: str = ""


@dataclass
class Episode:
    """
    TIMELY-Bench Episode - 统一的多模态临床数据包

    每个Episode代表一次ICU住院的完整临床记录，包含：
    1. 结构化时序数据（生命体征、实验室检验）
    2. 非结构化临床文本（医生/护士笔记、放射报告）
    3. 推理构件（疾病图、生理模式、时序-文本对齐）
    4. 多层次标签（结局标签 + 过程标签）

    设计目标：
    - 为临床LLM提供完整的推理上下文
    - 支持多模态融合研究
    - 评估临床推理能力（不仅仅是预测能力）
    """

    # 唯一标识
    episode_id: str                       # 格式: "TIMELY_v2_{stay_id}"
    stay_id: int                          # MIMIC stay_id

    # 患者信息
    patient: PatientDemographics = field(default_factory=PatientDemographics)

    # 多模态数据
    timeseries: TimeSeriesData = field(default_factory=TimeSeriesData)
    clinical_text: ClinicalText = field(default_factory=ClinicalText)

    # 推理构件
    reasoning: ReasoningArtefacts = field(default_factory=ReasoningArtefacts)

    # 标签
    labels: Labels = field(default_factory=Labels)

    # 元数据
    metadata: EpisodeMetadata = field(default_factory=EpisodeMetadata)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于JSON序列化）"""
        return {
            "episode_id": self.episode_id,
            "stay_id": self.stay_id,
            "patient": {
                "age": self.patient.age,
                "gender": self.patient.gender,
                "subject_id": self.patient.subject_id,
                "hadm_id": self.patient.hadm_id
            },
            "timeseries": {
                "vitals": [vars(v) for v in self.timeseries.vitals],
                "labs": [vars(l) for l in self.timeseries.labs],
                "start_hour": self.timeseries.start_hour,
                "end_hour": self.timeseries.end_hour,
                "resolution_hours": self.timeseries.resolution_hours,
                "n_timepoints": self.timeseries.n_timepoints,
                "missing_rate": self.timeseries.missing_rate
            },
            "clinical_text": {
                "notes": [vars(n) for n in self.clinical_text.notes],
                "llm_features": [
                    {
                        "note_id": f.note_id,
                        "pneumonia": f.pneumonia,
                        "edema": f.edema,
                        "pleural_effusion": f.pleural_effusion,
                        "pneumothorax": f.pneumothorax,
                        "tubes_lines": f.tubes_lines,
                        "extended_features": f.extended_features,
                        "extraction_confidence": f.extraction_confidence,
                        "model_version": f.model_version
                    } for f in self.clinical_text.llm_features
                ],
                "n_notes": self.clinical_text.n_notes,
                "note_types": self.clinical_text.note_types,
                "coverage_hours": self.clinical_text.coverage_hours
            },
            "reasoning": {
                "condition_graph": {
                    "nodes": [vars(n) for n in self.reasoning.condition_graph.nodes],
                    "edges": [vars(e) for e in self.reasoning.condition_graph.edges],
                    "primary_condition": self.reasoning.condition_graph.primary_condition,
                    "complexity_score": self.reasoning.condition_graph.complexity_score
                } if self.reasoning.condition_graph else None,
                "physiology_templates": [vars(t) for t in self.reasoning.physiology_templates],
                "detected_patterns": [vars(p) for p in self.reasoning.detected_patterns],
                "pattern_annotations": [vars(a) for a in self.reasoning.pattern_annotations],
                "n_patterns_detected": self.reasoning.n_patterns_detected,
                "n_alignments": self.reasoning.n_alignments,
                "n_supportive": self.reasoning.n_supportive,
                "n_contradictory": self.reasoning.n_contradictory
            },
            "labels": {
                "outcome": vars(self.labels.outcome),
                "process": vars(self.labels.process),
                "has_sepsis": self.labels.has_sepsis,
                "has_aki": self.labels.has_aki,
                "has_ards": self.labels.has_ards,
                "icd_codes": self.labels.icd_codes,
                "diagnoses_text": self.labels.diagnoses_text
            },
            "metadata": vars(self.metadata)
        }

    def to_json(self, indent: int = 2) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        """从字典创建Episode"""
        # 这里需要实现反序列化逻辑
        # 由于结构复杂，建议使用专门的Builder类
        raise NotImplementedError("Use EpisodeBuilder for construction")


# ==========================================
# 7. JSON Schema导出（用于验证）
# ==========================================

EPISODE_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "TIMELY-Bench Episode",
    "description": "Multi-modal clinical reasoning benchmark data package",
    "type": "object",
    "required": ["episode_id", "stay_id", "timeseries", "labels"],
    "properties": {
        "episode_id": {
            "type": "string",
            "pattern": "^TIMELY_v2_\\d+$",
            "description": "Unique episode identifier"
        },
        "stay_id": {
            "type": "integer",
            "description": "MIMIC ICU stay identifier"
        },
        "patient": {
            "type": "object",
            "properties": {
                "age": {"type": ["integer", "null"]},
                "gender": {"type": ["string", "null"], "enum": ["M", "F", None]},
                "subject_id": {"type": ["integer", "null"]},
                "hadm_id": {"type": ["integer", "null"]}
            }
        },
        "timeseries": {
            "type": "object",
            "required": ["vitals", "labs"],
            "properties": {
                "vitals": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["hour"],
                        "properties": {
                            "hour": {"type": "integer", "minimum": 0},
                            "timestamp": {"type": ["string", "null"]},
                            "heart_rate": {"type": ["number", "null"]},
                            "sbp": {"type": ["number", "null"]},
                            "dbp": {"type": ["number", "null"]},
                            "mbp": {"type": ["number", "null"]},
                            "resp_rate": {"type": ["number", "null"]},
                            "temperature": {"type": ["number", "null"]},
                            "spo2": {"type": ["number", "null"]},
                            "gcs": {"type": ["number", "null"]},
                            "urineoutput": {"type": ["number", "null"]}
                        }
                    }
                },
                "labs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["hour"],
                        "properties": {
                            "hour": {"type": "integer"},
                            "timestamp": {"type": ["string", "null"]},
                            "creatinine": {"type": ["number", "null"]},
                            "bun": {"type": ["number", "null"]},
                            "sodium": {"type": ["number", "null"]},
                            "potassium": {"type": ["number", "null"]},
                            "bicarbonate": {"type": ["number", "null"]},
                            "chloride": {"type": ["number", "null"]},
                            "ph": {"type": ["number", "null"]},
                            "lactate": {"type": ["number", "null"]},
                            "wbc": {"type": ["number", "null"]},
                            "hemoglobin": {"type": ["number", "null"]},
                            "hematocrit": {"type": ["number", "null"]},
                            "platelet": {"type": ["number", "null"]}
                            ,
                            "glucose": {"type": ["number", "null"]},
                            "albumin": {"type": ["number", "null"]},
                            "bilirubin_total": {"type": ["number", "null"]}
                        }
                    }
                },
                "interventions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["hour"],
                        "properties": {
                            "hour": {"type": "integer"},
                            "timestamp": {"type": ["string", "null"]},
                            "vasopressors": {"type": ["integer", "null"]},
                            "rrt": {"type": ["integer", "null"]}
                        }
                    }
                },
                "start_hour": {"type": "integer", "default": 0},
                "end_hour": {"type": "integer", "default": 24},
                "resolution_hours": {"type": "integer", "default": 1},
                "n_timepoints": {"type": "integer"},
                "missing_rate": {"type": "object"}
            }
        },
        "clinical_text": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["note_id", "note_type", "chart_hour"],
                        "properties": {
                            "note_id": {"type": "string"},
                            "note_type": {"type": "string"},
                            "note_category": {"type": "string"},
                            "chart_hour": {"type": "number"},
                            "chart_time": {"type": ["string", "null"]},
                            "text_full": {"type": "string"},
                            "text_relevant": {"type": "string"},
                            "text_length": {"type": ["integer", "null"]},
                            "has_llm_features": {"type": ["boolean", "null"]}
                        }
                    }
                },
                "llm_features": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "note_id": {"type": "string"},
                            "pneumonia": {"type": ["integer", "null"]},
                            "edema": {"type": ["integer", "null"]},
                            "pleural_effusion": {"type": ["integer", "null"]},
                            "pneumothorax": {"type": ["integer", "null"]},
                            "tubes_lines": {"type": ["integer", "null"]},
                            "extended_features": {"type": "object"},
                            "extraction_confidence": {"type": ["number", "null"]},
                            "model_version": {"type": ["string", "null"]}
                        }
                    }
                },
                "n_notes": {"type": "integer"},
                "note_types": {"type": "array", "items": {"type": "string"}},
                "coverage_hours": {"type": "array", "items": {"type": "integer"}},
                "aligned_spans": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["note_id", "note_type", "chart_hour", "span_start_hour", "span_end_hour", "text"],
                        "properties": {
                            "note_id": {"type": "string"},
                            "note_type": {"type": "string"},
                            "chart_hour": {"type": "number"},
                            "span_start_hour": {"type": "number"},
                            "span_end_hour": {"type": "number"},
                            "text": {"type": "string"},
                            "keywords": {"type": "array", "items": {"type": "string"}},
                            "relevance_score": {"type": ["number", "null"]}
                        }
                    }
                }
            }
        },
        "reasoning": {
            "type": "object",
            "properties": {
                "condition_graph": {
                    "type": ["object", "null"],
                    "properties": {
                        "nodes": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "name", "level", "onset_hour"],
                                "properties": {
                                    "id": {"type": "string"},
                                    "name": {"type": "string"},
                                    "level": {"type": "string", "enum": ["pattern", "condition"]},
                                    "onset_hour": {"type": "integer"},
                                    "value": {"type": ["number", "null"]},
                                    "severity": {"type": ["string", "null"]},
                                    "source": {"type": ["string", "null"]}
                                }
                            }
                        },
                        "edges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["source_id", "target_id", "relationship"],
                                "properties": {
                                    "source_id": {"type": "string"},
                                    "target_id": {"type": "string"},
                                    "relationship": {
                                        "type": "string",
                                        "enum": ["indicates", "contributes_to", "progresses_to", "cascade_causes"]
                                    },
                                    "confidence": {"type": ["number", "null"]},
                                    "time_delta_hours": {"type": ["number", "null"]},
                                    "clinical_rule": {"type": ["string", "null"]}
                                }
                            }
                        },
                        "primary_condition": {"type": ["string", "null"]},
                        "n_pattern_nodes": {"type": ["integer", "null"]},
                        "n_condition_nodes": {"type": ["integer", "null"]},
                        "n_edges": {"type": ["integer", "null"]}
                    }
                },
                "physiology_templates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["name", "pattern_type", "disease", "feature"],
                        "properties": {
                            "name": {"type": "string"},
                            "pattern_type": {"type": "string"},
                            "disease": {"type": "string"},
                            "feature": {"type": "string"},
                            "threshold": {"type": ["number", "null"]},
                            "direction": {"type": ["string", "null"]},
                            "delta_threshold": {"type": ["number", "null"]},
                            "delta_window_hours": {"type": ["integer", "null"]},
                            "description": {"type": "string"},
                            "unit": {"type": "string"},
                            "severity": {"type": "string"},
                            "clinical_source": {"type": "string"},
                            "reference_pmid": {"type": ["string", "null"]},
                            "evidence_level": {"type": "string"}
                        },
                        "additionalProperties": True
                    }
                },
                "detected_patterns": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["pattern_name", "detection_hour", "value"],
                        "properties": {
                            "pattern_name": {"type": "string"},
                            "detection_hour": {"type": "integer"},
                            "value": {"type": "number"},
                            "threshold": {"type": ["number", "null"]},
                            "severity": {"type": "string", "enum": ["mild", "moderate", "severe"]},
                            "disease": {"type": ["string", "null"]},
                            "feature": {"type": ["string", "null"]},
                            "unit": {"type": ["string", "null"]},
                            "description": {"type": ["string", "null"]},
                            "start_hour": {"type": ["integer", "null"]},
                            "end_hour": {"type": ["integer", "null"]},
                            "duration_hours": {"type": ["integer", "null"]},
                            "clinical_source": {"type": ["string", "null"]},
                            "reference_pmid": {"type": ["string", "null"]},
                            "evidence_level": {"type": ["string", "null"]}
                        }
                    }
                },
                "pattern_annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "pattern_name": {"type": "string"},
                            "pattern_hour": {"type": "integer"},
                            "note_id": {"type": "string"},
                            "note_hour": {"type": ["integer", "null"]},
                            "note_type": {"type": ["string", "null"]},
                            "time_delta_hours": {"type": ["number", "null"]},
                            "alignment_quality": {"type": "string"},
                            "aligned_text": {"type": ["string", "null"]},
                            "annotation_category": {
                                "type": ["string", "null"],
                                "enum": ["SUPPORTIVE", "CONTRADICTORY", "UNRELATED", "AMBIGUOUS", None]
                            },
                            "annotation_confidence": {"type": ["number", "null"]},
                            "annotation_reasoning": {"type": ["string", "null"]}
                        }
                    }
                },
                "n_patterns_detected": {"type": ["integer", "null"]},
                "n_alignments": {"type": ["integer", "null"]},
                "n_supportive": {"type": ["integer", "null"]},
                "n_contradictory": {"type": ["integer", "null"]},
                "n_unrelated": {"type": ["integer", "null"]}
            }
        },
        "labels": {
            "type": "object",
            "required": ["outcome"],
            "properties": {
                "outcome": {
                    "type": "object",
                    "required": ["mortality", "prolonged_los"],
                    "properties": {
                        "mortality": {"type": "integer", "enum": [0, 1]},
                        "prolonged_los": {
                            "type": "integer",
                            "enum": [0, 1],
                            "description": "Prolonged ICU length-of-stay label (ICU LOS > 7 days)"
                        },
                        "readmission_30d": {"type": ["integer", "null"]},
                        "los_days": {"type": ["number", "null"]}
                    }
                },
                "process": {
                    "type": "object",
                    "properties": {
                        "sepsis_onset_hour": {"type": ["integer", "null"]},
                        "aki_onset_hour": {"type": ["integer", "null"]},
                        "ards_onset_hour": {"type": ["integer", "null"]}
                    }
                },
                "has_sepsis": {"type": "boolean"},
                "has_aki": {"type": "boolean"},
                "has_ards": {"type": "boolean"},
                "icd_codes": {"type": "array", "items": {"type": "string"}}
            }
        },
        "metadata": {
            "type": "object",
            "properties": {
                "schema_version": {"type": "string"},
                "created_at": {"type": ["string", "null"]},
                "source_database": {"type": "string"},
                "source_version": {"type": ["string", "null"]},
                "observation_window_hours": {"type": "integer"},
                "data_quality_score": {"type": "number"},
                "completeness": {"type": "object"},
                "preprocessing_version": {"type": ["string", "null"]},
                "llm_annotation_model": {"type": ["string", "null"]}
            }
        }
    }
}


def export_json_schema(output_path: str):
    """导出JSON Schema到文件"""
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(EPISODE_JSON_SCHEMA, f, indent=2, ensure_ascii=False)
    print(f"JSON Schema exported to: {output_path}")


# ==========================================
# 8. 示例Episode
# ==========================================

def create_example_episode() -> Episode:
    """创建一个示例Episode用于演示"""

    episode = Episode(
        episode_id="TIMELY_v2_12345",
        stay_id=12345,
        patient=PatientDemographics(
            age=65,
            gender="M",
            subject_id=10001,
            hadm_id=20001
        )
    )

    # 添加时序数据
    for hour in range(24):
        episode.timeseries.vitals.append(VitalSign(
            hour=hour,
            heart_rate=80 + (hour % 5) * 2,
            sbp=120 - (hour % 3) * 5,
            resp_rate=16 + (hour % 4),
            temperature=37.2 if hour < 12 else 38.5,
            spo2=96 - (hour % 3)
        ))

        if hour % 6 == 0:  # 每6小时一次化验
            episode.timeseries.labs.append(LabValue(
                hour=hour,
                creatinine=1.2 + hour * 0.05,
                lactate=1.5 + hour * 0.1,
                wbc=12.5
            ))

    episode.timeseries.n_timepoints = 24

    # 添加临床文本
    episode.clinical_text.notes.append(NoteSpan(
        note_id="note_001",
        note_type="Radiology",
        note_category="Chest X-ray",
        chart_hour=4,
        text_full="CXR shows bilateral infiltrates consistent with pneumonia. No pleural effusion.",
        text_relevant="bilateral infiltrates consistent with pneumonia"
    ))
    episode.clinical_text.n_notes = 1

    # 添加检测到的模式
    episode.reasoning.detected_patterns.append(DetectedPattern(
        pattern_name="fever",
        detection_hour=12,
        value=38.5,
        threshold=38.3,
        disease="Sepsis",
        feature="temperature",
        severity="moderate",
        description="Fever: temperature > 38.3°C"
    ))

    # 添加模式-文本对齐
    episode.reasoning.pattern_annotations.append(PatternTextAlignment(
        pattern_name="fever",
        pattern_hour=12,
        note_id="note_001",
        note_hour=4,
        note_type="Radiology",
        time_delta_hours=8,
        alignment_quality="moderate",
        aligned_text="bilateral infiltrates consistent with pneumonia",
        annotation_category="SUPPORTIVE",
        annotation_confidence=0.85,
        annotation_reasoning="Pneumonia finding supports the fever pattern"
    ))

    episode.reasoning.n_patterns_detected = 1
    episode.reasoning.n_alignments = 1
    episode.reasoning.n_supportive = 1

    # 添加疾病图
    episode.reasoning.condition_graph = ConditionGraph(
        nodes=[
            ConditionGraphNode(condition="Sepsis", is_present=True, onset_hour=12),
            ConditionGraphNode(condition="Pneumonia", is_present=True, onset_hour=4),
        ],
        edges=[
            ConditionGraphEdge(
                source="Pneumonia",
                target="Sepsis",
                relationship="causes",
                temporal_order="before",
                time_delta_hours=8
            )
        ],
        primary_condition="Sepsis"
    )

    # 添加标签
    episode.labels = Labels(
        outcome=OutcomeLabels(mortality=0, prolonged_los=1, los_days=10.5),
        process=ProcessLabels(sepsis_onset_hour=12),
        has_sepsis=True,
        has_aki=False,
        has_ards=False
    )

    # 元数据
    episode.metadata = EpisodeMetadata(
        schema_version="2.0",
        created_at=datetime.now().isoformat(),
        source_database="MIMIC-IV",
        source_version="3.1",
        observation_window_hours=24,
        data_quality_score=0.85
    )

    return episode


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from config import ROOT_DIR

    print("=" * 70)
    print("TIMELY-Bench Episode Schema v2.0")
    print("=" * 70)

    # 创建示例
    print("\n1. Creating example episode...")
    example = create_example_episode()

    # 导出为JSON
    print("\n2. Exporting example to JSON...")
    import os

    doc_dir = ROOT_DIR / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)

    example_path = doc_dir / 'example_episode.json'
    with open(example_path, 'w', encoding='utf-8') as f:
        f.write(example.to_json(indent=2))
    print(f"   Saved: {example_path}")

    # 导出Schema
    print("\n3. Exporting JSON Schema...")
    schema_path = doc_dir / 'episode_schema.json'
    export_json_schema(str(schema_path))

    # 打印摘要
    print("\n" + "=" * 70)
    print("Episode Structure Summary:")
    print("=" * 70)
    print(f"""
    Episode
    ├── episode_id: str
    ├── stay_id: int
    ├── patient
    │   ├── age, gender
    │   └── subject_id, hadm_id
    ├── timeseries
    │   ├── vitals: List[VitalSign]  (hourly vital signs)
    │   ├── labs: List[LabValue]     (lab values)
    │   └── missing_rate: Dict
    ├── clinical_text
    │   ├── notes: List[NoteSpan]
    │   └── llm_features: List[LLMExtractedFeatures]
    ├── reasoning
    │   ├── condition_graph: ConditionGraph
    │   ├── physiology_templates: List[PhysiologyTemplate]
    │   ├── detected_patterns: List[DetectedPattern]
    │   └── pattern_annotations: List[PatternTextAlignment]
    ├── labels
    │   ├── outcome: OutcomeLabels (mortality, prolonged_los)
    │   ├── process: ProcessLabels (onset times)
    │   └── has_sepsis, has_aki, has_ards
    └── metadata
        ├── schema_version
        ├── source_database
        └── data_quality_score
    """)

    print("\nEpisode Schema Definition Complete!")
