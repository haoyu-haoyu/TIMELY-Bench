"""
Clinical Pattern Templates
定义临床模式模板并在时序数据中检测

模式基于临床标准: Sepsis-3, KDIGO AKI, Berlin ARDS, SIRS
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import json
import os

# ==========================================
# 1. 模式类型定义
# ==========================================

class PatternType(Enum):
    """模式检测类型"""
    THRESHOLD = "threshold"      # 超过/低于阈值
    DELTA = "delta"              # 变化幅度
    TREND = "trend"              # 趋势（上升/下降）
    DURATION = "duration"        # 持续时间
    COMBINATION = "combination"  # 多条件组合

class Direction(Enum):
    """阈值方向"""
    ABOVE = "above"  # 高于阈值
    BELOW = "below"  # 低于阈值

@dataclass
class PatternTemplate:
    """单个模式模板"""
    name: str                          # 模式名称
    pattern_type: PatternType          # 检测类型
    feature: str                       # 对应的时序特征
    description: str                   # 临床描述
    clinical_source: str               # 临床标准来源

    # Threshold类型参数
    threshold: Optional[float] = None
    direction: Optional[Direction] = None

    # Delta类型参数
    delta_threshold: Optional[float] = None
    delta_window_hours: Optional[int] = None

    # Duration类型参数
    min_duration_hours: Optional[int] = None

    # 额外约束
    unit: str = ""
    severity: str = "moderate"  # mild, moderate, severe

    # 添加文献引用字段
    reference_pmid: Optional[str] = None      # PubMed ID
    reference_doi: Optional[str] = None       # DOI
    evidence_level: str = "guideline"         # guideline, meta-analysis, rct, observational, expert

@dataclass
class DiseasePatternSet:
    """疾病相关的模式集合"""
    disease: str
    clinical_standard: str
    reference: str
    patterns: List[PatternTemplate] = field(default_factory=list)

# ==========================================
# 2. Sepsis模式模板 (基于Sepsis-3)
# ==========================================

SEPSIS_PATTERNS = DiseasePatternSet(
    disease="Sepsis",
    clinical_standard="Sepsis-3 (Singer et al., JAMA 2016)",
    reference="SOFA score criteria + SIRS criteria; PMID: 26903338, 1303622",
    patterns=[
        # === SIRS Criteria (Bone et al., Chest 1992; PMID: 1303622) ===
        PatternTemplate(
            name="fever",
            pattern_type=PatternType.THRESHOLD,
            feature="temperature",
            threshold=38.3,
            direction=Direction.ABOVE,
            unit="°C",
            description="Fever: temperature > 38.3°C",
            clinical_source="SIRS criteria (Bone et al., 1992)",
            severity="moderate",
            reference_pmid="1303622",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="hypothermia",
            pattern_type=PatternType.THRESHOLD,
            feature="temperature",
            threshold=36.0,
            direction=Direction.BELOW,
            unit="°C",
            description="Hypothermia: temperature < 36°C",
            clinical_source="SIRS criteria (Bone et al., 1992)",
            severity="moderate",
            reference_pmid="1303622",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="tachycardia",
            pattern_type=PatternType.THRESHOLD,
            feature="heart_rate",
            threshold=90,
            direction=Direction.ABOVE,
            unit="bpm",
            description="Tachycardia: heart rate > 90 bpm",
            clinical_source="SIRS criteria (Bone et al., 1992)",
            severity="mild",
            reference_pmid="1303622",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="tachypnea",
            pattern_type=PatternType.THRESHOLD,
            feature="resp_rate",
            threshold=20,
            direction=Direction.ABOVE,
            unit="breaths/min",
            description="Tachypnea: respiratory rate > 20/min",
            clinical_source="SIRS criteria (Bone et al., 1992)",
            severity="mild",
            reference_pmid="1303622",
            evidence_level="guideline"
        ),

        # === SOFA - Cardiovascular (Singer et al., JAMA 2016; PMID: 26903338) ===
        PatternTemplate(
            name="hypotension",
            pattern_type=PatternType.THRESHOLD,
            feature="sbp",
            threshold=90,
            direction=Direction.BELOW,
            unit="mmHg",
            description="Hypotension: SBP < 90 mmHg",
            clinical_source="SOFA cardiovascular (Sepsis-3)",
            severity="severe",
            reference_pmid="26903338",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="map_low",
            pattern_type=PatternType.THRESHOLD,
            feature="mbp",
            threshold=70,
            direction=Direction.BELOW,
            unit="mmHg",
            description="Low MAP: mean arterial pressure < 70 mmHg",
            clinical_source="SOFA cardiovascular (Sepsis-3)",
            severity="moderate",
            reference_pmid="26903338",
            evidence_level="guideline"
        ),

        # === SOFA - Respiratory ===
        PatternTemplate(
            name="hypoxemia",
            pattern_type=PatternType.THRESHOLD,
            feature="spo2",
            threshold=94,
            direction=Direction.BELOW,
            unit="%",
            description="Hypoxemia: SpO2 < 94%",
            clinical_source="SOFA respiratory (Sepsis-3)",
            severity="moderate",
            reference_pmid="26903338",
            evidence_level="guideline"
        ),

        # === Lactate (Septic Shock indicator) ===
        PatternTemplate(
            name="lactate_elevated",
            pattern_type=PatternType.THRESHOLD,
            feature="lactate",
            threshold=2.0,
            direction=Direction.ABOVE,
            unit="mmol/L",
            description="Elevated lactate: > 2 mmol/L (septic shock indicator)",
            clinical_source="Sepsis-3 Septic Shock (Singer et al., 2016)",
            severity="severe",
            reference_pmid="26903338",
            evidence_level="guideline"
        ),

        # === SOFA - Coagulation ===
        PatternTemplate(
            name="thrombocytopenia",
            pattern_type=PatternType.THRESHOLD,
            feature="platelet",
            threshold=150,
            direction=Direction.BELOW,
            unit="×10³/µL",
            description="Thrombocytopenia: platelet < 150 ×10³/µL",
            clinical_source="SOFA coagulation (Sepsis-3)",
            severity="moderate",
            reference_pmid="26903338",
            evidence_level="guideline"
        ),

        # === SOFA - Liver ===
        PatternTemplate(
            name="hyperbilirubinemia",
            pattern_type=PatternType.THRESHOLD,
            feature="bilirubin_total",
            threshold=1.2,
            direction=Direction.ABOVE,
            unit="mg/dL",
            description="Hyperbilirubinemia: bilirubin > 1.2 mg/dL",
            clinical_source="SOFA liver (Sepsis-3)",
            severity="moderate",
            reference_pmid="26903338",
            evidence_level="guideline"
        ),

        # === WBC abnormality ===
        PatternTemplate(
            name="leukocytosis",
            pattern_type=PatternType.THRESHOLD,
            feature="wbc",
            threshold=12.0,
            direction=Direction.ABOVE,
            unit="×10³/µL",
            description="Leukocytosis: WBC > 12 ×10³/µL",
            clinical_source="SIRS criteria (Bone et al., 1992)",
            severity="mild",
            reference_pmid="1303622",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="leukopenia",
            pattern_type=PatternType.THRESHOLD,
            feature="wbc",
            threshold=4.0,
            direction=Direction.BELOW,
            unit="×10³/µL",
            description="Leukopenia: WBC < 4 ×10³/µL",
            clinical_source="SIRS criteria (Bone et al., 1992)",
            severity="moderate",
            reference_pmid="1303622",
            evidence_level="guideline"
        ),
    ]
)

# ==========================================
# 3. AKI模式模板 (基于KDIGO)
# ==========================================

AKI_PATTERNS = DiseasePatternSet(
    disease="AKI",
    clinical_standard="KDIGO AKI Guidelines (Kidney International Supplements 2012)",
    reference="Creatinine and urine output criteria; PMID: 25018915",
    patterns=[
        # === Creatinine Threshold (KDIGO 2012; PMID: 25018915) ===
        PatternTemplate(
            name="creatinine_elevated",
            pattern_type=PatternType.THRESHOLD,
            feature="creatinine",
            threshold=1.2,
            direction=Direction.ABOVE,
            unit="mg/dL",
            description="Elevated creatinine: > 1.2 mg/dL (upper normal limit)",
            clinical_source="KDIGO AKI Guidelines 2012",
            severity="mild",
            reference_pmid="25018915",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="creatinine_severe",
            pattern_type=PatternType.THRESHOLD,
            feature="creatinine",
            threshold=4.0,
            direction=Direction.ABOVE,
            unit="mg/dL",
            description="Severe creatinine elevation: >= 4.0 mg/dL (Stage 3 AKI)",
            clinical_source="KDIGO Stage 3 (≥4.0 mg/dL or initiation of RRT)",
            severity="severe",
            reference_pmid="25018915",
            evidence_level="guideline"
        ),

        # === Creatinine Delta (48h) - KDIGO Stage 1 ===
        PatternTemplate(
            name="creatinine_rise_acute",
            pattern_type=PatternType.DELTA,
            feature="creatinine",
            delta_threshold=0.3,
            delta_window_hours=24,
            unit="mg/dL",
            description="Acute creatinine rise: increase >= 0.3 mg/dL within 24h (aligned to 0-24h window)",
            clinical_source="KDIGO Stage 1 (window adapted to 24h for strict 0-24h benchmark)",
            severity="moderate",
            reference_pmid="25018915",
            evidence_level="guideline"
        ),

        # === BUN (Lab reference range) ===
        PatternTemplate(
            name="bun_elevated",
            pattern_type=PatternType.THRESHOLD,
            feature="bun",
            threshold=20,
            direction=Direction.ABOVE,
            unit="mg/dL",
            description="Elevated BUN: > 20 mg/dL (upper normal limit: 7-20 mg/dL)",
            clinical_source="Lab reference range (Mayo Clinic)",
            severity="mild",
            reference_pmid="25018915",
            evidence_level="observational"
        ),
        PatternTemplate(
            name="bun_severe",
            pattern_type=PatternType.THRESHOLD,
            feature="bun",
            threshold=40,
            direction=Direction.ABOVE,
            unit="mg/dL",
            description="Severe BUN elevation: > 40 mg/dL (>2x upper normal)",
            clinical_source="Clinical marker of severe renal dysfunction",
            severity="severe",
            reference_pmid="25018915",
            evidence_level="observational"
        ),

        # === Urine Output (KDIGO criteria) ===
        PatternTemplate(
            name="oliguria",
            pattern_type=PatternType.THRESHOLD,
            feature="urineoutput",
            threshold=500,
            direction=Direction.BELOW,
            unit="mL/day",
            description="Oliguria: urine output < 500 mL/day (KDIGO: <0.5mL/kg/h for 6h)",
            clinical_source="KDIGO AKI Stage 1 urine output criterion",
            severity="moderate",
            reference_pmid="25018915",
            evidence_level="guideline"
        ),

        # === Electrolyte Disturbances (AKI complications) ===
        PatternTemplate(
            name="hyperkalemia",
            pattern_type=PatternType.THRESHOLD,
            feature="potassium",
            threshold=5.5,
            direction=Direction.ABOVE,
            unit="mEq/L",
            description="Hyperkalemia: K+ > 5.5 mEq/L",
            clinical_source="AKI complication; critical value",
            severity="severe",
            reference_pmid="25018915",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="metabolic_acidosis",
            pattern_type=PatternType.THRESHOLD,
            feature="bicarbonate",
            threshold=22,
            direction=Direction.BELOW,
            unit="mEq/L",
            description="Metabolic acidosis: bicarbonate < 22 mEq/L",
            clinical_source="AKI complication; normal range 22-28 mEq/L",
            severity="moderate",
            reference_pmid="25018915",
            evidence_level="guideline"
        ),
    ]
)

# ==========================================
# 4. ARDS模式模板 (基于Berlin Definition)
# ==========================================

ARDS_PATTERNS = DiseasePatternSet(
    disease="ARDS",
    clinical_standard="Berlin Definition (Ranieri et al., JAMA 2012)",
    reference="PaO2/FiO2 ratio with PEEP requirement; PMID: 22797452",
    patterns=[
        # === Oxygenation (Berlin Definition; PMID: 22797452) ===
        PatternTemplate(
            name="hypoxemia_mild",
            pattern_type=PatternType.THRESHOLD,
            feature="pao2_fio2",
            threshold=300,
            direction=Direction.BELOW,
            unit="mmHg",
            description="Mild hypoxemia: P/F ratio 200-300 mmHg (Berlin Mild ARDS)",
            clinical_source="Berlin Definition - Mild ARDS: 200 < PaO2/FiO2 ≤ 300",
            severity="mild",
            reference_pmid="22797452",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="hypoxemia_moderate",
            pattern_type=PatternType.THRESHOLD,
            feature="pao2_fio2",
            threshold=200,
            direction=Direction.BELOW,
            unit="mmHg",
            description="Moderate hypoxemia: P/F ratio 100-200 mmHg (Berlin Moderate ARDS)",
            clinical_source="Berlin Definition - Moderate ARDS: 100 < PaO2/FiO2 ≤ 200",
            severity="moderate",
            reference_pmid="22797452",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="hypoxemia_severe",
            pattern_type=PatternType.THRESHOLD,
            feature="pao2_fio2",
            threshold=100,
            direction=Direction.BELOW,
            unit="mmHg",
            description="Severe hypoxemia: P/F ratio <= 100 mmHg (Berlin Severe ARDS)",
            clinical_source="Berlin Definition - Severe ARDS: PaO2/FiO2 ≤ 100",
            severity="severe",
            reference_pmid="22797452",
            evidence_level="guideline"
        ),

        # === SpO2 proxy (WHO oxygen therapy) ===
        PatternTemplate(
            name="spo2_low",
            pattern_type=PatternType.THRESHOLD,
            feature="spo2",
            threshold=90,
            direction=Direction.BELOW,
            unit="%",
            description="Severe oxygen desaturation: SpO2 < 90%",
            clinical_source="WHO oxygen therapy threshold; respiratory failure indicator",
            severity="severe",
            reference_pmid="22797452",
            evidence_level="guideline"
        ),

        # === Respiratory distress (Clinical sign) ===
        PatternTemplate(
            name="respiratory_distress",
            pattern_type=PatternType.THRESHOLD,
            feature="resp_rate",
            threshold=30,
            direction=Direction.ABOVE,
            unit="breaths/min",
            description="Respiratory distress: respiratory rate > 30/min",
            clinical_source="Clinical sign of respiratory failure; ventilation threshold",
            severity="severe",
            reference_pmid="22797452",
            evidence_level="observational"
        ),
    ]
)

# ==========================================
# 5. 通用危重症模式
# ==========================================

CRITICAL_PATTERNS = DiseasePatternSet(
    disease="Critical Illness",
    clinical_standard="General ICU monitoring standards",
    reference="AHA/ACC Guidelines; ACLS; Standard vital sign thresholds; PMIDs: 29724560, 20956258",
    patterns=[
        # === Heart Rate (AHA/ACC; PMID: 29724560) ===
        PatternTemplate(
            name="bradycardia",
            pattern_type=PatternType.THRESHOLD,
            feature="heart_rate",
            threshold=60,
            direction=Direction.BELOW,
            unit="bpm",
            description="Bradycardia: heart rate < 60 bpm",
            clinical_source="AHA/ACC Guidelines; ACLS bradycardia protocol",
            severity="moderate",
            reference_pmid="29724560",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="severe_tachycardia",
            pattern_type=PatternType.THRESHOLD,
            feature="heart_rate",
            threshold=120,
            direction=Direction.ABOVE,
            unit="bpm",
            description="Severe tachycardia: heart rate > 120 bpm",
            clinical_source="AHA/ACC Guidelines; hemodynamic instability threshold",
            severity="severe",
            reference_pmid="29724560",
            evidence_level="guideline"
        ),

        # === Blood Pressure (ACC/AHA Hypertension Guidelines; PMID: 29133354) ===
        PatternTemplate(
            name="hypertensive_crisis",
            pattern_type=PatternType.THRESHOLD,
            feature="sbp",
            threshold=180,
            direction=Direction.ABOVE,
            unit="mmHg",
            description="Hypertensive crisis: SBP > 180 mmHg",
            clinical_source="ACC/AHA Hypertension Guidelines; hypertensive emergency threshold",
            severity="severe",
            reference_pmid="29133354",
            evidence_level="guideline"
        ),

        # === Hemoglobin (AABB Transfusion Guidelines; PMID: 26402112) ===
        PatternTemplate(
            name="anemia",
            pattern_type=PatternType.THRESHOLD,
            feature="hemoglobin",
            threshold=10,
            direction=Direction.BELOW,
            unit="g/dL",
            description="Anemia: hemoglobin < 10 g/dL",
            clinical_source="WHO anemia definition; clinical threshold",
            severity="moderate",
            reference_pmid="26402112",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="severe_anemia",
            pattern_type=PatternType.THRESHOLD,
            feature="hemoglobin",
            threshold=7,
            direction=Direction.BELOW,
            unit="g/dL",
            description="Severe anemia: hemoglobin < 7 g/dL (transfusion threshold)",
            clinical_source="AABB Transfusion Guidelines; restrictive transfusion threshold",
            severity="severe",
            reference_pmid="26402112",
            evidence_level="guideline"
        ),

        # === GCS (Teasdale & Jennett; PMID: 4136544) ===
        PatternTemplate(
            name="altered_consciousness",
            pattern_type=PatternType.THRESHOLD,
            feature="gcs",
            threshold=14,
            direction=Direction.BELOW,
            unit="score",
            description="Altered consciousness: GCS < 14",
            clinical_source="Glasgow Coma Scale; altered mental status threshold",
            severity="moderate",
            reference_pmid="4136544",
            evidence_level="guideline"
        ),
        PatternTemplate(
            name="coma",
            pattern_type=PatternType.THRESHOLD,
            feature="gcs",
            threshold=8,
            direction=Direction.BELOW,
            unit="score",
            description="Coma: GCS <= 8 (intubation threshold)",
            clinical_source="Glasgow Coma Scale; coma/intubation threshold",
            severity="severe",
            reference_pmid="4136544",
            evidence_level="guideline"
        ),
    ]
)

# ==========================================
# 6. 所有模式的注册表
# ==========================================

PATTERN_REGISTRY = {
    "sepsis": SEPSIS_PATTERNS,
    "aki": AKI_PATTERNS,
    "ards": ARDS_PATTERNS,
    "critical": CRITICAL_PATTERNS,
}

def get_all_patterns() -> Dict[str, DiseasePatternSet]:
    """获取所有已注册的模式集"""
    return PATTERN_REGISTRY

def get_patterns_for_disease(disease: str) -> DiseasePatternSet:
    """获取特定疾病的模式集"""
    return PATTERN_REGISTRY.get(disease.lower())

def get_feature_to_patterns_mapping() -> Dict[str, List[PatternTemplate]]:
    """获取特征到模式的映射（一个特征可能对应多个模式）"""
    mapping = {}
    for disease_set in PATTERN_REGISTRY.values():
        for pattern in disease_set.patterns:
            if pattern.feature not in mapping:
                mapping[pattern.feature] = []
            mapping[pattern.feature].append(pattern)
    return mapping

# ==========================================
# 7. 导出模式为JSON（用于文档）
# ==========================================

def export_patterns_to_json(output_path: str):
    """导出所有模式模板为JSON格式"""
    export_data = {}

    for disease_key, disease_set in PATTERN_REGISTRY.items():
        export_data[disease_key] = {
            "disease": disease_set.disease,
            "clinical_standard": disease_set.clinical_standard,
            "reference": disease_set.reference,
            "patterns": []
        }

        for p in disease_set.patterns:
            pattern_dict = {
                "name": p.name,
                "type": p.pattern_type.value,
                "feature": p.feature,
                "description": p.description,
                "clinical_source": p.clinical_source,
                "severity": p.severity,
                "unit": p.unit,
                # 导出新增的文献引用字段
                "reference_pmid": p.reference_pmid,
                "evidence_level": p.evidence_level,
            }

            if p.threshold is not None:
                pattern_dict["threshold"] = p.threshold
                pattern_dict["direction"] = p.direction.value

            if p.delta_threshold is not None:
                pattern_dict["delta_threshold"] = p.delta_threshold
                pattern_dict["delta_window_hours"] = p.delta_window_hours

            export_data[disease_key]["patterns"].append(pattern_dict)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"Exported patterns to: {output_path}")

def print_pattern_summary():
    """打印模式模板摘要"""
    print("\n" + "=" * 70)
    print("CLINICAL PATTERN TEMPLATES SUMMARY")
    print("=" * 70)
    
    for disease_key, disease_set in PATTERN_REGISTRY.items():
        print(f"\n{disease_set.disease}")
        print(f"   Standard: {disease_set.clinical_standard}")
        print(f"   Patterns: {len(disease_set.patterns)}")
        
        for p in disease_set.patterns:
            if p.pattern_type == PatternType.THRESHOLD:
                op = ">" if p.direction == Direction.ABOVE else "<"
                print(f"   - {p.name}: {p.feature} {op} {p.threshold} {p.unit} [{p.severity}]")
            elif p.pattern_type == PatternType.DELTA:
                print(f"   - {p.name}: Δ{p.feature} ≥ {p.delta_threshold} in {p.delta_window_hours}h [{p.severity}]")
    
    # 统计
    total_patterns = sum(len(ps.patterns) for ps in PATTERN_REGISTRY.values())
    unique_features = len(get_feature_to_patterns_mapping())
    
    print(f"\nTotal: {total_patterns} patterns across {unique_features} features")

# ==========================================
# Main
# ==========================================

if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from config import ROOT_DIR

    # 打印摘要
    print_pattern_summary()

    # 导出JSON
    doc_dir = ROOT_DIR / 'documentation'
    doc_dir.mkdir(parents=True, exist_ok=True)
    export_patterns_to_json(str(doc_dir / 'pattern_templates.json'))

    # 打印特征映射
    print("\n" + "=" * 70)
    print("FEATURE TO PATTERN MAPPING")
    print("=" * 70)

    mapping = get_feature_to_patterns_mapping()
    for feature, patterns in sorted(mapping.items()):
        pattern_names = [p.name for p in patterns]
        print(f"   {feature}: {pattern_names}")
