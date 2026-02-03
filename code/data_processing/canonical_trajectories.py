"""
Canonical Trajectories Module for TIMELY-Bench
定义临床状态的典型时序演变模式

包含:
1. TrajectoryType - 轨迹类型枚举
2. TrajectoryPhase - 轨迹阶段定义
3. CanonicalTrajectory - 典型轨迹模板
4. 导出为JSON格式供文档和验证使用
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Optional, Dict, Any
import json
from pathlib import Path
from datetime import datetime


class TrajectoryType(Enum):
    """轨迹类型"""
    RECOVERY = "recovery"           # 恢复轨迹
    WORSENING = "worsening"         # 恶化轨迹
    STABLE = "stable"               # 稳定轨迹
    OSCILLATING = "oscillating"     # 波动轨迹
    BIPHASIC = "biphasic"          # 双相轨迹 (先恶化后恢复)


class SeverityLevel(Enum):
    """严重程度"""
    NORMAL = "normal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


@dataclass
class TrajectoryPhase:
    """轨迹阶段"""
    name: str
    duration_hours: tuple  # (min, max)
    expected_direction: str  # "improving", "worsening", "stable"
    key_features: List[str]
    expected_values: Dict[str, tuple]  # feature: (min, max)
    description: str = ""


@dataclass
class CanonicalTrajectory:
    """典型轨迹模板"""
    condition: str
    trajectory_type: TrajectoryType
    phases: List[TrajectoryPhase]
    probability: float  # 该轨迹在该condition中的预期比例
    clinical_significance: str
    reference: str
    expected_outcome: str  # "survival", "mortality", "prolonged_stay"
    features_monitored: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "condition": self.condition,
            "trajectory_type": self.trajectory_type.value,
            "phases": [
                {
                    "name": p.name,
                    "duration_hours": list(p.duration_hours),
                    "expected_direction": p.expected_direction,
                    "key_features": p.key_features,
                    "expected_values": {k: list(v) for k, v in p.expected_values.items()},
                    "description": p.description
                }
                for p in self.phases
            ],
            "probability": self.probability,
            "clinical_significance": self.clinical_significance,
            "reference": self.reference,
            "expected_outcome": self.expected_outcome,
            "features_monitored": self.features_monitored
        }


# ============================================================
# Sepsis Trajectories
# ============================================================

SEPSIS_RECOVERY_TRAJECTORY = CanonicalTrajectory(
    condition="Sepsis",
    trajectory_type=TrajectoryType.RECOVERY,
    phases=[
        TrajectoryPhase(
            name="acute_phase",
            duration_hours=(0, 6),
            expected_direction="worsening",
            key_features=["temperature", "heart_rate", "lactate", "sbp"],
            expected_values={
                "temperature": (38.0, 40.0),
                "heart_rate": (100, 140),
                "lactate": (2.0, 6.0),
                "sbp": (70, 100)
            },
            description="Initial septic shock phase with hypotension and elevated lactate"
        ),
        TrajectoryPhase(
            name="resuscitation_phase",
            duration_hours=(6, 24),
            expected_direction="improving",
            key_features=["lactate", "sbp", "urine_output"],
            expected_values={
                "lactate": (1.5, 4.0),
                "sbp": (90, 120),
                "urine_output": (0.5, 1.5)
            },
            description="Response to fluid resuscitation and vasopressors"
        ),
        TrajectoryPhase(
            name="recovery_phase",
            duration_hours=(24, 72),
            expected_direction="improving",
            key_features=["temperature", "wbc", "creatinine"],
            expected_values={
                "temperature": (36.5, 38.0),
                "wbc": (8, 15),
                "creatinine": (0.7, 1.5)
            },
            description="Gradual normalization of inflammatory markers"
        )
    ],
    probability=0.65,
    clinical_significance="Represents successful sepsis management with early source control",
    reference="Surviving Sepsis Campaign Guidelines 2021; PMID: 34599691",
    expected_outcome="survival",
    features_monitored=["temperature", "heart_rate", "sbp", "lactate", "wbc", "creatinine", "urine_output"]
)

SEPSIS_WORSENING_TRAJECTORY = CanonicalTrajectory(
    condition="Sepsis",
    trajectory_type=TrajectoryType.WORSENING,
    phases=[
        TrajectoryPhase(
            name="initial_sepsis",
            duration_hours=(0, 6),
            expected_direction="worsening",
            key_features=["temperature", "heart_rate", "lactate"],
            expected_values={
                "temperature": (38.5, 40.5),
                "heart_rate": (110, 150),
                "lactate": (2.5, 6.0)
            },
            description="Onset of sepsis with systemic inflammatory response"
        ),
        TrajectoryPhase(
            name="refractory_shock",
            duration_hours=(6, 24),
            expected_direction="worsening",
            key_features=["sbp", "lactate", "vasopressor_dose"],
            expected_values={
                "sbp": (60, 85),
                "lactate": (4.0, 10.0),
                "creatinine": (2.0, 4.0)
            },
            description="Progressive shock despite resuscitation"
        ),
        TrajectoryPhase(
            name="multi_organ_failure",
            duration_hours=(24, 72),
            expected_direction="worsening",
            key_features=["creatinine", "bilirubin", "platelet"],
            expected_values={
                "creatinine": (3.0, 6.0),
                "bilirubin": (2.0, 8.0),
                "platelet": (50, 100)
            },
            description="Development of MODS (Multiple Organ Dysfunction Syndrome)"
        )
    ],
    probability=0.25,
    clinical_significance="Indicates treatment failure, need for escalation",
    reference="PMID: 27099011; Rhodes et al., Intensive Care Med 2017",
    expected_outcome="mortality",
    features_monitored=["temperature", "heart_rate", "sbp", "lactate", "creatinine", "bilirubin", "platelet"]
)

# ============================================================
# AKI Trajectories
# ============================================================

AKI_RECOVERY_TRAJECTORY = CanonicalTrajectory(
    condition="AKI",
    trajectory_type=TrajectoryType.RECOVERY,
    phases=[
        TrajectoryPhase(
            name="injury_phase",
            duration_hours=(0, 24),
            expected_direction="worsening",
            key_features=["creatinine", "urine_output"],
            expected_values={
                "creatinine": (1.5, 3.0),
                "urine_output": (0.3, 0.5)
            },
            description="Initial kidney injury with rising creatinine"
        ),
        TrajectoryPhase(
            name="plateau_phase",
            duration_hours=(24, 72),
            expected_direction="stable",
            key_features=["creatinine", "bun"],
            expected_values={
                "creatinine": (2.0, 3.5),
                "bun": (40, 80)
            },
            description="Peak creatinine, stabilization"
        ),
        TrajectoryPhase(
            name="recovery_phase",
            duration_hours=(72, 168),
            expected_direction="improving",
            key_features=["creatinine", "urine_output"],
            expected_values={
                "creatinine": (1.0, 2.0),
                "urine_output": (0.8, 2.0)
            },
            description="Gradual recovery of kidney function"
        )
    ],
    probability=0.60,
    clinical_significance="Transient AKI with complete or partial recovery",
    reference="KDIGO AKI Guidelines 2012; PMID: 22890468",
    expected_outcome="survival",
    features_monitored=["creatinine", "bun", "urine_output", "potassium"]
)

AKI_PROGRESSION_TRAJECTORY = CanonicalTrajectory(
    condition="AKI",
    trajectory_type=TrajectoryType.WORSENING,
    phases=[
        TrajectoryPhase(
            name="stage1",
            duration_hours=(0, 24),
            expected_direction="worsening",
            key_features=["creatinine"],
            expected_values={
                "creatinine": (1.5, 2.0)
            },
            description="KDIGO Stage 1: 1.5-1.9x baseline"
        ),
        TrajectoryPhase(
            name="stage2",
            duration_hours=(24, 48),
            expected_direction="worsening",
            key_features=["creatinine", "urine_output"],
            expected_values={
                "creatinine": (2.0, 3.0),
                "urine_output": (0.3, 0.5)
            },
            description="KDIGO Stage 2: 2.0-2.9x baseline"
        ),
        TrajectoryPhase(
            name="stage3",
            duration_hours=(48, 96),
            expected_direction="worsening",
            key_features=["creatinine", "potassium"],
            expected_values={
                "creatinine": (3.5, 8.0),
                "potassium": (5.5, 7.0)
            },
            description="KDIGO Stage 3: ≥3x baseline or ≥4.0 mg/dL, may require RRT"
        )
    ],
    probability=0.30,
    clinical_significance="Progressive AKI requiring close monitoring, possible RRT",
    reference="KDIGO AKI Guidelines 2012; PMID: 22890468",
    expected_outcome="prolonged_stay",
    features_monitored=["creatinine", "bun", "potassium", "bicarbonate", "urine_output"]
)

# ============================================================
# ARDS Trajectories
# ============================================================

ARDS_RECOVERY_TRAJECTORY = CanonicalTrajectory(
    condition="ARDS",
    trajectory_type=TrajectoryType.RECOVERY,
    phases=[
        TrajectoryPhase(
            name="exudative_phase",
            duration_hours=(0, 72),
            expected_direction="worsening",
            key_features=["pao2_fio2_ratio", "peep", "fio2"],
            expected_values={
                "pao2_fio2_ratio": (100, 200),
                "peep": (8, 15),
                "fio2": (0.5, 0.8)
            },
            description="Acute lung injury with bilateral infiltrates"
        ),
        TrajectoryPhase(
            name="proliferative_phase",
            duration_hours=(72, 168),
            expected_direction="stable",
            key_features=["pao2_fio2_ratio", "compliance"],
            expected_values={
                "pao2_fio2_ratio": (150, 250),
                "peep": (5, 10)
            },
            description="Gradual improvement in oxygenation"
        ),
        TrajectoryPhase(
            name="resolution_phase",
            duration_hours=(168, 336),
            expected_direction="improving",
            key_features=["pao2_fio2_ratio", "spo2"],
            expected_values={
                "pao2_fio2_ratio": (250, 400),
                "spo2": (94, 100)
            },
            description="Lung healing and weaning from ventilation"
        )
    ],
    probability=0.55,
    clinical_significance="Typical ARDS recovery with appropriate supportive care",
    reference="Berlin Definition; PMID: 22797452",
    expected_outcome="survival",
    features_monitored=["pao2_fio2_ratio", "spo2", "peep", "fio2", "resp_rate"]
)

# ============================================================
# Heart Failure Trajectories
# ============================================================

HF_COMPENSATION_TRAJECTORY = CanonicalTrajectory(
    condition="Heart Failure",
    trajectory_type=TrajectoryType.RECOVERY,
    phases=[
        TrajectoryPhase(
            name="acute_decompensation",
            duration_hours=(0, 24),
            expected_direction="worsening",
            key_features=["bnp", "sbp", "spo2"],
            expected_values={
                "bnp": (500, 2000),
                "sbp": (90, 140),
                "spo2": (85, 92)
            },
            description="Acute pulmonary congestion and symptom exacerbation"
        ),
        TrajectoryPhase(
            name="diuresis_phase",
            duration_hours=(24, 72),
            expected_direction="improving",
            key_features=["urine_output", "weight", "bnp"],
            expected_values={
                "urine_output": (1.5, 4.0),
                "bnp": (200, 800)
            },
            description="Response to diuretics with fluid removal"
        ),
        TrajectoryPhase(
            name="optimization_phase",
            duration_hours=(72, 168),
            expected_direction="improving",
            key_features=["bnp", "creatinine", "sbp"],
            expected_values={
                "bnp": (100, 400),
                "creatinine": (0.8, 1.5),
                "sbp": (100, 130)
            },
            description="Neurohormonal optimization and stabilization"
        )
    ],
    probability=0.70,
    clinical_significance="Successful management of acute decompensated heart failure",
    reference="ESC Guidelines 2016; PMID: 27206819",
    expected_outcome="survival",
    features_monitored=["bnp", "sbp", "heart_rate", "spo2", "creatinine", "urine_output"]
)


# ============================================================
# Registry of all trajectories
# ============================================================

CANONICAL_TRAJECTORIES = {
    "sepsis_recovery": SEPSIS_RECOVERY_TRAJECTORY,
    "sepsis_worsening": SEPSIS_WORSENING_TRAJECTORY,
    "aki_recovery": AKI_RECOVERY_TRAJECTORY,
    "aki_progression": AKI_PROGRESSION_TRAJECTORY,
    "ards_recovery": ARDS_RECOVERY_TRAJECTORY,
    "hf_compensation": HF_COMPENSATION_TRAJECTORY,
}


def export_trajectories_to_json(output_path: Path) -> None:
    """导出所有轨迹到JSON文件"""
    export_data = {
        "schema": "TIMELY-Bench-CanonicalTrajectories/1.0",
        "generated_at": datetime.now().isoformat(),
        "description": "Canonical clinical trajectories for temporal pattern analysis",
        "trajectories": {
            name: traj.to_dict()
            for name, traj in CANONICAL_TRAJECTORIES.items()
        },
        "trajectory_types": [t.value for t in TrajectoryType],
        "severity_levels": [s.value for s in SeverityLevel],
        "references": [
            "Surviving Sepsis Campaign Guidelines 2021 (PMID: 34599691)",
            "KDIGO AKI Guidelines 2012 (PMID: 22890468)",
            "Berlin Definition of ARDS 2012 (PMID: 22797452)",
            "ESC Heart Failure Guidelines 2016 (PMID: 27206819)"
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(CANONICAL_TRAJECTORIES)} canonical trajectories to {output_path}")


def get_trajectory(condition: str, trajectory_type: str) -> Optional[CanonicalTrajectory]:
    """获取指定条件和类型的轨迹"""
    key = f"{condition.lower()}_{trajectory_type.lower()}"
    return CANONICAL_TRAJECTORIES.get(key)


def list_trajectories_for_condition(condition: str) -> List[str]:
    """列出某个condition的所有轨迹"""
    return [
        name for name in CANONICAL_TRAJECTORIES.keys()
        if name.startswith(condition.lower())
    ]


if __name__ == "__main__":
    # 导出到documentation目录
    project_root = Path(__file__).parent.parent.parent
    output_file = project_root / "documentation" / "canonical_trajectories.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    export_trajectories_to_json(output_file)

    # 打印摘要
    print("\nCanonical Trajectories Summary:")
    print("=" * 50)
    for name, traj in CANONICAL_TRAJECTORIES.items():
        print(f"\n{name}:")
        print(f"  Condition: {traj.condition}")
        print(f"  Type: {traj.trajectory_type.value}")
        print(f"  Phases: {len(traj.phases)}")
        print(f"  Expected Outcome: {traj.expected_outcome}")
        print(f"  Probability: {traj.probability:.0%}")
