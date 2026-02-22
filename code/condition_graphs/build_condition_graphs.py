"""
Build condition graphs from pattern templates (opt-in).
"""

import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_templates import (
    AKI_PATTERNS,
    DELIRIUM_PATTERNS,
    SEPSIS_PATTERNS,
    STROKE_PATTERNS,
)


OUT_DIR = Path(__file__).resolve().parent / "graphs"

LAB_FEATURES = {
    # Labs used by TIMELY-Bench episodes (subset; extend as needed)
    "albumin",
    "bicarbonate",
    "bilirubin_total",
    "bun",
    "chloride",
    "creatinine",
    "glucose",
    "hematocrit",
    "hemoglobin",
    "lactate",
    "ph",
    "platelet",
    "potassium",
    "sodium",
    "wbc",
}

VITAL_FEATURES = {
    # Vitals used by TIMELY-Bench episodes
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "temperature",
    "spo2",
}

SYMPTOM_FEATURES = {
    # Measured outputs/signs that are closer to symptom/sign semantics
    "urineoutput",
    # Neuro status is recorded as a score, but it is clinically interpreted as a symptom/sign.
    "gcs",
}


def _to_direction(direction):
    if direction is None:
        return ""
    return getattr(direction, "value", str(direction))


def _domain_for_feature(feature: str) -> str:
    """
    Map an underlying feature name to a higher-level clinical domain category.
    Keep node.type stable ("structured_indicator"/"pattern_event"/...) while exposing
    supervisor-expected categories (lab markers, vital signs, symptoms, etc.).
    """
    if not feature:
        return "other"
    f = str(feature).strip().lower()
    if f in LAB_FEATURES:
        return "lab_marker"
    # Prefer symptom/sign semantics over pure "vital" categorization when ambiguous.
    if f in SYMPTOM_FEATURES:
        return "symptom"
    if f in VITAL_FEATURES:
        return "vital_sign"
    return "other"


def build_graph(pattern_set, graph_name):
    nodes = []
    edges = []

    condition_id = f"condition_{pattern_set.disease.lower()}"
    nodes.append({
        "id": condition_id,
        "type": "condition",
        "domain": "condition",
        "label": pattern_set.disease,
        "source": pattern_set.clinical_standard
    })

    for p in pattern_set.patterns:
        struct_id = f"struct_{p.feature}_{p.name}"
        pattern_id = f"pattern_{p.name}"
        text_id = f"text_{p.name}"
        domain = _domain_for_feature(p.feature)

        nodes.append({
            "id": struct_id,
            "type": "structured_indicator",
            "domain": domain,
            "label": f"{p.feature} {p.description}",
            "feature": p.feature,
            "threshold": p.threshold if p.threshold is not None else p.delta_threshold,
            "direction": _to_direction(p.direction),
            "unit": p.unit,
            "severity": p.severity,
            "source": p.clinical_source
        })

        nodes.append({
            "id": pattern_id,
            "type": "pattern_event",
            "domain": domain,
            "label": p.name,
            "severity": p.severity,
            "source": p.clinical_source
        })

        nodes.append({
            "id": text_id,
            "type": "text_evidence",
            "domain": "text_evidence",
            "label": f"text evidence for {p.name}",
            "source": "temporal_textual_alignment"
        })

        edges.append({
            "source": struct_id,
            "target": pattern_id,
            "relation": "triggers",
            "evidence": f"{p.feature} threshold/criterion"
        })
        edges.append({
            "source": text_id,
            "target": pattern_id,
            "relation": "supports",
            "evidence": "note_id/note_hour alignment"
        })
        edges.append({
            "source": text_id,
            "target": pattern_id,
            "relation": "contradicts",
            "evidence": "note_id/note_hour alignment"
        })
        edges.append({
            "source": pattern_id,
            "target": condition_id,
            "relation": "aggregates",
            "evidence": pattern_set.clinical_standard
        })

    # Minimal medication-domain anchors to align with the supervisor's Condition Graph framing.
    # These nodes are guideline-level and do not require that medication variables exist in the
    # released episode time-series schema.
    if pattern_set.disease.lower() == "sepsis":
        med_id = "struct_vasopressors_required"
        pat_id = "pattern_on_vasopressors"
        txt_id = "text_on_vasopressors"
        nodes.extend([
            {
                "id": med_id,
                "type": "structured_indicator",
                "domain": "medication",
                "label": "Vasopressors required to maintain MAP >= 65 mmHg",
                "feature": "vasopressors",
                "threshold": 1,
                "direction": "above",
                "unit": "binary",
                "severity": "severe",
                "source": "Sepsis-3 Septic Shock criteria (Singer et al., JAMA 2016)"
            },
            {
                "id": pat_id,
                "type": "pattern_event",
                "domain": "medication",
                "label": "on_vasopressors",
                "severity": "severe",
                "source": "Sepsis-3 Septic Shock criteria (Singer et al., JAMA 2016)"
            },
            {
                "id": txt_id,
                "type": "text_evidence",
                "domain": "text_evidence",
                "label": "text evidence for vasopressor use",
                "source": "temporal_textual_alignment"
            },
        ])
        edges.extend([
            {"source": med_id, "target": pat_id, "relation": "triggers", "evidence": "septic shock criterion"},
            {"source": txt_id, "target": pat_id, "relation": "supports", "evidence": "note_id/note_hour alignment"},
            {"source": pat_id, "target": condition_id, "relation": "aggregates", "evidence": "Sepsis-3 Septic Shock"},
        ])

        # Symptom/sign anchors (supervisor expects symptom clusters alongside labs/vitals/meds).
        # We use GCS as a measurable proxy for altered mental status (SOFA neurological component).
        sym_id = "struct_gcs_altered_mental_status"
        sym_pat_id = "pattern_altered_mental_status"
        sym_txt_id = "text_altered_mental_status"
        nodes.extend([
            {
                "id": sym_id,
                "type": "structured_indicator",
                "domain": "symptom",
                "label": "gcs Altered mental status: GCS < 15 (SOFA neurological component proxy)",
                "feature": "gcs",
                "threshold": 15,
                "direction": "below",
                "unit": "score",
                "severity": "moderate",
                "source": "Sepsis-3 / SOFA (neurologic dysfunction)"
            },
            {
                "id": sym_pat_id,
                "type": "pattern_event",
                "domain": "symptom",
                "label": "altered_mental_status",
                "severity": "moderate",
                "source": "Sepsis-3 / SOFA (neurologic dysfunction)"
            },
            {
                "id": sym_txt_id,
                "type": "text_evidence",
                "domain": "text_evidence",
                "label": "text evidence for altered mental status",
                "source": "temporal_textual_alignment"
            },
        ])
        edges.extend([
            {"source": sym_id, "target": sym_pat_id, "relation": "triggers", "evidence": "GCS threshold/criterion"},
            {"source": sym_txt_id, "target": sym_pat_id, "relation": "supports", "evidence": "note_id/note_hour alignment"},
            {"source": sym_txt_id, "target": sym_pat_id, "relation": "contradicts", "evidence": "note_id/note_hour alignment"},
            {"source": sym_pat_id, "target": condition_id, "relation": "aggregates", "evidence": "SOFA neurological component"},
        ])

    if pattern_set.disease.lower() == "aki":
        proc_id = "struct_rrt_initiated"
        pat_id = "pattern_rrt_initiated"
        txt_id = "text_rrt_initiated"
        nodes.extend([
            {
                "id": proc_id,
                "type": "structured_indicator",
                "domain": "medication",
                "label": "Renal replacement therapy initiated",
                "feature": "rrt",
                "threshold": 1,
                "direction": "above",
                "unit": "binary",
                "severity": "critical",
                "source": "KDIGO Stage 3 (initiation of RRT)"
            },
            {
                "id": pat_id,
                "type": "pattern_event",
                "domain": "medication",
                "label": "rrt_initiated",
                "severity": "critical",
                "source": "KDIGO Stage 3 (initiation of RRT)"
            },
            {
                "id": txt_id,
                "type": "text_evidence",
                "domain": "text_evidence",
                "label": "text evidence for RRT initiation",
                "source": "temporal_textual_alignment"
            },
        ])
        edges.extend([
            {"source": proc_id, "target": pat_id, "relation": "triggers", "evidence": "KDIGO Stage 3 criterion"},
            {"source": txt_id, "target": pat_id, "relation": "supports", "evidence": "note_id/note_hour alignment"},
            {"source": pat_id, "target": condition_id, "relation": "aggregates", "evidence": "KDIGO Stage 3"},
        ])

    # Minimal multimorbidity anchors to align with the supervisor's Condition Graph framing.
    # These are intended as guideline-level comorbidity nodes (not episode time-series variables).
    if pattern_set.disease.lower() == "sepsis":
        mm_id = "multi_ckd"
        nodes.append(
            {
                "id": mm_id,
                "type": "structured_indicator",
                "domain": "multimorbidity",
                "label": "Chronic kidney disease (CKD) comorbidity",
                "feature": "ckd",
                "threshold": 1,
                "direction": "above",
                "unit": "binary",
                "severity": "moderate",
                "source": "Comorbidity risk factor (multimorbidity interaction)"
            }
        )
        edges.append(
            {
                "source": mm_id,
                "target": condition_id,
                "relation": "contributes_to",
                "evidence": "Reduced physiologic reserve / higher susceptibility"
            }
        )

    if pattern_set.disease.lower() == "aki":
        mm_id = "multi_ckd"
        nodes.append(
            {
                "id": mm_id,
                "type": "structured_indicator",
                "domain": "multimorbidity",
                "label": "Pre-existing chronic kidney disease (CKD)",
                "feature": "ckd",
                "threshold": 1,
                "direction": "above",
                "unit": "binary",
                "severity": "moderate",
                "source": "KDIGO AKI Guidelines (risk factor: CKD)"
            }
        )
        edges.append(
            {
                "source": mm_id,
                "target": condition_id,
                "relation": "contributes_to",
                "evidence": "CKD increases AKI risk and affects recovery"
            }
        )

    graph = {
        "graph_name": graph_name,
        "version": "v1",
        "condition": pattern_set.disease,
        "description": pattern_set.reference,
        "nodes": nodes,
        "edges": edges,
        "sources": [
            {"name": pattern_set.clinical_standard, "reference": pattern_set.reference}
        ],
        "metadata": {
            "generated_by": "build_condition_graphs.py",
            "generated_at": datetime.now().isoformat(timespec="seconds")
        }
    }
    return graph


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    graphs = [
        (SEPSIS_PATTERNS, "sepsis_sirs_graph.json"),
        (AKI_PATTERNS, "aki_kdigo_graph.json"),
        (STROKE_PATTERNS, "stroke_neuro_graph.json"),
        (DELIRIUM_PATTERNS, "delirium_icu_graph.json"),
    ]

    expected = {name for _, name in graphs}
    for existing in OUT_DIR.glob("*.json"):
        if existing.name.startswith("._"):
            existing.unlink()
            continue
        if existing.name not in expected:
            existing.unlink()
            print(f"Removed stale graph: {existing.name}")

    for pattern_set, fname in graphs:
        graph = build_graph(pattern_set, fname)
        out_path = OUT_DIR / fname
        with out_path.open("w") as f:
            json.dump(graph, f, indent=2, ensure_ascii=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
