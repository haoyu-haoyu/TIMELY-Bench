"""
Build condition graphs from pattern templates (opt-in).
"""

import json
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_templates import SEPSIS_PATTERNS, AKI_PATTERNS


OUT_DIR = Path(__file__).resolve().parent / "graphs"


def _to_direction(direction):
    if direction is None:
        return ""
    return getattr(direction, "value", str(direction))


def build_graph(pattern_set, graph_name):
    nodes = []
    edges = []

    condition_id = f"condition_{pattern_set.disease.lower()}"
    nodes.append({
        "id": condition_id,
        "type": "condition",
        "label": pattern_set.disease,
        "source": pattern_set.clinical_standard
    })

    for p in pattern_set.patterns:
        struct_id = f"struct_{p.feature}_{p.name}"
        pattern_id = f"pattern_{p.name}"
        text_id = f"text_{p.name}"

        nodes.append({
            "id": struct_id,
            "type": "structured_indicator",
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
            "label": p.name,
            "severity": p.severity,
            "source": p.clinical_source
        })

        nodes.append({
            "id": text_id,
            "type": "text_evidence",
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
    ]

    for pattern_set, fname in graphs:
        graph = build_graph(pattern_set, fname)
        out_path = OUT_DIR / fname
        with out_path.open("w") as f:
            json.dump(graph, f, indent=2, ensure_ascii=True)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
