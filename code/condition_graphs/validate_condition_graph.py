"""
Validate condition graph JSON files against minimal schema and sanity checks.
"""

import json
from pathlib import Path
import sys
from datetime import datetime

import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_templates import SEPSIS_PATTERNS, AKI_PATTERNS
from episode_schema import VitalSign, LabValue

SCHEMA_PATH = Path(__file__).resolve().parent / "condition_graph_schema.json"
GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"


def load_schema():
    with SCHEMA_PATH.open() as f:
        return json.load(f)


def allowed_features():
    feats = set()
    for pset in (SEPSIS_PATTERNS, AKI_PATTERNS):
        for p in pset.patterns:
            if p.feature:
                feats.add(p.feature)
    return feats


def episode_schema_features():
    feats = set(VitalSign.__dataclass_fields__.keys()) | set(LabValue.__dataclass_fields__.keys())
    return feats - {"hour", "timestamp"}


ALIAS_MAP = {
    "meanbp": "mbp",
    "map": "mbp",
}

MISSING_NOTES = {
    "bilirubin_total": "not in episode schema (optional, requires lab extension)",
}


def validate_graph(graph):
    errors = []
    required = ["graph_name", "version", "condition", "nodes", "edges", "sources", "metadata"]
    for key in required:
        if key not in graph:
            errors.append(f"missing key: {key}")

    node_ids = set()
    for n in graph.get("nodes", []):
        if "id" not in n or "type" not in n or "label" not in n:
            errors.append("node missing id/type/label")
            continue
        if n["id"] in node_ids:
            errors.append(f"duplicate node id: {n['id']}")
        node_ids.add(n["id"])

        if n.get("type") == "structured_indicator":
            feat = n.get("feature")
            if not feat:
                errors.append(f"structured_indicator missing feature for {n['id']}")
            elif feat not in allowed_features():
                errors.append(f"feature not in templates: {feat}")

    for e in graph.get("edges", []):
        if e.get("source") not in node_ids or e.get("target") not in node_ids:
            errors.append(f"edge references missing node: {e.get('source')} -> {e.get('target')}")

    # required feature coverage by condition
    struct_feats = {n.get("feature") for n in graph.get("nodes", []) if n.get("type") == "structured_indicator"}
    condition = str(graph.get("condition", "")).lower()
    if "aki" in condition:
        if not (struct_feats & {"creatinine", "urineoutput"}):
            errors.append("missing required AKI features: creatinine or urineoutput")
    if "sepsis" in condition:
        if not (struct_feats & {"mbp", "sbp"}):
            errors.append("missing required sepsis hemodynamics: mbp or sbp")
        if not (struct_feats & {"spo2", "lactate"}):
            errors.append("missing required sepsis respiratory/metabolic: spo2 or lactate")

    return errors


def run_mapping_check(graph_files, output_path: Path):
    schema_feats = episode_schema_features()
    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "available_features": sorted(schema_feats),
        "graphs": {},
        "missing_fields": [],
        "mapping_notes": {},
    }

    all_missing = set()
    for gf in graph_files:
        with gf.open() as f:
            graph = json.load(f)
        struct_feats = sorted({n.get("feature") for n in graph.get("nodes", []) if n.get("type") == "structured_indicator"})
        mapped = {}
        missing = []
        for feat in struct_feats:
            if feat in schema_feats:
                mapped[feat] = feat
            elif feat in ALIAS_MAP and ALIAS_MAP[feat] in schema_feats:
                mapped[feat] = ALIAS_MAP[feat]
            else:
                missing.append(feat)
                if feat in MISSING_NOTES:
                    report["mapping_notes"][feat] = MISSING_NOTES[feat]
        all_missing.update(missing)
        report["graphs"][gf.name] = {
            "structured_features": struct_feats,
            "mapped_features": mapped,
            "missing_features": missing,
        }

    report["missing_fields"] = sorted(all_missing)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Wrote {output_path}")

    # PASS if missing fields are all explained in mapping_notes
    unexplained = [f for f in report["missing_fields"] if f not in report["mapping_notes"]]
    if unexplained:
        print(f"FAIL: missing fields without mapping notes: {unexplained}")
        return False
    if report["missing_fields"]:
        print(f"WARN: missing fields with notes: {report['missing_fields']}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-mapping", action="store_true")
    parser.add_argument("--mapping-output", type=str, default=str(Path(__file__).resolve().parent / "mapping_report.json"))
    args = parser.parse_args()

    _ = load_schema()  # minimal sanity use
    graph_files = sorted(GRAPHS_DIR.glob("*.json"))
    if not graph_files:
        print("FAIL: no graph files found")
        sys.exit(1)

    any_fail = False
    for gf in graph_files:
        with gf.open() as f:
            graph = json.load(f)
        errors = validate_graph(graph)
        if errors:
            any_fail = True
            print(f"FAIL: {gf.name}")
            for err in errors:
                print(f"  - {err}")
        else:
            print(f"PASS: {gf.name}")

    mapping_ok = True
    if args.check_mapping:
        mapping_ok = run_mapping_check(graph_files, Path(args.mapping_output))

    sys.exit(1 if any_fail or not mapping_ok else 0)


if __name__ == "__main__":
    main()
