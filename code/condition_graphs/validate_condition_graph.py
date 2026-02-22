"""
Validate condition graph JSON files against minimal schema and sanity checks.
"""

import json
from pathlib import Path
import sys
from datetime import datetime

import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pattern_templates import (
    AKI_PATTERNS,
    DELIRIUM_PATTERNS,
    SEPSIS_PATTERNS,
    STROKE_PATTERNS,
)
from episode_schema import VitalSign, LabValue, Intervention

SCHEMA_PATH = Path(__file__).resolve().parent / "condition_graph_schema.json"
GRAPHS_DIR = Path(__file__).resolve().parent / "graphs"

CANONICAL_GRAPH_NAMES = {
    # Canonical condition graph set that is published to final_release.
    "sepsis_sirs_graph.json",
    "aki_kdigo_graph.json",
    "stroke_neuro_graph.json",
    "delirium_icu_graph.json",
}


def load_schema():
    # Do not rely on the process locale encoding (CREATE can default to ASCII).
    # All released JSON artefacts are written as UTF-8.
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def allowed_features():
    feats = set()
    for pset in (SEPSIS_PATTERNS, AKI_PATTERNS, STROKE_PATTERNS, DELIRIUM_PATTERNS):
        for p in pset.patterns:
            if p.feature:
                feats.add(p.feature)
    return feats


def episode_schema_features():
    feats = (
        set(VitalSign.__dataclass_fields__.keys())
        | set(LabValue.__dataclass_fields__.keys())
        | set(Intervention.__dataclass_fields__.keys())
    )
    return feats - {"hour", "timestamp"}


ALIAS_MAP = {
    "meanbp": "mbp",
    "map": "mbp",
}

MISSING_NOTES = {
    "bilirubin_total": "currently unobserved in episode schema; can be promoted to derived after lab-extension pipeline",
    "vasopressors": "derived feature from medication/infusion records (not in current episode time-series columns)",
    "rrt": "derived feature from procedure/treatment records (not in current episode time-series columns)",
    "ckd": "external static comorbidity feature from diagnosis history (not a time-series variable)",
}

STATUS_PRIORITY = {
    "mapped": 0,
    "derived": 1,
    "external_static": 2,
    "unobserved": 3,
}

FEATURE_STATUS_OVERRIDES = {
    "vasopressors": "derived",
    "rrt": "derived",
    "ckd": "external_static",
    "bilirubin_total": "unobserved",
}


def classify_feature_status(feature: str, schema_feats: set) -> str:
    if feature in schema_feats:
        return "mapped"
    if feature in ALIAS_MAP and ALIAS_MAP[feature] in schema_feats:
        return "mapped"
    if feature in FEATURE_STATUS_OVERRIDES:
        return FEATURE_STATUS_OVERRIDES[feature]
    return "unobserved"


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
            else:
                domain = str(n.get("domain", "")).strip().lower()
                # Allow non-template anchors (e.g., medications/procedures) without requiring a
                # corresponding PatternTemplate definition.
                if domain not in {"medication", "multimorbidity", "symptom"} and feat not in allowed_features():
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
        "feature_status": {},
        "status_counts": {
            "mapped": 0,
            "derived": 0,
            "external_static": 0,
            "unobserved": 0,
        },
        "mapping_notes": {},
    }

    all_missing = set()
    global_status = {}
    for gf in graph_files:
        # Explicit UTF-8 avoids UnicodeDecodeError under non-UTF8 locales.
        with gf.open(encoding="utf-8") as f:
            graph = json.load(f)
        struct_feats = sorted({n.get("feature") for n in graph.get("nodes", []) if n.get("type") == "structured_indicator"})
        mapped = {}
        missing = []
        per_graph_status = {}
        for feat in struct_feats:
            status = classify_feature_status(feat, schema_feats)
            per_graph_status[feat] = status
            prev = global_status.get(feat)
            if prev is None or STATUS_PRIORITY[status] < STATUS_PRIORITY[prev]:
                global_status[feat] = status

            if status == "mapped":
                mapped[feat] = feat if feat in schema_feats else ALIAS_MAP[feat]
            else:
                missing.append(feat)
                if feat in MISSING_NOTES:
                    report["mapping_notes"][feat] = MISSING_NOTES[feat]
        all_missing.update(missing)
        report["graphs"][gf.name] = {
            "structured_features": struct_feats,
            "mapped_features": mapped,
            "missing_features": missing,
            "feature_status": per_graph_status,
        }

    report["missing_fields"] = sorted(all_missing)
    report["feature_status"] = dict(sorted(global_status.items()))
    for status in report["status_counts"]:
        report["status_counts"][status] = sum(1 for s in global_status.values() if s == status)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Wrote {output_path}")

    # PASS if every non-mapped feature has a note.
    unresolved = [
        f for f, s in report["feature_status"].items()
        if s != "mapped" and f not in report["mapping_notes"]
    ]
    if unresolved:
        print(f"FAIL: non-mapped fields without notes: {unresolved}")
        return False
    if report["missing_fields"]:
        print(f"WARN: missing fields with notes: {report['missing_fields']}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all-graphs",
        action="store_true",
        help="Validate all JSON files under graphs/ (including draft/non-released graphs).",
    )
    parser.add_argument("--check-mapping", action="store_true")
    parser.add_argument("--mapping-output", type=str, default=str(Path(__file__).resolve().parent / "mapping_report.json"))
    args = parser.parse_args()

    _ = load_schema()  # minimal sanity use
    # Ignore macOS AppleDouble files (resource forks) that can appear after rsync
    # from a Mac client. These are binary and will not decode as JSON.
    graph_files = sorted(
        p for p in GRAPHS_DIR.glob("*.json") if not p.name.startswith("._")
    )
    if not graph_files:
        print("FAIL: no graph files found")
        sys.exit(1)

    if not args.all_graphs:
        graph_files = [p for p in graph_files if p.name in CANONICAL_GRAPH_NAMES]
        if not graph_files:
            print("FAIL: no canonical graph files found")
            sys.exit(1)

    any_fail = False
    for gf in graph_files:
        # Explicit UTF-8 avoids UnicodeDecodeError under non-UTF8 locales.
        with gf.open(encoding="utf-8") as f:
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
