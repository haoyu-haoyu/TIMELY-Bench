#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cres_v3.constants import (  # type: ignore
    DEFAULT_GRAPH_NODE_REGISTRY,
    DEFAULT_KNOWLEDGE_GAP_REPORT,
    DEFAULT_KNOWLEDGE_SUMMARY,
    DEFAULT_REFERENCE_CONDITION_GRAPHS,
    DEFAULT_REFERENCE_PHYSIOLOGY_TEMPLATES,
    V3_KNOWLEDGE_DIR,
    ensure_cres_v3_directories,
)
from cres_v3.knowledge_mappings import (  # type: ignore
    CORE_REQUIREMENTS,
    heuristic_mapping,
    node_override,
    normalize_text,
)
from v3.constants import DEFAULT_DIAGNOSIS_PATHWAY_EVENTS, DEFAULT_FEATURE_DICTIONARY_CSV, ROOT_DIR  # type: ignore
from v3.io_utils import portable_path, read_table, relativize_value, table_exists, write_table  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build executable knowledge-layer artefacts for TIMELY-Bench v3.")
    p.add_argument("--condition-graphs-json", default=str(DEFAULT_REFERENCE_CONDITION_GRAPHS))
    p.add_argument("--physiology-templates-json", default=str(DEFAULT_REFERENCE_PHYSIOLOGY_TEMPLATES))
    p.add_argument("--feature-dictionary-csv", default=str(DEFAULT_FEATURE_DICTIONARY_CSV))
    p.add_argument("--pathway-events", default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS))
    p.add_argument("--registry-csv", default=str(DEFAULT_GRAPH_NODE_REGISTRY))
    p.add_argument("--gap-report-json", default=str(DEFAULT_KNOWLEDGE_GAP_REPORT))
    p.add_argument("--summary-json", default=str(DEFAULT_KNOWLEDGE_SUMMARY))
    return p.parse_args()


def _load_feature_dict(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows[str(row["name"]).strip()] = row
    return rows


def _load_pathway_rules(path: Path) -> set[str]:
    if not table_exists(path):
        return set()
    df = read_table(path)
    rules = set()
    for col in ("event_rule", "event_name"):
        if col in df.columns:
            for value in df[col].dropna().astype(str):
                value = value.strip()
                if value:
                    rules.add(value)
    return rules


def _condition_slug(condition_name: str) -> str:
    return "stroke_proxy" if condition_name == "stroke" else condition_name


def _is_core_requirement_met(condition_slug: str, registry_rows: list[dict]) -> tuple[list[str], list[str]]:
    covered = {
        row["mapped_feature_name"]
        for row in registry_rows
        if row["support_level"] in {"core_executable", "proxy_executable"}
    }
    required = CORE_REQUIREMENTS[condition_slug]
    missing = sorted(required - covered)
    return sorted(required & covered), missing


def _resolve_support(
    condition_key: str,
    node_id: str,
    node_label: str,
    source_raw: str,
    feature_dict: dict[str, dict],
    pathway_rules: set[str],
) -> tuple[str, str, str, str]:
    override = node_override(condition_key, node_id)
    if override is not None:
        mapped, support, mode, notes = override
    else:
        mapped, support, mode, notes = heuristic_mapping(condition_key, node_id, node_label, source_raw)

    if mapped in feature_dict and support == "proxy_executable" and mode == "derived_feature":
        return mapped, support, "direct_feature", notes
    if mapped in feature_dict and support == "reference_only":
        return mapped, "proxy_executable", "direct_feature", "Promoted by feature-dictionary match."
    if mapped.startswith("diagnosis_pathway::") and mapped.split("::", 1)[1] in pathway_rules:
        return mapped, "proxy_executable", "event_rule", notes
    if mapped in {
        "nephrotoxic_drug_active",
        "contrast_exposure",
        "diuretic_exposure",
        "ace_arb_exposure",
        "sleep_wake_note_proxy",
        "hallucination_note_proxy",
        "stroke_cohort_membership",
        "cognitive_impairment_context",
        "sedation_burden",
    }:
        return mapped, support, mode, notes
    return mapped, support, mode, notes


def _build_registry_rows(
    graphs_obj: dict,
    feature_dict: dict[str, dict],
    pathway_rules: set[str],
) -> list[dict]:
    rows: list[dict] = []
    for condition_key, spec in graphs_obj["conditions"].items():
        condition_slug = _condition_slug(condition_key)
        for node in spec.get("nodes", []):
            node_id = str(node.get("id", "")).strip()
            node_label = str(node.get("label", "")).strip()
            source_raw = str(node.get("mimic_source", "")).strip()
            itemid_raw = node.get("mimic_itemid")
            mapped, support, mode, notes = _resolve_support(
                condition_key,
                node_id,
                node_label,
                source_raw,
                feature_dict,
                pathway_rules,
            )
            rows.append(
                {
                    "condition": condition_slug,
                    "node_id": node_id,
                    "node_label": node_label,
                    "node_type": str(node.get("type", "")),
                    "reference_source": str(node.get("ref", "")),
                    "mimic_source_raw": source_raw,
                    "mimic_itemid_raw": json.dumps(itemid_raw, ensure_ascii=False),
                    "mapped_feature_name": mapped,
                    "support_level": support,
                    "implementation_mode": mode,
                    "blocking": False,
                    "notes": notes,
                }
            )
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    for condition, items in by_condition.items():
        _, missing = _is_core_requirement_met(condition, items)
        for req in missing:
            rows.append(
                {
                    "condition": condition,
                    "node_id": f"__required::{req}",
                    "node_label": f"Required executable feature: {req}",
                    "node_type": "stage",
                    "reference_source": "wave1_requirement",
                    "mimic_source_raw": "",
                    "mimic_itemid_raw": "",
                    "mapped_feature_name": req,
                    "support_level": "reference_only",
                    "implementation_mode": "not_implemented",
                    "blocking": True,
                    "notes": "Core requirement missing from executable subgraph.",
                }
            )
    return rows


def _phase_needs_blocking(condition_slug: str, template_obj: dict, registry_rows: list[dict]) -> list[dict]:
    available = {
        row["mapped_feature_name"]
        for row in registry_rows
        if row["condition"] == condition_slug and row["support_level"] != "reference_only"
    }
    blocking_rows: list[dict] = []
    if condition_slug == "aki":
        phase_requirements = {
            "aki_phase_0": {"creatinine", "urineoutput", "map_merged"},
            "aki_phase_1": {"creatinine", "urineoutput", "kdigo_stage"},
            "aki_phase_2": {"creatinine", "urineoutput", "rrt_active"},
            "aki_phase_3": {"creatinine", "urineoutput"},
        }
    elif condition_slug == "delirium":
        phase_requirements = {
            "del_phase_0": {"rass", "gcs_total"},
            "del_phase_1": {"delirium_assessment", "rass"},
            "del_phase_2": {"delirium_assessment", "rass", "gcs_total"},
            "del_phase_3": {"delirium_assessment", "rass"},
        }
    else:
        phase_requirements = {
            "stroke_phase_0": {"gcs_total", "rass", "stroke_cohort_membership"},
            "stroke_phase_1": {"gcs_total", "rass", "map_merged"},
            "stroke_phase_2": {"gcs_total", "ventilation_status", "sedation_burden"},
            "stroke_phase_3": {"gcs_total", "rass", "ventilation_status"},
        }
    for phase in template_obj["conditions"][condition_slug.replace("_proxy", "")]["phases"]:
        phase_id = str(phase.get("phase_id", "")).strip()
        for req in sorted(phase_requirements.get(phase_id, set()) - available):
            blocking_rows.append(
                {
                    "condition": condition_slug,
                    "node_id": f"__phase_required::{phase_id}::{req}",
                    "node_label": f"Required phase feature: {req}",
                    "node_type": "stage",
                    "reference_source": "phase_requirement",
                    "mimic_source_raw": "",
                    "mimic_itemid_raw": "",
                    "mapped_feature_name": req,
                    "support_level": "reference_only",
                    "implementation_mode": "not_implemented",
                    "blocking": True,
                    "notes": f"Phase {phase_id} cannot be executed without {req}.",
                }
            )
    return blocking_rows


def _build_executable_graph(spec: dict, registry_rows: list[dict], condition_slug: str) -> dict:
    support_by_node = {
        row["node_id"]: row
        for row in registry_rows
        if row["condition"] == condition_slug and not row["node_id"].startswith("__required::") and not row["node_id"].startswith("__phase_required::")
    }
    executable_nodes = []
    reference_nodes = []
    for node in spec.get("nodes", []):
        node_id = str(node.get("id", "")).strip()
        registry = support_by_node.get(node_id)
        if registry is None:
            continue
        merged = dict(node)
        merged["mapped_feature_name"] = registry["mapped_feature_name"]
        merged["support_level"] = registry["support_level"]
        merged["implementation_mode"] = registry["implementation_mode"]
        merged["execution_notes"] = registry["notes"]
        if registry["support_level"] == "reference_only":
            reference_nodes.append(merged)
        else:
            executable_nodes.append(merged)
    executable_ids = {n["id"] for n in executable_nodes}
    executable_edges = []
    dropped_edges = []
    for edge in spec.get("edges", []):
        src = str(edge.get("from", "")).strip()
        dst = str(edge.get("to", "")).strip()
        if src in executable_ids and dst in executable_ids:
            executable_edges.append(edge)
        else:
            dropped_edges.append(edge)
    return {
        "condition": condition_slug,
        "name": spec.get("name", condition_slug),
        "references": spec.get("references", []),
        "nodes": executable_nodes,
        "reference_only_nodes": reference_nodes,
        "edges": executable_edges,
        "dropped_edges": dropped_edges,
    }


def _map_template_marker(condition_slug: str, marker_name: str, mimic_source: str, feature_dict: dict[str, dict]) -> tuple[str, str, str, str]:
    mapped, support, mode, notes = heuristic_mapping(condition_slug.replace("_proxy", ""), marker_name, marker_name, mimic_source)
    if mapped in feature_dict and support == "reference_only":
        return mapped, "proxy_executable", "direct_feature", "Promoted by feature dictionary."
    return mapped, support, mode, notes


def _build_executable_template(spec: dict, registry_rows: list[dict], feature_dict: dict[str, dict], condition_slug: str) -> dict:
    phase_out = []
    for phase in spec.get("phases", []):
        executable_trajectories = []
        reference_trajectories = []
        for traj in phase.get("trajectories", []):
            mapped, support, mode, notes = _map_template_marker(
                condition_slug,
                str(traj.get("marker", "")),
                str(traj.get("mimic_source", "")),
                feature_dict,
            )
            merged = dict(traj)
            merged["mapped_feature_name"] = mapped
            merged["support_level"] = support
            merged["implementation_mode"] = mode
            merged["execution_notes"] = notes
            if support == "reference_only":
                reference_trajectories.append(merged)
            else:
                executable_trajectories.append(merged)
        phase_out.append(
            {
                "phase_id": phase.get("phase_id"),
                "name": phase.get("name"),
                "time_relative_to_anchor": phase.get("time_relative_to_anchor"),
                "description": phase.get("description"),
                "trajectories": executable_trajectories,
                "reference_only_trajectories": reference_trajectories,
            }
        )
    out = {
        "condition": condition_slug,
        "name": spec.get("name", condition_slug),
        "anchor_event": spec.get("anchor_event"),
        "anchor_mimic": spec.get("anchor_mimic"),
        "total_typical_duration": spec.get("total_typical_duration"),
        "references": spec.get("references", []),
        "phases": phase_out,
        "atypical_variants": spec.get("atypical_variants", []),
    }
    return out


def main() -> None:
    args = parse_args()
    ensure_cres_v3_directories()

    graphs_obj = json.loads(Path(args.condition_graphs_json).read_text(encoding="utf-8"))
    templates_obj = json.loads(Path(args.physiology_templates_json).read_text(encoding="utf-8"))
    feature_dict = _load_feature_dict(Path(args.feature_dictionary_csv))
    pathway_rules = _load_pathway_rules(Path(args.pathway_events))

    registry_rows = _build_registry_rows(graphs_obj, feature_dict, pathway_rules)
    for condition_slug in ("aki", "delirium", "stroke_proxy"):
        registry_rows.extend(_phase_needs_blocking(condition_slug, templates_obj, registry_rows))
    registry_df = pd.DataFrame(registry_rows).sort_values(["condition", "blocking", "node_id"], kind="mergesort")

    registry_written = write_table(registry_df, args.registry_csv, index=False)

    gap_report = {
        "conditions": {},
        "inputs": {
            "condition_graphs_json": portable_path(args.condition_graphs_json, root=ROOT_DIR),
            "physiology_templates_json": portable_path(args.physiology_templates_json, root=ROOT_DIR),
            "feature_dictionary_csv": portable_path(args.feature_dictionary_csv, root=ROOT_DIR),
            "pathway_events": portable_path(args.pathway_events, root=ROOT_DIR),
        },
    }

    summary = {
        "registry_csv": portable_path(registry_written, root=ROOT_DIR),
        "graph_outputs": {},
        "template_outputs": {},
        "blocking_conditions": [],
    }

    for condition_key, spec in graphs_obj["conditions"].items():
        condition_slug = _condition_slug(condition_key)
        graph_obj = _build_executable_graph(spec, registry_rows, condition_slug)
        template_spec = templates_obj["conditions"][condition_key]
        template_obj = _build_executable_template(template_spec, registry_rows, feature_dict, condition_slug)

        graph_path = V3_KNOWLEDGE_DIR / f"condition_graph_{condition_slug}_executable.json"
        template_path = V3_KNOWLEDGE_DIR / f"physiology_template_{condition_slug}_executable.json"
        graph_path.write_text(json.dumps(graph_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        template_path.write_text(json.dumps(template_obj, ensure_ascii=False, indent=2), encoding="utf-8")
        summary["graph_outputs"][condition_slug] = portable_path(graph_path, root=ROOT_DIR)
        summary["template_outputs"][condition_slug] = portable_path(template_path, root=ROOT_DIR)

        cond_rows = registry_df[registry_df["condition"] == condition_slug].copy()
        support_counts = Counter(cond_rows["support_level"].astype(str))
        implementation_counts = Counter(cond_rows["implementation_mode"].astype(str))
        required_present, required_missing = _is_core_requirement_met(condition_slug, cond_rows.to_dict(orient="records"))
        phase_blockers = cond_rows[cond_rows["node_id"].astype(str).str.startswith("__phase_required::")]
        node_blockers = cond_rows[cond_rows["blocking"] == True]  # noqa: E712
        gap_report["conditions"][condition_slug] = {
            "support_counts": dict(support_counts),
            "implementation_counts": dict(implementation_counts),
            "required_present": required_present,
            "required_missing": required_missing,
            "n_executable_nodes": int(len(graph_obj["nodes"])),
            "n_reference_only_nodes": int(len(graph_obj["reference_only_nodes"])),
            "n_executable_edges": int(len(graph_obj["edges"])),
            "n_dropped_edges": int(len(graph_obj["dropped_edges"])),
            "blocking_items": node_blockers[
                [
                    "node_id",
                    "mapped_feature_name",
                    "support_level",
                    "implementation_mode",
                    "notes",
                ]
            ].to_dict(orient="records"),
            "phase_blocker_count": int(len(phase_blockers)),
        }
        if len(node_blockers) > 0:
            summary["blocking_conditions"].append(condition_slug)

    gap_path = Path(args.gap_report_json)
    gap_path.parent.mkdir(parents=True, exist_ok=True)
    gap_path.write_text(
        json.dumps(relativize_value(gap_report, root=ROOT_DIR), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(relativize_value(summary, root=ROOT_DIR), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {registry_written}")
    print(f"Wrote {gap_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
