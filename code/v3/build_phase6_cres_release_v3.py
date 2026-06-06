#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import ROOT_DIR, V3_PROCESSED_DIR
from v3.io_utils import relativize_value, write_table


RESULTS_DIR = ROOT_DIR / "results"
CRES_RESULTS_DIR = RESULTS_DIR / "cres_v3"
CRES_PROCESSED_DIR = V3_PROCESSED_DIR / "cres"


MASTER_MANIFEST_PATH = CRES_PROCESSED_DIR / "master_instance_manifest.parquet"
DELIRIUM_META_PATH = V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_stay_metadata.parquet"


REPRESENTATION_PATHS = {
    "aki": {
        "A": V3_PROCESSED_DIR / "aki" / "representations" / "aki_A_anchor_stats.parquet",
        "B1_anchor_index": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B1_anchor_index.parquet",
        "B1_bank": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B1_hourly_sequence_bank.parquet",
        "B2_index": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B2_original_context_index.parquet",
        "B3_anchor_index": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B3_anchor_index.parquet",
        "B3_bank": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B3_state_bank.parquet",
        "phase_labels_hourly": V3_PROCESSED_DIR / "aki" / "state_space" / "aki_phase_labels_hourly.parquet",
        "phase_episode_summary": V3_PROCESSED_DIR / "aki" / "state_space" / "aki_phase_episode_summary.parquet",
        "trajectory_tiers": V3_PROCESSED_DIR / "aki" / "state_space" / "aki_trajectory_tiers.parquet",
        "state_prototypes": V3_PROCESSED_DIR / "aki" / "state_space" / "aki_state_prototypes.parquet",
        "transition_matrix": V3_PROCESSED_DIR / "aki" / "state_space" / "aki_transition_matrix.json",
    },
    "delirium": {
        "A": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_A_anchor_stats.parquet",
        "B1_anchor_index": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B1_anchor_index.parquet",
        "B1_bank": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B1_hourly_sequence_bank.parquet",
        "B2_index": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B2_original_context_index.parquet",
        "B3_anchor_index": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B3_anchor_index.parquet",
        "B3_bank": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B3_state_bank.parquet",
        "phase_labels_hourly": V3_PROCESSED_DIR / "delirium" / "state_space" / "delirium_phase_labels_hourly.parquet",
        "phase_episode_summary": V3_PROCESSED_DIR / "delirium" / "state_space" / "delirium_phase_episode_summary.parquet",
        "trajectory_tiers": V3_PROCESSED_DIR / "delirium" / "state_space" / "delirium_trajectory_tiers.parquet",
        "state_prototypes": V3_PROCESSED_DIR / "delirium" / "state_space" / "delirium_state_prototypes.parquet",
        "transition_matrix": V3_PROCESSED_DIR / "delirium" / "state_space" / "delirium_transition_matrix.json",
    },
    "sepsis": {
        "A": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_A_anchor_stats.parquet",
        "B1_anchor_index": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B1_anchor_index.parquet",
        "B1_bank": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B1_hourly_sequence_bank.parquet",
        "B2_index": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B2_original_context_index.parquet",
        "B3_anchor_index": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B3_anchor_index.parquet",
        "B3_bank": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B3_state_bank.parquet",
        "phase_labels_hourly": V3_PROCESSED_DIR / "sepsis" / "state_space" / "sepsis_phase_labels_hourly.parquet",
        "phase_episode_summary": V3_PROCESSED_DIR / "sepsis" / "state_space" / "sepsis_phase_episode_summary.parquet",
        "trajectory_tiers": V3_PROCESSED_DIR / "sepsis" / "state_space" / "sepsis_trajectory_tiers.parquet",
        "state_prototypes": V3_PROCESSED_DIR / "sepsis" / "state_space" / "sepsis_state_prototypes.parquet",
        "transition_matrix": V3_PROCESSED_DIR / "sepsis" / "state_space" / "sepsis_transition_matrix.json",
    },
}


STROKE_REPRESENTATION_PATHS = {
    "temporal": {
        "A": V3_PROCESSED_DIR / "stroke" / "representations" / "stroke_A_anchor_stats.parquet",
        "B1_anchor_index": V3_PROCESSED_DIR / "stroke" / "representations" / "stroke_B1_anchor_index.parquet",
        "B1_bank": V3_PROCESSED_DIR / "stroke" / "representations" / "stroke_B1_hourly_sequence_bank.parquet",
        "B2_index": V3_PROCESSED_DIR / "stroke" / "representations" / "stroke_B2_original_temporal_index.parquet",
    },
    "retrospective": {
        "B2_index": V3_PROCESSED_DIR / "stroke" / "representations" / "stroke_B2_original_retrospective_index.parquet",
    },
}


def _read_manifest() -> pd.DataFrame:
    return pd.read_parquet(MASTER_MANIFEST_PATH)


def _norm_text(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    return out.replace({"<NA>": pd.NA})


def _norm_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _resolve_rep_paths(condition: str, branch: str, task_mode: str) -> dict[str, str]:
    if condition == "stroke":
        mapping = STROKE_REPRESENTATION_PATHS["retrospective" if task_mode == "retrospective" else "temporal"]
    else:
        mapping = REPRESENTATION_PATHS[condition]
    out: dict[str, str] = {}
    for key, path in mapping.items():
        if branch == "A" and key == "A":
            out["representation_table_path"] = relativize_value(path, root=ROOT_DIR)
        elif branch == "B1" and key in {"B1_anchor_index", "B1_bank"}:
            out[key.lower() + "_path"] = relativize_value(path, root=ROOT_DIR)
        elif branch == "B2_original" and key == "B2_index":
            out["b2_index_path"] = relativize_value(path, root=ROOT_DIR)
        elif branch == "B3" and key in {
            "B3_anchor_index",
            "B3_bank",
            "phase_labels_hourly",
            "phase_episode_summary",
            "trajectory_tiers",
            "state_prototypes",
            "transition_matrix",
        }:
            out[key.lower() + "_path"] = relativize_value(path, root=ROOT_DIR)
    return out


def _repair_master_manifest(master: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    repaired: list[str] = []
    delirium_mask = master["condition"].astype("string").eq("delirium")
    if bool(delirium_mask.any()) and int(master.loc[delirium_mask, "left_censored"].notna().sum()) == 0:
        del_meta = pd.read_parquet(DELIRIUM_META_PATH)[["stay_id", "left_censored"]].copy()
        del_meta["stay_id"] = pd.to_numeric(del_meta["stay_id"], errors="coerce").astype("Int64")
        del_meta["left_censored"] = pd.to_numeric(del_meta["left_censored"], errors="coerce").astype("Int64")
        del_meta = del_meta.dropna(subset=["stay_id"]).drop_duplicates(subset=["stay_id"])
        left_map = del_meta.set_index("stay_id")["left_censored"]
        filled = master.loc[delirium_mask, "stay_id"].map(left_map)
        master.loc[delirium_mask, "left_censored"] = pd.to_numeric(filled, errors="coerce").astype("Int64")
        repaired.append("delirium.left_censored")
    return master, repaired


def _stratification_signature(row: pd.Series) -> str:
    condition = row["condition"]
    if condition == "aki":
        return "trajectory_tier"
    if condition == "delirium":
        return "trajectory_tier|left_censored"
    if condition == "sepsis":
        return "trajectory_tier|onset_confidence|shock_before_sepsis_onset|shock_at_sepsis_onset"
    return "stroke_layer|stroke_tier|stroke_subtype_priority"


def build_eval_metadata_manifest(master: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    eval_cols = [
        "instance_id",
        "condition",
        "task_id",
        "task_mode",
        "layer",
        "stay_id",
        "anchor_hour",
        "horizon_hours",
        "trajectory_tier",
        "mean_template_executable_support_score",
        "n_supported_atypical_flags",
        "left_censored",
        "onset_confidence",
        "shock_before_sepsis_onset",
        "shock_at_sepsis_onset",
        "shock_onset_hour",
        "stroke_layer",
        "stroke_tier",
        "stroke_subtype_priority",
        "stroke_subtype_mixed",
        "stroke_sensitivity_subset",
        "representation_profile",
        "available_representations",
        "has_A",
        "has_B1",
        "has_B2_original",
        "has_B3",
        "phase5_available",
    ]
    eval_df = master[eval_cols].copy()
    eval_df["condition"] = _norm_text(eval_df["condition"])
    eval_df["task_id"] = _norm_text(eval_df["task_id"])
    eval_df["task_mode"] = _norm_text(eval_df["task_mode"])
    eval_df["layer"] = _norm_text(eval_df["layer"])
    eval_df["trajectory_tier"] = _norm_text(eval_df["trajectory_tier"])
    eval_df["onset_confidence"] = _norm_text(eval_df["onset_confidence"])
    eval_df["stroke_layer"] = _norm_text(eval_df["stroke_layer"])
    eval_df["stroke_tier"] = _norm_text(eval_df["stroke_tier"])
    eval_df["stroke_subtype_priority"] = _norm_text(eval_df["stroke_subtype_priority"])
    eval_df["stroke_subtype_mixed"] = _norm_text(eval_df["stroke_subtype_mixed"])
    for col in ["left_censored", "shock_before_sepsis_onset", "shock_at_sepsis_onset"]:
        eval_df[col] = _norm_int(eval_df[col])
    eval_df["stratification_signature"] = eval_df.apply(_stratification_signature, axis=1).astype("string")
    eval_df["requires_stratified_reporting"] = True

    path = CRES_PROCESSED_DIR / "cres_eval_metadata_manifest.parquet"
    write_table(eval_df, path)

    summary = {
        "rows": int(len(eval_df)),
        "unique_stays": int(eval_df["stay_id"].nunique()),
        "path": relativize_value(path, root=ROOT_DIR),
        "by_condition": (
            eval_df.groupby("condition", sort=False)
            .agg(rows=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        ),
        "trajectory_tier_rows": int(eval_df["trajectory_tier"].notna().sum()),
        "left_censored_rows": int(eval_df["left_censored"].notna().sum()),
        "onset_confidence_rows": int(eval_df["onset_confidence"].notna().sum()),
        "stroke_layer_rows": int(eval_df["stroke_layer"].notna().sum()),
    }
    return eval_df, summary


def _build_branch_manifest(master: pd.DataFrame, branch: str) -> tuple[pd.DataFrame, dict[str, object]]:
    branch_flag = {
        "A": "has_A",
        "B1": "has_B1",
        "B2_original": "has_B2_original",
        "B3": "has_B3",
    }[branch]
    df = master.loc[master[branch_flag].fillna(False)].copy()
    keep_cols = [
        "instance_id",
        "condition",
        "task_id",
        "task_family",
        "task_mode",
        "layer",
        "stay_id",
        "subject_id",
        "hadm_id",
        "anchor_hour",
        "horizon_hours",
        "primary_label_key",
        "primary_label_type",
        "primary_label_binary",
        "primary_label_numeric",
        "primary_label_text",
        "representation_profile",
        "available_representations",
        "task_source_file",
    ]
    df = df[keep_cols].copy()
    df["representation_branch"] = branch
    df["lookup_key_fields"] = "instance_id|stay_id|anchor_hour|task_id"

    records = []
    for condition, task_mode in zip(df["condition"].tolist(), df["task_mode"].tolist()):
        records.append(_resolve_rep_paths(condition, branch, task_mode))
    rep_paths = pd.DataFrame.from_records(records)
    df = pd.concat([df.reset_index(drop=True), rep_paths], axis=1)

    path = CRES_PROCESSED_DIR / f"cres_{branch}_manifest.parquet"
    write_table(df, path)

    summary = {
        "representation_branch": branch,
        "rows": int(len(df)),
        "unique_stays": int(df["stay_id"].nunique()),
        "path": relativize_value(path, root=ROOT_DIR),
        "by_condition": (
            df.groupby("condition", sort=False)
            .agg(rows=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        ),
        "by_task_mode": (
            df.groupby(["condition", "task_mode"], sort=False)
            .agg(rows=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        ),
    }
    return df, summary


def _existing_branch_summary(path: Path, branch: str, master: pd.DataFrame) -> dict[str, object]:
    df = master.loc[master[{"A": "has_A", "B1": "has_B1", "B2_original": "has_B2_original", "B3": "has_B3"}[branch]].fillna(False)].copy()
    return {
        "representation_branch": branch,
        "rows": int(len(df)),
        "unique_stays": int(df["stay_id"].nunique()),
        "path": relativize_value(path, root=ROOT_DIR),
        "by_condition": (
            df.groupby("condition", sort=False)
            .agg(rows=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        ),
        "by_task_mode": (
            df.groupby(["condition", "task_mode"], sort=False)
            .agg(rows=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        ),
        "reused_existing_file": True,
    }


def _value_counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.astype("string").fillna("<NA>").value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _bool_counts_dict(series: pd.Series) -> dict[str, int]:
    counts = series.astype("boolean").astype("string").fillna("<NA>").value_counts(dropna=False)
    return {str(k): int(v) for k, v in counts.items()}


def _condition_summary(master: pd.DataFrame, condition: str) -> dict[str, object]:
    subset = master.loc[master["condition"].eq(condition)].copy()
    summary = {
        "condition": condition,
        "instances": int(len(subset)),
        "unique_stays": int(subset["stay_id"].nunique()),
        "tasks": (
            subset.groupby(["task_id", "task_mode", "layer"], sort=False)
            .agg(instances=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
            .reset_index()
            .to_dict(orient="records")
        ),
        "representations": {
            "A_rows": int(subset["has_A"].fillna(False).sum()),
            "B1_rows": int(subset["has_B1"].fillna(False).sum()),
            "B2_original_rows": int(subset["has_B2_original"].fillna(False).sum()),
            "B3_rows": int(subset["has_B3"].fillna(False).sum()),
        },
    }
    if condition in {"aki", "delirium", "sepsis"}:
        summary["trajectory_tier"] = _value_counts_dict(subset["trajectory_tier"])
    if condition == "delirium":
        summary["left_censored"] = _bool_counts_dict(subset["left_censored"])
    if condition == "sepsis":
        summary["onset_confidence"] = _value_counts_dict(subset["onset_confidence"])
        summary["shock_before_sepsis_onset"] = _bool_counts_dict(subset["shock_before_sepsis_onset"])
        summary["shock_at_sepsis_onset"] = _bool_counts_dict(subset["shock_at_sepsis_onset"])
    if condition == "stroke":
        summary["stroke_layer"] = _value_counts_dict(subset["stroke_layer"])
        summary["stroke_tier"] = _value_counts_dict(subset["stroke_tier"])
        summary["stroke_subtype_priority"] = _value_counts_dict(subset["stroke_subtype_priority"])
    return summary


def _missing_output_flags() -> list[str]:
    flags: list[str] = []
    expected_paths = [MASTER_MANIFEST_PATH]
    for condition, mapping in REPRESENTATION_PATHS.items():
        del condition
        expected_paths.extend(mapping.values())
    for mapping in STROKE_REPRESENTATION_PATHS.values():
        expected_paths.extend(mapping.values())
    for path in expected_paths:
        if not path.exists() and not path.with_name(f"{path.name}.parts").exists():
            flags.append(f"missing_output::{path.relative_to(ROOT_DIR)}")
    return flags


def build_phase6c_6e() -> dict[str, object]:
    CRES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CRES_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    master = _read_manifest()
    master, repaired_master_fields = _repair_master_manifest(master)
    if repaired_master_fields:
        write_table(master, MASTER_MANIFEST_PATH)

    eval_meta_df, eval_meta_summary = build_eval_metadata_manifest(master)

    branch_summaries = {}
    for branch in ["A", "B1", "B2_original", "B3"]:
        _, branch_summary = _build_branch_manifest(master, branch)
        branch_summaries[branch] = branch_summary

    condition_summaries = {}
    for condition in ["aki", "delirium", "sepsis", "stroke"]:
        condition_summary = _condition_summary(master, condition)
        condition_summaries[condition] = condition_summary
        path = CRES_RESULTS_DIR / f"{condition}_cres_build_summary.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(relativize_value(condition_summary, root=ROOT_DIR), f, indent=2)

    eval_summary_path = CRES_RESULTS_DIR / "cres_eval_stratification_summary.json"
    release_summary_path = CRES_RESULTS_DIR / "cres_release_manifest_summary.json"
    combined_summary_path = CRES_RESULTS_DIR / "cres_v3_build_summary.json"

    eval_payload = {
        "phase6_scope": "phase6c_eval_metadata",
        **eval_meta_summary,
        "flags": [],
    }
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(relativize_value(eval_payload, root=ROOT_DIR), f, indent=2)

    release_payload = {
        "phase6_scope": "phase6d_release_manifests",
        "branches": branch_summaries,
        "outputs": relativize_value(
            {
                "A_manifest": CRES_PROCESSED_DIR / "cres_A_manifest.parquet",
                "B1_manifest": CRES_PROCESSED_DIR / "cres_B1_manifest.parquet",
                "B2_original_manifest": CRES_PROCESSED_DIR / "cres_B2_original_manifest.parquet",
                "B3_manifest": CRES_PROCESSED_DIR / "cres_B3_manifest.parquet",
            },
            root=ROOT_DIR,
        ),
        "flags": [],
    }
    with open(release_summary_path, "w", encoding="utf-8") as f:
        json.dump(relativize_value(release_payload, root=ROOT_DIR), f, indent=2)

    combined_payload = {
        "phase6_scope": "assembly_complete_pre_eval_runs",
        "master_manifest_rows": int(len(master)),
        "master_unique_stays": int(master["stay_id"].nunique()),
        "repaired_master_fields": repaired_master_fields,
        "eval_metadata": eval_meta_summary,
        "branch_manifests": branch_summaries,
        "conditions": condition_summaries,
        "outputs": relativize_value(
            {
                "master_manifest": MASTER_MANIFEST_PATH,
                "eval_metadata_manifest": CRES_PROCESSED_DIR / "cres_eval_metadata_manifest.parquet",
                "A_manifest": CRES_PROCESSED_DIR / "cres_A_manifest.parquet",
                "B1_manifest": CRES_PROCESSED_DIR / "cres_B1_manifest.parquet",
                "B2_original_manifest": CRES_PROCESSED_DIR / "cres_B2_original_manifest.parquet",
                "B3_manifest": CRES_PROCESSED_DIR / "cres_B3_manifest.parquet",
                "eval_summary": eval_summary_path,
                "release_summary": release_summary_path,
                "combined_summary": combined_summary_path,
            },
            root=ROOT_DIR,
        ),
        "flags": _missing_output_flags(),
    }
    with open(combined_summary_path, "w", encoding="utf-8") as f:
        json.dump(relativize_value(combined_payload, root=ROOT_DIR), f, indent=2)
    return combined_payload


def main() -> None:
    summary = build_phase6c_6e()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
