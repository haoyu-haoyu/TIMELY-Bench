#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import DEFAULT_V3_COHORT_FILE, ROOT_DIR, V3_PROCESSED_DIR
from v3.io_utils import relativize_value, write_table


RESULTS_DIR = ROOT_DIR / "results"
CRES_RESULTS_DIR = RESULTS_DIR / "cres_v3"
CRES_PROCESSED_DIR = V3_PROCESSED_DIR / "cres"


REPRESENTATION_PROFILES = {
    "full_ab1b2b3": {
        "available_representations": ["A", "B1", "B2_original", "B3"],
        "has_A": True,
        "has_B1": True,
        "has_B2_original": True,
        "has_B3": True,
        "phase5_available": True,
    },
    "stroke_temporal_ab1b2": {
        "available_representations": ["A", "B1", "B2_original"],
        "has_A": True,
        "has_B1": True,
        "has_B2_original": True,
        "has_B3": False,
        "phase5_available": False,
    },
    "stroke_retrospective_b2_only": {
        "available_representations": ["B2_original"],
        "has_A": False,
        "has_B1": False,
        "has_B2_original": True,
        "has_B3": False,
        "phase5_available": False,
    },
}


TASK_SPECS = [
    {
        "condition": "aki",
        "task_id": "AKI-T1",
        "task_family": "temporal_progression",
        "task_mode": "temporal",
        "layer": "single_layer",
        "source_path": V3_PROCESSED_DIR / "aki" / "tasks" / "aki_stage2plus_instances.parquet",
        "anchor_col": "anchor_hour",
        "horizon_col": "horizon_hours",
        "eligible_col": "eligible",
        "label_key": "label",
        "label_type": "binary",
        "representation_profile": "full_ab1b2b3",
    },
    {
        "condition": "aki",
        "task_id": "AKI-S1",
        "task_family": "support_escalation_proxy",
        "task_mode": "temporal",
        "layer": "single_layer",
        "source_path": V3_PROCESSED_DIR / "aki" / "tasks" / "aki_rrt_proxy_instances.parquet",
        "anchor_col": "anchor_hour",
        "horizon_col": "horizon_hours",
        "eligible_col": "eligible",
        "label_key": "label",
        "label_type": "binary",
        "representation_profile": "full_ab1b2b3",
    },
    {
        "condition": "delirium",
        "task_id": "DEL-T1",
        "task_family": "persistence",
        "task_mode": "temporal",
        "layer": "single_layer",
        "source_path": V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_persistence_instances.parquet",
        "anchor_col": "prediction_hour",
        "horizon_col": "horizon_hours",
        "eligible_col": "eligible",
        "label_key": "label",
        "label_type": "binary",
        "representation_profile": "full_ab1b2b3",
    },
    {
        "condition": "delirium",
        "task_id": "DEL-S1",
        "task_family": "resolution",
        "task_mode": "temporal",
        "layer": "single_layer",
        "source_path": V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_resolution_instances.parquet",
        "anchor_col": "prediction_hour",
        "horizon_col": "horizon_hours",
        "eligible_col": "eligible",
        "label_key": "label",
        "label_type": "binary",
        "representation_profile": "full_ab1b2b3",
    },
    {
        "condition": "sepsis",
        "task_id": "SEP-T1",
        "task_family": "shock_progression",
        "task_mode": "temporal",
        "layer": "single_layer",
        "source_path": V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_shock_instances.parquet",
        "anchor_col": "prediction_hour",
        "horizon_col": "horizon_hours",
        "label_key": "label",
        "label_type": "binary",
        "representation_profile": "full_ab1b2b3",
    },
    {
        "condition": "sepsis",
        "task_id": "SEP-S1",
        "task_family": "lactate_clearance",
        "task_mode": "temporal",
        "layer": "single_layer",
        "source_path": V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_lactate_clearance_instances.parquet",
        "anchor_col": "prediction_hour",
        "horizon_col": "horizon_hours",
        "label_key": "label",
        "label_type": "binary",
        "representation_profile": "full_ab1b2b3",
    },
    {
        "condition": "stroke",
        "task_id": "S-T1",
        "task_family": "strength_worsening",
        "task_mode": "temporal",
        "layer": "Layer1",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T1_instances.parquet",
        "anchor_col": "anchor_hour",
        "horizon_col": "horizon_hours",
        "label_key": "label_strength_worsening",
        "label_type": "binary",
        "representation_profile": "stroke_temporal_ab1b2",
    },
    {
        "condition": "stroke",
        "task_id": "S-T2",
        "task_family": "laterality_reasoning",
        "task_mode": "temporal",
        "layer": "Layer1",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T2_instances.parquet",
        "anchor_col": "anchor_hour",
        "label_key": "label_affected_side",
        "label_type": "categorical",
        "representation_profile": "stroke_temporal_ab1b2",
    },
    {
        "condition": "stroke",
        "task_id": "S-T3",
        "task_family": "imaging_clinical_consistency",
        "task_mode": "temporal",
        "layer": "Layer1",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T3_instances.parquet",
        "anchor_col": "anchor_hour",
        "label_key": "label_consistency",
        "label_type": "categorical",
        "representation_profile": "stroke_temporal_ab1b2",
    },
    {
        "condition": "stroke",
        "task_id": "S-T4",
        "task_family": "sequence_signature",
        "task_mode": "temporal",
        "layer": "Layer1",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T4_instances.parquet",
        "anchor_col": "anchor_hour",
        "label_key": "label_sequence_signature",
        "label_type": "categorical",
        "representation_profile": "stroke_temporal_ab1b2",
    },
    {
        "condition": "stroke",
        "task_id": "S-R1",
        "task_family": "mechanism_attribution",
        "task_mode": "retrospective",
        "layer": "Layer2",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R1_instances.parquet",
        "label_key": "label_mechanism",
        "label_type": "categorical",
        "representation_profile": "stroke_retrospective_b2_only",
    },
    {
        "condition": "stroke",
        "task_id": "S-R2",
        "task_family": "secondary_prevention_strategy",
        "task_mode": "retrospective",
        "layer": "Layer2",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R2_instances.parquet",
        "label_key": "label_strategy",
        "label_type": "categorical",
        "representation_profile": "stroke_retrospective_b2_only",
    },
    {
        "condition": "stroke",
        "task_id": "S-R3",
        "task_family": "nihss_retrospective_extraction",
        "task_mode": "retrospective",
        "layer": "Layer2",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R3_instances.parquet",
        "label_key": "label_nihss_peak",
        "label_type": "numeric",
        "representation_profile": "stroke_retrospective_b2_only",
    },
    {
        "condition": "stroke",
        "task_id": "S-R4",
        "task_family": "complication_understanding",
        "task_mode": "retrospective",
        "layer": "Layer2",
        "source_path": V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R4_instances.parquet",
        "label_key": "label_any_complication",
        "label_type": "binary",
        "representation_profile": "stroke_retrospective_b2_only",
    },
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _nullable_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Float64")


def _nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _normalize_text(series: pd.Series) -> pd.Series:
    out = series.astype("string")
    return out.replace({"<NA>": pd.NA})


def _format_anchor(anchor: object) -> str:
    if pd.isna(anchor):
        return "na"
    value = float(anchor)
    if value.is_integer():
        return str(int(value))
    return str(value)


def _build_instance_id(task_id: str, stay_id: int, anchor_hour: object | None) -> str:
    if anchor_hour is None or pd.isna(anchor_hour):
        return f"{task_id}::{stay_id}"
    return f"{task_id}::{stay_id}::h{_format_anchor(anchor_hour)}"


def _cohort_lookup() -> pd.DataFrame:
    cohort = pd.read_csv(DEFAULT_V3_COHORT_FILE, low_memory=False)
    keep_cols = [
        "stay_id",
        "subject_id",
        "hadm_id",
        "stroke_subtype_priority",
        "stroke_subtype_mixed",
    ]
    cohort = cohort[[c for c in keep_cols if c in cohort.columns]].copy()
    cohort["stay_id"] = _nullable_int(cohort["stay_id"])
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")
    if "subject_id" in cohort.columns:
        cohort["subject_id"] = _nullable_int(cohort["subject_id"])
    if "hadm_id" in cohort.columns:
        cohort["hadm_id"] = _nullable_int(cohort["hadm_id"])
    return cohort.drop_duplicates(subset=["stay_id"])


def _load_phase5_metadata() -> dict[str, pd.DataFrame]:
    meta: dict[str, pd.DataFrame] = {}

    for condition in ["aki", "delirium", "sepsis"]:
        tiers = _read_table(V3_PROCESSED_DIR / condition / "state_space" / f"{condition}_trajectory_tiers.parquet")
        tiers["stay_id"] = _nullable_int(tiers["stay_id"])
        tiers = tiers.dropna(subset=["stay_id"]).copy()
        tiers["stay_id"] = tiers["stay_id"].astype("int64")
        meta[f"{condition}_tiers"] = tiers

    del_meta = _read_table(V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_stay_metadata.parquet")
    del_meta = del_meta[["stay_id", "left_censored"]].copy()
    del_meta["stay_id"] = _nullable_int(del_meta["stay_id"])
    del_meta["left_censored"] = _nullable_int(del_meta["left_censored"])
    del_meta = del_meta.dropna(subset=["stay_id"]).copy()
    del_meta["stay_id"] = del_meta["stay_id"].astype("int64")
    meta["delirium_meta"] = del_meta

    sep_meta = _read_table(V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_stay_metadata.parquet")
    sep_meta = sep_meta[
        [
            "stay_id",
            "onset_confidence",
            "shock_before_sepsis_onset",
            "shock_at_sepsis_onset",
            "shock_onset_hour",
        ]
    ].copy()
    sep_meta["stay_id"] = _nullable_int(sep_meta["stay_id"])
    sep_meta["shock_before_sepsis_onset"] = _nullable_int(sep_meta["shock_before_sepsis_onset"])
    sep_meta["shock_at_sepsis_onset"] = _nullable_int(sep_meta["shock_at_sepsis_onset"])
    sep_meta["shock_onset_hour"] = _nullable_float(sep_meta["shock_onset_hour"])
    sep_meta = sep_meta.dropna(subset=["stay_id"]).copy()
    sep_meta["stay_id"] = sep_meta["stay_id"].astype("int64")
    meta["sepsis_meta"] = sep_meta

    stroke_meta = _read_table(V3_PROCESSED_DIR / "stroke" / "stroke_stay_metadata.parquet")
    stroke_meta["stay_id"] = _nullable_int(stroke_meta["stay_id"])
    stroke_meta = stroke_meta.dropna(subset=["stay_id"]).copy()
    stroke_meta["stay_id"] = stroke_meta["stay_id"].astype("int64")
    meta["stroke_meta"] = stroke_meta
    return meta


def _attach_condition_metadata(df: pd.DataFrame, spec: dict, meta: dict[str, pd.DataFrame]) -> pd.DataFrame:
    condition = spec["condition"]
    if condition in {"aki", "delirium", "sepsis"}:
        tiers = meta[f"{condition}_tiers"]
        df = df.merge(tiers, on="stay_id", how="left")
    if condition == "delirium":
        df = df.merge(meta["delirium_meta"], on="stay_id", how="left", suffixes=("", "_delirium_meta"))
        if "left_censored_delirium_meta" in df.columns:
            base = df["left_censored"] if "left_censored" in df.columns else pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
            df["left_censored"] = _nullable_int(base).combine_first(_nullable_int(df["left_censored_delirium_meta"]))
            df = df.drop(columns=["left_censored_delirium_meta"])
    if condition == "sepsis":
        df = df.merge(meta["sepsis_meta"], on="stay_id", how="left", suffixes=("", "_sepsis_meta"))
        for col in [
            "onset_confidence",
            "shock_before_sepsis_onset",
            "shock_at_sepsis_onset",
            "shock_onset_hour",
        ]:
            shadow = f"{col}_sepsis_meta"
            if shadow in df.columns:
                if col in {"shock_before_sepsis_onset", "shock_at_sepsis_onset"}:
                    df[col] = _nullable_int(df.get(col, pd.Series(pd.array([pd.NA] * len(df), dtype="Int64")))).combine_first(_nullable_int(df[shadow]))
                elif col == "shock_onset_hour":
                    df[col] = _nullable_float(df.get(col, pd.Series(pd.array([pd.NA] * len(df), dtype="Float64")))).combine_first(_nullable_float(df[shadow]))
                else:
                    base = df.get(col, pd.Series(pd.array([pd.NA] * len(df), dtype="string")))
                    df[col] = _normalize_text(base).combine_first(_normalize_text(df[shadow]))
                df = df.drop(columns=[shadow])
    if condition == "stroke":
        stroke_meta = meta["stroke_meta"]
        keep_cols = [
            "stay_id",
            "stroke_subtype_priority",
            "stroke_subtype_mixed",
            "tier",
            "layer1_eligible",
            "layer2_eligible",
            "has_discharge_summary",
            "primary_dx_flag",
        ]
        use_cols = [c for c in keep_cols if c in stroke_meta.columns]
        df = df.merge(stroke_meta[use_cols], on="stay_id", how="left", suffixes=("", "_stroke_meta"))
    return df


def _make_primary_label(df: pd.DataFrame, label_key: str, label_type: str) -> pd.DataFrame:
    df["primary_label_key"] = label_key
    df["primary_label_type"] = label_type
    df["primary_label_binary"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
    df["primary_label_numeric"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Float64"))
    df["primary_label_text"] = pd.Series(pd.array([pd.NA] * len(df), dtype="string"))
    if label_type == "binary":
        df["primary_label_binary"] = _nullable_int(df[label_key])
    elif label_type == "numeric":
        df["primary_label_numeric"] = _nullable_float(df[label_key])
    else:
        df["primary_label_text"] = _normalize_text(df[label_key])
    return df


def _available_repr_fields(profile_name: str) -> dict[str, object]:
    profile = REPRESENTATION_PROFILES[profile_name]
    return {
        "representation_profile": profile_name,
        "available_representations": "|".join(profile["available_representations"]),
        "has_A": profile["has_A"],
        "has_B1": profile["has_B1"],
        "has_B2_original": profile["has_B2_original"],
        "has_B3": profile["has_B3"],
        "phase5_available": profile["phase5_available"],
    }


def _build_task_frame(spec: dict, cohort: pd.DataFrame, meta: dict[str, pd.DataFrame]) -> pd.DataFrame:
    source_path = Path(spec["source_path"])
    df = _read_table(source_path)

    if spec.get("eligible_col") and spec["eligible_col"] in df.columns:
        eligible = df[spec["eligible_col"]]
        if str(eligible.dtype) in {"bool", "boolean"}:
            df = df.loc[eligible.fillna(False)].copy()
        else:
            df = df.loc[pd.to_numeric(eligible, errors="coerce").fillna(0).astype(int).eq(1)].copy()

    df["stay_id"] = _nullable_int(df["stay_id"])
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")

    if "hadm_id" in df.columns:
        df["hadm_id"] = _nullable_int(df["hadm_id"])

    df = df.merge(cohort, on="stay_id", how="left", suffixes=("", "_cohort"))
    if "hadm_id_cohort" in df.columns:
        df["hadm_id"] = df["hadm_id"].combine_first(df["hadm_id_cohort"]) if "hadm_id" in df.columns else df["hadm_id_cohort"]
        df = df.drop(columns=["hadm_id_cohort"])
    if "subject_id" not in df.columns:
        df["subject_id"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
    else:
        df["subject_id"] = _nullable_int(df["subject_id"])
    if "hadm_id" not in df.columns:
        df["hadm_id"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))
    else:
        df["hadm_id"] = _nullable_int(df["hadm_id"])

    df = _attach_condition_metadata(df, spec, meta)

    anchor_col = spec.get("anchor_col")
    if anchor_col and anchor_col in df.columns:
        df["anchor_hour"] = _nullable_float(df[anchor_col])
    else:
        df["anchor_hour"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Float64"))
    horizon_col = spec.get("horizon_col")
    if horizon_col and horizon_col in df.columns:
        df["horizon_hours"] = _nullable_int(df[horizon_col])
    else:
        df["horizon_hours"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))

    df = _make_primary_label(df, spec["label_key"], spec["label_type"])
    repr_fields = _available_repr_fields(spec["representation_profile"])
    for k, v in repr_fields.items():
        df[k] = v

    if "trajectory_tier" not in df.columns:
        df["trajectory_tier"] = pd.Series(pd.array([pd.NA] * len(df), dtype="string"))
    else:
        df["trajectory_tier"] = _normalize_text(df["trajectory_tier"])
    if "mean_template_executable_support_score" in df.columns:
        df["mean_template_executable_support_score"] = _nullable_float(df["mean_template_executable_support_score"])
    else:
        df["mean_template_executable_support_score"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Float64"))
    if "n_supported_atypical_flags" in df.columns:
        df["n_supported_atypical_flags"] = _nullable_int(df["n_supported_atypical_flags"])
    else:
        df["n_supported_atypical_flags"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))

    for col in [
        "left_censored",
        "shock_before_sepsis_onset",
        "shock_at_sepsis_onset",
        "layer1_eligible",
        "layer2_eligible",
        "has_discharge_summary",
        "primary_dx_flag",
    ]:
        if col in df.columns:
            df[col] = _nullable_int(df[col])
        else:
            df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int64"))

    for col in [
        "onset_confidence",
        "stroke_subtype_priority",
        "stroke_subtype_mixed",
        "tier",
    ]:
        if col in df.columns:
            df[col] = _normalize_text(df[col])
        else:
            df[col] = pd.Series(pd.array([pd.NA] * len(df), dtype="string"))

    if "shock_onset_hour" in df.columns:
        df["shock_onset_hour"] = _nullable_float(df["shock_onset_hour"])
    else:
        df["shock_onset_hour"] = pd.Series(pd.array([pd.NA] * len(df), dtype="Float64"))

    df["condition"] = spec["condition"]
    df["task_id"] = spec["task_id"]
    df["task_family"] = spec["task_family"]
    df["task_mode"] = spec["task_mode"]
    df["layer"] = spec["layer"]
    df["stroke_layer"] = df["layer"].where(df["condition"].eq("stroke"), pd.NA).astype("string")
    df["stroke_tier"] = df["tier"].where(df["condition"].eq("stroke"), pd.NA).astype("string")
    df["stroke_sensitivity_subset"] = False
    df["task_source_file"] = str(source_path.relative_to(ROOT_DIR))
    df["instance_id"] = [
        _build_instance_id(spec["task_id"], int(stay_id), anchor)
        for stay_id, anchor in zip(df["stay_id"].tolist(), df["anchor_hour"].tolist())
    ]

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
        "has_A",
        "has_B1",
        "has_B2_original",
        "has_B3",
        "phase5_available",
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
        "task_source_file",
    ]
    return df[keep_cols].copy()


def _schema_payload() -> dict[str, object]:
    return {
        "phase6_scope": "assembly_mvp",
        "description": "CRES benchmark assembly schema for TIMELY-Bench v3. Baseline evaluation runs are intentionally out of scope at this stage.",
        "manifest_path": "data/processed/v3/cres/master_instance_manifest.parquet",
        "summary_path": "results/cres_v3/cres_master_manifest_summary.json",
        "fields": [
            {"name": "instance_id", "dtype": "string", "description": "Unique CRES instance identifier."},
            {"name": "condition", "dtype": "string", "description": "Condition family: aki, delirium, sepsis, stroke."},
            {"name": "task_id", "dtype": "string", "description": "Condition-specific task identifier."},
            {"name": "task_family", "dtype": "string", "description": "High-level task family used in CRES reporting."},
            {"name": "task_mode", "dtype": "string", "description": "temporal or retrospective."},
            {"name": "layer", "dtype": "string", "description": "single_layer, Layer1, or Layer2."},
            {"name": "stay_id", "dtype": "int64", "description": "ICU stay identifier."},
            {"name": "subject_id", "dtype": "Int64", "description": "Patient identifier from cohort_v3."},
            {"name": "hadm_id", "dtype": "Int64", "description": "Hospital admission identifier."},
            {"name": "anchor_hour", "dtype": "Float64", "description": "Prediction anchor hour when applicable."},
            {"name": "horizon_hours", "dtype": "Int64", "description": "Prediction horizon when applicable."},
            {"name": "primary_label_key", "dtype": "string", "description": "Column name used as primary task target."},
            {"name": "primary_label_type", "dtype": "string", "description": "binary, categorical, or numeric."},
            {"name": "primary_label_binary", "dtype": "Int64", "description": "Binary target value if task is binary."},
            {"name": "primary_label_numeric", "dtype": "Float64", "description": "Numeric target value if task is numeric."},
            {"name": "primary_label_text", "dtype": "string", "description": "Categorical/text target value if task is categorical."},
            {"name": "representation_profile", "dtype": "string", "description": "Named availability profile for A/B1/B2/B3 branches."},
            {"name": "available_representations", "dtype": "string", "description": "Pipe-delimited list of representations available to the instance."},
            {"name": "has_A", "dtype": "bool", "description": "A baseline summary available."},
            {"name": "has_B1", "dtype": "bool", "description": "B1 hourly sequence available."},
            {"name": "has_B2_original", "dtype": "bool", "description": "B2 original context available."},
            {"name": "has_B3", "dtype": "bool", "description": "B3 state-space representation available."},
            {"name": "phase5_available", "dtype": "bool", "description": "Whether Phase 5 state-space is in scope for this instance."},
            {"name": "trajectory_tier", "dtype": "string", "description": "Phase 5 trajectory tier for AKI/Delirium/Sepsis."},
            {"name": "mean_template_executable_support_score", "dtype": "Float64", "description": "Mean executable template support score for the stay trajectory."},
            {"name": "n_supported_atypical_flags", "dtype": "Int64", "description": "Count of supported atypical variants flagged for the stay."},
            {"name": "left_censored", "dtype": "Int64", "description": "Delirium-specific left-censoring flag."},
            {"name": "onset_confidence", "dtype": "string", "description": "Sepsis onset confidence: high or low."},
            {"name": "shock_before_sepsis_onset", "dtype": "Int64", "description": "Sepsis metadata stratifier."},
            {"name": "shock_at_sepsis_onset", "dtype": "Int64", "description": "Sepsis metadata stratifier."},
            {"name": "shock_onset_hour", "dtype": "Float64", "description": "First shock-after-sepsis-onset hour when applicable."},
            {"name": "stroke_layer", "dtype": "string", "description": "Stroke Layer1/Layer2 reporting field."},
            {"name": "stroke_tier", "dtype": "string", "description": "Stroke tier A/B/C used in CRES reporting."},
            {"name": "stroke_subtype_priority", "dtype": "string", "description": "Priority-based stroke subtype assignment."},
            {"name": "stroke_subtype_mixed", "dtype": "string", "description": "Mixed-aware stroke subtype assignment."},
            {"name": "stroke_sensitivity_subset", "dtype": "bool", "description": "Whether instance belongs to pure-ischaemic sensitivity subset. Currently false in master manifest."},
            {"name": "task_source_file", "dtype": "string", "description": "Relative path to the source task parquet."},
        ],
        "representation_profiles": REPRESENTATION_PROFILES,
        "task_catalog": [
            {
                "task_id": spec["task_id"],
                "condition": spec["condition"],
                "task_family": spec["task_family"],
                "task_mode": spec["task_mode"],
                "layer": spec["layer"],
                "primary_label_key": spec["label_key"],
                "primary_label_type": spec["label_type"],
                "representation_profile": spec["representation_profile"],
                "source_path": str(Path(spec["source_path"]).relative_to(ROOT_DIR)),
            }
            for spec in TASK_SPECS
        ],
    }


def _schema_markdown(schema: dict[str, object]) -> str:
    lines = [
        "# CRES v3 Schema",
        "",
        "This schema freezes the Phase 6A assembly scope for TIMELY-Bench v3.",
        "",
        "- Scope: assembly only",
        "- Out of scope: baseline model evaluation runs",
        "",
        "## Representation Profiles",
        "",
    ]
    for name, profile in schema["representation_profiles"].items():
        lines.append(f"- `{name}`: `{ '|'.join(profile['available_representations']) }`")
    lines.extend(["", "## Manifest Fields", ""])
    for field in schema["fields"]:
        lines.append(f"- `{field['name']}` (`{field['dtype']}`): {field['description']}")
    lines.extend(["", "## Task Catalog", ""])
    for task in schema["task_catalog"]:
        lines.append(
            f"- `{task['task_id']}`: condition=`{task['condition']}`, mode=`{task['task_mode']}`, "
            f"layer=`{task['layer']}`, primary_label=`{task['primary_label_key']}`, "
            f"representations=`{task['representation_profile']}`"
        )
    return "\n".join(lines) + "\n"


def build_phase6a_6b() -> dict[str, object]:
    CRES_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CRES_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    cohort = _cohort_lookup()
    meta = _load_phase5_metadata()

    manifest_frames = [_build_task_frame(spec, cohort, meta) for spec in TASK_SPECS]
    manifest = pd.concat(manifest_frames, ignore_index=True)
    manifest = manifest.sort_values(["condition", "task_id", "stay_id", "anchor_hour"], kind="mergesort").reset_index(drop=True)

    manifest_path = CRES_PROCESSED_DIR / "master_instance_manifest.parquet"
    schema_json_path = CRES_RESULTS_DIR / "cres_schema_v3.json"
    schema_md_path = CRES_RESULTS_DIR / "cres_schema_v3.md"
    summary_path = CRES_RESULTS_DIR / "cres_master_manifest_summary.json"

    write_table(manifest, manifest_path)

    schema = _schema_payload()
    with open(schema_json_path, "w", encoding="utf-8") as f:
        json.dump(relativize_value(schema, root=ROOT_DIR), f, indent=2)
    schema_md_path.write_text(_schema_markdown(schema), encoding="utf-8")

    by_condition = (
        manifest.groupby("condition", sort=False)
        .agg(instances=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )
    by_task = (
        manifest.groupby(["condition", "task_id", "task_mode", "layer"], sort=False)
        .agg(instances=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )
    by_profile = (
        manifest.groupby(["representation_profile", "condition"], sort=False)
        .agg(instances=("instance_id", "size"), unique_stays=("stay_id", "nunique"))
        .reset_index()
        .to_dict(orient="records")
    )

    summary = {
        "phase6_scope": "assembly_a_b_only",
        "manifest_rows": int(len(manifest)),
        "unique_stays": int(manifest["stay_id"].nunique()),
        "conditions": by_condition,
        "tasks": by_task,
        "representation_profiles": by_profile,
        "outputs": relativize_value(
            {
                "manifest": manifest_path,
                "schema_json": schema_json_path,
                "schema_md": schema_md_path,
                "summary": summary_path,
            },
            root=ROOT_DIR,
        ),
        "flags": [],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    summary = build_phase6a_6b()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
