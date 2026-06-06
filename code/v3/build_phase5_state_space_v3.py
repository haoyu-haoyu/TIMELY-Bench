#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import DEFAULT_V3_COHORT_FILE, ROOT_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR  # type: ignore
from v3.io_utils import iter_table_chunks, read_table, relativize_value, write_table  # type: ignore


PROJECT_ROOT = ROOT_DIR.parent
KNOWLEDGE_DIR = ROOT_DIR / "results" / "v3" / "knowledge"


CONDITION_CONFIGS = {
    "aki": {
        "template_json": KNOWLEDGE_DIR / "physiology_template_aki_executable.json",
        "b3_anchor_index": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B3_anchor_index.parquet",
        "b3_state_bank": V3_PROCESSED_DIR / "aki" / "representations" / "aki_B3_state_bank.parquet",
        "onsets_parquet": V3_PROCESSED_DIR / "aki" / "tasks" / "aki_kdigo_onsets_v3.parquet",
        "kdigo_parquet": V3_PROCESSED_DIR / "aki" / "tasks" / "kdigo_staged_v3.parquet",
        "cohort_file": DEFAULT_V3_COHORT_FILE,
        "outputs_dir": V3_PROCESSED_DIR / "aki" / "state_space",
        "summary_json": V3_RESULTS_DIR / "aki" / "aki_state_space_build_summary.json",
        "align_anchor_col": "stage1_onset_hour",
        "core_features": [
            "map_merged",
            "lactate",
            "creatinine",
            "urineoutput",
            "bun",
            "potassium",
            "bicarbonate",
            "rrt_active",
            "fluid_balance",
            "ckd",
            "kdigo_stage_ffill",
        ],
    },
    "delirium": {
        "template_json": KNOWLEDGE_DIR / "physiology_template_delirium_executable.json",
        "b3_anchor_index": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B3_anchor_index.parquet",
        "b3_state_bank": V3_PROCESSED_DIR / "delirium" / "representations" / "delirium_B3_state_bank.parquet",
        "metadata_parquet": V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_stay_metadata.parquet",
        "delirium_hourly_parquet": V3_PROCESSED_DIR / "hourly_features" / "delirium_neuro_hourly_v3.parquet",
        "restraint_hourly_parquet": V3_PROCESSED_DIR / "hourly_features" / "restraint_hourly_v3.parquet",
        "outputs_dir": V3_PROCESSED_DIR / "delirium" / "state_space",
        "summary_json": V3_RESULTS_DIR / "delirium" / "delirium_state_space_build_summary.json",
        "align_anchor_col": "delirium_onset_hour",
        "core_features": [
            "rass",
            "gcs_total",
            "glucose_merged",
            "delirium_assessment",
            "delirium_positive",
            "delirium_negative",
            "delirium_uta",
            "restraint_active",
            "sedation_burden",
            "delirium_resolution",
            "heart_rate",
            "sbp",
            "mbp",
            "propofol_rate",
            "midazolam_rate",
            "fentanyl_rate",
            "left_censored",
        ],
    },
    "sepsis": {
        "template_json": KNOWLEDGE_DIR / "physiology_template_sepsis_executable.json",
        "b3_anchor_index": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B3_anchor_index.parquet",
        "b3_state_bank": V3_PROCESSED_DIR / "sepsis" / "representations" / "sepsis_B3_state_bank.parquet",
        "metadata_parquet": V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_stay_metadata.parquet",
        "shock_hourly_parquet": V3_PROCESSED_DIR / "sepsis" / "tasks" / "septic_shock_hourly_v3.parquet",
        "outputs_dir": V3_PROCESSED_DIR / "sepsis" / "state_space",
        "summary_json": V3_RESULTS_DIR / "sepsis" / "sepsis_state_space_build_summary.json",
        "align_anchor_col": "sepsis_onset_hour",
        "core_features": [
            "sepsis_onset",
            "temperature_c",
            "wbc",
            "lactate",
            "sofa_total",
            "map_merged",
            "sofa_respiration",
            "creatinine",
            "bilirubin_total",
            "urineoutput",
            "vasopressors_active",
            "fluid_balance",
            "is_septic_shock",
            "shock_after_onset",
            "onset_confidence",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 Phase 5 template-driven state-space outputs.")
    p.add_argument("--condition", required=True, choices=sorted(CONDITION_CONFIGS.keys()))
    return p.parse_args()


def _parts_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}.parts")


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _load_template(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_signed_hours(token: str) -> int | None:
    token = token.strip().lower()
    m = re.fullmatch(r"([+-]?\d+)\s*([hd])", token)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        value *= 24
    return value


def _phase_window_from_text(text: str) -> tuple[int | None, int | None]:
    normalized = text.strip().lower().replace("−", "-")
    if "to" in normalized:
        left, right = [s.strip() for s in normalized.split("to", 1)]
        start = _parse_signed_hours(left)
        end_match = re.match(r"([+-]?\d+\s*[hd])", right)
        end = _parse_signed_hours(end_match.group(1)) if end_match else None
        return start, end
    if normalized.startswith("before"):
        return None, 0
    m = re.search(r"([+-]?\d+\s*[hd])\s+onward", normalized)
    if m:
        return _parse_signed_hours(m.group(1)), None
    return None, None


def _phase_supported_features(template: dict, available_columns: set[str]) -> dict[str, list[str]]:
    supported: dict[str, list[str]] = {}
    for phase in template.get("phases", []):
        phase_id = str(phase.get("phase_id", "")).strip()
        cols: list[str] = []
        for traj in phase.get("trajectories", []):
            feat = str(traj.get("mapped_feature_name", "")).strip()
            if feat and feat in available_columns:
                cols.append(feat)
        supported[phase_id] = sorted(dict.fromkeys(cols))
    return supported


def _future_nonpositive_resolution(flags_neg: np.ndarray, flags_pos: np.ndarray, start_idx: int, window: int) -> bool:
    end = min(len(flags_neg), start_idx + window)
    if end - start_idx < window:
        return False
    return bool(flags_pos[start_idx:end].sum() == 0 and flags_neg[start_idx:end].sum() >= 1)


def _load_aki_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    template = _load_template(Path(cfg["template_json"]))
    onset_df = read_table(cfg["onsets_parquet"])[["stay_id", "stage1_onset_hour", "stage2plus_onset_hour", "stage3_onset_hour"]].copy()
    onset_df["stay_id"] = pd.to_numeric(onset_df["stay_id"], errors="coerce").astype("Int64")
    onset_df = onset_df.dropna(subset=["stay_id", "stage1_onset_hour"]).copy()
    onset_df["stay_id"] = onset_df["stay_id"].astype("int64")
    for col in ["stage1_onset_hour", "stage2plus_onset_hour", "stage3_onset_hour"]:
        onset_df[col] = pd.to_numeric(onset_df[col], errors="coerce")

    kdigo_df = read_table(cfg["kdigo_parquet"])[["stay_id", "hour", "aki_stage", "aki_stage_ffill"]].copy()
    kdigo_df["stay_id"] = pd.to_numeric(kdigo_df["stay_id"], errors="coerce").astype("Int64")
    kdigo_df["hour"] = pd.to_numeric(kdigo_df["hour"], errors="coerce")
    kdigo_df = kdigo_df.dropna(subset=["stay_id", "hour"]).copy()
    kdigo_df["stay_id"] = kdigo_df["stay_id"].astype("int64")
    kdigo_df["hour"] = kdigo_df["hour"].astype("int64")

    cohort_df = read_table(cfg["cohort_file"])
    cohort_keep = [c for c in ["stay_id", "ckd"] if c in cohort_df.columns]
    cohort_df = cohort_df[cohort_keep].copy()
    cohort_df["stay_id"] = pd.to_numeric(cohort_df["stay_id"], errors="coerce").astype("Int64")
    cohort_df = cohort_df.dropna(subset=["stay_id"]).copy()
    cohort_df["stay_id"] = cohort_df["stay_id"].astype("int64")
    if "ckd" in cohort_df.columns:
        cohort_df["ckd"] = pd.to_numeric(cohort_df["ckd"], errors="coerce").fillna(0).astype(int)

    stay_ids = set(onset_df["stay_id"].tolist())
    b3_frames: list[pd.DataFrame] = []
    keep_cols = [
        "stay_id",
        "state_hour",
        "map_merged",
        "lactate",
        "creatinine",
        "urineoutput",
        "bun",
        "potassium",
        "bicarbonate",
        "rrt_active",
        "fluid_balance",
    ]
    for chunk in iter_table_chunks(cfg["b3_state_bank"]):
        cols = [c for c in keep_cols if c in chunk.columns]
        df = chunk[cols].copy()
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        df["state_hour"] = pd.to_numeric(df["state_hour"], errors="coerce")
        df = df.dropna(subset=["stay_id", "state_hour"]).copy()
        df["stay_id"] = df["stay_id"].astype("int64")
        df["state_hour"] = df["state_hour"].astype("int64")
        df = df.loc[df["stay_id"].isin(stay_ids)].copy()
        if not df.empty:
            b3_frames.append(df)
    if not b3_frames:
        raise RuntimeError("No B3 state rows found for AKI Phase 5")
    b3_df = pd.concat(b3_frames, ignore_index=True)

    merged = b3_df.merge(onset_df, on="stay_id", how="inner")
    merged = merged.merge(kdigo_df, left_on=["stay_id", "state_hour"], right_on=["stay_id", "hour"], how="left")
    merged = merged.drop(columns=["hour"], errors="ignore")
    merged = merged.merge(cohort_df, on="stay_id", how="left")
    if "ckd" not in merged.columns:
        merged["ckd"] = 0
    merged["relative_hour"] = merged["state_hour"] - merged["stage1_onset_hour"]
    merged = merged.loc[(merged["relative_hour"] >= -48) & (merged["relative_hour"] <= 168)].copy()
    merged["kdigo_stage_ffill"] = pd.to_numeric(merged["aki_stage_ffill"], errors="coerce")
    merged["kdigo_stage"] = pd.to_numeric(merged["aki_stage"], errors="coerce")
    merged["rrt_active"] = pd.to_numeric(merged["rrt_active"], errors="coerce")
    return merged, onset_df, cohort_df, template


def _load_delirium_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    template = _load_template(Path(cfg["template_json"]))
    meta_df = read_table(cfg["metadata_parquet"])[
        [
            "stay_id",
            "delirium_onset_hour",
            "left_censored",
            "has_any_positive",
            "n_positive_hours",
            "n_negative_hours",
            "n_uta_hours",
            "n_rass_hours",
            "n_gcs_hours",
            "n_restraint_hours",
            "has_rass_support",
            "has_gcs_support",
            "has_restraint_support",
        ]
    ].copy()
    meta_df["stay_id"] = pd.to_numeric(meta_df["stay_id"], errors="coerce").astype("Int64")
    meta_df["delirium_onset_hour"] = pd.to_numeric(meta_df["delirium_onset_hour"], errors="coerce")
    meta_df = meta_df.dropna(subset=["stay_id", "delirium_onset_hour"]).copy()
    meta_df["stay_id"] = meta_df["stay_id"].astype("int64")
    meta_df["delirium_onset_hour"] = meta_df["delirium_onset_hour"].astype("int64")
    for col in [
        "left_censored",
        "has_any_positive",
        "n_positive_hours",
        "n_negative_hours",
        "n_uta_hours",
        "n_rass_hours",
        "n_gcs_hours",
        "n_restraint_hours",
        "has_rass_support",
        "has_gcs_support",
        "has_restraint_support",
    ]:
        meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce").fillna(0).astype(int)

    stay_ids = set(meta_df["stay_id"].tolist())
    keep_cols = [
        "stay_id",
        "state_hour",
        "heart_rate",
        "sbp",
        "mbp",
        "glucose_merged",
        "gcs_total",
        "rass",
        "delirium_assessment",
        "propofol_rate",
        "midazolam_rate",
        "fentanyl_rate",
    ]
    b3_frames: list[pd.DataFrame] = []
    for chunk in iter_table_chunks(cfg["b3_state_bank"]):
        cols = [c for c in keep_cols if c in chunk.columns]
        df = chunk[cols].copy()
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        df["state_hour"] = pd.to_numeric(df["state_hour"], errors="coerce")
        df = df.dropna(subset=["stay_id", "state_hour"]).copy()
        df["stay_id"] = df["stay_id"].astype("int64")
        df["state_hour"] = df["state_hour"].astype("int64")
        df = df.loc[df["stay_id"].isin(stay_ids)].copy()
        if not df.empty:
            b3_frames.append(df)
    if not b3_frames:
        raise RuntimeError("No B3 state rows found for Delirium Phase 5")
    b3_df = pd.concat(b3_frames, ignore_index=True)

    delirium_hourly = read_table(cfg["delirium_hourly_parquet"])[
        ["stay_id", "hour", "rass", "delirium_positive", "delirium_negative", "delirium_uta", "cam_component_recorded"]
    ].copy()
    delirium_hourly["stay_id"] = pd.to_numeric(delirium_hourly["stay_id"], errors="coerce").astype("Int64")
    delirium_hourly["hour"] = pd.to_numeric(delirium_hourly["hour"], errors="coerce")
    delirium_hourly = delirium_hourly.dropna(subset=["stay_id", "hour"]).copy()
    delirium_hourly["stay_id"] = delirium_hourly["stay_id"].astype("int64")
    delirium_hourly["hour"] = delirium_hourly["hour"].astype("int64")
    delirium_hourly = delirium_hourly.loc[delirium_hourly["stay_id"].isin(stay_ids)].copy()
    for col in ["delirium_positive", "delirium_negative", "delirium_uta", "cam_component_recorded"]:
        delirium_hourly[col] = pd.to_numeric(delirium_hourly[col], errors="coerce").fillna(0).astype(int)
    delirium_hourly["rass_hourly"] = pd.to_numeric(delirium_hourly["rass"], errors="coerce")
    delirium_hourly = delirium_hourly.drop(columns=["rass"], errors="ignore")

    restraint_hourly = read_table(cfg["restraint_hourly_parquet"])[["stay_id", "hour", "restraint_active"]].copy()
    restraint_hourly["stay_id"] = pd.to_numeric(restraint_hourly["stay_id"], errors="coerce").astype("Int64")
    restraint_hourly["hour"] = pd.to_numeric(restraint_hourly["hour"], errors="coerce")
    restraint_hourly = restraint_hourly.dropna(subset=["stay_id", "hour"]).copy()
    restraint_hourly["stay_id"] = restraint_hourly["stay_id"].astype("int64")
    restraint_hourly["hour"] = restraint_hourly["hour"].astype("int64")
    restraint_hourly = restraint_hourly.loc[restraint_hourly["stay_id"].isin(stay_ids)].copy()
    restraint_hourly["restraint_active"] = pd.to_numeric(restraint_hourly["restraint_active"], errors="coerce").fillna(0).astype(int)

    merged = b3_df.merge(meta_df, on="stay_id", how="inner")
    merged = merged.merge(delirium_hourly, left_on=["stay_id", "state_hour"], right_on=["stay_id", "hour"], how="left")
    merged = merged.drop(columns=["hour"], errors="ignore")
    merged = merged.merge(restraint_hourly, left_on=["stay_id", "state_hour"], right_on=["stay_id", "hour"], how="left")
    merged = merged.drop(columns=["hour"], errors="ignore")
    merged["relative_hour"] = merged["state_hour"] - merged["delirium_onset_hour"]
    merged = merged.loc[(merged["relative_hour"] >= -48) & (merged["relative_hour"] <= 168)].copy()

    merged["rass_hourly"] = pd.to_numeric(merged.get("rass_hourly"), errors="coerce")
    merged["rass"] = pd.to_numeric(merged.get("rass"), errors="coerce")
    merged.loc[merged["rass"].isna(), "rass"] = merged.loc[merged["rass"].isna(), "rass_hourly"]
    merged = merged.drop(columns=["rass_hourly"], errors="ignore")

    for col in ["propofol_rate", "midazolam_rate", "fentanyl_rate"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    merged["sedation_burden"] = merged["propofol_rate"] + merged["midazolam_rate"] + merged["fentanyl_rate"]
    merged["restraint_active"] = pd.to_numeric(merged["restraint_active"], errors="coerce").fillna(0).astype(int)
    merged["glucose_merged"] = pd.to_numeric(merged["glucose_merged"], errors="coerce")
    merged["gcs_total"] = pd.to_numeric(merged["gcs_total"], errors="coerce")
    merged["delirium_positive"] = pd.to_numeric(merged["delirium_positive"], errors="coerce").fillna(0).astype(int)
    merged["delirium_negative"] = pd.to_numeric(merged["delirium_negative"], errors="coerce").fillna(0).astype(int)
    merged["delirium_uta"] = pd.to_numeric(merged["delirium_uta"], errors="coerce").fillna(0).astype(int)
    merged["cam_component_recorded"] = pd.to_numeric(merged["cam_component_recorded"], errors="coerce").fillna(0).astype(int)
    merged["delirium_assessment"] = merged["delirium_positive"]

    positive_hours = (
        merged.loc[merged["delirium_positive"] == 1, ["stay_id", "state_hour"]]
        .groupby("stay_id", sort=False)["state_hour"]
        .max()
        .reset_index(name="last_positive_hour")
    )
    resolution_rows: list[dict[str, int | None]] = []
    grouped = (
        merged[["stay_id", "state_hour", "delirium_positive", "delirium_negative"]]
        .drop_duplicates(subset=["stay_id", "state_hour"])
        .sort_values(["stay_id", "state_hour"], kind="mergesort")
        .groupby("stay_id", sort=False)
    )
    for stay_id, g in grouped:
        upper_t = int(g["state_hour"].max())
        flags_pos = g.set_index("state_hour")["delirium_positive"].reindex(range(upper_t + 1), fill_value=0).to_numpy()
        flags_neg = g.set_index("state_hour")["delirium_negative"].reindex(range(upper_t + 1), fill_value=0).to_numpy()
        resolution_start = None
        for cand in range(0, upper_t + 1):
            if _future_nonpositive_resolution(flags_neg, flags_pos, cand, 24):
                resolution_start = cand
                break
        resolution_rows.append({"stay_id": int(stay_id), "resolution_start_hour": resolution_start})
    resolution_df = pd.DataFrame(resolution_rows)

    merged = merged.merge(positive_hours, on="stay_id", how="left")
    merged = merged.merge(resolution_df, on="stay_id", how="left")
    merged["last_positive_hour"] = pd.to_numeric(merged["last_positive_hour"], errors="coerce")
    merged["resolution_start_hour"] = pd.to_numeric(merged["resolution_start_hour"], errors="coerce")
    merged["delirium_resolution"] = (
        merged["resolution_start_hour"].notna() & (merged["state_hour"] >= merged["resolution_start_hour"])
    ).astype(int)
    return merged, meta_df, template


def _load_sepsis_inputs(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    template = _load_template(Path(cfg["template_json"]))
    meta_df = read_table(cfg["metadata_parquet"])[
        [
            "stay_id",
            "sepsis_onset_hour",
            "onset_source",
            "onset_confidence",
            "shock_first_hour_any",
            "shock_onset_hour",
            "n_hours",
            "has_map",
            "has_lactate",
            "has_vasopressor",
            "has_sofa_total",
            "first_lactate_hour",
            "has_any_shock",
            "has_progression_shock",
            "shock_before_sepsis_onset",
            "shock_at_sepsis_onset",
        ]
    ].copy()
    meta_df["stay_id"] = pd.to_numeric(meta_df["stay_id"], errors="coerce").astype("Int64")
    meta_df["sepsis_onset_hour"] = pd.to_numeric(meta_df["sepsis_onset_hour"], errors="coerce")
    meta_df = meta_df.dropna(subset=["stay_id", "sepsis_onset_hour"]).copy()
    meta_df["stay_id"] = meta_df["stay_id"].astype("int64")
    meta_df["sepsis_onset_hour"] = meta_df["sepsis_onset_hour"].astype("int64")
    for col in [
        "shock_first_hour_any",
        "shock_onset_hour",
        "first_lactate_hour",
    ]:
        meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce")
    for col in [
        "n_hours",
        "has_map",
        "has_lactate",
        "has_vasopressor",
        "has_sofa_total",
        "has_any_shock",
        "has_progression_shock",
        "shock_before_sepsis_onset",
        "shock_at_sepsis_onset",
    ]:
        meta_df[col] = pd.to_numeric(meta_df[col], errors="coerce").fillna(0).astype(int)

    stay_ids = set(meta_df["stay_id"].tolist())
    keep_cols = [
        "stay_id",
        "state_hour",
        "map_merged",
        "temperature_c",
        "creatinine",
        "wbc",
        "lactate",
        "urineoutput",
        "bilirubin_total",
        "vasopressors_active",
        "sofa_total",
        "sofa_respiration",
        "fluid_balance",
    ]
    b3_frames: list[pd.DataFrame] = []
    for chunk in iter_table_chunks(cfg["b3_state_bank"]):
        cols = [c for c in keep_cols if c in chunk.columns]
        df = chunk[cols].copy()
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        df["state_hour"] = pd.to_numeric(df["state_hour"], errors="coerce")
        df = df.dropna(subset=["stay_id", "state_hour"]).copy()
        df["stay_id"] = df["stay_id"].astype("int64")
        df["state_hour"] = df["state_hour"].astype("int64")
        df = df.loc[df["stay_id"].isin(stay_ids)].copy()
        if not df.empty:
            b3_frames.append(df)
    if not b3_frames:
        raise RuntimeError("No B3 state rows found for Sepsis Phase 5")
    b3_df = pd.concat(b3_frames, ignore_index=True)

    shock_hourly = read_table(cfg["shock_hourly_parquet"])[["stay_id", "hour", "is_septic_shock"]].copy()
    shock_hourly["stay_id"] = pd.to_numeric(shock_hourly["stay_id"], errors="coerce").astype("Int64")
    shock_hourly["hour"] = pd.to_numeric(shock_hourly["hour"], errors="coerce")
    shock_hourly = shock_hourly.dropna(subset=["stay_id", "hour"]).copy()
    shock_hourly["stay_id"] = shock_hourly["stay_id"].astype("int64")
    shock_hourly["hour"] = shock_hourly["hour"].astype("int64")
    shock_hourly = shock_hourly.loc[shock_hourly["stay_id"].isin(stay_ids)].copy()
    shock_hourly["is_septic_shock"] = pd.to_numeric(shock_hourly["is_septic_shock"], errors="coerce").fillna(0).astype(int)

    merged = b3_df.merge(meta_df, on="stay_id", how="inner")
    merged = merged.merge(shock_hourly, left_on=["stay_id", "state_hour"], right_on=["stay_id", "hour"], how="left")
    merged = merged.drop(columns=["hour"], errors="ignore")
    merged["is_septic_shock"] = pd.to_numeric(merged["is_septic_shock"], errors="coerce").fillna(0).astype(int)
    merged["relative_hour"] = merged["state_hour"] - merged["sepsis_onset_hour"]
    merged = merged.loc[(merged["relative_hour"] >= -48) & (merged["relative_hour"] <= 168)].copy()

    for col in [
        "map_merged",
        "temperature_c",
        "creatinine",
        "wbc",
        "lactate",
        "urineoutput",
        "bilirubin_total",
        "sofa_total",
        "sofa_respiration",
        "fluid_balance",
    ]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged["vasopressors_active"] = pd.to_numeric(merged["vasopressors_active"], errors="coerce").fillna(0).astype(int)
    merged["sepsis_onset"] = (merged["relative_hour"] <= 0).astype(int)
    merged["shock_after_onset"] = (
        merged["is_septic_shock"].eq(1) & (merged["state_hour"] >= merged["sepsis_onset_hour"])
    ).astype(int)
    return merged, meta_df, template


def _assign_aki_phase(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    phase_windows: dict[str, tuple[int | None, int | None]] = {}
    phase_names: dict[str, str] = {}
    for phase in template.get("phases", []):
        pid = str(phase.get("phase_id"))
        phase_windows[pid] = _phase_window_from_text(str(phase.get("time_relative_to_anchor", "")))
        phase_names[pid] = str(phase.get("name", pid))

    df = df.copy()
    rel = pd.to_numeric(df["relative_hour"], errors="coerce")
    base_phase = np.select(
        [
            (rel >= -48) & (rel < 0),
            (rel >= 0) & (rel < 24),
            (rel >= 24) & (rel < 72),
            (rel >= 72) & (rel <= 168),
        ],
        [
            "risk_exposure",
            "early_injury_stage1",
            "progressive_injury_stage2plus",
            "recovery_or_nonrecovery",
        ],
        default="out_of_template_window",
    )
    df["phase_id"] = base_phase
    df["phase_name"] = df["phase_id"].map(lambda x: phase_names.get(str(x), "Out of Template Window"))
    df["phase_source"] = np.where(df["phase_id"].eq("out_of_template_window"), "outside_template_window", "template_window")

    support_mask = (df["relative_hour"] >= 24) & (
        (df["rrt_active"].fillna(0) > 0) | (df["kdigo_stage_ffill"].fillna(0) >= 3)
    )
    df.loc[support_mask, "phase_id"] = "support_escalation"
    df.loc[support_mask, "phase_name"] = phase_names.get("support_escalation", "Support Escalation")
    df.loc[support_mask, "phase_source"] = "template_window_support_override"
    return df


def _assign_delirium_phase(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    phase_names = {str(phase.get("phase_id")): str(phase.get("name", phase.get("phase_id"))) for phase in template.get("phases", [])}
    df = df.copy()
    rel = pd.to_numeric(df["relative_hour"], errors="coerce")
    df["phase_id"] = np.select(
        [
            rel < 0,
            (rel >= 0) & (rel < 24),
            (rel >= 24) & (rel < 72),
            rel >= 72,
        ],
        [
            "at_risk_or_prepositive",
            "active_positive",
            "fluctuating_course",
            "resolution_or_persistence",
        ],
        default="out_of_template_window",
    )
    df["phase_name"] = df["phase_id"].map(lambda x: phase_names.get(str(x), "Out of Template Window"))
    df["phase_source"] = np.where(df["phase_id"].eq("out_of_template_window"), "outside_template_window", "template_window")
    return df


def _assign_sepsis_phase(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    phase_names = {str(phase.get("phase_id")): str(phase.get("name", phase.get("phase_id"))) for phase in template.get("phases", [])}
    df = df.copy()
    rel = pd.to_numeric(df["relative_hour"], errors="coerce")
    base_phase = np.select(
        [
            rel < 0,
            (rel >= 0) & (rel < 24),
            (rel >= 24) & (rel < 72),
            rel >= 72,
        ],
        [
            "suspected_infection",
            "early_sepsis",
            "organ_dysfunction_progression",
            "stabilization_or_nonrecovery",
        ],
        default="out_of_template_window",
    )
    df["phase_id"] = base_phase
    df["phase_name"] = df["phase_id"].map(lambda x: phase_names.get(str(x), "Out of Template Window"))
    df["phase_source"] = np.where(df["phase_id"].eq("out_of_template_window"), "outside_template_window", "template_window")

    shock_mask = pd.to_numeric(df["shock_after_onset"], errors="coerce").fillna(0).astype(int).eq(1)
    df.loc[shock_mask, "phase_id"] = "hemodynamic_compromise_or_shock"
    df.loc[shock_mask, "phase_name"] = phase_names.get("hemodynamic_compromise_or_shock", "Hemodynamic Compromise or Shock")
    df.loc[shock_mask, "phase_source"] = "template_window_shock_override"
    return df


def _attach_template_support(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    df = df.copy()
    supported = _phase_supported_features(template, set(df.columns))
    df["template_supported_feature_count"] = 0
    df["template_observed_feature_count"] = 0
    df["template_executable_support_score"] = np.nan

    for pid, feats in supported.items():
        mask = df["phase_id"] == pid
        if not mask.any():
            continue
        if not feats:
            continue
        obs = df.loc[mask, feats].notna().sum(axis=1).astype(int)
        df.loc[mask, "template_supported_feature_count"] = len(feats)
        df.loc[mask, "template_observed_feature_count"] = obs.to_numpy()
        df.loc[mask, "template_executable_support_score"] = (obs / float(len(feats))).to_numpy()
    return df


def _add_empirical_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    stage = pd.to_numeric(df["kdigo_stage_ffill"], errors="coerce").fillna(0)
    rrt = pd.to_numeric(df["rrt_active"], errors="coerce").fillna(0)
    df["severity_bin"] = np.select(
        [
            rrt > 0,
            stage >= 3,
            stage >= 2,
            stage >= 1,
        ],
        [
            "rrt_support",
            "stage3",
            "stage2plus",
            "stage1",
        ],
        default="prestage",
    )
    df["empirical_state_id"] = df["phase_id"].astype(str) + "::" + df["severity_bin"].astype(str)
    return df


def _add_delirium_empirical_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    positive = pd.to_numeric(df["delirium_positive"], errors="coerce").fillna(0).astype(int)
    negative = pd.to_numeric(df["delirium_negative"], errors="coerce").fillna(0).astype(int)
    uta = pd.to_numeric(df["delirium_uta"], errors="coerce").fillna(0).astype(int)
    resolved = pd.to_numeric(df["delirium_resolution"], errors="coerce").fillna(0).astype(int)
    rass = pd.to_numeric(df["rass"], errors="coerce")

    df["phenotype_bin"] = np.select(
        [
            resolved == 1,
            (positive == 1) & (rass <= -1),
            (positive == 1) & (rass >= 1),
            positive == 1,
            negative == 1,
            uta == 1,
        ],
        [
            "resolved",
            "hypoactive_positive",
            "hyperactive_positive",
            "positive_unspecified",
            "negative_assessment",
            "uta",
        ],
        default="no_assessment",
    )
    df["empirical_state_id"] = df["phase_id"].astype(str) + "::" + df["phenotype_bin"].astype(str)
    return df


def _add_sepsis_empirical_states(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    shock = pd.to_numeric(df["shock_after_onset"], errors="coerce").fillna(0).astype(int)
    vaso = pd.to_numeric(df["vasopressors_active"], errors="coerce").fillna(0).astype(int)
    lactate = pd.to_numeric(df["lactate"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    sofa = pd.to_numeric(df["sofa_total"], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
    cond_shock = (shock.eq(1) | vaso.gt(0)).to_numpy(dtype=bool)
    cond_high_sofa = np.nan_to_num(sofa, nan=-np.inf) >= 10
    cond_high_lactate = np.nan_to_num(lactate, nan=-np.inf) >= 4.0
    cond_moderate = np.nan_to_num(sofa, nan=-np.inf) >= 6
    df["severity_bin"] = np.select(
        [
            cond_shock,
            cond_high_sofa,
            cond_high_lactate,
            cond_moderate,
        ],
        [
            "shock_state",
            "high_sofa",
            "high_lactate",
            "moderate_dysfunction",
        ],
        default="lower_burden",
    )
    df["empirical_state_id"] = df["phase_id"].astype(str) + "::" + df["severity_bin"].astype(str)
    return df


def _build_episode_summary(df: pd.DataFrame) -> pd.DataFrame:
    frame = df[["stay_id", "state_hour", "relative_hour", "phase_id", "phase_name", "phase_source", "template_executable_support_score"]].copy()
    frame = frame.sort_values(["stay_id", "state_hour"], kind="mergesort").reset_index(drop=True)
    phase_change = (
        (frame["stay_id"] != frame["stay_id"].shift(1))
        | (frame["phase_id"] != frame["phase_id"].shift(1))
        | (frame["state_hour"] != frame["state_hour"].shift(1) + 1)
    )
    frame["episode_idx"] = phase_change.cumsum().astype(int)
    episodes = (
        frame.groupby(["stay_id", "episode_idx", "phase_id", "phase_name", "phase_source"], sort=False)
        .agg(
            start_state_hour=("state_hour", "min"),
            end_state_hour=("state_hour", "max"),
            start_relative_hour=("relative_hour", "min"),
            end_relative_hour=("relative_hour", "max"),
            n_hours=("state_hour", "size"),
            mean_template_executable_support_score=("template_executable_support_score", "mean"),
        )
        .reset_index()
    )
    return episodes


def _build_state_prototypes(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    keep = [c for c in feature_cols if c in df.columns]
    grouped = df.groupby("empirical_state_id", sort=False)
    proto = grouped[keep].mean(numeric_only=True).reset_index()
    proto["rows"] = grouped.size().values
    proto["unique_stays"] = grouped["stay_id"].nunique().values
    return proto


def _build_transition_matrix(df: pd.DataFrame) -> dict[str, object]:
    frame = df[["stay_id", "state_hour", "empirical_state_id"]].copy().sort_values(["stay_id", "state_hour"], kind="mergesort")
    frame["next_state"] = frame.groupby("stay_id", sort=False)["empirical_state_id"].shift(-1)
    trans = frame.dropna(subset=["next_state"]).copy()
    trans = trans.loc[trans["empirical_state_id"] != trans["next_state"]].copy()
    counts = (
        trans.groupby(["empirical_state_id", "next_state"], sort=False)
        .size()
        .reset_index(name="count")
    )
    return {
        "n_transitions": int(counts["count"].sum()) if not counts.empty else 0,
        "n_edges": int(len(counts)),
        "edges": counts.to_dict(orient="records"),
    }


def _build_aki_atypical_flags(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.sort_values(["stay_id", "state_hour"], kind="mergesort").groupby("stay_id", sort=False)
    variant_meta = {v["variant_id"]: v for v in template.get("atypical_variants", []) if "variant_id" in v}
    for stay_id, g in grouped:
        g = g.copy()
        rel = pd.to_numeric(g["relative_hour"], errors="coerce")
        uo_0_24 = g.loc[(rel >= 0) & (rel < 24), "urineoutput"].dropna()
        scr_0_72 = g.loc[(rel >= 0) & (rel <= 72), "creatinine"].dropna()
        scr_ge_72 = g.loc[rel >= 72, "creatinine"].dropna()
        stage_ge_120 = g.loc[rel >= 120, "kdigo_stage_ffill"].dropna()
        stage2plus_hour = pd.to_numeric(g["stage2plus_onset_hour"], errors="coerce").dropna()
        ckd = int(pd.to_numeric(g["ckd"], errors="coerce").fillna(0).max()) if "ckd" in g.columns else 0

        peak_72 = float(scr_0_72.max()) if not scr_0_72.empty else np.nan
        last_ge_72 = float(scr_ge_72.iloc[-1]) if not scr_ge_72.empty else np.nan
        non_oliguric = bool((not uo_0_24.empty) and (uo_0_24.min() >= 30.0) and (not stage2plus_hour.empty))
        rapidly_reversible = bool((not math.isnan(peak_72)) and (not math.isnan(last_ge_72)) and last_ge_72 <= 0.8 * peak_72 and stage2plus_hour.empty)
        non_recovery = bool((not stage_ge_120.empty) and (stage_ge_120.max() >= 1))

        rows.extend(
            [
                {
                    "stay_id": int(stay_id),
                    "variant_id": "aki_atyp_1",
                    "variant_name": variant_meta.get("aki_atyp_1", {}).get("name", "Non-oliguric AKI"),
                    "is_flagged": int(non_oliguric),
                    "rule_status": "supported",
                    "rule_name": "uo_preserved_despite_stage2plus",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "aki_atyp_2",
                    "variant_name": variant_meta.get("aki_atyp_2", {}).get("name", "Rapidly reversible (pre-renal) AKI"),
                    "is_flagged": int(rapidly_reversible),
                    "rule_status": "supported",
                    "rule_name": "creatinine_peak_then_decline_without_stage2plus",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "aki_atyp_3",
                    "variant_name": variant_meta.get("aki_atyp_3", {}).get("name", "AKI on CKD (acute-on-chronic)"),
                    "is_flagged": int(ckd == 1),
                    "rule_status": "supported",
                    "rule_name": "cohort_ckd_flag",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "aki_atyp_4",
                    "variant_name": variant_meta.get("aki_atyp_4", {}).get("name", "Delayed SCr rise (low muscle mass / liver disease)"),
                    "is_flagged": 0,
                    "rule_status": "unsupported",
                    "rule_name": "required_features_not_available",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "aki_atyp_5",
                    "variant_name": variant_meta.get("aki_atyp_5", {}).get("name", "Rhabdomyolysis-associated AKI"),
                    "is_flagged": 0,
                    "rule_status": "unsupported",
                    "rule_name": "ck_not_available",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "aki_atyp_6",
                    "variant_name": variant_meta.get("aki_atyp_6", {}).get("name", "Non-recovery / AKD transition"),
                    "is_flagged": int(non_recovery),
                    "rule_status": "supported",
                    "rule_name": "persistent_kdigo_stage_after_120h",
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_delirium_atypical_flags(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.sort_values(["stay_id", "state_hour"], kind="mergesort").groupby("stay_id", sort=False)
    variant_meta = {v["variant_id"]: v for v in template.get("atypical_variants", []) if "variant_id" in v}
    for stay_id, g in grouped:
        positive = g.loc[g["delirium_positive"] == 1].copy()
        rass = pd.to_numeric(positive["rass"], errors="coerce")
        sedation = pd.to_numeric(positive["sedation_burden"], errors="coerce").fillna(0.0)
        hr = pd.to_numeric(positive["heart_rate"], errors="coerce")
        sbp = pd.to_numeric(positive["sbp"], errors="coerce")
        midazolam = pd.to_numeric(positive["midazolam_rate"], errors="coerce").fillna(0.0)
        onset_hour = int(pd.to_numeric(g["delirium_onset_hour"], errors="coerce").dropna().iloc[0])
        max_state_hour = int(pd.to_numeric(g["state_hour"], errors="coerce").max())
        last_positive = pd.to_numeric(g["last_positive_hour"], errors="coerce").dropna()
        resolution = pd.to_numeric(g["delirium_resolution"], errors="coerce").fillna(0).astype(int)

        has_pos = not positive.empty
        last_pos_hour = int(last_positive.iloc[0]) if not last_positive.empty else None
        hypoactive = bool(has_pos and (rass.notna().sum() > 0) and (rass.le(-1).mean() >= 0.8) and (sedation.max() == 0))
        persistent = bool(has_pos and last_pos_hour is not None and (last_pos_hour - onset_hour >= 144) and (resolution.max() == 0))
        rapid_cycling = bool(has_pos and rass.le(-1).any() and rass.ge(1).any())
        alcohol_withdrawal_proxy = bool(
            has_pos
            and midazolam.gt(0).any()
            and (hr.max(skipna=True) >= 120 if hr.notna().any() else False)
            and (sbp.max(skipna=True) >= 160 if sbp.notna().any() else False)
        )

        rows.extend(
            [
                {
                    "stay_id": int(stay_id),
                    "variant_id": "del_atyp_1",
                    "variant_name": variant_meta.get("del_atyp_1", {}).get("name", "Purely hypoactive delirium"),
                    "is_flagged": int(hypoactive),
                    "rule_status": "supported",
                    "rule_name": "positive_hours_hypoactive_without_active_sedation",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "del_atyp_2",
                    "variant_name": variant_meta.get("del_atyp_2", {}).get("name", "Persistent delirium (>7 days)"),
                    "is_flagged": int(persistent),
                    "rule_status": "supported",
                    "rule_name": "positive_signal_persists_to_7d_without_resolution",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "del_atyp_3",
                    "variant_name": variant_meta.get("del_atyp_3", {}).get("name", "Delirium superimposed on dementia"),
                    "is_flagged": 0,
                    "rule_status": "unsupported",
                    "rule_name": "baseline_dementia_signal_not_available",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "del_atyp_4",
                    "variant_name": variant_meta.get("del_atyp_4", {}).get("name", "Alcohol withdrawal delirium"),
                    "is_flagged": int(alcohol_withdrawal_proxy),
                    "rule_status": "supported",
                    "rule_name": "benzo_plus_hyperadrenergic_proxy",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "del_atyp_5",
                    "variant_name": variant_meta.get("del_atyp_5", {}).get("name", "Rapid-cycling mixed subtype"),
                    "is_flagged": int(rapid_cycling),
                    "rule_status": "supported",
                    "rule_name": "rass_crosses_hypo_and_hyper_thresholds_during_positive_hours",
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_sepsis_atypical_flags(df: pd.DataFrame, template: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.sort_values(["stay_id", "state_hour"], kind="mergesort").groupby("stay_id", sort=False)
    variant_meta = {v["variant_id"]: v for v in template.get("atypical_variants", []) if "variant_id" in v}
    for stay_id, g in grouped:
        shock_onset = pd.to_numeric(g["shock_onset_hour"], errors="coerce").dropna()
        shock_hour = int(shock_onset.iloc[0]) if not shock_onset.empty else None
        onset_hour = int(pd.to_numeric(g["sepsis_onset_hour"], errors="coerce").dropna().iloc[0])
        late = g.loc[g["relative_hour"] >= 72].copy()

        shock_rows = g.loc[g["shock_after_onset"] == 1].copy()
        shock_lactate = pd.to_numeric(shock_rows["lactate"], errors="coerce").dropna()
        shock_map = pd.to_numeric(shock_rows["map_merged"], errors="coerce").dropna()
        shock_vaso = pd.to_numeric(shock_rows["vasopressors_active"], errors="coerce").fillna(0)
        lactate_low_shock = bool(
            (not shock_rows.empty)
            and (shock_vaso.gt(0).any() or (not shock_map.empty and shock_map.lt(65).any()))
            and ((shock_lactate.max() < 2.0) if not shock_lactate.empty else True)
        )

        late_sofa = pd.to_numeric(late["sofa_total"], errors="coerce").dropna()
        late_vaso = pd.to_numeric(late["vasopressors_active"], errors="coerce").fillna(0)
        persistent_failure = bool(
            (not late.empty)
            and (
                ((not late_sofa.empty) and (late_sofa.median() >= 8))
                or late_vaso.gt(0).any()
            )
        )

        rapid_recovery = False
        if shock_hour is not None:
            recovery = g.loc[(g["state_hour"] >= shock_hour) & (g["state_hour"] <= shock_hour + 24)].copy()
            rec_map = pd.to_numeric(recovery["map_merged"], errors="coerce")
            rec_vaso = pd.to_numeric(recovery["vasopressors_active"], errors="coerce").fillna(0)
            rec_sofa = pd.to_numeric(recovery["sofa_total"], errors="coerce")
            rapid_recovery = bool(
                not recovery.empty
                and ((rec_vaso.eq(0) & rec_map.ge(65)).sum() >= 6)
                and ((rec_sofa.max() >= 4) if rec_sofa.notna().any() else False)
            )

        rows.extend(
            [
                {
                    "stay_id": int(stay_id),
                    "variant_id": "sepsis_atyp_1",
                    "variant_name": variant_meta.get("sepsis_atyp_1", {}).get("name", "Lactate-low shock"),
                    "is_flagged": int(lactate_low_shock),
                    "rule_status": "supported",
                    "rule_name": "shock_after_onset_with_normal_lactate",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "sepsis_atyp_2",
                    "variant_name": variant_meta.get("sepsis_atyp_2", {}).get("name", "Persistent multiorgan failure"),
                    "is_flagged": int(persistent_failure),
                    "rule_status": "supported",
                    "rule_name": "late_sofa_or_vasopressor_persistence_after_72h",
                },
                {
                    "stay_id": int(stay_id),
                    "variant_id": "sepsis_atyp_3",
                    "variant_name": variant_meta.get("sepsis_atyp_3", {}).get("name", "Rapid hemodynamic recovery"),
                    "is_flagged": int(rapid_recovery),
                    "rule_status": "supported",
                    "rule_name": "shock_then_map_recovers_with_vasopressor_off_within_24h",
                },
            ]
        )
    return pd.DataFrame(rows)


def _build_trajectory_tiers(df: pd.DataFrame, atypical_df: pd.DataFrame) -> pd.DataFrame:
    support = (
        df.groupby("stay_id", sort=False)["template_executable_support_score"]
        .mean()
        .reset_index(name="mean_template_executable_support_score")
    )
    atypical_supported = (
        atypical_df.loc[(atypical_df["rule_status"] == "supported") & (atypical_df["is_flagged"] == 1)]
        .groupby("stay_id", sort=False)
        .size()
        .reset_index(name="n_supported_atypical_flags")
    )
    tiers = support.merge(atypical_supported, on="stay_id", how="left")
    tiers["n_supported_atypical_flags"] = tiers["n_supported_atypical_flags"].fillna(0).astype(int)
    tiers["trajectory_tier"] = np.select(
        [
            (tiers["n_supported_atypical_flags"] > 0) & (tiers["mean_template_executable_support_score"] < 0.5),
            tiers["n_supported_atypical_flags"] > 0,
            tiers["mean_template_executable_support_score"] < 0.5,
        ],
        [
            "atypical_low_support",
            "atypical_supported",
            "low_template_support",
        ],
        default="typical_supported",
    )
    return tiers


def _write_transition_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _summarize_phase5(
    condition: str,
    template: dict,
    aligned_df: pd.DataFrame,
    phase_labels_df: pd.DataFrame,
    episodes_df: pd.DataFrame,
    atypical_df: pd.DataFrame,
    tiers_df: pd.DataFrame,
    state_proto_df: pd.DataFrame,
    transition_payload: dict,
    outputs: dict[str, Path],
    extra_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    atypical_supported = atypical_df.loc[atypical_df["rule_status"] == "supported"].copy()
    supported_counts = (
        atypical_supported.groupby("variant_id", sort=False)["is_flagged"].sum().astype(int).to_dict()
        if not atypical_supported.empty
        else {}
    )
    unsupported = sorted(atypical_df.loc[atypical_df["rule_status"] == "unsupported", "variant_id"].unique().tolist())
    summary = {
        "condition": condition,
        "phase5_mode": "template_driven_state_space_mvp",
        "template_source": relativize_value(str(CONDITION_CONFIGS[condition]["template_json"]), root=ROOT_DIR),
        "template_phase_ids": [p.get("phase_id") for p in template.get("phases", [])],
        "template_atypical_variant_ids": [v.get("variant_id") for v in template.get("atypical_variants", [])],
        "aligned_trajectories": {
            "rows": int(len(aligned_df)),
            "unique_stays": int(aligned_df["stay_id"].nunique()),
            "min_relative_hour": float(aligned_df["relative_hour"].min()) if not aligned_df.empty else None,
            "max_relative_hour": float(aligned_df["relative_hour"].max()) if not aligned_df.empty else None,
            "outputs": relativize_value(str(outputs["aligned"]), root=ROOT_DIR),
        },
        "phase_labels_hourly": {
            "rows": int(len(phase_labels_df)),
            "unique_stays": int(phase_labels_df["stay_id"].nunique()),
            "phase_distribution": {str(k): int(v) for k, v in phase_labels_df["phase_id"].value_counts(dropna=False).to_dict().items()},
            "mean_template_executable_support_score": float(phase_labels_df["template_executable_support_score"].dropna().mean()) if phase_labels_df["template_executable_support_score"].notna().any() else None,
            "outputs": relativize_value(str(outputs["phase_labels"]), root=ROOT_DIR),
        },
        "phase_episode_summary": {
            "rows": int(len(episodes_df)),
            "unique_stays": int(episodes_df["stay_id"].nunique()),
            "outputs": relativize_value(str(outputs["episodes"]), root=ROOT_DIR),
        },
        "state_prototypes": {
            "rows": int(len(state_proto_df)),
            "outputs": relativize_value(str(outputs["prototypes"]), root=ROOT_DIR),
        },
        "transition_matrix": {
            "n_transitions": int(transition_payload.get("n_transitions", 0)),
            "n_edges": int(transition_payload.get("n_edges", 0)),
            "outputs": relativize_value(str(outputs["transitions"]), root=ROOT_DIR),
        },
        "atypical_variants": {
            "rows": int(len(atypical_df)),
            "unique_stays": int(atypical_df["stay_id"].nunique()) if not atypical_df.empty else 0,
            "supported_flag_counts": supported_counts,
            "unsupported_variants": unsupported,
            "outputs": relativize_value(str(outputs["atypical"]), root=ROOT_DIR),
        },
        "trajectory_tiers": {
            "rows": int(len(tiers_df)),
            "unique_stays": int(tiers_df["stay_id"].nunique()),
            "tier_distribution": {str(k): int(v) for k, v in tiers_df["trajectory_tier"].value_counts(dropna=False).to_dict().items()},
            "outputs": relativize_value(str(outputs["tiers"]), root=ROOT_DIR),
        },
        "flags": [],
    }
    if extra_summary:
        summary.update(extra_summary)
    return summary


def build_aki_phase5() -> dict[str, object]:
    cfg = CONDITION_CONFIGS["aki"]
    outputs_dir = Path(cfg["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    aligned_df, onset_df, cohort_df, template = _load_aki_inputs(cfg)
    aligned_df = _assign_aki_phase(aligned_df, template)
    aligned_df = _attach_template_support(aligned_df, template)
    aligned_df = _add_empirical_states(aligned_df)

    phase_labels_df = aligned_df[
        [
            "stay_id",
            "state_hour",
            "relative_hour",
            "stage1_onset_hour",
            "stage2plus_onset_hour",
            "phase_id",
            "phase_name",
            "phase_source",
            "template_supported_feature_count",
            "template_observed_feature_count",
            "template_executable_support_score",
            "empirical_state_id",
            "severity_bin",
        ]
    ].copy()

    aligned_out = outputs_dir / "aki_aligned_trajectories.parquet"
    phase_labels_out = outputs_dir / "aki_phase_labels_hourly.parquet"
    episodes_out = outputs_dir / "aki_phase_episode_summary.parquet"
    atypical_out = outputs_dir / "aki_atypical_variant_flags.parquet"
    tiers_out = outputs_dir / "aki_trajectory_tiers.parquet"
    prototypes_out = outputs_dir / "aki_state_prototypes.parquet"
    transitions_out = outputs_dir / "aki_transition_matrix.json"
    summary_out = Path(cfg["summary_json"])

    write_table(
        aligned_df[
            [
                "stay_id",
                "state_hour",
                "relative_hour",
                "stage1_onset_hour",
                "stage2plus_onset_hour",
                "stage3_onset_hour",
                "phase_id",
                "phase_name",
                "phase_source",
                "template_executable_support_score",
                "empirical_state_id",
                "severity_bin",
                "map_merged",
                "lactate",
                "creatinine",
                "urineoutput",
                "bun",
                "potassium",
                "bicarbonate",
                "rrt_active",
                "fluid_balance",
                "kdigo_stage",
                "kdigo_stage_ffill",
                "ckd",
            ]
        ],
        aligned_out,
    )
    write_table(phase_labels_df, phase_labels_out)

    episodes_df = _build_episode_summary(phase_labels_df)
    write_table(episodes_df, episodes_out)

    atypical_df = _build_aki_atypical_flags(aligned_df, template)
    write_table(atypical_df, atypical_out)

    tiers_df = _build_trajectory_tiers(aligned_df, atypical_df)
    write_table(tiers_df, tiers_out)

    state_proto_df = _build_state_prototypes(
        aligned_df,
        feature_cols=["map_merged", "lactate", "creatinine", "urineoutput", "bun", "potassium", "bicarbonate", "rrt_active", "fluid_balance", "kdigo_stage_ffill", "template_executable_support_score"],
    )
    write_table(state_proto_df, prototypes_out)

    transition_payload = _build_transition_matrix(aligned_df)
    _write_transition_json(transition_payload, transitions_out)

    summary = _summarize_phase5(
        condition="aki",
        template=template,
        aligned_df=aligned_df,
        phase_labels_df=phase_labels_df,
        episodes_df=episodes_df,
        atypical_df=atypical_df,
        tiers_df=tiers_df,
        state_proto_df=state_proto_df,
        transition_payload=transition_payload,
        outputs={
            "aligned": aligned_out,
            "phase_labels": phase_labels_out,
            "episodes": episodes_out,
            "atypical": atypical_out,
            "tiers": tiers_out,
            "prototypes": prototypes_out,
            "transitions": transitions_out,
        },
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(relativize_value(summary, root=ROOT_DIR), f, indent=2)
    return summary


def build_delirium_phase5() -> dict[str, object]:
    cfg = CONDITION_CONFIGS["delirium"]
    outputs_dir = Path(cfg["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    aligned_df, meta_df, template = _load_delirium_inputs(cfg)
    aligned_df = _assign_delirium_phase(aligned_df, template)
    aligned_df = _attach_template_support(aligned_df, template)
    aligned_df = _add_delirium_empirical_states(aligned_df)

    phase_labels_df = aligned_df[
        [
            "stay_id",
            "state_hour",
            "relative_hour",
            "delirium_onset_hour",
            "left_censored",
            "phase_id",
            "phase_name",
            "phase_source",
            "template_supported_feature_count",
            "template_observed_feature_count",
            "template_executable_support_score",
            "empirical_state_id",
            "phenotype_bin",
        ]
    ].copy()

    aligned_out = outputs_dir / "delirium_aligned_trajectories.parquet"
    phase_labels_out = outputs_dir / "delirium_phase_labels_hourly.parquet"
    episodes_out = outputs_dir / "delirium_phase_episode_summary.parquet"
    atypical_out = outputs_dir / "delirium_atypical_variant_flags.parquet"
    tiers_out = outputs_dir / "delirium_trajectory_tiers.parquet"
    prototypes_out = outputs_dir / "delirium_state_prototypes.parquet"
    transitions_out = outputs_dir / "delirium_transition_matrix.json"
    summary_out = Path(cfg["summary_json"])

    write_table(
        aligned_df[
            [
                "stay_id",
                "state_hour",
                "relative_hour",
                "delirium_onset_hour",
                "left_censored",
                "phase_id",
                "phase_name",
                "phase_source",
                "template_executable_support_score",
                "empirical_state_id",
                "phenotype_bin",
                "rass",
                "gcs_total",
                "glucose_merged",
                "delirium_assessment",
                "delirium_positive",
                "delirium_negative",
                "delirium_uta",
                "restraint_active",
                "sedation_burden",
                "delirium_resolution",
                "heart_rate",
                "sbp",
                "mbp",
                "propofol_rate",
                "midazolam_rate",
                "fentanyl_rate",
            ]
        ],
        aligned_out,
    )
    write_table(phase_labels_df, phase_labels_out)

    episodes_df = _build_episode_summary(phase_labels_df)
    write_table(episodes_df, episodes_out)

    atypical_df = _build_delirium_atypical_flags(aligned_df, template)
    write_table(atypical_df, atypical_out)

    tiers_df = _build_trajectory_tiers(aligned_df, atypical_df)
    write_table(tiers_df, tiers_out)

    state_proto_df = _build_state_prototypes(
        aligned_df,
        feature_cols=[
            "rass",
            "gcs_total",
            "glucose_merged",
            "delirium_assessment",
            "delirium_positive",
            "delirium_negative",
            "delirium_uta",
            "restraint_active",
            "sedation_burden",
            "delirium_resolution",
            "heart_rate",
            "sbp",
            "mbp",
            "propofol_rate",
            "midazolam_rate",
            "fentanyl_rate",
            "template_executable_support_score",
        ],
    )
    write_table(state_proto_df, prototypes_out)

    transition_payload = _build_transition_matrix(aligned_df)
    _write_transition_json(transition_payload, transitions_out)

    extra_summary = {
        "left_censored": {
            "stays": int(meta_df["left_censored"].sum()),
            "rate": float(meta_df["left_censored"].mean()) if len(meta_df) else None,
        }
    }
    summary = _summarize_phase5(
        condition="delirium",
        template=template,
        aligned_df=aligned_df,
        phase_labels_df=phase_labels_df,
        episodes_df=episodes_df,
        atypical_df=atypical_df,
        tiers_df=tiers_df,
        state_proto_df=state_proto_df,
        transition_payload=transition_payload,
        outputs={
            "aligned": aligned_out,
            "phase_labels": phase_labels_out,
            "episodes": episodes_out,
            "atypical": atypical_out,
            "tiers": tiers_out,
            "prototypes": prototypes_out,
            "transitions": transitions_out,
        },
        extra_summary=extra_summary,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(relativize_value(summary, root=ROOT_DIR), f, indent=2)
    return summary


def build_sepsis_phase5() -> dict[str, object]:
    cfg = CONDITION_CONFIGS["sepsis"]
    outputs_dir = Path(cfg["outputs_dir"])
    outputs_dir.mkdir(parents=True, exist_ok=True)

    aligned_df, meta_df, template = _load_sepsis_inputs(cfg)
    aligned_df = _assign_sepsis_phase(aligned_df, template)
    aligned_df = _attach_template_support(aligned_df, template)
    aligned_df = _add_sepsis_empirical_states(aligned_df)

    phase_labels_df = aligned_df[
        [
            "stay_id",
            "state_hour",
            "relative_hour",
            "sepsis_onset_hour",
            "onset_confidence",
            "shock_first_hour_any",
            "shock_onset_hour",
            "shock_before_sepsis_onset",
            "shock_at_sepsis_onset",
            "phase_id",
            "phase_name",
            "phase_source",
            "template_supported_feature_count",
            "template_observed_feature_count",
            "template_executable_support_score",
            "empirical_state_id",
            "severity_bin",
        ]
    ].copy()

    aligned_out = outputs_dir / "sepsis_aligned_trajectories.parquet"
    phase_labels_out = outputs_dir / "sepsis_phase_labels_hourly.parquet"
    episodes_out = outputs_dir / "sepsis_phase_episode_summary.parquet"
    atypical_out = outputs_dir / "sepsis_atypical_variant_flags.parquet"
    tiers_out = outputs_dir / "sepsis_trajectory_tiers.parquet"
    prototypes_out = outputs_dir / "sepsis_state_prototypes.parquet"
    transitions_out = outputs_dir / "sepsis_transition_matrix.json"
    summary_out = Path(cfg["summary_json"])

    write_table(
        aligned_df[
            [
                "stay_id",
                "state_hour",
                "relative_hour",
                "sepsis_onset_hour",
                "onset_confidence",
                "shock_first_hour_any",
                "shock_onset_hour",
                "shock_before_sepsis_onset",
                "shock_at_sepsis_onset",
                "phase_id",
                "phase_name",
                "phase_source",
                "template_executable_support_score",
                "empirical_state_id",
                "severity_bin",
                "sepsis_onset",
                "temperature_c",
                "wbc",
                "lactate",
                "sofa_total",
                "map_merged",
                "sofa_respiration",
                "creatinine",
                "bilirubin_total",
                "urineoutput",
                "vasopressors_active",
                "fluid_balance",
                "is_septic_shock",
                "shock_after_onset",
            ]
        ],
        aligned_out,
    )
    write_table(phase_labels_df, phase_labels_out)

    episodes_df = _build_episode_summary(phase_labels_df)
    write_table(episodes_df, episodes_out)

    atypical_df = _build_sepsis_atypical_flags(aligned_df, template)
    write_table(atypical_df, atypical_out)

    tiers_df = _build_trajectory_tiers(aligned_df, atypical_df)
    write_table(tiers_df, tiers_out)

    state_proto_df = _build_state_prototypes(
        aligned_df,
        feature_cols=[
            "sepsis_onset",
            "temperature_c",
            "wbc",
            "lactate",
            "sofa_total",
            "map_merged",
            "sofa_respiration",
            "creatinine",
            "bilirubin_total",
            "urineoutput",
            "vasopressors_active",
            "fluid_balance",
            "is_septic_shock",
            "shock_after_onset",
            "template_executable_support_score",
        ],
    )
    write_table(state_proto_df, prototypes_out)

    transition_payload = _build_transition_matrix(aligned_df)
    _write_transition_json(transition_payload, transitions_out)

    extra_summary = {
        "onset_confidence": {
            "high_stays": int((meta_df["onset_confidence"] == "high").sum()),
            "low_stays": int((meta_df["onset_confidence"] == "low").sum()),
        },
        "shock_timing": {
            "stays_with_any_shock": int(meta_df["has_any_shock"].sum()),
            "stays_with_progression_shock": int(meta_df["has_progression_shock"].sum()),
            "shock_before_sepsis_onset_stays": int(meta_df["shock_before_sepsis_onset"].sum()),
            "shock_at_sepsis_onset_stays": int(meta_df["shock_at_sepsis_onset"].sum()),
        },
    }
    summary = _summarize_phase5(
        condition="sepsis",
        template=template,
        aligned_df=aligned_df,
        phase_labels_df=phase_labels_df,
        episodes_df=episodes_df,
        atypical_df=atypical_df,
        tiers_df=tiers_df,
        state_proto_df=state_proto_df,
        transition_payload=transition_payload,
        outputs={
            "aligned": aligned_out,
            "phase_labels": phase_labels_out,
            "episodes": episodes_out,
            "atypical": atypical_out,
            "tiers": tiers_out,
            "prototypes": prototypes_out,
            "transitions": transitions_out,
        },
        extra_summary=extra_summary,
    )
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(relativize_value(summary, root=ROOT_DIR), f, indent=2)
    return summary


def main() -> None:
    args = parse_args()
    if args.condition == "aki":
        summary = build_aki_phase5()
    elif args.condition == "delirium":
        summary = build_delirium_phase5()
    elif args.condition == "sepsis":
        summary = build_sepsis_phase5()
    else:
        raise NotImplementedError("Phase 5 builder is implemented condition-by-condition.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
