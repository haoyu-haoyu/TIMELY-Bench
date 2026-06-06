#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import DEFAULT_HOURLY_STATE_GRID, ROOT_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR  # type: ignore
from v3.io_utils import chunk_dir_path, iter_table_chunks, read_table, relativize_value, table_exists, write_table  # type: ignore


MAX_HOUR = 167
RECENT_WINDOW_HOURS = 24
PART_ROWS = 50000

SEQUENCE_STATIC_COLS = [
    "stay_id",
    "hour",
    "hadm_id",
    "anchor_age",
    "gender",
    "ckd",
    "is_within_observed_icu_los",
    "hours_until_discharge",
    "hours_until_death",
    "hour_normalized_168h",
]

SUMMARY_FEATURES = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "map_merged",
    "resp_rate",
    "temperature_c",
    "spo2",
    "albumin",
    "bun",
    "creatinine",
    "glucose_merged",
    "sodium",
    "potassium",
    "bicarbonate",
    "chloride",
    "aniongap",
    "wbc",
    "hemoglobin",
    "hematocrit",
    "platelet",
    "gcs_total",
    "gcs_motor",
    "gcs_verbal",
    "gcs_eye",
    "urineoutput_hourly",
    "lactate",
    "ph",
    "pao2",
    "paco2",
    "pao2_fio2_ratio",
    "sofa_total",
    "sofa_respiration",
    "bilirubin_total",
    "vasopressors_active",
    "vasopressor_dose_norepi_equiv",
    "rrt_active",
    "propofol_rate",
    "midazolam_rate",
    "fentanyl_rate",
    "fluid_input_hourly",
    "fluid_balance",
    "rass",
    "delirium_positive",
    "delirium_negative",
    "delirium_uta",
    "cam_component_recorded",
    "restraint_active",
    "fio2",
    "peep",
    "tidal_volume",
    "minute_volume",
    "plateau_pressure",
]


CONDITION_CONFIGS = {
    "aki": {
        "task_files": [
            V3_PROCESSED_DIR / "aki" / "tasks" / "aki_stage2plus_instances.parquet",
            V3_PROCESSED_DIR / "aki" / "tasks" / "aki_rrt_proxy_instances.parquet",
        ],
        "anchor_col": "anchor_hour",
        "representations_dir": V3_PROCESSED_DIR / "aki" / "representations",
        "b1_summary_json": V3_RESULTS_DIR / "aki" / "aki_B1_build_summary.json",
        "a_summary_json": V3_RESULTS_DIR / "aki" / "aki_A_build_summary.json",
        "scope": "temporal",
    },
    "delirium": {
        "task_files": [
            V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_persistence_instances.parquet",
            V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_resolution_instances.parquet",
        ],
        "anchor_col": "prediction_hour",
        "representations_dir": V3_PROCESSED_DIR / "delirium" / "representations",
        "b1_summary_json": V3_RESULTS_DIR / "delirium" / "delirium_B1_build_summary.json",
        "a_summary_json": V3_RESULTS_DIR / "delirium" / "delirium_A_build_summary.json",
        "scope": "temporal",
    },
    "sepsis": {
        "task_files": [
            V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_shock_instances.parquet",
            V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_lactate_clearance_instances.parquet",
        ],
        "anchor_col": "prediction_hour",
        "representations_dir": V3_PROCESSED_DIR / "sepsis" / "representations",
        "b1_summary_json": V3_RESULTS_DIR / "sepsis" / "sepsis_B1_build_summary.json",
        "a_summary_json": V3_RESULTS_DIR / "sepsis" / "sepsis_A_build_summary.json",
        "scope": "temporal",
    },
    "stroke": {
        "task_files": [
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T1_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T2_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T3_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T4_instances.parquet",
        ],
        "anchor_col": "anchor_hour",
        "representations_dir": V3_PROCESSED_DIR / "stroke" / "representations",
        "b1_summary_json": V3_RESULTS_DIR / "stroke" / "stroke_B1_build_summary.json",
        "a_summary_json": V3_RESULTS_DIR / "stroke" / "stroke_A_build_summary.json",
        "scope": "stroke_temporal_only",
    },
}


class ParquetPartWriter:
    def __init__(self, output_path: Path, part_rows: int = PART_ROWS) -> None:
        self.output_path = output_path
        self.parts_dir = output_path.with_name(f"{output_path.name}.parts")
        if self.parts_dir.exists():
            shutil.rmtree(self.parts_dir)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.part_rows = int(part_rows)
        self.frames: list[pd.DataFrame] = []
        self.buffer_rows = 0
        self.part_idx = 0
        self.total_rows = 0
        self.columns: list[str] | None = None

    def write_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        if self.columns is None:
            self.columns = df.columns.tolist()
        self.frames.append(df)
        self.buffer_rows += len(df)
        if self.buffer_rows >= self.part_rows:
            self.flush()

    def flush(self) -> None:
        if not self.frames:
            return
        self.part_idx += 1
        part_path = self.parts_dir / f"part_{self.part_idx:05d}.parquet"
        pd.concat(self.frames, ignore_index=True).to_parquet(part_path, index=False)
        self.total_rows += self.buffer_rows
        self.frames = []
        self.buffer_rows = 0

    def close(self) -> dict[str, Any]:
        self.flush()
        return {
            "parts_dir": str(self.parts_dir),
            "n_parts": int(self.part_idx),
            "rows": int(self.total_rows),
            "columns": self.columns or [],
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 Phase 4D B1 + A representations.")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["aki", "delirium", "sepsis", "stroke"],
        choices=sorted(CONDITION_CONFIGS.keys()),
    )
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument(
        "--combined-summary-json",
        default=str(V3_RESULTS_DIR / "representations" / "phase4d_B1_A_build_summary.json"),
    )
    return p.parse_args()


def _aggregate_unique_join(df: pd.DataFrame, group_cols: list[str], value_col: str, output_col: str) -> pd.DataFrame:
    if value_col not in df.columns:
        out = df[group_cols].drop_duplicates().copy()
        out[output_col] = None
        return out
    base = df[group_cols + [value_col]].dropna().copy()
    if base.empty:
        out = df[group_cols].drop_duplicates().copy()
        out[output_col] = None
        return out
    base[value_col] = base[value_col].astype(str)
    base = base.loc[base[value_col] != ""].drop_duplicates()
    if base.empty:
        out = df[group_cols].drop_duplicates().copy()
        out[output_col] = None
        return out
    out = (
        base.sort_values(group_cols + [value_col], kind="mergesort")
        .groupby(group_cols, sort=False)[value_col]
        .agg("|".join)
        .reset_index()
        .rename(columns={value_col: output_col})
    )
    return out


def _existing_columns_for_fast_read(path: Path, requested: list[str]) -> list[str]:
    path = path.expanduser().resolve(strict=False)
    target = path
    if not target.exists():
        parts_dir = chunk_dir_path(target)
        if parts_dir.exists():
            part_files = sorted([p for p in parts_dir.iterdir() if p.is_file() and p.suffix == ".parquet"])
            if part_files:
                target = part_files[0]
    if target.suffix != ".parquet" or not target.exists():
        return requested
    schema_cols = set(pq.ParquetFile(target).schema_arrow.names)
    return [c for c in requested if c in schema_cols]


def _load_anchor_index(task_files: list[Path], anchor_col: str, condition: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    task_row_counts: dict[str, int] = {}
    for path in task_files:
        if not table_exists(path):
            raise FileNotFoundError(path)
        requested = _existing_columns_for_fast_read(path, ["stay_id", anchor_col, "eligible"])
        df = read_table(path, columns=requested) if requested else read_table(path)
        if "eligible" in df.columns:
            df = df.loc[pd.to_numeric(df["eligible"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if anchor_col not in df.columns:
            raise KeyError(f"{path} missing anchor column {anchor_col}")
        frame = df[["stay_id", anchor_col]].copy()
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
        frame["anchor_hour_requested"] = pd.to_numeric(frame[anchor_col], errors="coerce")
        frame = frame.dropna(subset=["stay_id", "anchor_hour_requested"]).copy()
        frame["stay_id"] = frame["stay_id"].astype("int64")
        frame["anchor_hour_requested"] = frame["anchor_hour_requested"].astype("int64")
        frame["anchor_hour_clipped"] = frame["anchor_hour_requested"].clip(upper=MAX_HOUR)
        frame["anchor_was_clipped"] = (frame["anchor_hour_requested"] > MAX_HOUR).astype(int)
        frame["task_id"] = path.stem
        task_row_counts[path.stem] = int(len(frame))
        frames.append(frame)

    merged = pd.concat(frames, ignore_index=True)
    group_cols = ["stay_id", "anchor_hour_requested", "anchor_hour_clipped"]
    base = (
        merged[group_cols + ["anchor_was_clipped"]]
        .groupby(group_cols, sort=False)["anchor_was_clipped"]
        .max()
        .reset_index()
    )
    task_base = merged[group_cols + ["task_id"]].drop_duplicates()
    task_ids = (
        task_base.sort_values(group_cols + ["task_id"], kind="mergesort")
        .groupby(group_cols, sort=False)["task_id"]
        .agg("|".join)
        .reset_index()
        .rename(columns={"task_id": "source_task_ids"})
    )
    task_counts = task_base.groupby(group_cols, sort=False).size().reset_index(name="n_source_tasks")
    agg = base.merge(task_ids, on=group_cols, how="left").merge(task_counts, on=group_cols, how="left")
    agg["condition"] = condition
    agg["representation_id"] = (
        condition + ":" + agg["stay_id"].astype(str) + ":" + agg["anchor_hour_clipped"].astype(str)
    )
    agg["history_start_hour"] = 0
    agg["history_end_hour"] = agg["anchor_hour_clipped"]
    agg["history_length_hours"] = agg["anchor_hour_clipped"] + 1
    agg = agg.sort_values(["stay_id", "anchor_hour_clipped"], kind="mergesort").reset_index(drop=True)
    return agg, task_row_counts


def _sequence_columns(hourly_cols: list[str]) -> tuple[list[str], list[str]]:
    feature_cols = [c for c in SUMMARY_FEATURES if c in hourly_cols]
    missing_cols = [f"{c}__missing" for c in feature_cols if f"{c}__missing" in hourly_cols]
    seq_cols = [c for c in SEQUENCE_STATIC_COLS if c in hourly_cols] + feature_cols + missing_cols
    return seq_cols, feature_cols


def _anchor_slices(anchor_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, tuple[int, int]]]:
    if anchor_df.empty:
        return anchor_df, {}
    ordered = anchor_df.sort_values(["stay_id", "anchor_hour_clipped"], kind="mergesort").reset_index(drop=True)
    stay_ids = ordered["stay_id"].to_numpy()
    boundaries = np.flatnonzero(np.r_[True, stay_ids[1:] != stay_ids[:-1], True])
    slices: dict[int, tuple[int, int]] = {}
    for i in range(len(boundaries) - 1):
        start = int(boundaries[i])
        end = int(boundaries[i + 1])
        slices[int(stay_ids[start])] = (start, end)
    return ordered, slices


def _build_anchor_stats(stay_df: pd.DataFrame, anchor_rows: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    stay_df = stay_df.sort_values("hour", kind="mergesort").reset_index(drop=True).copy()
    stats_cols: dict[str, Any] = {"hour": stay_df["hour"].astype(int).to_numpy()}
    n_features = max(len(feature_cols), 1)
    observed_row = stay_df[feature_cols].notna().sum(axis=1).astype(float) if feature_cols else pd.Series([0.0] * len(stay_df))
    observed_cells_all = observed_row.cumsum()
    observed_cells_recent24h = observed_row.rolling(RECENT_WINDOW_HOURS, min_periods=1).sum()
    stats_cols["observed_cells_all"] = observed_cells_all
    stats_cols["observed_cells_recent24h"] = observed_cells_recent24h
    stats_cols["observed_ratio_all"] = observed_cells_all / ((pd.RangeIndex(len(stay_df)) + 1) * n_features)
    stats_cols["observed_ratio_recent24h"] = observed_cells_recent24h / (
        pd.Series(range(1, len(stay_df) + 1)).clip(upper=RECENT_WINDOW_HOURS) * n_features
    )
    stats_cols["history_steps"] = pd.RangeIndex(len(stay_df)) + 1
    stats_cols["recent_window_steps"] = pd.Series(range(1, len(stay_df) + 1)).clip(upper=RECENT_WINDOW_HOURS)

    for feat in feature_cols:
        s = pd.to_numeric(stay_df[feat], errors="coerce")
        stats_cols[f"{feat}__last"] = s
        stats_cols[f"{feat}__mean_all"] = s.expanding(min_periods=1).mean()
        stats_cols[f"{feat}__min_all"] = s.expanding(min_periods=1).min()
        stats_cols[f"{feat}__max_all"] = s.expanding(min_periods=1).max()
        stats_cols[f"{feat}__mean_recent24h"] = s.rolling(RECENT_WINDOW_HOURS, min_periods=1).mean()
        stats_cols[f"{feat}__min_recent24h"] = s.rolling(RECENT_WINDOW_HOURS, min_periods=1).min()
        stats_cols[f"{feat}__max_recent24h"] = s.rolling(RECENT_WINDOW_HOURS, min_periods=1).max()

    stats = pd.DataFrame(stats_cols)
    out = anchor_rows.merge(
        stats,
        left_on="anchor_hour_clipped",
        right_on="hour",
        how="left",
    ).drop(columns=["hour"])
    out["representation_branch"] = "A"
    out["recent_window_hours"] = RECENT_WINDOW_HOURS
    return out


def _process_condition(condition: str, cfg: dict[str, Any], hourly_grid_path: Path) -> dict[str, Any]:
    anchor_df, task_row_counts = _load_anchor_index([Path(p) for p in cfg["task_files"]], str(cfg["anchor_col"]), condition)
    anchor_df, anchor_slices = _anchor_slices(anchor_df)
    reps_dir = Path(cfg["representations_dir"])
    reps_dir.mkdir(parents=True, exist_ok=True)
    anchor_index_path = reps_dir / f"{condition}_B1_anchor_index.parquet"
    b1_output_path = reps_dir / f"{condition}_B1_hourly_sequence_bank.parquet"
    a_output_path = reps_dir / f"{condition}_A_anchor_stats.parquet"

    write_table(anchor_df, anchor_index_path, index=False)

    relevant_stays = set(anchor_slices.keys())
    seen_stays: set[int] = set()
    overlap_stays: set[int] = set()

    first_chunk = next(iter_table_chunks(hourly_grid_path))
    hourly_cols = first_chunk.columns.tolist()
    seq_cols, feature_cols = _sequence_columns(hourly_cols)
    # Re-iterate from the beginning after schema probe.
    chunk_iter = iter_table_chunks(hourly_grid_path)

    b1_writer = ParquetPartWriter(b1_output_path)
    a_writer = ParquetPartWriter(a_output_path)
    carry_df: pd.DataFrame | None = None

    def handle_stay(stay_df: pd.DataFrame) -> None:
        stay_id = int(stay_df["stay_id"].iloc[0])
        seq_df = stay_df[seq_cols].copy()
        seq_df.insert(0, "condition", condition)
        seq_df = seq_df.rename(columns={"hour": "sequence_hour"})
        seq_df["representation_branch"] = "B1"
        b1_writer.write_df(seq_df)
        anchor_span = anchor_slices.get(stay_id)
        if anchor_span is None:
            return
        anchor_rows = anchor_df.iloc[anchor_span[0] : anchor_span[1]].reset_index(drop=True)
        a_df = _build_anchor_stats(stay_df[["stay_id", "hour"] + feature_cols].copy(), anchor_rows, feature_cols)
        a_writer.write_df(a_df)

    for chunk in chunk_iter:
        df = chunk.loc[chunk["stay_id"].isin(relevant_stays)].copy()
        if df.empty:
            continue
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df = df.dropna(subset=["stay_id", "hour"]).copy()
        df["stay_id"] = df["stay_id"].astype("int64")
        df["hour"] = df["hour"].astype("int64")
        df = df.sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)
        if carry_df is not None and not carry_df.empty:
            df = pd.concat([carry_df, df], ignore_index=True).sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)
            carry_df = None
        last_stay = int(df["stay_id"].iloc[-1])
        carry_df = df.loc[df["stay_id"] == last_stay].copy()
        process_df = df.loc[df["stay_id"] != last_stay].copy()
        if process_df.empty:
            continue
        for stay_id, stay_df in process_df.groupby("stay_id", sort=False):
            stay_id = int(stay_id)
            if stay_id in seen_stays:
                overlap_stays.add(stay_id)
            seen_stays.add(stay_id)
            handle_stay(stay_df.reset_index(drop=True))

    if carry_df is not None and not carry_df.empty:
        stay_id = int(carry_df["stay_id"].iloc[0])
        if stay_id in seen_stays:
            overlap_stays.add(stay_id)
        handle_stay(carry_df.reset_index(drop=True))
        seen_stays.add(stay_id)

    b1_info = b1_writer.close()
    a_info = a_writer.close()

    flags: list[str] = []
    if overlap_stays:
        flags.append(f"stay_ids_seen_in_multiple_hourly_chunks={len(overlap_stays)}")
    if int(a_info["rows"]) != int(len(anchor_df)):
        flags.append(f"a_rows_mismatch_anchor_rows={a_info['rows']}!={len(anchor_df)}")

    b1_summary = {
        "condition": condition,
        "representation_family": "B1",
        "representation_layout": "hourly_sequence_bank_plus_anchor_index",
        "task_scope": cfg["scope"],
        "anchor_index": {
            "rows": int(len(anchor_df)),
            "unique_stays": int(anchor_df["stay_id"].nunique()),
            "min_anchor_hour_requested": int(anchor_df["anchor_hour_requested"].min()) if len(anchor_df) else None,
            "max_anchor_hour_requested": int(anchor_df["anchor_hour_requested"].max()) if len(anchor_df) else None,
            "clipped_anchor_rows": int(anchor_df["anchor_was_clipped"].sum()),
            "outputs": relativize_value(str(anchor_index_path), root=ROOT_DIR),
        },
        "sequence_bank": {
            "rows": int(b1_info["rows"]),
            "unique_stays": int(len(seen_stays)),
            "n_parts": int(b1_info["n_parts"]),
            "n_columns": int(len(b1_info["columns"])),
            "outputs": relativize_value(str(b1_info["parts_dir"]), root=ROOT_DIR),
        },
        "settings": {
            "sequence_columns": [c for c in seq_cols if c not in {"stay_id", "hour"}],
            "anchor_clip_policy": "clip_to_167",
            "hour_rule": "Use raw hourly_state_grid rows where sequence_hour <= clipped anchor hour",
        },
        "inputs": {"task_row_counts": task_row_counts},
        "flags": flags,
    }

    a_summary = {
        "condition": condition,
        "representation_family": "A",
        "representation_layout": "anchor_level_statistical_summaries",
        "task_scope": cfg["scope"],
        "anchor_rows": int(len(anchor_df)),
        "unique_stays": int(anchor_df["stay_id"].nunique()),
        "outputs": {
            "anchor_stats_parts_dir": relativize_value(str(a_info["parts_dir"]), root=ROOT_DIR),
            "summary_json": relativize_value(str(cfg["a_summary_json"]), root=ROOT_DIR),
        },
        "build_stats": {
            "rows_written": int(a_info["rows"]),
            "n_parts": int(a_info["n_parts"]),
            "n_columns": int(len(a_info["columns"])),
        },
        "settings": {
            "summary_features": feature_cols,
            "recent_window_hours": RECENT_WINDOW_HOURS,
            "stats_per_feature": [
                "last",
                "mean_all",
                "min_all",
                "max_all",
                "mean_recent24h",
                "min_recent24h",
                "max_recent24h",
            ],
            "missingness_summary": [
                "observed_cells_all",
                "observed_cells_recent24h",
                "observed_ratio_all",
                "observed_ratio_recent24h",
            ],
        },
        "flags": flags,
    }

    with open(cfg["b1_summary_json"], "w", encoding="utf-8") as f:
        json.dump(relativize_value(b1_summary, root=ROOT_DIR), f, indent=2)
    with open(cfg["a_summary_json"], "w", encoding="utf-8") as f:
        json.dump(relativize_value(a_summary, root=ROOT_DIR), f, indent=2)

    return {
        "B1": relativize_value(b1_summary, root=ROOT_DIR),
        "A": relativize_value(a_summary, root=ROOT_DIR),
    }


def main() -> None:
    args = parse_args()
    hourly_grid_path = Path(args.hourly_grid).resolve()
    combined: dict[str, Any] = {
        "phase": "4D",
        "representation_families": ["B1", "A"],
        "source_hourly_grid": relativize_value(str(hourly_grid_path), root=ROOT_DIR),
        "conditions": {},
    }
    for condition in args.conditions:
        combined["conditions"][condition] = _process_condition(condition, CONDITION_CONFIGS[condition], hourly_grid_path)
    combined_path = Path(args.combined_summary_json)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(relativize_value(combined, root=ROOT_DIR), f, indent=2)
    print(f"[phase4d] wrote combined summary to {combined_path}")


if __name__ == "__main__":
    main()
