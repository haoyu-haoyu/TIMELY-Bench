#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import shutil

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    ROOT_DIR,
    DEFAULT_DIAGNOSIS_COMORBIDITIES,
    DEFAULT_HOURLY_STATE_GRID,
    V3_PROCESSED_DIR,
    ensure_v3_directories,
)
from v3.io_utils import chunk_dir_path, iter_table_chunks, read_table, relativize_value, write_table  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 stroke-proxy cohort and hourly markers.")
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--comorbidities", default=str(DEFAULT_DIAGNOSIS_COMORBIDITIES))
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--cohort-out", default=str(V3_PROCESSED_DIR / "stroke_proxy" / "stroke_proxy_cohort_v3.parquet"))
    p.add_argument("--markers-out", default=str(V3_PROCESSED_DIR / "stroke_proxy" / "stroke_proxy_markers_v3.parquet"))
    p.add_argument("--summary-json", default=str(V3_PROCESSED_DIR / "stroke_proxy" / "stroke_proxy_summary_v3.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    comorb = read_table(args.comorbidities)

    if "stroke_family" not in comorb.columns:
        raise ValueError("Comorbidity table must contain stroke_family.")

    cohort_path = Path(args.cohort_out)
    markers_path = Path(args.markers_out)
    summary_path = Path(args.summary_json)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    markers_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_parts_dir = chunk_dir_path(cohort_path)
    markers_parts_dir = chunk_dir_path(markers_path)
    if cohort_parts_dir.exists():
        shutil.rmtree(cohort_parts_dir)
    if markers_parts_dir.exists():
        shutil.rmtree(markers_parts_dir)
    cohort_parts_dir.mkdir(parents=True, exist_ok=True)
    markers_parts_dir.mkdir(parents=True, exist_ok=True)

    stroke_cohort_total = 0
    marker_rows_total = 0
    deterioration_stays = 0
    part_count = 0

    for chunk_idx, grid in enumerate(iter_table_chunks(args.hourly_grid), start=1):
        if args.stay_limit is not None:
            keep = grid["stay_id"].drop_duplicates().head(int(args.stay_limit)).tolist()
            grid = grid[grid["stay_id"].isin(keep)].copy()
            comorb_chunk = comorb[comorb["stay_id"].isin(keep)].copy()
        else:
            comorb_chunk = comorb[comorb["stay_id"].isin(grid["stay_id"].drop_duplicates())].copy()

        stroke_cohort = comorb_chunk[comorb_chunk["stroke_family"] == 1].copy()
        stroke_ids = set(int(v) for v in stroke_cohort["stay_id"].tolist())
        stroke_grid = grid[grid["stay_id"].isin(stroke_ids)].copy()
        stroke_grid = stroke_grid.sort_values(["stay_id", "hour"], kind="mergesort")
        if stroke_grid.empty:
            continue

        for col in ["gcs_total", "rass", "map_merged", "vasopressors_active", "propofol_rate", "midazolam_rate", "fentanyl_rate", "restraint_active"]:
            if col not in stroke_grid.columns:
                stroke_grid[col] = pd.NA if col in {"gcs_total", "rass", "map_merged", "propofol_rate", "midazolam_rate", "fentanyl_rate"} else 0

        stroke_grid["gcs_total"] = pd.to_numeric(stroke_grid["gcs_total"], errors="coerce")
        stroke_grid["rass"] = pd.to_numeric(stroke_grid["rass"], errors="coerce")
        stroke_grid["map_merged"] = pd.to_numeric(stroke_grid["map_merged"], errors="coerce")
        for col in ["vasopressors_active", "restraint_active"]:
            stroke_grid[col] = pd.to_numeric(stroke_grid[col], errors="coerce").fillna(0).astype(int)

        stroke_grid["gcs_baseline_12h"] = stroke_grid.groupby("stay_id", sort=False)["gcs_total"].transform(
            lambda s: s.rolling(window=12, min_periods=1).max().shift(1)
        )
        stroke_grid["gcs_drop_ge_2"] = ((stroke_grid["gcs_baseline_12h"] - stroke_grid["gcs_total"]) >= 2).fillna(False).astype(int)
        stroke_grid["gcs_drop_ge_4"] = ((stroke_grid["gcs_baseline_12h"] - stroke_grid["gcs_total"]) >= 4).fillna(False).astype(int)

        stroke_grid["rass_prev"] = stroke_grid.groupby("stay_id", sort=False)["rass"].shift(1)
        stroke_grid["rass_abs_change_ge_2"] = ((stroke_grid["rass"] - stroke_grid["rass_prev"]).abs() >= 2).fillna(False).astype(int)

        sedation_cols = [c for c in ["propofol_rate", "midazolam_rate", "fentanyl_rate"] if c in stroke_grid.columns]
        for col in sedation_cols:
            stroke_grid[col] = pd.to_numeric(stroke_grid[col], errors="coerce").fillna(0.0)
        stroke_grid["sedation_burden"] = stroke_grid[sedation_cols].sum(axis=1) if sedation_cols else 0.0
        stroke_grid["sedation_prev"] = stroke_grid.groupby("stay_id", sort=False)["sedation_burden"].shift(1).fillna(0.0)
        stroke_grid["sedation_intensification"] = (stroke_grid["sedation_burden"] > stroke_grid["sedation_prev"]).astype(int)

        vent_col = "ventilation_status" if "ventilation_status" in stroke_grid.columns else None
        if vent_col:
            stroke_grid["ventilation_status"] = stroke_grid["ventilation_status"].fillna("none").astype(str)
            stroke_grid["vent_prev"] = stroke_grid.groupby("stay_id", sort=False)["ventilation_status"].shift(1).fillna("none")
            stroke_grid["ventilation_escalation"] = (
                (stroke_grid["ventilation_status"] != stroke_grid["vent_prev"])
                & (stroke_grid["ventilation_status"].str.lower() != "none")
            ).astype(int)
        else:
            stroke_grid["ventilation_escalation"] = 0

        stroke_grid["map_low"] = (stroke_grid["map_merged"] < 65).fillna(False).astype(int)
        stroke_grid["neuro_deterioration_proxy"] = (
            (stroke_grid["gcs_drop_ge_2"] == 1)
            | (stroke_grid["rass_abs_change_ge_2"] == 1)
            | (stroke_grid["ventilation_escalation"] == 1)
            | (stroke_grid["restraint_active"] == 1)
            | (stroke_grid["sedation_intensification"] == 1)
        ).astype(int)

        marker_cols = [
            "stay_id",
            "hour",
            "gcs_total",
            "rass",
            "map_merged",
            "vasopressors_active",
            "restraint_active",
            "sedation_burden",
            "gcs_drop_ge_2",
            "gcs_drop_ge_4",
            "rass_abs_change_ge_2",
            "sedation_intensification",
            "ventilation_escalation",
            "map_low",
            "neuro_deterioration_proxy",
        ]
        if "ventilation_status" in stroke_grid.columns:
            marker_cols.insert(marker_cols.index("ventilation_escalation"), "ventilation_status")

        cohort_written = write_table(stroke_cohort, cohort_parts_dir / f"part_{chunk_idx:05d}.parquet", index=False)
        markers_written = write_table(stroke_grid[marker_cols], markers_parts_dir / f"part_{chunk_idx:05d}.parquet", index=False)
        print(f"Wrote {cohort_written}")
        print(f"Wrote {markers_written}")
        stroke_cohort_total += int(stroke_cohort["stay_id"].nunique())
        marker_rows_total += int(len(stroke_grid))
        deterioration_stays += int(stroke_grid.groupby("stay_id", sort=False)["neuro_deterioration_proxy"].max().sum())
        part_count += 1

    summary = {
        "stroke_cohort_stays": stroke_cohort_total,
        "marker_rows": marker_rows_total,
        "stays_with_any_neuro_deterioration_proxy": deterioration_stays,
        "parts": part_count,
        "outputs": relativize_value({"cohort_parts": str(cohort_parts_dir), "markers_parts": str(markers_parts_dir)}, root=ROOT_DIR),
    }
    summary_path.write_text(
        json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
