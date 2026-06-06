#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import shutil

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    ROOT_DIR,
    DEFAULT_BQ_FEATURE_DIR,
    DEFAULT_HOURLY_STATE_GRID,
    DEFAULT_V3_COHORT_FILE,
    DEFAULT_V3_TIMESERIES_FILE,
    V3_MAX_HOURS,
    V3_HOURLY_FEATURES_DIR,
    ensure_v3_directories,
)
from v3.io_utils import read_table, relativize_value, write_table  # type: ignore
from v3.io_utils import iter_table_chunks, table_exists  # type: ignore
from v3.mappings import CORE_BACKBONE_FEATURES  # type: ignore


STATIC_CONTEXT_COLS = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "deathtime",
    "anchor_age",
    "gender",
    "label_mortality",
    "los_hours",
    "los_days",
    "prolonged_los_3d",
    "prolonged_los_5d",
    "prolonged_los_7d",
    "ckd",
    "readmission_30d",
]

DEFAULT_EXTRA_HOURLY_FILES = {
    "bilirubin_total": "bilirubin_total_hourly.csv",
    "vasopressors": "vasopressors_hourly.csv",
    "rrt": "rrt_hourly.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a time-aware 168h hourly state grid for TIMELY-Bench v3.")
    p.add_argument("--timeseries-csv", default=str(DEFAULT_V3_TIMESERIES_FILE))
    p.add_argument("--cohort-csv", default=str(DEFAULT_V3_COHORT_FILE))
    p.add_argument("--bq-feature-dir", default=str(DEFAULT_BQ_FEATURE_DIR))
    p.add_argument("--hours", type=int, default=V3_MAX_HOURS)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--v3-hourly-feature-dir", default=str(V3_HOURLY_FEATURES_DIR))
    p.add_argument("--out", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--meta-json", default="")
    p.add_argument("--include-missingness-masks", action="store_true")
    p.add_argument("--add-empty-backbone-cols", action="store_true")
    p.add_argument("--chunk-size-stays", type=int, default=2000)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _load_cohort(path: Path, stay_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")
    if stay_limit is not None:
        df = df.head(int(stay_limit)).copy()
    for col in ("intime", "outtime", "deathtime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _build_fixed_hour_grid(stay_ids: np.ndarray, hours: int) -> pd.DataFrame:
    repeated_stays = np.repeat(stay_ids, hours)
    repeated_hours = np.tile(np.arange(hours, dtype=np.int16), len(stay_ids))
    return pd.DataFrame({"stay_id": repeated_stays, "hour": repeated_hours})


def _load_hourly_table(path: Path, stay_ids: Iterable[int]) -> pd.DataFrame:
    if not table_exists(path):
        return pd.DataFrame(columns=["stay_id", "hour"])
    stay_id_set = set(int(v) for v in stay_ids)
    parts = []
    for chunk in iter_table_chunks(path):
        if "stay_id" not in chunk.columns or "hour" not in chunk.columns:
            continue
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk["hour"] = pd.to_numeric(chunk["hour"], errors="coerce")
        chunk = chunk.dropna(subset=["stay_id", "hour"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype("int64")
        chunk["hour"] = chunk["hour"].astype("int64")
        chunk = chunk[chunk["stay_id"].isin(stay_id_set)].copy()
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["stay_id", "hour"])
    return pd.concat(parts, ignore_index=True)


def _iter_stay_batches(cohort: pd.DataFrame, chunk_size_stays: int):
    if chunk_size_stays <= 0:
        raise ValueError("chunk_size_stays must be positive.")
    for start in range(0, len(cohort), chunk_size_stays):
        yield cohort.iloc[start : start + chunk_size_stays].copy()


def _existing_part_paths(parts_dir: Path) -> list[Path]:
    return sorted(parts_dir.glob("part_*.parquet"))


def _part_row_count(path: Path) -> int:
    if pq is not None:
        return int(pq.ParquetFile(path).metadata.num_rows)
    return int(len(pd.read_parquet(path)))


def _merge_extra_hourly_sources(df: pd.DataFrame, bq_feature_dir: Path, stay_ids: np.ndarray) -> pd.DataFrame:
    out = df
    for _, file_name in DEFAULT_EXTRA_HOURLY_FILES.items():
        path = bq_feature_dir / file_name
        extra = _load_hourly_table(path, stay_ids)
        if extra.empty:
            continue
        value_cols = [col for col in extra.columns if col not in {"stay_id", "hour"}]
        if not value_cols:
            continue
        out = out.merge(extra, on=["stay_id", "hour"], how="left", suffixes=("", "_extra"))
        for col in value_cols:
            extra_col = f"{col}_extra"
            if extra_col in out.columns:
                if col in out.columns:
                    out[col] = out[col].combine_first(out[extra_col])
                    out = out.drop(columns=[extra_col])
                else:
                    out = out.rename(columns={extra_col: col})
    return out


def _merge_directory_hourly_sources(df: pd.DataFrame, directory: Path, stay_ids: np.ndarray) -> pd.DataFrame:
    if not directory.exists():
        return df
    out = df
    candidate_paths = sorted(
        [p for p in directory.iterdir() if p.is_file() and p.suffix in {".csv", ".parquet"}]
    )
    for path in candidate_paths:
        extra = _load_hourly_table(path, stay_ids)
        if extra.empty:
            continue
        value_cols = [col for col in extra.columns if col not in {"stay_id", "hour"}]
        if not value_cols:
            continue
        out = out.merge(extra, on=["stay_id", "hour"], how="left", suffixes=("", "_extra"))
        for col in value_cols:
            extra_col = f"{col}_extra"
            if extra_col in out.columns:
                if col in out.columns:
                    out[col] = out[col].combine_first(out[extra_col])
                    out = out.drop(columns=[extra_col])
                else:
                    out = out.rename(columns={extra_col: col})
    return out


def _normalize_temperature(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.where(s > 80, (s - 32.0) / 1.8, s), index=series.index, dtype="float64")


def _ensure_backbone_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in CORE_BACKBONE_FEATURES:
        if col not in df.columns:
            df[col] = np.nan
    return df


def _derive_columns(df: pd.DataFrame, include_missingness_masks: bool) -> pd.DataFrame:
    if "mbp" in df.columns and "map_merged" not in df.columns:
        df["map_merged"] = pd.to_numeric(df["mbp"], errors="coerce")
    elif "map_merged" not in df.columns:
        df["map_merged"] = np.nan

    if "temperature" in df.columns:
        df["temperature_c"] = _normalize_temperature(df["temperature"])
    elif "temperature_c" not in df.columns:
        df["temperature_c"] = np.nan

    if "glucose_lab" in df.columns or "glucose_chart" in df.columns:
        glucose_lab = pd.to_numeric(df.get("glucose_lab"), errors="coerce") if "glucose_lab" in df.columns else np.nan
        glucose_chart = pd.to_numeric(df.get("glucose_chart"), errors="coerce") if "glucose_chart" in df.columns else np.nan
        if isinstance(glucose_lab, pd.Series):
            df["glucose_merged"] = glucose_lab
            if isinstance(glucose_chart, pd.Series):
                df["glucose_merged"] = df["glucose_merged"].combine_first(glucose_chart)
        elif isinstance(glucose_chart, pd.Series):
            df["glucose_merged"] = glucose_chart
    elif "glucose_merged" not in df.columns:
        df["glucose_merged"] = np.nan

    if "gcs_total" not in df.columns:
        if "gcs_min" in df.columns:
            df["gcs_total"] = pd.to_numeric(df["gcs_min"], errors="coerce")
            df["gcs_total_is_proxy"] = 1
        else:
            df["gcs_total"] = np.nan
            df["gcs_total_is_proxy"] = 0

    if "vasopressors" in df.columns and "vasopressors_active" not in df.columns:
        df["vasopressors_active"] = pd.to_numeric(df["vasopressors"], errors="coerce").fillna(0).astype(int)
    elif "vasopressors_active" not in df.columns:
        df["vasopressors_active"] = 0

    if "rrt" in df.columns and "rrt_active" not in df.columns:
        df["rrt_active"] = pd.to_numeric(df["rrt"], errors="coerce").fillna(0).astype(int)
    elif "rrt_active" not in df.columns:
        df["rrt_active"] = 0

    if "urineoutput" in df.columns and "urineoutput_hourly" not in df.columns:
        df["urineoutput_hourly"] = pd.to_numeric(df["urineoutput"], errors="coerce")

    if "los_hours" in df.columns:
        df["is_within_observed_icu_los"] = (pd.to_numeric(df["hour"], errors="coerce") < pd.to_numeric(df["los_hours"], errors="coerce")).astype(int)
        df["hours_until_discharge"] = pd.to_numeric(df["los_hours"], errors="coerce") - pd.to_numeric(df["hour"], errors="coerce")
    else:
        df["is_within_observed_icu_los"] = 1
        df["hours_until_discharge"] = np.nan

    if "deathtime" in df.columns and "intime" in df.columns:
        dt_diff = (pd.to_datetime(df["deathtime"], errors="coerce") - pd.to_datetime(df["intime"], errors="coerce")).dt.total_seconds() / 3600.0
        df["hours_until_death"] = dt_diff - pd.to_numeric(df["hour"], errors="coerce")
    else:
        df["hours_until_death"] = np.nan

    df["hour_normalized_168h"] = pd.to_numeric(df["hour"], errors="coerce") / max(1, (int(df["hour"].max()) if len(df) else 167))

    if include_missingness_masks:
        feature_cols = [col for col in CORE_BACKBONE_FEATURES if col in df.columns]
        for col in feature_cols:
            df[f"{col}__missing"] = df[col].isna().astype(int)

    return df


def main() -> None:
    args = parse_args()
    ensure_v3_directories()

    timeseries_path = Path(args.timeseries_csv)
    cohort_path = Path(args.cohort_csv)
    bq_feature_dir = Path(args.bq_feature_dir)
    v3_hourly_feature_dir = Path(args.v3_hourly_feature_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_parts_dir = out_path.with_name(f"{out_path.name}.parts")
    if out_parts_dir.exists() and not args.resume:
        shutil.rmtree(out_parts_dir)
    out_parts_dir.mkdir(parents=True, exist_ok=True)

    cohort = _load_cohort(cohort_path, args.stay_limit)
    static_cols = [col for col in STATIC_CONTEXT_COLS if col in cohort.columns]

    existing_parts = _existing_part_paths(out_parts_dir) if args.resume else []
    completed_parts = len(existing_parts)
    part_paths: list[str] = [str(path) for path in existing_parts]
    total_rows = sum(_part_row_count(path) for path in existing_parts)
    total_parts = completed_parts

    for part_idx, cohort_chunk in enumerate(_iter_stay_batches(cohort, int(args.chunk_size_stays)), start=1):
        if args.resume and part_idx <= completed_parts:
            continue
        stay_ids = cohort_chunk["stay_id"].to_numpy(dtype=np.int64)
        grid = _build_fixed_hour_grid(stay_ids, int(args.hours))
        static_df = cohort_chunk[static_cols].copy()
        grid = grid.merge(static_df, on="stay_id", how="left")

        ts = _load_hourly_table(timeseries_path, stay_ids)
        if not ts.empty:
            ts = ts[ts["hour"].between(0, int(args.hours) - 1)].copy()
            grid = grid.merge(ts, on=["stay_id", "hour"], how="left", suffixes=("", "_ts"))

        grid = _merge_extra_hourly_sources(grid, bq_feature_dir, stay_ids)
        grid = _merge_directory_hourly_sources(grid, v3_hourly_feature_dir, stay_ids)
        grid = _derive_columns(grid, include_missingness_masks=bool(args.include_missingness_masks))
        if args.add_empty_backbone_cols:
            grid = _ensure_backbone_columns(grid)

        grid = grid.sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)
        part_path = out_parts_dir / f"part_{part_idx:05d}.parquet"
        written = write_table(grid, part_path, index=False)
        part_paths.append(str(written))
        total_rows += int(len(grid))
        total_parts += 1
        print(
            f"Wrote {written} "
            f"(part {part_idx}, stays={cohort_chunk['stay_id'].nunique()}, rows={len(grid)})",
            flush=True,
        )
        del grid
        del ts
        del static_df
        del cohort_chunk
        del stay_ids
        gc.collect()

    print(f"Wrote partitioned hourly state grid to {out_parts_dir} ({total_parts} parts)")

    if args.meta_json:
        meta = {
            "timeseries_csv": relativize_value(str(timeseries_path), root=ROOT_DIR),
            "cohort_csv": relativize_value(str(cohort_path), root=ROOT_DIR),
            "bq_feature_dir": relativize_value(str(bq_feature_dir), root=ROOT_DIR),
            "hours": int(args.hours),
            "n_stays": int(cohort["stay_id"].nunique()),
            "n_rows": total_rows,
            "n_parts": total_parts,
            "part_paths": relativize_value(part_paths, root=ROOT_DIR),
            "partition_dir": relativize_value(str(out_parts_dir), root=ROOT_DIR),
        }
        meta_path = Path(args.meta_json)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
