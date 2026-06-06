#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    DEFAULT_DIAGNOSIS_PATHWAY_EVENTS,
    DEFAULT_HOURLY_STATE_GRID,
    DEFAULT_STATE_VECTORS,
    ensure_v3_directories,
)
from v3.io_utils import chunk_dir_path, iter_table_chunks, read_table, table_exists, write_table  # type: ignore
from v3.mappings import CORE_BACKBONE_FEATURES  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 state-vector trajectories.")
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--pathway-events", default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS))
    p.add_argument("--out", default=str(DEFAULT_STATE_VECTORS))
    p.add_argument("--stay-limit", type=int, default=None)
    return p.parse_args()


def _load_events(path: Path, stay_ids: set[int]) -> pd.DataFrame:
    if not table_exists(path):
        return pd.DataFrame(columns=["stay_id", "event_time_hour", "event_type", "is_proxy"])
    df = read_table(path)
    if "stay_id" not in df.columns:
        return pd.DataFrame(columns=["stay_id", "event_time_hour", "event_type", "is_proxy"])
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df["event_time_hour"] = pd.to_numeric(df.get("event_time_hour"), errors="coerce")
    df = df.dropna(subset=["stay_id", "event_time_hour"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")
    df["event_hour_floor"] = df["event_time_hour"].astype(int)
    return df[df["stay_id"].isin(stay_ids)].copy()


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_parts_dir = chunk_dir_path(out_path)
    if out_parts_dir.exists():
        shutil.rmtree(out_parts_dir)
    out_parts_dir.mkdir(parents=True, exist_ok=True)

    part_count = 0
    for chunk_idx, grid in enumerate(iter_table_chunks(args.hourly_grid), start=1):
        if args.stay_limit is not None:
            keep = grid["stay_id"].drop_duplicates().head(int(args.stay_limit)).tolist()
            grid = grid[grid["stay_id"].isin(keep)].copy()
        stay_ids = set(int(v) for v in grid["stay_id"].drop_duplicates().tolist())
        events = _load_events(Path(args.pathway_events), stay_ids)

        keep_cols = ["stay_id", "hour", "hour_normalized_168h", "is_within_observed_icu_los", "hours_until_discharge", "hours_until_death"]
        keep_cols.extend([col for col in CORE_BACKBONE_FEATURES if col in grid.columns])
        keep_cols.extend([col for col in grid.columns if col.endswith("__missing")])
        vectors = grid[keep_cols].copy()

        if not events.empty:
            events["event_any_flag"] = 1
            events["event_proxy_flag"] = events["is_proxy"].fillna(0).astype(int)
            type_pivot = (
                pd.get_dummies(events["event_type"], prefix="evt")
                .groupby([events["stay_id"], events["event_hour_floor"]], sort=False)
                .max()
                .reset_index()
                .rename(columns={"event_hour_floor": "hour"})
            )
            base_flags = (
                events.groupby(["stay_id", "event_hour_floor"], sort=False)[["event_any_flag", "event_proxy_flag"]]
                .max()
                .reset_index()
                .rename(columns={"event_hour_floor": "hour"})
            )
            vectors = vectors.merge(base_flags, on=["stay_id", "hour"], how="left")
            vectors = vectors.merge(type_pivot, on=["stay_id", "hour"], how="left")
        else:
            vectors["event_any_flag"] = 0
            vectors["event_proxy_flag"] = 0

        flag_cols = [col for col in vectors.columns if col.startswith("evt_") or col.endswith("_flag")]
        for col in flag_cols:
            vectors[col] = pd.to_numeric(vectors[col], errors="coerce").fillna(0).astype(int)

        vectors = vectors.sort_values(["stay_id", "hour"], kind="mergesort")
        written = write_table(vectors, out_parts_dir / f"part_{chunk_idx:05d}.parquet", index=False)
        print(f"Wrote {written}")
        part_count += 1

    print(f"Wrote partitioned state vectors to {out_parts_dir} ({part_count} parts)")


if __name__ == "__main__":
    main()
