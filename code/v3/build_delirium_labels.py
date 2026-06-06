#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    ROOT_DIR,
    DEFAULT_HOURLY_STATE_GRID,
    V3_PROCESSED_DIR,
    ensure_v3_directories,
)
from v3.io_utils import chunk_dir_path, iter_table_chunks, relativize_value, write_table  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 delirium labels.")
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--prediction-interval", type=int, default=4)
    p.add_argument("--lookahead-hours", type=int, default=24)
    p.add_argument("--resolution-window", type=int, default=24)
    p.add_argument("--max-hour", type=int, default=168)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--cohort-out", default=str(V3_PROCESSED_DIR / "delirium" / "delirium_cohort_v3.parquet"))
    p.add_argument("--labels-out", default=str(V3_PROCESSED_DIR / "delirium" / "delirium_labels_v3.parquet"))
    p.add_argument("--summary-json", default=str(V3_PROCESSED_DIR / "delirium" / "delirium_label_summary_v3.json"))
    return p.parse_args()


def _future_nonpositive_resolution(flags_neg: np.ndarray, flags_pos: np.ndarray, start_idx: int, window: int) -> bool:
    end = min(len(flags_neg), start_idx + window)
    if end - start_idx < window:
        return False
    return bool(flags_pos[start_idx:end].sum() == 0 and flags_neg[start_idx:end].sum() >= 1)


def _build_chunk_outputs(df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"stay_id", "hour", "delirium_positive", "delirium_negative", "delirium_uta"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required delirium columns in hourly grid: {missing}")

    df = df.sort_values(["stay_id", "hour"], kind="mergesort").copy()
    for col in ["delirium_positive", "delirium_negative", "delirium_uta"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    cohort_rows = []
    label_rows = []
    for stay_id, g in df.groupby("stay_id", sort=False):
        g = g[g["hour"].between(0, int(args.max_hour), inclusive="both")].copy()
        if g.empty:
            continue
        pos_hours = g.loc[g["delirium_positive"] == 1, "hour"]
        if pos_hours.empty:
            continue
        onset_hour = int(pos_hours.min())
        observed_before_onset = g[g["hour"] < onset_hour]
        left_censored = int(observed_before_onset.empty or observed_before_onset["delirium_negative"].sum() == 0)
        cohort_rows.append(
            {
                "stay_id": int(stay_id),
                "delirium_onset_hour": onset_hour,
                "left_censored": left_censored,
                "has_any_positive": 1,
                "n_positive_hours": int(g["delirium_positive"].sum()),
                "n_negative_hours": int(g["delirium_negative"].sum()),
                "n_uta_hours": int(g["delirium_uta"].sum()),
            }
        )
        upper_t = int(g["hour"].max())
        t = onset_hour + int(args.prediction_interval)
        flags_pos = g.set_index("hour")["delirium_positive"].reindex(range(upper_t + 1), fill_value=0).to_numpy()
        flags_neg = g.set_index("hour")["delirium_negative"].reindex(range(upper_t + 1), fill_value=0).to_numpy()
        while t <= upper_t:
            horizon_end = min(upper_t, t + int(args.lookahead_hours))
            persistent = int(flags_pos[t + 1 : horizon_end + 1].sum() > 0) if t + 1 <= horizon_end else 0
            resolution = 0
            for cand in range(t + 1, horizon_end + 1):
                if _future_nonpositive_resolution(flags_neg, flags_pos, cand, int(args.resolution_window)):
                    resolution = 1
                    break
            label_rows.append(
                {
                    "stay_id": int(stay_id),
                    "prediction_hour": int(t),
                    "delirium_onset_hour": onset_hour,
                    "left_censored": left_censored,
                    "label_persistent_delirium": persistent,
                    "label_resolution": resolution,
                }
            )
            t += int(args.prediction_interval)
    return pd.DataFrame(cohort_rows), pd.DataFrame(label_rows)


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    cohort_path = Path(args.cohort_out)
    labels_path = Path(args.labels_out)
    summary_path = Path(args.summary_json)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_parts_dir = chunk_dir_path(cohort_path)
    labels_parts_dir = chunk_dir_path(labels_path)
    if cohort_parts_dir.exists():
        shutil.rmtree(cohort_parts_dir)
    if labels_parts_dir.exists():
        shutil.rmtree(labels_parts_dir)
    cohort_parts_dir.mkdir(parents=True, exist_ok=True)
    labels_parts_dir.mkdir(parents=True, exist_ok=True)

    n_stays_in_grid = 0
    delirium_cohort_stays = 0
    label_rows_total = 0
    left_censored_sum = 0
    persistent_sum = 0
    resolution_sum = 0
    part_count = 0
    cohort_written = ""
    labels_written = ""
    for chunk_idx, df in enumerate(iter_table_chunks(args.hourly_grid), start=1):
        if args.stay_limit is not None:
            keep = df["stay_id"].drop_duplicates().head(int(args.stay_limit)).tolist()
            df = df[df["stay_id"].isin(keep)].copy()
        cohort_df, labels_df = _build_chunk_outputs(df, args)
        cohort_part = cohort_parts_dir / f"part_{chunk_idx:05d}.parquet"
        labels_part = labels_parts_dir / f"part_{chunk_idx:05d}.parquet"
        cohort_written = str(write_table(cohort_df, cohort_part, index=False))
        labels_written = str(write_table(labels_df, labels_part, index=False))
        print(f"Wrote {cohort_written}")
        print(f"Wrote {labels_written}")
        n_stays_in_grid += int(df["stay_id"].nunique())
        delirium_cohort_stays += int(len(cohort_df))
        label_rows_total += int(len(labels_df))
        left_censored_sum += int(cohort_df["left_censored"].sum()) if len(cohort_df) else 0
        persistent_sum += int(labels_df["label_persistent_delirium"].sum()) if len(labels_df) else 0
        resolution_sum += int(labels_df["label_resolution"].sum()) if len(labels_df) else 0
        part_count += 1
    summary = {
        "n_stays_in_grid": n_stays_in_grid,
        "delirium_cohort_stays": delirium_cohort_stays,
        "left_censored_rate": (left_censored_sum / delirium_cohort_stays) if delirium_cohort_stays else None,
        "label_rows": label_rows_total,
        "persistent_positive_rate": (persistent_sum / label_rows_total) if label_rows_total else None,
        "resolution_positive_rate": (resolution_sum / label_rows_total) if label_rows_total else None,
        "parts": part_count,
        "outputs": relativize_value({"cohort_parts": str(cohort_parts_dir), "labels_parts": str(labels_parts_dir)}, root=ROOT_DIR),
    }
    summary_path.write_text(
        json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
