#!/usr/bin/env python3
"""
Build structured features for progression tasks at (stay_id, prediction_hour).

Windows:
- W6:    [T-6,  T]
- W12:   [T-12, T]
- W24:   [T-24, T]
- leaked:[T-24, T+24] clipped by per-stay max available hour
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


WINDOWS = {
    "W6": 6,
    "W12": 12,
    "W24": 24,
    "leaked": 24,
}

EXCLUDE_COLS = {
    "stay_id",
    "hour",
    "subject_id",
    "hadm_id",
    "intime",
    "charttime",
    "deathtime",
    "outtime",
}

STATS = [
    "min",
    "max",
    "mean",
    "last",
    "first",
    "std",
    "missing_rate",
    "n_measurements",
    "delta_last_first",
    "slope_per_hour",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build progression structured features.")
    p.add_argument("--timeseries-csv", default="data/processed/timeseries_sorted_72h.csv")
    p.add_argument("--timepoints-csv", default="data/processed/progression_timepoints.csv")
    p.add_argument("--output-dir", default="data/processed/progression_features")
    p.add_argument("--summary-json", default="results/audit/progression_structured_feature_summary.json")
    p.add_argument("--max-hour", type=int, default=72)
    p.add_argument("--flush-rows", type=int, default=2000)
    p.add_argument("--report-every-stays", type=int, default=500)
    p.add_argument("--max-stays", type=int, default=None)
    return p.parse_args()


def _make_grid(ts_stay: pd.DataFrame, feature_cols: List[str], max_hour: int) -> Tuple[np.ndarray, int]:
    grid = np.full((max_hour + 1, len(feature_cols)), np.nan, dtype=np.float32)
    if ts_stay.empty:
        return grid, 0
    hourly = ts_stay.groupby("hour", sort=False)[feature_cols].mean(numeric_only=True).reset_index()
    hours = pd.to_numeric(hourly["hour"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    valid = (hours >= 0) & (hours <= max_hour)
    if valid.any():
        grid[hours[valid], :] = hourly.loc[valid, feature_cols].to_numpy(dtype=np.float32, copy=False)
        max_obs = int(hours[valid].max())
    else:
        max_obs = 0
    return grid, max_obs


def _compute_stats(
    grid: np.ndarray,
    start: int,
    end: int,
    feature_cols: List[str],
) -> Dict[str, float]:
    seg = grid[start : end + 1, :]
    n_rows = int(seg.shape[0])
    out: Dict[str, float] = {}
    if n_rows <= 0:
        for col in feature_cols:
            for st in STATS:
                out[f"{col}_{st}"] = np.nan
        return out

    obs = ~np.isnan(seg)
    cnt = obs.sum(axis=0).astype(np.int32)

    mn = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    mx = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    avg = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    std = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    valid_cols = np.where(cnt > 0)[0]
    if valid_cols.size > 0:
        s2 = seg[:, valid_cols]
        mn[valid_cols] = np.nanmin(s2, axis=0)
        mx[valid_cols] = np.nanmax(s2, axis=0)
        avg[valid_cols] = np.nanmean(s2, axis=0)
        std[valid_cols] = np.nanstd(s2, axis=0, ddof=0)
    miss_rate = 1.0 - (cnt / float(n_rows))

    for j, col in enumerate(feature_cols):
        c = int(cnt[j])
        if c == 0:
            first = last = delta = slope = np.nan
            mn_j = mx_j = avg_j = std_j = np.nan
        else:
            idx = np.flatnonzero(obs[:, j])
            fi = int(idx[0])
            li = int(idx[-1])
            first = float(seg[fi, j])
            last = float(seg[li, j])
            if c == 1:
                delta = 0.0
                slope = 0.0
                std_j = 0.0
            else:
                delta = last - first
                hour_delta = float(li - fi)
                slope = (delta / hour_delta) if hour_delta > 0 else 0.0
                std_j = float(std[j]) if np.isfinite(std[j]) else 0.0
            mn_j = float(mn[j]) if np.isfinite(mn[j]) else np.nan
            mx_j = float(mx[j]) if np.isfinite(mx[j]) else np.nan
            avg_j = float(avg[j]) if np.isfinite(avg[j]) else np.nan

        out[f"{col}_min"] = mn_j
        out[f"{col}_max"] = mx_j
        out[f"{col}_mean"] = avg_j
        out[f"{col}_last"] = last
        out[f"{col}_first"] = first
        out[f"{col}_std"] = std_j
        out[f"{col}_missing_rate"] = float(miss_rate[j])
        out[f"{col}_n_measurements"] = float(c)
        out[f"{col}_delta_last_first"] = delta
        out[f"{col}_slope_per_hour"] = slope

    return out


def _flush_rows(
    rows: List[Dict[str, object]],
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
) -> Optional[pq.ParquetWriter]:
    if not rows:
        return writer
    df = pd.DataFrame(rows)
    for c in ["stay_id", "prediction_hour", "window_start", "window_end"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    if "window_hours_actual" in df.columns:
        df["window_hours_actual"] = pd.to_numeric(df["window_hours_actual"], errors="coerce").astype(np.float32)
    for c in df.columns:
        if c.endswith(
            (
                "_min",
                "_max",
                "_mean",
                "_last",
                "_first",
                "_std",
                "_missing_rate",
                "_n_measurements",
                "_delta_last_first",
                "_slope_per_hour",
            )
        ):
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
    writer.write_table(table)
    rows.clear()
    return writer


def main() -> None:
    args = parse_args()
    t0 = time.time()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_json)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    tp = pd.read_csv(args.timepoints_csv)
    tp["stay_id"] = pd.to_numeric(tp["stay_id"], errors="coerce").astype("Int64")
    tp["prediction_hour"] = pd.to_numeric(tp["prediction_hour"], errors="coerce")
    tp = tp.dropna(subset=["stay_id", "prediction_hour"]).copy()
    tp["stay_id"] = tp["stay_id"].astype(np.int64)
    tp["prediction_hour"] = tp["prediction_hour"].astype(np.int16).clip(0, int(args.max_hour))
    tp = tp.drop_duplicates(subset=["stay_id", "prediction_hour"]).sort_values(
        ["stay_id", "prediction_hour"], kind="mergesort"
    )

    stays = tp["stay_id"].drop_duplicates().tolist()
    if args.max_stays is not None and args.max_stays > 0:
        stays = stays[: int(args.max_stays)]
    stay_set = set(stays)
    tp = tp[tp["stay_id"].isin(stay_set)].copy()
    tp_group = tp.groupby("stay_id", sort=False)["prediction_hour"].apply(list).to_dict()

    print(f"Loading timeseries: {args.timeseries_csv}")
    ts = pd.read_csv(args.timeseries_csv)
    ts["stay_id"] = pd.to_numeric(ts["stay_id"], errors="coerce").astype("Int64")
    ts["hour"] = pd.to_numeric(ts["hour"], errors="coerce")
    ts = ts.dropna(subset=["stay_id", "hour"]).copy()
    ts["stay_id"] = ts["stay_id"].astype(np.int64)
    ts["hour"] = ts["hour"].astype(np.int16)
    ts = ts[(ts["hour"] >= 0) & (ts["hour"] <= int(args.max_hour))]
    ts = ts[ts["stay_id"].isin(stay_set)].copy()

    feature_cols = []
    for c in ts.columns:
        if c in EXCLUDE_COLS:
            continue
        if pd.api.types.is_numeric_dtype(ts[c]):
            feature_cols.append(c)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found in timeseries input.")

    ts = ts.sort_values(["stay_id", "hour"], kind="mergesort")
    ts_groups = ts.groupby("stay_id", sort=False).groups

    writers: Dict[str, Optional[pq.ParquetWriter]] = {}
    buffers: Dict[str, List[Dict[str, object]]] = {}
    out_paths: Dict[str, Path] = {}
    rows_written: Dict[str, int] = {}
    for w in WINDOWS:
        out = out_dir / f"structured_{w}.parquet"
        if out.exists():
            out.unlink()
        writers[w] = None
        buffers[w] = []
        out_paths[w] = out
        rows_written[w] = 0

    print(f"Building structured progression features for {len(stays):,} stays ...")
    for i, sid in enumerate(stays, start=1):
        idx = ts_groups.get(int(sid))
        stay_ts = ts.loc[idx] if idx is not None else ts.iloc[:0]
        grid, max_obs_hour = _make_grid(stay_ts, feature_cols, int(args.max_hour))
        t_list = tp_group.get(int(sid), [])
        if not t_list:
            continue
        t_list = sorted(set(int(t) for t in t_list))

        for t in t_list:
            for wname, hours in WINDOWS.items():
                if wname == "leaked":
                    start = max(0, t - int(hours))
                    end = min(max_obs_hour, t + int(hours))
                else:
                    start = max(0, t - int(hours))
                    end = min(int(args.max_hour), t)
                if end < start:
                    end = start

                stats = _compute_stats(grid=grid, start=start, end=end, feature_cols=feature_cols)
                row = {
                    "stay_id": int(sid),
                    "prediction_hour": int(t),
                    "window_start": int(start),
                    "window_end": int(end),
                    "window_hours_actual": float(end - start + 1),
                }
                row.update(stats)
                buffers[wname].append(row)
                rows_written[wname] += 1
                if len(buffers[wname]) >= int(args.flush_rows):
                    writers[wname] = _flush_rows(buffers[wname], writers[wname], out_paths[wname])

        if i % int(args.report_every_stays) == 0:
            elapsed = (time.time() - t0) / 60.0
            print(f"  processed stays={i:,}/{len(stays):,} elapsed={elapsed:.1f}m")

    for wname in WINDOWS:
        writers[wname] = _flush_rows(buffers[wname], writers[wname], out_paths[wname])
        if writers[wname] is not None:
            writers[wname].close()

    summary = {
        "n_stays": int(len(stays)),
        "n_timepoints": int(len(tp)),
        "feature_cols_count": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "rows_written": rows_written,
        "outputs": {k: str(v) for k, v in out_paths.items()},
        "elapsed_seconds": float(time.time() - t0),
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved: {summary_out}")


if __name__ == "__main__":
    main()
