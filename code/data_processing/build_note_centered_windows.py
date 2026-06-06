#!/usr/bin/env python3
"""
Build note-centered structured features with spec-compliant windows.

Windows:
- W6/W12/W24: lookback [T-W, T]
- D0: calendar day up to T -> [floor(T/24)*24, T]
- leaked: [T-24, T+24] (intentional future leakage)

Output:
- data/processed/note_centered/note_window_structured_W6.parquet
- data/processed/note_centered/note_window_structured_W12.parquet
- data/processed/note_centered/note_window_structured_W24.parquet
- data/processed/note_centered/note_window_structured_D0.parquet
- data/processed/note_centered/note_window_structured_leaked.parquet
- results/audit/phase2_note_centered_windows_summary.json
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

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR  # type: ignore


WINDOWS: Dict[str, Dict[str, object]] = {
    "W6": {"type": "lookback", "hours": 6},
    "W12": {"type": "lookback", "hours": 12},
    "W24": {"type": "lookback", "hours": 24},
    "D0": {"type": "calendar_day_up_to_t"},
    "leaked": {"type": "symmetric", "hours": 24},
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

EXCLUDE_COLS = {"stay_id", "hour", "hour_offset", "subject_id", "hadm_id", "intime", "charttime"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build note-centered structured windows.")
    p.add_argument(
        "--timeseries-csv",
        default=str(ROOT_DIR / "data" / "processed" / "timeseries_sorted_72h.csv"),
    )
    p.add_argument(
        "--note-metadata-csv",
        default=str(ROOT_DIR / "data" / "processed" / "text_embeddings" / "note_level_metadata_48h.csv"),
    )
    p.add_argument(
        "--cohort-csv",
        default=str(ROOT_DIR / "data" / "raw" / "cohort_with_conditions.csv"),
        help="Used for deterministic --max-stays selection.",
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "processed" / "note_centered"),
    )
    p.add_argument(
        "--summary-json",
        default=str(ROOT_DIR / "results" / "audit" / "phase2_note_centered_windows_summary.json"),
    )
    p.add_argument("--max-hour", type=int, default=71)
    p.add_argument("--chunk-stays", type=int, default=1000)
    p.add_argument("--flush-rows", type=int, default=20000)
    p.add_argument("--report-every", type=int, default=200000)
    p.add_argument("--max-stays", type=int, default=None, help="Optional debug cap.")
    p.add_argument(
        "--note-types",
        default="nursing,radiology,lab_comment",
        help="Comma-separated note types to include.",
    )
    return p.parse_args()


def _window_bounds(t_hour: int, window_id: str, max_hour: int) -> Tuple[int, int, bool]:
    cfg = WINDOWS[window_id]
    wtype = cfg["type"]
    if wtype == "lookback":
        w = int(cfg["hours"])
        raw_start = t_hour - w
        start = max(0, raw_start)
        end = t_hour
        is_truncated_left = raw_start < 0
    elif wtype == "calendar_day_up_to_t":
        start = (t_hour // 24) * 24
        end = t_hour
        is_truncated_left = False
    elif wtype == "symmetric":
        w = int(cfg["hours"])
        raw_start = t_hour - w
        start = max(0, raw_start)
        end = min(max_hour, t_hour + w)
        is_truncated_left = raw_start < 0
    else:
        raise ValueError(f"Unknown window type: {wtype}")

    start = max(0, min(start, max_hour))
    end = max(0, min(end, max_hour))
    if end < start:
        end = start
    return start, end, is_truncated_left


def _build_hour_grid(ts_stay: pd.DataFrame, feature_cols: List[str], max_hour: int) -> np.ndarray:
    grid = np.full((max_hour + 1, len(feature_cols)), np.nan, dtype=np.float32)
    if ts_stay.empty:
        return grid
    g = ts_stay.groupby("hour", sort=False)[feature_cols].mean(numeric_only=True).reset_index()
    hours = pd.to_numeric(g["hour"], errors="coerce").fillna(-1).astype(int).to_numpy()
    valid = (hours >= 0) & (hours <= max_hour)
    if valid.any():
        grid[hours[valid], :] = g.loc[valid, feature_cols].to_numpy(dtype=np.float32, copy=False)
    return grid


def _compute_interval_stats(
    grid: np.ndarray,
    start: int,
    end: int,
    feature_cols: List[str],
    out_cols: List[str],
) -> Dict[str, float]:
    seg = grid[start : end + 1, :]
    n_rows = int(seg.shape[0])

    out: Dict[str, float] = {}
    if n_rows == 0:
        for col in feature_cols:
            out[f"{col}_min"] = np.nan
            out[f"{col}_max"] = np.nan
            out[f"{col}_mean"] = np.nan
            out[f"{col}_last"] = np.nan
            out[f"{col}_first"] = np.nan
            out[f"{col}_std"] = np.nan
            out[f"{col}_missing_rate"] = 1.0
            out[f"{col}_n_measurements"] = 0.0
            out[f"{col}_delta_last_first"] = np.nan
            out[f"{col}_slope_per_hour"] = np.nan
        return out

    obs = ~np.isnan(seg)
    cnt = obs.sum(axis=0).astype(np.int32)
    mn = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    mx = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    mean = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    std = np.full((seg.shape[1],), np.nan, dtype=np.float32)
    valid_cols = np.where(cnt > 0)[0]
    if valid_cols.size > 0:
        seg_valid = seg[:, valid_cols]
        mn[valid_cols] = np.nanmin(seg_valid, axis=0)
        mx[valid_cols] = np.nanmax(seg_valid, axis=0)
        mean[valid_cols] = np.nanmean(seg_valid, axis=0)
        std[valid_cols] = np.nanstd(seg_valid, axis=0, ddof=0)
    miss_rate = 1.0 - (cnt / float(n_rows))

    for j, col in enumerate(feature_cols):
        c = int(cnt[j])
        if c == 0:
            first = np.nan
            last = np.nan
            delta = np.nan
            slope = np.nan
            mn_j = np.nan
            mx_j = np.nan
            mean_j = np.nan
            std_j = np.nan
        else:
            idx = np.flatnonzero(obs[:, j])
            fi = int(idx[0])
            li = int(idx[-1])
            first = float(seg[fi, j])
            last = float(seg[li, j])
            if c == 1:
                delta = 0.0
                slope = 0.0
            else:
                delta = last - first
                hour_delta = float((start + li) - (start + fi))
                slope = (delta / hour_delta) if hour_delta > 0 else 0.0
            mn_j = float(mn[j]) if np.isfinite(mn[j]) else np.nan
            mx_j = float(mx[j]) if np.isfinite(mx[j]) else np.nan
            mean_j = float(mean[j]) if np.isfinite(mean[j]) else np.nan
            std_j = float(std[j]) if np.isfinite(std[j]) else 0.0
            if c == 1:
                std_j = 0.0

        out[f"{col}_min"] = mn_j
        out[f"{col}_max"] = mx_j
        out[f"{col}_mean"] = mean_j
        out[f"{col}_last"] = last
        out[f"{col}_first"] = first
        out[f"{col}_std"] = std_j
        out[f"{col}_missing_rate"] = float(miss_rate[j])
        out[f"{col}_n_measurements"] = float(c)
        out[f"{col}_delta_last_first"] = delta
        out[f"{col}_slope_per_hour"] = slope

    # Ensure all expected keys exist.
    for c in out_cols:
        if c not in out:
            out[c] = np.nan
    return out


def _flush_rows(
    rows: List[Dict[str, object]],
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
    stat_cols: List[str],
) -> Optional[pq.ParquetWriter]:
    if not rows:
        return writer

    df = pd.DataFrame(rows)
    if not df.empty:
        int_cols = ["stay_id", "note_idx", "window_start", "window_end"]
        bool_cols = ["is_truncated_left", "contains_future_data"]
        float_meta = ["chart_hour", "window_hours_actual"]
        for c in int_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in bool_cols:
            if c in df.columns:
                df[c] = df[c].astype(bool)
        for c in float_meta:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(np.float32)
        for c in stat_cols:
            if c in df.columns:
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

    ts_path = Path(args.timeseries_csv)
    notes_path = Path(args.note_metadata_csv)
    cohort_path = Path(args.cohort_csv)
    out_dir = Path(args.output_dir)
    summary_path = Path(args.summary_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    allowed_note_types = {x.strip() for x in args.note_types.split(",") if x.strip()}

    print(f"[1/6] Loading note metadata: {notes_path}")
    notes = pd.read_csv(
        notes_path,
        usecols=["stay_id", "note_idx", "note_type", "chart_hour"],
        dtype={"stay_id": "int64", "note_idx": "int64", "note_type": "string", "chart_hour": "float32"},
    )
    notes = notes[notes["note_type"].isin(list(allowed_note_types))].copy()
    notes = notes[notes["stay_id"] >= 0].copy()
    notes = notes[notes["chart_hour"].notna()].copy()
    notes["chart_hour"] = pd.to_numeric(notes["chart_hour"], errors="coerce")
    notes["chart_hour_int"] = np.floor(notes["chart_hour"]).astype(np.int16).clip(0, args.max_hour)
    notes = notes.sort_values(["stay_id", "note_idx"], kind="mergesort")

    selected_stays_from_cohort: Optional[List[int]] = None
    if args.max_stays is not None and args.max_stays > 0:
        cohort = pd.read_csv(cohort_path, usecols=["stay_id"])
        selected_stays_from_cohort = (
            cohort["stay_id"].dropna().astype(np.int64).drop_duplicates().head(args.max_stays).tolist()
        )
        selected_set = set(int(s) for s in selected_stays_from_cohort)
        notes = notes[notes["stay_id"].isin(selected_set)].copy()

    note_stays = notes["stay_id"].drop_duplicates().tolist()
    if selected_stays_from_cohort is None:
        print(f"  notes={len(notes):,}, stays_with_notes={len(note_stays):,}")
    else:
        print(
            f"  notes={len(notes):,}, stays_with_notes={len(note_stays):,}, "
            f"selected_stays={len(selected_stays_from_cohort):,}"
        )

    print(f"[2/6] Loading timeseries: {ts_path}")
    if args.max_stays is not None and args.max_stays > 0:
        if selected_stays_from_cohort is not None:
            selected = set(int(s) for s in selected_stays_from_cohort)
        else:
            selected = set(int(s) for s in note_stays)
        chunks = []
        for chunk in pd.read_csv(ts_path, chunksize=300000):
            stay_col = pd.to_numeric(chunk["stay_id"], errors="coerce").fillna(-1).astype(np.int64)
            mask = stay_col.isin(selected)
            if mask.any():
                chunks.append(chunk.loc[mask].copy())
        ts = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=["stay_id", "hour"])
    else:
        ts = pd.read_csv(ts_path)

    if "hour" not in ts.columns and "hour_offset" in ts.columns:
        ts = ts.rename(columns={"hour_offset": "hour"})
    ts["stay_id"] = pd.to_numeric(ts["stay_id"], errors="coerce").fillna(-1).astype(np.int64)
    ts["hour"] = pd.to_numeric(ts["hour"], errors="coerce").fillna(-1).astype(np.int16)
    ts = ts[(ts["stay_id"] >= 0) & (ts["hour"] >= 0) & (ts["hour"] <= args.max_hour)].copy()
    ts = ts.sort_values(["stay_id", "hour"], kind="mergesort")

    feature_cols = [c for c in ts.columns if c not in EXCLUDE_COLS]
    if feature_cols:
        ts[feature_cols] = ts[feature_cols].apply(pd.to_numeric, errors="coerce").astype(np.float32)
    stat_cols = [f"{c}_{s}" for c in feature_cols for s in STATS]
    print(f"  rows={len(ts):,}, feature_cols={len(feature_cols)}")

    print("[3/6] Preparing writers and index maps...")
    ts_groups = ts.groupby("stay_id", sort=False).groups
    note_groups = notes.groupby("stay_id", sort=False).groups

    writers: Dict[str, Optional[pq.ParquetWriter]] = {}
    buffers: Dict[str, List[Dict[str, object]]] = {}
    row_counts: Dict[str, int] = {}
    out_paths: Dict[str, Path] = {}
    for win in WINDOWS.keys():
        p = out_dir / f"note_window_structured_{win}.parquet"
        if p.exists():
            p.unlink()
        writers[win] = None
        buffers[win] = []
        row_counts[win] = 0
        out_paths[win] = p

    print("[4/6] Building note-centered structured features...")
    processed = 0
    total_records = 0
    chunk_size = max(1, int(args.chunk_stays))
    stay_chunks = [note_stays[i : i + chunk_size] for i in range(0, len(note_stays), chunk_size)]

    for chunk_idx, stay_chunk in enumerate(stay_chunks, start=1):
        for sid in stay_chunk:
            sid = int(sid)
            note_idx = note_groups.get(sid)
            if note_idx is None:
                continue
            ndf = notes.loc[note_idx, ["stay_id", "note_idx", "note_type", "chart_hour", "chart_hour_int"]]
            ndf = ndf.sort_values(["note_idx"], kind="mergesort")

            ts_idx = ts_groups.get(sid)
            if ts_idx is None:
                ts_sid = ts.iloc[0:0][["hour"] + feature_cols]
            else:
                ts_sid = ts.loc[ts_idx, ["hour"] + feature_cols]
            grid = _build_hour_grid(ts_sid, feature_cols=feature_cols, max_hour=args.max_hour)

            interval_cache: Dict[Tuple[int, int], Dict[str, float]] = {}

            for row in ndf.itertuples(index=False):
                note_id = int(row.note_idx)
                chart_hour = float(row.chart_hour)
                t_hour = int(row.chart_hour_int)
                note_type = str(row.note_type)

                for win in WINDOWS.keys():
                    start, end, trunc_left = _window_bounds(t_hour=t_hour, window_id=win, max_hour=args.max_hour)
                    key = (start, end)
                    if key not in interval_cache:
                        interval_cache[key] = _compute_interval_stats(
                            grid=grid,
                            start=start,
                            end=end,
                            feature_cols=feature_cols,
                            out_cols=stat_cols,
                        )

                    rec: Dict[str, object] = {
                        "stay_id": sid,
                        "note_idx": note_id,
                        "note_type": note_type,
                        "chart_hour": chart_hour,
                        "window_id": win,
                        "window_start": start,
                        "window_end": end,
                        "window_hours_actual": float(end - start),
                        "is_truncated_left": bool(trunc_left),
                        "contains_future_data": bool(win == "leaked"),
                    }
                    rec.update(interval_cache[key])
                    buffers[win].append(rec)
                    row_counts[win] += 1
                    total_records += 1

                    if len(buffers[win]) >= args.flush_rows:
                        writers[win] = _flush_rows(
                            rows=buffers[win],
                            writer=writers[win],
                            out_path=out_paths[win],
                            stat_cols=stat_cols,
                        )

                    if total_records % args.report_every == 0:
                        elapsed = (time.time() - t0) / 60.0
                        print(
                            f"  progress records={total_records:,} "
                            f"stays={processed:,}/{len(note_stays):,} elapsed={elapsed:.1f}m"
                        )

            processed += 1

        if chunk_idx % 5 == 0:
            print(f"  processed stay chunks {chunk_idx}/{len(stay_chunks)}")

    print("[5/6] Flushing remaining buffers...")
    for win in WINDOWS.keys():
        writers[win] = _flush_rows(
            rows=buffers[win],
            writer=writers[win],
            out_path=out_paths[win],
            stat_cols=stat_cols,
        )
        if writers[win] is not None:
            writers[win].close()

    elapsed_sec = time.time() - t0
    summary = {
        "inputs": {
            "timeseries_csv": str(ts_path),
            "note_metadata_csv": str(notes_path),
            "cohort_csv": str(cohort_path),
            "max_hour": int(args.max_hour),
            "chunk_stays": int(args.chunk_stays),
            "flush_rows": int(args.flush_rows),
            "note_types": sorted(list(allowed_note_types)),
            "max_stays": args.max_stays,
        },
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "stats_per_feature": len(STATS),
        "output_files": {w: str(out_paths[w]) for w in WINDOWS.keys()},
        "rows_per_window": row_counts,
        "total_records": total_records,
        "elapsed_seconds": elapsed_sec,
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[6/6] Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
