#!/usr/bin/env python3
"""
Build text features for progression tasks at (stay_id, prediction_hour).

Outputs:
- data/processed/progression_features/text_W24_original.parquet
- data/processed/progression_features/text_W24_weighted_no_after.parquet
- data/processed/progression_features/text_W24_leaked.parquet

Critical validation:
- leaked text must differ from original text in >= 10% rows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


EMB_DIM = 768
METHODS = ["original", "weighted_no_after", "leaked"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build progression text features.")
    p.add_argument("--timepoints-csv", default="data/processed/progression_timepoints.csv")
    p.add_argument(
        "--metadata-csv",
        default="data/processed/text_embeddings/note_level_metadata_48h.csv",
    )
    p.add_argument(
        "--embeddings-npy",
        default="data/processed/text_embeddings/note_level_embeddings_48h.npy",
    )
    p.add_argument(
        "--doctime-parquet",
        default="results/doctime_rel/unified_doctime_classifications_48h.parquet",
    )
    p.add_argument("--output-dir", default="data/processed/progression_features")
    p.add_argument(
        "--summary-json",
        default="results/audit/progression_text_feature_summary.json",
    )
    p.add_argument("--max-note-hour", type=int, default=71)
    p.add_argument("--flush-rows", type=int, default=1000)
    p.add_argument("--report-every-stays", type=int, default=500)
    p.add_argument("--min-diff-pct", type=float, default=0.10)
    p.add_argument("--max-stays", type=int, default=None)
    p.add_argument("--metadata-chunksize", type=int, default=1_000_000)
    return p.parse_args()


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if x.size == 0:
        return np.zeros((EMB_DIM,), dtype=np.float32)
    if w is None:
        return x.mean(axis=0).astype(np.float32)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if len(w) != x.shape[0]:
        raise ValueError("Weight length does not match embedding rows.")
    s = float(np.sum(w))
    if s <= 0:
        return np.zeros((EMB_DIM,), dtype=np.float32)
    return ((x * w[:, None]).sum(axis=0) / s).astype(np.float32)


def _flush_rows(
    rows: List[Dict[str, object]],
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
) -> Optional[pq.ParquetWriter]:
    if not rows:
        return writer

    df = pd.DataFrame(rows)
    emb = np.vstack(df.pop("embedding").tolist()).astype(np.float32)
    emb_cols = [f"emb_{i:04d}" for i in range(EMB_DIM)]
    emb_df = pd.DataFrame(emb, columns=emb_cols)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df["prediction_hour"] = pd.to_numeric(df["prediction_hour"], errors="coerce").astype("Int64")
    df["text_has_notes"] = df["text_has_notes"].astype(bool)
    table = pa.Table.from_pandas(df, preserve_index=False)

    if writer is None:
        writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
    writer.write_table(table)
    rows.clear()
    return writer


def _load_filtered_metadata(
    metadata_csv: Path,
    keep_stays: set[int],
    chunksize: int,
) -> pd.DataFrame:
    usecols = ["stay_id", "note_idx", "note_type", "chart_hour", "embedding_row_idx"]
    chunks = []
    for chunk in pd.read_csv(
        metadata_csv,
        usecols=usecols,
        chunksize=max(10_000, int(chunksize)),
        dtype={
            "stay_id": "Int64",
            "note_idx": "Int64",
            "note_type": "string",
            "chart_hour": "float32",
            "embedding_row_idx": "Int64",
        },
    ):
        part = chunk[chunk["stay_id"].isin(keep_stays)]
        if not part.empty:
            chunks.append(part)

    if not chunks:
        return pd.DataFrame(columns=usecols + ["chart_hour_int"])

    meta = pd.concat(chunks, axis=0, ignore_index=True)
    meta = meta.dropna(subset=["stay_id", "note_idx", "chart_hour", "embedding_row_idx"]).copy()
    meta["stay_id"] = meta["stay_id"].astype(np.int64)
    meta["note_idx"] = meta["note_idx"].astype(np.int64)
    meta["embedding_row_idx"] = meta["embedding_row_idx"].astype(np.int64)
    meta["chart_hour"] = pd.to_numeric(meta["chart_hour"], errors="coerce")
    meta = meta.dropna(subset=["chart_hour"]).copy()
    return meta


def main() -> None:
    args = parse_args()
    t0 = time.time()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_out = Path(args.summary_json)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading timepoints: {args.timepoints_csv}")
    tp = pd.read_csv(args.timepoints_csv, usecols=["stay_id", "prediction_hour"])
    tp["stay_id"] = pd.to_numeric(tp["stay_id"], errors="coerce").astype("Int64")
    tp["prediction_hour"] = pd.to_numeric(tp["prediction_hour"], errors="coerce")
    tp = tp.dropna(subset=["stay_id", "prediction_hour"]).copy()
    tp["stay_id"] = tp["stay_id"].astype(np.int64)
    tp["prediction_hour"] = tp["prediction_hour"].astype(np.int16)
    tp = tp.sort_values(["stay_id", "prediction_hour"], kind="mergesort").drop_duplicates()

    stays = tp["stay_id"].drop_duplicates().tolist()
    if args.max_stays is not None and args.max_stays > 0:
        stays = stays[: int(args.max_stays)]
    keep_stays = set(int(x) for x in stays)
    tp = tp[tp["stay_id"].isin(keep_stays)].copy()
    tp_group = tp.groupby("stay_id", sort=False)["prediction_hour"].apply(list).to_dict()
    print(f"  stays={len(stays):,}  timepoints={len(tp):,}")

    print(f"[2/6] Loading filtered metadata: {args.metadata_csv}")
    meta = _load_filtered_metadata(
        metadata_csv=Path(args.metadata_csv),
        keep_stays=keep_stays,
        chunksize=int(args.metadata_chunksize),
    )
    meta["chart_hour_int"] = (
        np.floor(pd.to_numeric(meta["chart_hour"], errors="coerce").fillna(-1))
        .astype(np.int16)
        .clip(0, int(args.max_note_hour))
    )
    meta = meta.sort_values(["stay_id", "note_idx"], kind="mergesort")
    print(f"  filtered notes={len(meta):,}")

    print(f"[3/6] Loading DocTime ratios: {args.doctime_parquet}")
    dt_cols = ["stay_id", "note_idx", "doctime_after_ratio"]
    try:
        dt = pd.read_parquet(
            args.doctime_parquet,
            columns=dt_cols,
            filters=[("stay_id", "in", [int(x) for x in stays])],
        )
    except Exception:
        dt = pd.read_parquet(args.doctime_parquet, columns=dt_cols)
        dt = dt[dt["stay_id"].isin(keep_stays)].copy()

    dt["stay_id"] = pd.to_numeric(dt["stay_id"], errors="coerce").astype("Int64")
    dt["note_idx"] = pd.to_numeric(dt["note_idx"], errors="coerce").astype("Int64")
    dt["doctime_after_ratio"] = pd.to_numeric(dt["doctime_after_ratio"], errors="coerce")
    dt = dt.dropna(subset=["stay_id", "note_idx"]).copy()
    dt["stay_id"] = dt["stay_id"].astype(np.int64)
    dt["note_idx"] = dt["note_idx"].astype(np.int64)

    meta = meta.merge(dt, on=["stay_id", "note_idx"], how="left")
    meta["doctime_after_ratio"] = pd.to_numeric(
        meta["doctime_after_ratio"], errors="coerce"
    ).fillna(0.0)
    meta["w_weighted_no_after"] = (1.0 - meta["doctime_after_ratio"]).clip(lower=0.0, upper=1.0).astype(np.float32)

    print(f"[4/6] Loading embedding mmap: {args.embeddings_npy}")
    emb = np.load(args.embeddings_npy, mmap_mode="r")
    print(f"  embeddings shape={emb.shape}")

    note_groups = meta.groupby("stay_id", sort=False).groups

    out_paths: Dict[str, Path] = {}
    writers: Dict[str, Optional[pq.ParquetWriter]] = {}
    buffers: Dict[str, List[Dict[str, object]]] = {}
    rows_written = {m: 0 for m in METHODS}
    for m in METHODS:
        out = out_dir / f"text_W24_{m}.parquet"
        if out.exists():
            out.unlink()
        out_paths[m] = out
        writers[m] = None
        buffers[m] = []

    print("[5/6] Building progression text features ...")
    diff_rows = 0
    total_rows = 0
    rows_with_notes_lookback = 0
    rows_with_notes_leaked = 0

    for i, sid in enumerate(stays, start=1):
        idx = note_groups.get(int(sid))
        sdf = meta.loc[idx].copy() if idx is not None else meta.iloc[:0].copy()
        if not sdf.empty:
            sdf = sdf.sort_values(["chart_hour_int", "note_idx"], kind="mergesort")
            note_hours = sdf["chart_hour_int"].to_numpy(dtype=np.int16, copy=False)
            note_emb_idx = sdf["embedding_row_idx"].to_numpy(dtype=np.int64, copy=False)
            note_weights = sdf["w_weighted_no_after"].to_numpy(dtype=np.float32, copy=False)
            note_emb = np.asarray(emb[note_emb_idx], dtype=np.float32)
        else:
            note_hours = np.zeros((0,), dtype=np.int16)
            note_weights = np.zeros((0,), dtype=np.float32)
            note_emb = np.zeros((0, EMB_DIM), dtype=np.float32)

        for t in sorted(set(int(x) for x in tp_group.get(int(sid), []))):
            start = max(0, t - 24)
            end = t
            end_leaked = min(int(args.max_note_hour), t + 24)

            m_lookback = (note_hours >= start) & (note_hours <= end)
            m_leaked = (note_hours >= start) & (note_hours <= end_leaked)

            emb_original = _weighted_mean(note_emb[m_lookback], None)
            emb_weighted = _weighted_mean(note_emb[m_lookback], note_weights[m_lookback])
            emb_leaked = _weighted_mean(note_emb[m_leaked], None)

            has_lookback = bool(m_lookback.any())
            has_leaked = bool(m_leaked.any())
            rows_with_notes_lookback += int(has_lookback)
            rows_with_notes_leaked += int(has_leaked)
            total_rows += 1
            if np.any(np.abs(emb_original - emb_leaked) > 1e-8):
                diff_rows += 1

            base = {
                "stay_id": int(sid),
                "prediction_hour": int(t),
            }

            buffers["original"].append({**base, "text_has_notes": has_lookback, "embedding": emb_original})
            buffers["weighted_no_after"].append(
                {**base, "text_has_notes": has_lookback, "embedding": emb_weighted}
            )
            buffers["leaked"].append({**base, "text_has_notes": has_leaked, "embedding": emb_leaked})

            for m in METHODS:
                rows_written[m] += 1
                if len(buffers[m]) >= int(args.flush_rows):
                    writers[m] = _flush_rows(buffers[m], writers[m], out_paths[m])

        if i % int(args.report_every_stays) == 0:
            elapsed = (time.time() - t0) / 60.0
            print(f"  processed stays={i:,}/{len(stays):,} elapsed={elapsed:.1f}m")

    for m in METHODS:
        writers[m] = _flush_rows(buffers[m], writers[m], out_paths[m])
        if writers[m] is not None:
            writers[m].close()

    diff_pct = float(diff_rows / total_rows) if total_rows > 0 else 0.0
    print(f"\nCRITICAL VERIFICATION: leaked != original rows = {diff_pct:.2%}")
    print(f"Expected >= {float(args.min_diff_pct):.0%}")
    if diff_pct < float(args.min_diff_pct):
        raise AssertionError(
            f"Text leakage not detectable: diff_pct={diff_pct:.2%} < {float(args.min_diff_pct):.0%}"
        )

    summary = {
        "n_stays": int(len(stays)),
        "n_timepoints": int(total_rows),
        "rows_written": rows_written,
        "rows_with_notes_lookback": int(rows_with_notes_lookback),
        "rows_with_notes_leaked": int(rows_with_notes_leaked),
        "leaked_vs_original_diff_rows": int(diff_rows),
        "leaked_vs_original_diff_pct": diff_pct,
        "outputs": {m: str(out_paths[m]) for m in METHODS},
        "elapsed_seconds": float(time.time() - t0),
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved: {summary_out}")


if __name__ == "__main__":
    main()
