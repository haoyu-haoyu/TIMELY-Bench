#!/usr/bin/env python3
"""
Build stay-level text features for note-centered windows.

Produces one parquet per (window, text_method):
- data/processed/note_centered/stay_level/text_{window}_{method}.parquet

Methods:
- original, hard, weighted, weighted_no_after
- original_typed, hard_typed, weighted_typed, weighted_typed_no_after
"""

from __future__ import annotations

import argparse
import json
import math
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


WINDOWS = ["W6", "W12", "W24", "D0", "leaked"]
NOTE_TYPES = ["nursing", "radiology", "lab_comment"]
TYPE_TO_IDX = {k: i for i, k in enumerate(NOTE_TYPES)}
EMB_DIM = 768
TYPED_DIM = EMB_DIM * 3

METHODS = {
    "original": {"typed": False, "weight_col": "w_original"},
    "hard": {"typed": False, "weight_col": "w_hard"},
    "weighted": {"typed": False, "weight_col": "w_weighted"},
    "weighted_no_after": {"typed": False, "weight_col": "w_weighted_no_after"},
    "original_typed": {"typed": True, "weight_col": "w_original"},
    "hard_typed": {"typed": True, "weight_col": "w_hard"},
    "weighted_typed": {"typed": True, "weight_col": "w_weighted"},
    "weighted_typed_no_after": {"typed": True, "weight_col": "w_weighted_no_after"},
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stay-level text features for note-centered windows.")
    p.add_argument(
        "--metadata-csv",
        default=str(ROOT_DIR / "data" / "processed" / "text_embeddings" / "note_level_metadata_48h.csv"),
    )
    p.add_argument(
        "--embeddings-npy",
        default=str(ROOT_DIR / "data" / "processed" / "text_embeddings" / "note_level_embeddings_48h.npy"),
    )
    p.add_argument(
        "--doctime-parquet",
        default=str(ROOT_DIR / "results" / "doctime_rel" / "unified_doctime_classifications_48h.parquet"),
    )
    p.add_argument(
        "--cohort-csv",
        default=str(ROOT_DIR / "data" / "raw" / "cohort_with_conditions.csv"),
    )
    p.add_argument(
        "--output-dir",
        default=str(ROOT_DIR / "data" / "processed" / "note_centered" / "stay_level"),
    )
    p.add_argument(
        "--summary-json",
        default=str(ROOT_DIR / "results" / "audit" / "phase2_note_centered_text_summary.json"),
    )
    p.add_argument("--max-hour", type=int, default=71)
    p.add_argument("--flush-rows", type=int, default=128)
    p.add_argument("--report-every", type=int, default=1000)
    p.add_argument("--max-stays", type=int, default=None, help="Optional debug cap.")
    return p.parse_args()


def _window_mask(note_hours: np.ndarray, t_anchor: int, window: str, max_hour: int) -> np.ndarray:
    if window == "W6":
        start, end = max(0, t_anchor - 6), t_anchor
    elif window == "W12":
        start, end = max(0, t_anchor - 12), t_anchor
    elif window == "W24":
        start, end = max(0, t_anchor - 24), t_anchor
    elif window == "D0":
        start, end = (t_anchor // 24) * 24, t_anchor
    elif window == "leaked":
        start, end = max(0, t_anchor - 24), min(max_hour, t_anchor + 24)
    else:
        raise ValueError(f"Unknown window: {window}")
    return (note_hours >= start) & (note_hours <= end)


def _weighted_mean(x: np.ndarray, w: Optional[np.ndarray]) -> np.ndarray:
    if x.size == 0:
        return np.zeros((x.shape[1] if x.ndim == 2 else EMB_DIM,), dtype=np.float32)
    if w is None:
        return np.mean(x, axis=0).astype(np.float32)
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    if len(w) != x.shape[0]:
        raise ValueError("Weight length mismatch")
    s = float(np.sum(w))
    if s <= 0:
        return np.zeros((x.shape[1],), dtype=np.float32)
    return (x * w[:, None]).sum(axis=0).astype(np.float32) / s


def _flush_buffer(
    rows: List[Dict[str, object]],
    writer: Optional[pq.ParquetWriter],
    out_path: Path,
    dim: int,
) -> Optional[pq.ParquetWriter]:
    if not rows:
        return writer

    # Build base columns and embedding matrix separately, then concatenate once.
    # This avoids repeated column insertion and DataFrame fragmentation warnings.
    df = pd.DataFrame(rows)
    emb_mat = np.vstack(df.pop("embedding").to_list()).astype(np.float32, copy=False)
    emb_col_names = [f"emb_{i:04d}" for i in range(dim)]
    emb_df = pd.DataFrame(emb_mat, columns=emb_col_names)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1, copy=False)

    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    for c in ["text_has_notes", "text_has_nursing", "text_has_radiology", "text_has_lab"]:
        if c in df.columns:
            df[c] = df[c].astype(bool)

    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
    writer.write_table(table)
    rows.clear()
    return writer

def main() -> None:
    args = parse_args()
    t0 = time.time()

    metadata_csv = Path(args.metadata_csv)
    embeddings_npy = Path(args.embeddings_npy)
    doctime_parquet = Path(args.doctime_parquet)
    cohort_csv = Path(args.cohort_csv)
    out_dir = Path(args.output_dir)
    summary_json = Path(args.summary_json)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading cohort stays: {cohort_csv}")
    cohort = pd.read_csv(cohort_csv, usecols=["stay_id"])
    all_stays = cohort["stay_id"].dropna().astype(np.int64).drop_duplicates().tolist()
    if args.max_stays is not None and args.max_stays > 0:
        all_stays = all_stays[: args.max_stays]
    selected_stays = set(int(x) for x in all_stays)

    print(f"[2/6] Loading note metadata: {metadata_csv}")
    meta_kwargs = {
        "usecols": ["stay_id", "note_idx", "note_type", "chart_hour", "embedding_row_idx"],
        "dtype": {
            "stay_id": "int64",
            "note_idx": "int64",
            "note_type": "string",
            "chart_hour": "float32",
            "embedding_row_idx": "int64",
        },
    }
    if args.max_stays is not None and args.max_stays > 0:
        chunks = []
        for chunk in pd.read_csv(metadata_csv, chunksize=300000, **meta_kwargs):
            part = chunk[chunk["stay_id"].isin(selected_stays)]
            if not part.empty:
                chunks.append(part)
        meta = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=meta_kwargs["usecols"])
    else:
        meta = pd.read_csv(metadata_csv, **meta_kwargs)
    meta = meta[meta["chart_hour"].notna()].copy()
    meta = meta[meta["stay_id"] >= 0].copy()
    meta["note_type"] = meta["note_type"].astype(str)
    meta["note_type"] = np.where(meta["note_type"].isin(NOTE_TYPES), meta["note_type"], "other")
    meta["chart_hour_int"] = np.floor(meta["chart_hour"]).astype(np.int16).clip(0, args.max_hour)
    meta = meta.sort_values(["stay_id", "note_idx"], kind="mergesort")

    print(f"[3/6] Loading DocTime weights: {doctime_parquet}")
    dt_cols = [
        "stay_id",
        "note_idx",
        "doctime_before_overlap_ratio",
        "doctime_after_ratio",
    ]
    if args.max_stays is not None and args.max_stays > 0:
        try:
            dt = pd.read_parquet(
                doctime_parquet,
                columns=dt_cols,
                filters=[("stay_id", "in", [int(x) for x in all_stays])],
            )
        except Exception:
            dt = pd.read_parquet(doctime_parquet, columns=dt_cols)
            dt = dt[dt["stay_id"].isin(selected_stays)].copy()
    else:
        dt = pd.read_parquet(doctime_parquet, columns=dt_cols)
    dt["stay_id"] = pd.to_numeric(dt["stay_id"], errors="coerce").fillna(-1).astype(np.int64)
    dt["note_idx"] = pd.to_numeric(dt["note_idx"], errors="coerce").fillna(-1).astype(np.int64)

    meta = meta.merge(dt, on=["stay_id", "note_idx"], how="left")
    # Note-level DocTimeRel weighting uses non-AFTER share.
    # For nursing notes, after_ratio=0.0 so weight remains 1.0.
    after_ratio = pd.to_numeric(meta["doctime_after_ratio"], errors="coerce")
    w_non_after = (1.0 - after_ratio)
    w_non_after = w_non_after.where(w_non_after.notna(), 1.0).clip(lower=0.0, upper=1.0)

    meta["w_original"] = 1.0
    meta["w_weighted"] = w_non_after.astype(np.float32)
    # In note-level pooling (without sentence embeddings), these are identical by design.
    meta["w_weighted_no_after"] = meta["w_weighted"]
    meta["w_hard"] = np.where(after_ratio.fillna(0.0) < 0.5, 1.0, 0.0).astype(np.float32)

    print(f"[4/6] Loading embeddings mmap: {embeddings_npy}")
    note_emb = np.load(embeddings_npy, mmap_mode="r")
    print(f"  embeddings shape={note_emb.shape}")

    meta = meta[meta["stay_id"].isin(all_stays)].copy()
    note_groups = meta.groupby("stay_id", sort=False).groups

    out_paths: Dict[Tuple[str, str], Path] = {}
    buffers: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    writers: Dict[Tuple[str, str], Optional[pq.ParquetWriter]] = {}
    rows_written: Dict[Tuple[str, str], int] = {}
    dim_map: Dict[Tuple[str, str], int] = {}
    for w in WINDOWS:
        for m, cfg in METHODS.items():
            path = out_dir / f"text_{w}_{m}.parquet"
            if path.exists():
                path.unlink()
            key = (w, m)
            out_paths[key] = path
            buffers[key] = []
            writers[key] = None
            rows_written[key] = 0
            dim_map[key] = TYPED_DIM if bool(cfg["typed"]) else EMB_DIM

    print("[5/6] Building stay-level text features...")
    processed = 0
    for sid in all_stays:
        sid = int(sid)
        idx = note_groups.get(sid)
        if idx is None:
            # No notes: emit zero vectors and False flags for all outputs.
            for w in WINDOWS:
                for m, cfg in METHODS.items():
                    key = (w, m)
                    dim = TYPED_DIM if bool(cfg["typed"]) else EMB_DIM
                    buffers[key].append(
                        {
                            "stay_id": sid,
                            "embedding": np.zeros((dim,), dtype=np.float32),
                            "text_has_notes": False,
                            "text_has_nursing": False,
                            "text_has_radiology": False,
                            "text_has_lab": False,
                        }
                    )
                    rows_written[key] += 1
                    if len(buffers[key]) >= args.flush_rows:
                        writers[key] = _flush_buffer(
                            rows=buffers[key],
                            writer=writers[key],
                            out_path=out_paths[key],
                            dim=dim_map[key],
                        )
            processed += 1
            continue

        sdf = meta.loc[idx].sort_values(["chart_hour", "note_idx"], kind="mergesort")
        note_hours = sdf["chart_hour_int"].to_numpy(dtype=np.int16, copy=False)
        t_anchor = int(note_hours.max())
        note_idx_arr = sdf["embedding_row_idx"].to_numpy(dtype=np.int64, copy=False)
        note_type_arr = sdf["note_type"].astype(str).to_numpy(copy=False)
        emb = np.asarray(note_emb[note_idx_arr], dtype=np.float32)
        w_original = sdf["w_original"].to_numpy(dtype=np.float32, copy=False)
        w_hard = sdf["w_hard"].to_numpy(dtype=np.float32, copy=False)
        w_weighted = sdf["w_weighted"].to_numpy(dtype=np.float32, copy=False)
        w_weighted_no_after = sdf["w_weighted_no_after"].to_numpy(dtype=np.float32, copy=False)

        weight_map = {
            "w_original": w_original,
            "w_hard": w_hard,
            "w_weighted": w_weighted,
            "w_weighted_no_after": w_weighted_no_after,
        }

        for w in WINDOWS:
            wmask = _window_mask(note_hours=note_hours, t_anchor=t_anchor, window=w, max_hour=args.max_hour)
            has_notes = bool(wmask.any())
            has_nursing = bool((wmask & (note_type_arr == "nursing")).any())
            has_radiology = bool((wmask & (note_type_arr == "radiology")).any())
            has_lab = bool((wmask & (note_type_arr == "lab_comment")).any())

            emb_w = emb[wmask] if has_notes else np.zeros((0, EMB_DIM), dtype=np.float32)
            type_w = note_type_arr[wmask] if has_notes else np.array([], dtype=object)

            for m, cfg in METHODS.items():
                key = (w, m)
                wcol = str(cfg["weight_col"])
                weight_arr = weight_map[wcol][wmask] if has_notes else np.zeros((0,), dtype=np.float32)
                typed = bool(cfg["typed"])

                if not typed:
                    vec = _weighted_mean(emb_w, weight_arr if has_notes else None)
                else:
                    chunks: List[np.ndarray] = []
                    for nt in NOTE_TYPES:
                        tmask = type_w == nt
                        if tmask.any():
                            sub_vec = _weighted_mean(emb_w[tmask], weight_arr[tmask])
                        else:
                            sub_vec = np.zeros((EMB_DIM,), dtype=np.float32)
                        chunks.append(sub_vec)
                    vec = np.concatenate(chunks, axis=0).astype(np.float32)

                buffers[key].append(
                    {
                        "stay_id": sid,
                        "embedding": vec,
                        "text_has_notes": has_notes,
                        "text_has_nursing": has_nursing,
                        "text_has_radiology": has_radiology,
                        "text_has_lab": has_lab,
                    }
                )
                rows_written[key] += 1
                if len(buffers[key]) >= args.flush_rows:
                    writers[key] = _flush_buffer(
                        rows=buffers[key],
                        writer=writers[key],
                        out_path=out_paths[key],
                        dim=dim_map[key],
                    )

        processed += 1
        if processed % args.report_every == 0:
            elapsed = (time.time() - t0) / 60.0
            print(f"  processed stays={processed:,}/{len(all_stays):,} elapsed={elapsed:.1f}m")

    print("[5.5/6] Flushing remaining buffers...")
    for key in out_paths.keys():
        writers[key] = _flush_buffer(
            rows=buffers[key],
            writer=writers[key],
            out_path=out_paths[key],
            dim=dim_map[key],
        )
        if writers[key] is not None:
            writers[key].close()

    elapsed = time.time() - t0
    summary = {
        "inputs": {
            "metadata_csv": str(metadata_csv),
            "embeddings_npy": str(embeddings_npy),
            "doctime_parquet": str(doctime_parquet),
            "cohort_csv": str(cohort_csv),
            "max_stays": args.max_stays,
            "max_hour": args.max_hour,
        },
        "stays_total": len(all_stays),
        "rows_written": {f"{w}|{m}": int(rows_written[(w, m)]) for w in WINDOWS for m in METHODS.keys()},
        "output_files": {f"{w}|{m}": str(out_paths[(w, m)]) for w in WINDOWS for m in METHODS.keys()},
        "elapsed_seconds": elapsed,
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[6/6] Done.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {summary_json}")


if __name__ == "__main__":
    main()
