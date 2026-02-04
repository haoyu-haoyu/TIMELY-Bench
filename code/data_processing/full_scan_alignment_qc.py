"""
Full-scan QC for alignment constraints on very large CSV files.

This script performs a streaming scan over the alignment CSV to produce
hard evidence counts for:
1) discharge note rows
2) note_hour outside 0<=hour<24
3) duplicates on key: stay_id + pattern_hour + pattern_name + note_id

Duplicate detection uses a hash-partitioned, disk-backed two-pass method
so it is rigorous without relying on sampling.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR


RESULTS_QC_DIR = ROOT_DIR / "results" / "qc"
FINAL_QC_DIR = ROOT_DIR / "final_release" / "qc"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def key_string(row: Dict[str, str]) -> str:
    return "\t".join(
        [
            str(row.get("stay_id", "")),
            str(row.get("pattern_hour", "")),
            str(row.get("pattern_name", "")),
            str(row.get("note_id", "")),
        ]
    )


def bucket_id(key: str, n_buckets: int) -> int:
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


class BucketWriter:
    """LRU-bounded bucket file writer to avoid open-file limits."""

    def __init__(self, bucket_dir: Path, n_buckets: int, max_open_files: int = 64):
        self.bucket_dir = bucket_dir
        self.n_buckets = n_buckets
        self.max_open_files = max_open_files
        self._handles: "OrderedDict[int, any]" = OrderedDict()
        self.bucket_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, bid: int) -> Path:
        return self.bucket_dir / f"bucket_{bid:05d}.txt"

    def _open_handle(self, bid: int):
        handle = self._handles.pop(bid, None)
        if handle is not None:
            # refresh LRU order
            self._handles[bid] = handle
            return handle

        if len(self._handles) >= self.max_open_files:
            old_bid, old_handle = self._handles.popitem(last=False)
            old_handle.close()

        path = self._path(bid)
        handle = path.open("a", buffering=1024 * 1024)
        self._handles[bid] = handle
        return handle

    def write(self, bid: int, line: str) -> None:
        handle = self._open_handle(bid)
        handle.write(line)
        handle.write("\n")

    def close_all(self) -> None:
        for _, handle in self._handles.items():
            handle.close()
        self._handles.clear()


@dataclass
class ScanCounts:
    total_rows: int = 0
    discharge_rows: int = 0
    note_hour_out_of_range_rows: int = 0
    parse_error_rows: int = 0


def scan_alignment_stream(
    alignment_path: Path,
    bucket_dir: Path,
    n_buckets: int,
    max_open_files: int,
    max_rows: int,
) -> Tuple[ScanCounts, Path]:
    counts = ScanCounts()
    key_buckets_dir = bucket_dir / "alignment_keys"
    if key_buckets_dir.exists():
        shutil.rmtree(key_buckets_dir)
    writer = BucketWriter(key_buckets_dir, n_buckets=n_buckets, max_open_files=max_open_files)

    start = time.time()
    with alignment_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if max_rows and i > max_rows:
                break

            counts.total_rows += 1

            note_type = str(row.get("note_type", "")).strip().lower()
            if note_type == "discharge":
                counts.discharge_rows += 1

            note_hour_raw = row.get("note_hour", "")
            try:
                note_hour = float(note_hour_raw)
                if note_hour < 0 or note_hour >= 24:
                    counts.note_hour_out_of_range_rows += 1
            except Exception:
                counts.note_hour_out_of_range_rows += 1
                counts.parse_error_rows += 1

            key = key_string(row)
            bid = bucket_id(key, n_buckets=n_buckets)
            writer.write(bid, key)

            # progress print every 5M rows
            if counts.total_rows % 5_000_000 == 0:
                elapsed = time.time() - start
                rate = counts.total_rows / max(elapsed, 1e-6)
                print(
                    f"Scanned {counts.total_rows:,} rows "
                    f"({rate:,.0f} rows/s); discharge={counts.discharge_rows:,}; "
                    f"out_of_range={counts.note_hour_out_of_range_rows:,}"
                )

    writer.close_all()
    elapsed = time.time() - start
    print(f"Finished streaming scan: {counts.total_rows:,} rows in {elapsed/60:.1f} min")
    return counts, key_buckets_dir


def count_duplicates_in_buckets(bucket_dir: Path) -> Dict[str, object]:
    dup_rows = 0
    dup_keys = 0
    bucket_summaries: List[Dict[str, object]] = []
    sample_dups: List[Tuple[str, int]] = []

    bucket_files = sorted(bucket_dir.glob("bucket_*.txt"))
    if not bucket_files:
        return {
            "duplicate_rows": 0,
            "duplicate_keys": 0,
            "bucket_summaries": [],
            "sample_duplicates": [],
        }

    for bf in bucket_files:
        counts: Dict[str, int] = {}
        with bf.open() as f:
            for line in f:
                key = line.rstrip("\n")
                counts[key] = counts.get(key, 0) + 1

        bucket_dup_keys = 0
        bucket_dup_rows = 0
        for k, c in counts.items():
            if c > 1:
                bucket_dup_keys += 1
                bucket_dup_rows += c - 1
                if len(sample_dups) < 20:
                    sample_dups.append((k, c))

        dup_keys += bucket_dup_keys
        dup_rows += bucket_dup_rows
        bucket_summaries.append(
            {
                "bucket": bf.name,
                "n_keys": len(counts),
                "duplicate_keys": bucket_dup_keys,
                "duplicate_rows": bucket_dup_rows,
            }
        )
        print(
            f"Bucket {bf.name}: keys={len(counts):,}, dup_keys={bucket_dup_keys:,}, dup_rows={bucket_dup_rows:,}"
        )

    return {
        "duplicate_rows": dup_rows,
        "duplicate_keys": dup_keys,
        "bucket_summaries": bucket_summaries,
        "sample_duplicates": [
            {"key": k, "count": c} for k, c in sample_dups
        ],
    }


def scan_annotation_set(annotation_set_path: Path) -> Dict[str, object]:
    if not annotation_set_path.exists():
        return {
            "path": str(annotation_set_path),
            "exists": False,
        }

    df_rows = 0
    discharge_rows = 0
    out_of_range_rows = 0
    dup_rows = 0
    seen = set()

    with annotation_set_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            df_rows += 1
            note_type = str(row.get("note_type", "")).strip().lower()
            if note_type == "discharge":
                discharge_rows += 1
            try:
                note_hour = float(row.get("note_hour", ""))
                if note_hour < 0 or note_hour >= 24:
                    out_of_range_rows += 1
            except Exception:
                out_of_range_rows += 1

            key = key_string(row)
            if key in seen:
                dup_rows += 1
            else:
                seen.add(key)

    return {
        "path": str(annotation_set_path),
        "exists": True,
        "sha256": sha256_file(annotation_set_path),
        "total_rows": df_rows,
        "discharge_rows": discharge_rows,
        "note_hour_out_of_range_rows": out_of_range_rows,
        "duplicate_rows": dup_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alignment-path",
        type=str,
        default=str(TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"),
    )
    parser.add_argument(
        "--annotation-set-path",
        type=str,
        default=str(ROOT_DIR / "results" / "llm_annotations" / "llm_annotation_set.csv"),
    )
    parser.add_argument("--n-buckets", type=int, default=2048)
    parser.add_argument("--max-open-files", type=int, default=64)
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(RESULTS_QC_DIR / "_dup_buckets"),
        help="Directory for temporary bucket files.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional local testing limit; 0 means full scan.",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove temporary bucket files after counting duplicates.",
    )
    args = parser.parse_args()

    alignment_path = Path(args.alignment_path)
    if not alignment_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {alignment_path}")

    RESULTS_QC_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_QC_DIR.mkdir(parents=True, exist_ok=True)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    max_rows = int(args.max_rows) if args.max_rows else 0

    print("Starting full-scan alignment QC")
    print(f"alignment_path={alignment_path}")
    print(f"n_buckets={args.n_buckets}, max_open_files={args.max_open_files}, max_rows={max_rows or 'FULL'}")

    counts, key_bucket_dir = scan_alignment_stream(
        alignment_path=alignment_path,
        bucket_dir=work_dir,
        n_buckets=args.n_buckets,
        max_open_files=args.max_open_files,
        max_rows=max_rows,
    )

    dup_info = count_duplicates_in_buckets(key_bucket_dir)

    annotation_info = scan_annotation_set(Path(args.annotation_set_path))

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "alignment": {
            "path": str(alignment_path),
            "total_rows": counts.total_rows,
            "discharge_rows": counts.discharge_rows,
            "note_hour_out_of_range_rows": counts.note_hour_out_of_range_rows,
            "parse_error_rows": counts.parse_error_rows,
            "duplicate_rows": dup_info["duplicate_rows"],
            "duplicate_keys": dup_info["duplicate_keys"],
            "duplicate_key": "stay_id+pattern_hour+pattern_name+note_id",
        },
        "annotation_set": annotation_info,
        "duplicate_scan": {
            "n_buckets": args.n_buckets,
            "bucket_dir": str(key_bucket_dir),
            "bucket_summaries": dup_info["bucket_summaries"],
            "sample_duplicates": dup_info["sample_duplicates"],
        },
        "parameters": {
            "n_buckets": args.n_buckets,
            "max_open_files": args.max_open_files,
            "max_rows": max_rows,
            "work_dir": str(work_dir),
        },
    }

    out_path = RESULTS_QC_DIR / "full_alignment_qc.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"Wrote {out_path}")

    final_out = FINAL_QC_DIR / out_path.name
    final_out.write_bytes(out_path.read_bytes())
    print(f"Copied to {final_out}")

    if args.cleanup:
        shutil.rmtree(key_bucket_dir, ignore_errors=True)
        print(f"Cleaned up {key_bucket_dir}")

    # Strict success condition for full scan
    if max_rows == 0:
        bad = []
        if result["alignment"]["discharge_rows"] != 0:
            bad.append("discharge_rows")
        if result["alignment"]["note_hour_out_of_range_rows"] != 0:
            bad.append("note_hour_out_of_range_rows")
        if result["alignment"]["duplicate_rows"] != 0:
            bad.append("duplicate_rows")
        if bad:
            raise SystemExit(f"FAIL: non-zero QC counts: {bad}")
        print("PASS: full alignment QC counts all zero")


if __name__ == "__main__":
    main()
