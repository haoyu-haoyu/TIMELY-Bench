"""
Merge extracted BigQuery feature extensions into the canonical 0-24h hourly timeseries
and cohort files, producing "extended" versions that are preferred by config.py when present.

Inputs:
  - data/raw/timeseries_sorted.csv
  - data/processed/merge_output/cohort_final.csv
  - data/processed/bq_features/*.csv (from bq_extract_feature_extensions.py)

Outputs:
  - data/raw/timeseries_sorted_extended.csv
  - data/processed/merge_output/cohort_final_extended.csv
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import sys

import pandas as pd

# Allow importing `config` when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RAW_DATA_DIR, MERGE_OUTPUT_DIR, PROCESSED_DIR


@dataclass
class MergeMeta:
    generated_at: str
    timeseries_in: str
    timeseries_out: str
    cohort_in: str
    cohort_out: str
    n_timeseries_rows: int
    n_cohort_rows: int
    added_columns: list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bq-dir",
        type=Path,
        default=PROCESSED_DIR / "bq_features",
        help="Directory containing bq feature CSVs.",
    )
    parser.add_argument(
        "--timeseries-in",
        type=Path,
        default=RAW_DATA_DIR / "timeseries_sorted.csv",
    )
    parser.add_argument(
        "--timeseries-out",
        type=Path,
        default=RAW_DATA_DIR / "timeseries_sorted_extended.csv",
    )
    parser.add_argument(
        "--cohort-in",
        type=Path,
        default=MERGE_OUTPUT_DIR / "cohort_final.csv",
    )
    parser.add_argument(
        "--cohort-out",
        type=Path,
        default=MERGE_OUTPUT_DIR / "cohort_final_extended.csv",
    )
    args = parser.parse_args()

    if not args.timeseries_in.exists():
        raise FileNotFoundError(f"Missing timeseries: {args.timeseries_in}")
    if not args.cohort_in.exists():
        raise FileNotFoundError(f"Missing cohort: {args.cohort_in}")

    bili_path = args.bq_dir / "bilirubin_total_hourly.csv"
    vaso_path = args.bq_dir / "vasopressors_hourly.csv"
    rrt_path = args.bq_dir / "rrt_hourly.csv"
    ckd_path = args.bq_dir / "ckd_static.csv"

    for p in (bili_path, vaso_path, rrt_path, ckd_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing bq feature file: {p}")

    print(f"[merge] loading timeseries: {args.timeseries_in}")
    ts = pd.read_csv(args.timeseries_in)
    ts["stay_id"] = pd.to_numeric(ts["stay_id"], errors="coerce").fillna(-1).astype("int64")
    ts["hour"] = pd.to_numeric(ts["hour"], errors="coerce").fillna(-1).astype("int64")

    print(f"[merge] loading feature extensions: {args.bq_dir}")
    bili = pd.read_csv(bili_path)
    vaso = pd.read_csv(vaso_path)
    rrt = pd.read_csv(rrt_path)

    for df in (bili, vaso, rrt):
        if not df.empty:
            df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").fillna(-1).astype("int64")
            df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(-1).astype("int64")

    # Merge
    ts2 = ts.merge(bili, on=["stay_id", "hour"], how="left")
    ts2 = ts2.merge(vaso, on=["stay_id", "hour"], how="left")
    ts2 = ts2.merge(rrt, on=["stay_id", "hour"], how="left")

    # Binary derived features: absence means "off".
    if "vasopressors" in ts2.columns:
        ts2["vasopressors"] = pd.to_numeric(ts2["vasopressors"], errors="coerce").fillna(0).astype("int64")
    if "rrt" in ts2.columns:
        ts2["rrt"] = pd.to_numeric(ts2["rrt"], errors="coerce").fillna(0).astype("int64")

    # Ensure stable ordering for downstream consumers.
    ts2 = ts2.sort_values(["stay_id", "hour"], kind="mergesort")

    print(f"[merge] writing extended timeseries: {args.timeseries_out}")
    ts2.to_csv(args.timeseries_out, index=False)

    # Cohort extension: add CKD static flag.
    print(f"[merge] loading cohort: {args.cohort_in}")
    cohort = pd.read_csv(args.cohort_in)
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").fillna(-1).astype("int64")

    ckd = pd.read_csv(ckd_path)
    if not ckd.empty:
        ckd["stay_id"] = pd.to_numeric(ckd["stay_id"], errors="coerce").fillna(-1).astype("int64")
        ckd["ckd"] = pd.to_numeric(ckd["ckd"], errors="coerce").fillna(0).astype("int64")
    else:
        ckd = pd.DataFrame({"stay_id": cohort["stay_id"].astype("int64"), "ckd": 0})

    cohort2 = cohort.merge(ckd, on="stay_id", how="left")
    cohort2["ckd"] = pd.to_numeric(cohort2["ckd"], errors="coerce").fillna(0).astype("int64")

    print(f"[merge] writing extended cohort: {args.cohort_out}")
    cohort2.to_csv(args.cohort_out, index=False)

    meta = MergeMeta(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        timeseries_in=str(args.timeseries_in),
        timeseries_out=str(args.timeseries_out),
        cohort_in=str(args.cohort_in),
        cohort_out=str(args.cohort_out),
        n_timeseries_rows=int(len(ts2)),
        n_cohort_rows=int(len(cohort2)),
        added_columns=[c for c in ["bilirubin_total", "vasopressors", "rrt", "ckd"] if c in (list(ts2.columns) + list(cohort2.columns))],
    )
    (args.bq_dir / "bq_feature_merge_meta.json").write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print("[merge] done")


if __name__ == "__main__":
    main()
