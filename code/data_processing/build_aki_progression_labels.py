#!/usr/bin/env python3
"""
Build labels for Task A: AKI Stage 1 -> Stage 2+ progression.

Design:
- Cohort: stays with first Stage 1 onset within 48h of ICU admission.
- Prediction anchors: T = onset_stage1 + 4h, then every 4h.
- Positive label: first Stage2+ happens in (T, T+24].
- No predictions at/after Stage2 onset.

Outputs:
- data/processed/aki/kdigo_staged.parquet
- data/processed/labels_aki_progression.csv
- results/audit/aki_progression_label_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover - imported only in runtime env
    bigquery = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build AKI progression labels.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument(
        "--kdigo-table",
        default="physionet-data.mimiciv_3_1_derived.kdigo_stages",
        help="Fully-qualified BigQuery table: project.dataset.table",
    )
    p.add_argument(
        "--icustays-table",
        default="physionet-data.mimiciv_3_1_icu.icustays",
        help="Fully-qualified BigQuery table: project.dataset.table",
    )
    p.add_argument(
        "--cohort-csv",
        default="data/processed/merge_output/cohort_final.csv",
    )
    p.add_argument("--max-hour", type=int, default=72)
    p.add_argument("--stage1-onset-max-hour", type=int, default=48)
    p.add_argument("--prediction-interval", type=int, default=4)
    p.add_argument("--lookahead-hours", type=int, default=24)
    p.add_argument("--min-positive-rate", type=float, default=0.05)
    p.add_argument("--max-positive-rate", type=float, default=0.45)
    p.add_argument("--min-stays", type=int, default=3000)
    p.add_argument(
        "--kdigo-parquet-out",
        default="data/processed/aki/kdigo_staged.parquet",
    )
    p.add_argument(
        "--labels-out",
        default="data/processed/labels_aki_progression.csv",
    )
    p.add_argument(
        "--summary-json",
        default="results/audit/aki_progression_label_summary.json",
    )
    return p.parse_args()


def _run_bigquery_extract(args: argparse.Namespace) -> pd.DataFrame:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed in this environment.")

    client = bigquery.Client(project=args.billing_project)
    bt = "`"
    sql = (
        "SELECT "
        "k.stay_id, "
        "CAST(DATETIME_DIFF(DATETIME(k.charttime), DATETIME(i.intime), HOUR) AS INT64) AS hour, "
        "k.aki_stage "
        f"FROM {bt}{args.kdigo_table}{bt} k "
        f"JOIN {bt}{args.icustays_table}{bt} i "
        "ON k.stay_id = i.stay_id "
        "WHERE k.aki_stage IS NOT NULL "
        f"AND DATETIME_DIFF(DATETIME(k.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND {int(args.max_hour)} "
        "ORDER BY k.stay_id, hour"
    )
    return client.query(sql).result().to_dataframe()


def _build_labels(aki_cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    p_int = int(args.prediction_interval)
    lookahead = int(args.lookahead_hours)
    max_hour = int(args.max_hour)

    for _, row in aki_cohort.iterrows():
        sid = int(row["stay_id"])
        onset1 = int(row["stage1_onset_hour"])
        onset2 = row["stage2_onset_hour"]
        onset2 = float(onset2) if pd.notna(onset2) else np.inf

        upper_t = min(max_hour, int(onset2) - 1 if np.isfinite(onset2) else max_hour)
        t = onset1 + p_int
        while t <= upper_t:
            label = int(np.isfinite(onset2) and (onset2 > t) and (onset2 <= t + lookahead))
            rows.append(
                {
                    "stay_id": sid,
                    "prediction_hour": int(t),
                    "stage1_onset_hour": int(onset1),
                    "stage2_onset_hour": float(onset2) if np.isfinite(onset2) else np.nan,
                    "label": int(label),
                }
            )
            t += p_int

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    cohort_path = Path(args.cohort_csv)
    kdigo_out = Path(args.kdigo_parquet_out)
    labels_out = Path(args.labels_out)
    summary_out = Path(args.summary_json)
    kdigo_out.parent.mkdir(parents=True, exist_ok=True)
    labels_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)

    print("[1/5] Querying kdigo_stages from BigQuery ...")
    kdigo = _run_bigquery_extract(args)
    if kdigo.empty:
        raise RuntimeError("BigQuery extract returned 0 rows from kdigo_stages.")

    kdigo = kdigo.rename(columns={"aki_stage": "aki_stage"})
    kdigo["stay_id"] = pd.to_numeric(kdigo["stay_id"], errors="coerce").astype("Int64")
    kdigo["hour"] = pd.to_numeric(kdigo["hour"], errors="coerce")
    kdigo["aki_stage"] = pd.to_numeric(kdigo["aki_stage"], errors="coerce")
    kdigo = kdigo.dropna(subset=["stay_id", "hour", "aki_stage"]).copy()
    kdigo["stay_id"] = kdigo["stay_id"].astype(np.int64)
    kdigo["hour"] = kdigo["hour"].astype(np.int64)
    kdigo["aki_stage"] = kdigo["aki_stage"].astype(np.int16)
    kdigo = kdigo[(kdigo["hour"] >= 0) & (kdigo["hour"] <= int(args.max_hour))].copy()

    print("[2/5] Filtering to main cohort stays ...")
    cohort = pd.read_csv(cohort_path, usecols=["stay_id"])
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    kdigo = kdigo[kdigo["stay_id"].isin(set(cohort["stay_id"].tolist()))].copy()
    kdigo = kdigo.sort_values(["stay_id", "hour"], kind="mergesort")

    # Fix 3: ffill only on observed timepoints, no full hourly expansion.
    kdigo["aki_stage_ffill"] = (
        kdigo.groupby("stay_id", sort=False)["aki_stage"].transform("ffill").astype(np.int16)
    )
    kdigo.to_parquet(kdigo_out, index=False)

    print("[3/5] Building onset tables ...")
    first_stage1 = (
        kdigo[kdigo["aki_stage_ffill"] == 1]
        .groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "stage1_onset_hour"})
    )
    first_stage1 = first_stage1[first_stage1["stage1_onset_hour"] <= int(args.stage1_onset_max_hour)].copy()

    first_stage2 = (
        kdigo[kdigo["aki_stage_ffill"] >= 2]
        .groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "stage2_onset_hour"})
    )

    aki_cohort = first_stage1.merge(first_stage2, on="stay_id", how="left")

    print("[4/5] Building prediction labels ...")
    labels = _build_labels(aki_cohort, args)
    if labels.empty:
        raise RuntimeError("No label rows were generated for AKI progression.")
    labels.to_csv(labels_out, index=False)

    pos_rate = float(labels["label"].mean())
    n_stays = int(labels["stay_id"].nunique())
    stays_with_positive = int(labels.groupby("stay_id", sort=False)["label"].max().sum())

    print("[5/5] Final checks ...")
    if not (float(args.min_positive_rate) <= pos_rate <= float(args.max_positive_rate)):
        raise AssertionError(
            f"STOP: AKI positive rate {pos_rate:.2%} out of range "
            f"[{float(args.min_positive_rate):.0%}, {float(args.max_positive_rate):.0%}]"
        )
    if n_stays < int(args.min_stays):
        raise AssertionError(
            f"STOP: Only {n_stays} stays in AKI labels (< {int(args.min_stays)})."
        )

    summary: Dict[str, object] = {
        "kdigo_rows": int(len(kdigo)),
        "kdigo_stays": int(kdigo["stay_id"].nunique()),
        "aki_stage1_stays_onset_lte_48h": int(len(first_stage1)),
        "aki_stage2plus_stays_any": int(len(first_stage2)),
        "label_rows": int(len(labels)),
        "label_stays": n_stays,
        "positive_rate": pos_rate,
        "stays_with_any_positive": stays_with_positive,
        "outputs": {
            "kdigo_parquet": str(kdigo_out),
            "labels_csv": str(labels_out),
        },
        "settings": {
            "prediction_interval": int(args.prediction_interval),
            "lookahead_hours": int(args.lookahead_hours),
            "max_hour": int(args.max_hour),
            "stage1_onset_max_hour": int(args.stage1_onset_max_hour),
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nAKI progression labels complete.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Summary saved: {summary_out}")


if __name__ == "__main__":
    main()

