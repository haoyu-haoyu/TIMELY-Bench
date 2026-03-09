#!/usr/bin/env python3
"""
Build labels for Task B: Sepsis -> Septic Shock progression.

Design:
- Cohort: sepsis onset within first 48h.
- Prediction anchors: T = sepsis_onset + 4h, then every 4h.
- Positive label: first shock onset in (T, T+12].
- No predictions at/after shock onset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build sepsis->shock labels.")
    p.add_argument("--sepsis-cohort-csv", default="data/processed/sepsis/sepsis3_cohort.csv")
    p.add_argument("--shock-hourly-parquet", default="data/processed/sepsis/septic_shock_hourly.parquet")
    p.add_argument("--labels-out", default="data/processed/labels_sepsis_shock.csv")
    p.add_argument("--summary-json", default="results/audit/sepsis_shock_label_summary.json")
    p.add_argument("--prediction-interval", type=int, default=4)
    p.add_argument("--lookahead-hours", type=int, default=12)
    p.add_argument("--max-hour", type=int, default=72)
    p.add_argument("--min-positive-rate", type=float, default=0.05)
    p.add_argument("--max-positive-rate", type=float, default=0.50)
    p.add_argument("--min-stays", type=int, default=2000)
    return p.parse_args()


def _build_labels(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    p_int = int(args.prediction_interval)
    lookahead = int(args.lookahead_hours)
    max_hour = int(args.max_hour)

    for _, row in df.iterrows():
        sid = int(row["stay_id"])
        onset = int(row["sepsis_onset_hour"])
        shock_h = row["shock_onset_hour"]
        shock_h = float(shock_h) if pd.notna(shock_h) else np.inf

        upper_t = min(max_hour, int(shock_h) - 1 if np.isfinite(shock_h) else max_hour)
        t = onset + p_int
        while t <= upper_t:
            label = int(np.isfinite(shock_h) and (shock_h > t) and (shock_h <= t + lookahead))
            rows.append(
                {
                    "stay_id": sid,
                    "prediction_hour": int(t),
                    "sepsis_onset_hour": int(onset),
                    "shock_onset_hour": float(shock_h) if np.isfinite(shock_h) else np.nan,
                    "label": int(label),
                }
            )
            t += p_int

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    out_csv = Path(args.labels_out)
    out_json = Path(args.summary_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    sepsis = pd.read_csv(args.sepsis_cohort_csv)
    shock = pd.read_parquet(args.shock_hourly_parquet)

    sepsis["stay_id"] = pd.to_numeric(sepsis["stay_id"], errors="coerce").astype("Int64")
    sepsis["sepsis_onset_hour"] = pd.to_numeric(sepsis["sepsis_onset_hour"], errors="coerce")
    sepsis = sepsis.dropna(subset=["stay_id", "sepsis_onset_hour"]).copy()
    sepsis["stay_id"] = sepsis["stay_id"].astype("int64")
    sepsis["sepsis_onset_hour"] = sepsis["sepsis_onset_hour"].astype("int64")

    shock = shock[["stay_id", "hour", "is_septic_shock"]].copy()
    shock["stay_id"] = pd.to_numeric(shock["stay_id"], errors="coerce").astype("Int64")
    shock["hour"] = pd.to_numeric(shock["hour"], errors="coerce")
    shock = shock.dropna(subset=["stay_id", "hour"]).copy()
    shock["stay_id"] = shock["stay_id"].astype("int64")
    shock["hour"] = shock["hour"].astype("int64")
    shock["is_septic_shock"] = shock["is_septic_shock"].astype(bool)

    shock_pos = shock[shock["is_septic_shock"]].copy()
    # Keep only shock events after sepsis onset, then pick first.
    merged = shock_pos.merge(sepsis[["stay_id", "sepsis_onset_hour"]], on="stay_id", how="inner")
    merged = merged[merged["hour"] > merged["sepsis_onset_hour"]].copy()
    first_shock = (
        merged.groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "shock_onset_hour"})
    )

    cohort = sepsis.merge(first_shock, on="stay_id", how="left")
    labels = _build_labels(cohort, args)
    if labels.empty:
        raise RuntimeError("No label rows generated for sepsis_shock task.")

    labels.to_csv(out_csv, index=False)
    pos_rate = float(labels["label"].mean())
    n_stays = int(labels["stay_id"].nunique())
    stays_with_positive = int(labels.groupby("stay_id", sort=False)["label"].max().sum())

    print(f"Total timepoints: {len(labels)}")
    print(f"Positive rate: {pos_rate:.2%}")
    print(f"Stays with any positive: {stays_with_positive}")

    if not (float(args.min_positive_rate) <= pos_rate <= float(args.max_positive_rate)):
        raise AssertionError(
            f"STOP: positive rate {pos_rate:.1%} out of range "
            f"[{float(args.min_positive_rate):.0%}, {float(args.max_positive_rate):.0%}]"
        )
    if n_stays < int(args.min_stays):
        raise AssertionError(f"STOP: only {n_stays} stays in labels (< {args.min_stays}).")

    summary: Dict[str, object] = {
        "sepsis_stays": int(len(sepsis)),
        "stays_with_any_shock_after_onset": int(len(first_shock)),
        "label_rows": int(len(labels)),
        "label_stays": n_stays,
        "positive_rate": pos_rate,
        "stays_with_any_positive": stays_with_positive,
        "settings": {
            "prediction_interval": int(args.prediction_interval),
            "lookahead_hours": int(args.lookahead_hours),
            "max_hour": int(args.max_hour),
        },
        "outputs": {"labels_csv": str(out_csv)},
    }
    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved: {out_json}")


if __name__ == "__main__":
    main()

