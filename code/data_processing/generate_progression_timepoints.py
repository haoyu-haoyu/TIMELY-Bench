#!/usr/bin/env python3
"""
Merge AKI and Sepsis-shock prediction anchors into one progression_timepoints file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate unified progression_timepoints.csv")
    p.add_argument("--aki-labels", default="data/processed/labels_aki_progression.csv")
    p.add_argument("--sepsis-labels", default="data/processed/labels_sepsis_shock.csv")
    p.add_argument("--out-csv", default="data/processed/progression_timepoints.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    aki = pd.read_csv(args.aki_labels, usecols=["stay_id", "prediction_hour"])
    sepsis = pd.read_csv(args.sepsis_labels, usecols=["stay_id", "prediction_hour"])

    all_tp = pd.concat([aki, sepsis], axis=0, ignore_index=True).drop_duplicates()
    all_tp["stay_id"] = pd.to_numeric(all_tp["stay_id"], errors="coerce").astype("Int64")
    all_tp["prediction_hour"] = pd.to_numeric(all_tp["prediction_hour"], errors="coerce")
    all_tp = all_tp.dropna(subset=["stay_id", "prediction_hour"]).copy()
    all_tp["stay_id"] = all_tp["stay_id"].astype("int64")
    all_tp["prediction_hour"] = all_tp["prediction_hour"].astype("int64")
    all_tp = all_tp.sort_values(["stay_id", "prediction_hour"], kind="mergesort")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_tp.to_csv(out, index=False)
    print(f"Total unique (stay, T) pairs: {len(all_tp)}")
    print(f"Unique stays: {all_tp['stay_id'].nunique()}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

