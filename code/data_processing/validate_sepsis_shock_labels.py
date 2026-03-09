#!/usr/bin/env python3
"""
Validate sepsis->shock labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate labels_sepsis_shock.csv")
    p.add_argument("--labels-csv", default="data/processed/labels_sepsis_shock.csv")
    p.add_argument("--min-positive-rate", type=float, default=0.05)
    p.add_argument("--max-positive-rate", type=float, default=0.50)
    p.add_argument("--lookahead-hours", type=int, default=12)
    p.add_argument("--report-json", default="results/audit/sepsis_shock_validation_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)

    req = {"stay_id", "prediction_hour", "sepsis_onset_hour", "shock_onset_hour", "label"}
    miss = req - set(labels.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    # Test 1: all timepoints after sepsis onset.
    bad_t1 = int((labels["prediction_hour"] <= labels["sepsis_onset_hour"]).sum())
    pass_t1 = bad_t1 == 0

    # Test 2: no timepoint at/after shock onset.
    bad_stays_t2 = 0
    for sid, grp in labels.groupby("stay_id", sort=False):
        shock_h = grp["shock_onset_hour"].iloc[0]
        if not np.isnan(shock_h):
            bad = int((grp["prediction_hour"] >= shock_h).sum())
            if bad > 0:
                bad_stays_t2 += 1
    pass_t2 = bad_stays_t2 == 0

    # Test 3: positive window correctness (12h by default).
    pos = labels[labels["label"] == 1].copy()
    bad_pos = int(
        (
            ~(
                (pos["shock_onset_hour"] > pos["prediction_hour"])
                & (pos["shock_onset_hour"] <= pos["prediction_hour"] + int(args.lookahead_hours))
            )
        ).sum()
    )
    pass_t3 = bad_pos == 0

    # Test 4: positive rate.
    pos_rate = float(labels["label"].mean())
    pass_t4 = float(args.min_positive_rate) <= pos_rate <= float(args.max_positive_rate)

    print(f"{'PASS' if pass_t1 else 'FAIL'} Test 1: all rows after sepsis onset")
    print(f"{'PASS' if pass_t2 else 'FAIL'} Test 2: no rows at/after shock onset")
    print(f"{'PASS' if pass_t3 else 'FAIL'} Test 3: positive window correctness")
    print(
        f"{'PASS' if pass_t4 else 'FAIL'} Test 4: positive rate {pos_rate:.2%} "
        f"(expected {args.min_positive_rate:.0%}-{args.max_positive_rate:.0%})"
    )

    report = {
        "tests": {
            "after_sepsis_onset": {"pass": pass_t1, "bad_rows": bad_t1},
            "no_post_shock_rows": {"pass": pass_t2, "bad_stays": bad_stays_t2},
            "positive_window_correct": {"pass": pass_t3, "bad_rows": bad_pos},
            "positive_rate_range": {
                "pass": pass_t4,
                "value": pos_rate,
                "min": float(args.min_positive_rate),
                "max": float(args.max_positive_rate),
            },
        },
        "n_rows": int(len(labels)),
        "n_stays": int(labels["stay_id"].nunique()),
        "positive_rate": pos_rate,
    }

    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if not all(v["pass"] for v in report["tests"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

