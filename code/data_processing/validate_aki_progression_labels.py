#!/usr/bin/env python3
"""
Validate AKI progression labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate labels_aki_progression.csv")
    p.add_argument("--labels-csv", default="data/processed/labels_aki_progression.csv")
    p.add_argument("--min-positive-rate", type=float, default=0.05)
    p.add_argument("--max-positive-rate", type=float, default=0.45)
    p.add_argument("--min-stays", type=int, default=3000)
    p.add_argument("--lookahead-hours", type=int, default=24)
    p.add_argument("--report-json", default="results/audit/aki_progression_validation_report.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    labels = pd.read_csv(args.labels_csv)

    required = {
        "stay_id",
        "prediction_hour",
        "stage1_onset_hour",
        "stage2_onset_hour",
        "label",
    }
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    errors = []

    # Test 1: no prediction at/after stage2 onset.
    bad_stays_t1 = 0
    for sid, grp in labels.groupby("stay_id", sort=False):
        prog = grp["stage2_onset_hour"].iloc[0]
        if not np.isnan(prog):
            bad = int((grp["prediction_hour"] >= prog).sum())
            if bad > 0:
                bad_stays_t1 += 1
                if len(errors) < 10:
                    errors.append(f"Stay {sid}: {bad} rows at/after stage2 onset h{prog}")

    # Test 2: positive label window is correct.
    pos = labels[labels["label"] == 1].copy()
    bad_window = ~(
        (pos["stage2_onset_hour"] > pos["prediction_hour"])
        & (pos["stage2_onset_hour"] <= pos["prediction_hour"] + int(args.lookahead_hours))
    )
    bad_pos_rows = int(bad_window.sum())

    # Test 3/4 cohort stats.
    pos_rate = float(labels["label"].mean())
    n_stays = int(labels["stay_id"].nunique())

    pass_t1 = bad_stays_t1 == 0
    pass_t2 = bad_pos_rows == 0
    pass_t3 = float(args.min_positive_rate) <= pos_rate <= float(args.max_positive_rate)
    pass_t4 = n_stays >= int(args.min_stays)

    print(f"{'PASS' if pass_t1 else 'FAIL'} Test 1: no post-progression rows")
    print(f"{'PASS' if pass_t2 else 'FAIL'} Test 2: positive window correctness")
    print(
        f"{'PASS' if pass_t3 else 'FAIL'} Test 3: positive rate {pos_rate:.2%} "
        f"(expected {args.min_positive_rate:.0%}-{args.max_positive_rate:.0%})"
    )
    print(f"{'PASS' if pass_t4 else 'FAIL'} Test 4: stays={n_stays} (>= {args.min_stays})")
    if errors:
        print("Sample errors:")
        for e in errors:
            print(" ", e)

    report = {
        "tests": {
            "no_post_progression_rows": {"pass": pass_t1, "bad_stays": bad_stays_t1},
            "positive_window_correct": {"pass": pass_t2, "bad_rows": bad_pos_rows},
            "positive_rate_range": {
                "pass": pass_t3,
                "value": pos_rate,
                "min": float(args.min_positive_rate),
                "max": float(args.max_positive_rate),
            },
            "min_stays": {"pass": pass_t4, "value": n_stays, "min": int(args.min_stays)},
        },
        "n_rows": int(len(labels)),
        "n_stays": n_stays,
        "positive_rate": pos_rate,
    }
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    if not all(v["pass"] for v in report["tests"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

