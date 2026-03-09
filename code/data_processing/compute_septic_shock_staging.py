#!/usr/bin/env python3
"""
Compute septic shock hourly flags for sepsis cohort.

Core rule:
- Shock when MBP < 65 and vasopressor_active is True.
- If lactate exists at that hour (after short ffill), apply stricter lactate > 2 criterion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute septic shock hourly staging.")
    p.add_argument(
        "--timeseries-csv",
        default="data/processed/timeseries_sorted_72h.csv",
    )
    p.add_argument(
        "--sepsis-cohort-csv",
        default="data/processed/sepsis/sepsis3_cohort.csv",
    )
    p.add_argument(
        "--out-parquet",
        default="data/processed/sepsis/septic_shock_hourly.parquet",
    )
    p.add_argument(
        "--summary-json",
        default="results/audit/septic_shock_staging_summary.json",
    )
    p.add_argument("--mbp-threshold", type=float, default=65.0)
    p.add_argument("--lactate-threshold", type=float, default=2.0)
    p.add_argument("--vaso-ffill-limit-hours", type=int, default=4)
    p.add_argument("--lactate-ffill-limit-hours", type=int, default=12)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    base_required = ["stay_id", "hour", "mbp", "vasopressors"]
    optional = ["vasopressor_dose_norepi_equiv", "lactate"]

    # Read header first to avoid hard failure on optional columns.
    cols = pd.read_csv(args.timeseries_csv, nrows=1).columns.tolist()
    missing_required = [c for c in base_required if c not in cols]
    if missing_required:
        raise ValueError(f"Missing required columns in timeseries: {missing_required}")
    usecols = base_required + [c for c in optional if c in cols]

    ts = pd.read_csv(args.timeseries_csv, usecols=usecols)
    sepsis_cohort = pd.read_csv(args.sepsis_cohort_csv, usecols=["stay_id"])
    sepsis_ids = set(pd.to_numeric(sepsis_cohort["stay_id"], errors="coerce").dropna().astype("int64").tolist())

    ts["stay_id"] = pd.to_numeric(ts["stay_id"], errors="coerce").astype("Int64")
    ts["hour"] = pd.to_numeric(ts["hour"], errors="coerce")
    ts = ts.dropna(subset=["stay_id", "hour"]).copy()
    ts["stay_id"] = ts["stay_id"].astype("int64")
    ts["hour"] = ts["hour"].astype("int64")
    ts = ts[ts["stay_id"].isin(sepsis_ids)].copy()
    ts = ts.sort_values(["stay_id", "hour"], kind="mergesort")

    ts["vasopressors"] = pd.to_numeric(ts["vasopressors"], errors="coerce").fillna(0).astype("int8")
    ts["mbp"] = pd.to_numeric(ts["mbp"], errors="coerce")

    # Keep vasopressor status active over short gaps.
    ts["vasopressor_active"] = (
        ts.groupby("stay_id", sort=False)["vasopressors"]
        .transform(lambda x: x.replace(0, np.nan).ffill(limit=int(args.vaso_ffill_limit_hours)).fillna(0))
        .astype(bool)
    )

    ts["is_septic_shock"] = (ts["mbp"] < float(args.mbp_threshold)) & ts["vasopressor_active"]

    lactate_used = False
    if "lactate" in ts.columns:
        lactate_used = True
        ts["lactate"] = pd.to_numeric(ts["lactate"], errors="coerce")
        ts["lactate_ffill"] = (
            ts.groupby("stay_id", sort=False)["lactate"]
            .transform(lambda x: x.ffill(limit=int(args.lactate_ffill_limit_hours)))
        )
        has_lactate = ts["lactate_ffill"].notna()
        ts.loc[has_lactate, "is_septic_shock"] = (
            ts.loc[has_lactate, "is_septic_shock"]
            & (ts.loc[has_lactate, "lactate_ffill"] > float(args.lactate_threshold))
        )

    keep_cols = ["stay_id", "hour", "is_septic_shock", "mbp", "vasopressor_active"]
    if "vasopressor_dose_norepi_equiv" in ts.columns:
        ts["vasopressor_dose_norepi_equiv"] = pd.to_numeric(
            ts["vasopressor_dose_norepi_equiv"], errors="coerce"
        )
        keep_cols.append("vasopressor_dose_norepi_equiv")
    if lactate_used:
        keep_cols.extend(["lactate", "lactate_ffill"])

    out = ts[keep_cols].copy()
    out.to_parquet(out_path, index=False)

    hours_shock = int(out["is_septic_shock"].sum())
    pct_hours = float(out["is_septic_shock"].mean()) if len(out) else 0.0
    stays_with_shock = int(out.loc[out["is_septic_shock"], "stay_id"].nunique())

    print(f"Hours with septic shock: {hours_shock} ({pct_hours:.2%} of sepsis cohort hours)")
    print(f"Stays with any septic shock: {stays_with_shock}")

    summary = {
        "n_rows": int(len(out)),
        "n_stays": int(out["stay_id"].nunique()),
        "hours_with_septic_shock": hours_shock,
        "pct_hours_with_septic_shock": pct_hours,
        "stays_with_any_septic_shock": stays_with_shock,
        "lactate_rule_used": lactate_used,
        "input_timeseries": args.timeseries_csv,
        "input_sepsis_cohort": args.sepsis_cohort_csv,
        "output_parquet": str(out_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

