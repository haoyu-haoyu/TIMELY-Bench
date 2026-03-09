#!/usr/bin/env python3
"""
Load Sepsis-3 onset times and project to cohort.

Outputs:
- data/processed/sepsis/sepsis3_onset.csv
- data/processed/sepsis/sepsis3_cohort.csv
- results/audit/sepsis3_onset_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import pandas as pd

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract Sepsis-3 onset hour from BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument(
        "--sepsis3-table",
        default="physionet-data.mimiciv_3_1_derived.sepsis3",
    )
    p.add_argument(
        "--icustays-table",
        default="physionet-data.mimiciv_3_1_icu.icustays",
    )
    p.add_argument(
        "--cohort-csv",
        default="data/processed/merge_output/cohort_final.csv",
    )
    p.add_argument("--max-onset-hour", type=int, default=48)
    p.add_argument("--min-stays", type=int, default=5000)
    p.add_argument("--onset-column", default=None, help="Override onset timestamp column.")
    p.add_argument("--onset-out", default="data/processed/sepsis/sepsis3_onset.csv")
    p.add_argument("--cohort-out", default="data/processed/sepsis/sepsis3_cohort.csv")
    p.add_argument("--summary-json", default="results/audit/sepsis3_onset_summary.json")
    return p.parse_args()


def _split_table_ref(table_ref: str) -> Tuple[str, str, str]:
    parts = table_ref.split(".")
    if len(parts) != 3:
        raise ValueError(f"Table must be project.dataset.table, got: {table_ref}")
    return parts[0], parts[1], parts[2]


def _detect_onset_col(client: "bigquery.Client", table_ref: str, override: str | None) -> str:
    if override:
        return override

    p, d, t = _split_table_ref(table_ref)
    bt = "`"
    sql = (
        f"SELECT column_name FROM {bt}{p}.{d}.INFORMATION_SCHEMA.COLUMNS{bt} "
        f"WHERE table_name='{t}'"
    )
    cols = set(client.query(sql).result().to_dataframe()["column_name"].tolist())
    if "sofa_time" in cols:
        return "sofa_time"
    if "suspected_infection_time" in cols:
        return "suspected_infection_time"
    raise RuntimeError(
        f"Neither sofa_time nor suspected_infection_time found in {table_ref}. "
        f"Columns: {sorted(cols)}"
    )


def main() -> None:
    args = parse_args()
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed.")

    out_onset = Path(args.onset_out)
    out_cohort = Path(args.cohort_out)
    out_summary = Path(args.summary_json)
    out_onset.parent.mkdir(parents=True, exist_ok=True)
    out_cohort.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    client = bigquery.Client(project=args.billing_project)
    onset_col = _detect_onset_col(client, args.sepsis3_table, args.onset_column)
    bt = "`"
    sql = (
        "SELECT "
        "s.stay_id, "
        "s.subject_id, "
        f"CAST(DATETIME_DIFF(DATETIME(s.{onset_col}), DATETIME(i.intime), HOUR) AS INT64) AS sepsis_onset_hour, "
        "s.sofa_score "
        f"FROM {bt}{args.sepsis3_table}{bt} s "
        f"JOIN {bt}{args.icustays_table}{bt} i "
        "ON s.stay_id = i.stay_id "
        f"WHERE DATETIME_DIFF(DATETIME(s.{onset_col}), DATETIME(i.intime), HOUR) BETWEEN 0 AND {int(args.max_onset_hour)}"
    )
    sepsis = client.query(sql).result().to_dataframe()
    if sepsis.empty:
        raise RuntimeError("Sepsis query returned 0 rows.")

    sepsis["stay_id"] = pd.to_numeric(sepsis["stay_id"], errors="coerce").astype("Int64")
    sepsis["subject_id"] = pd.to_numeric(sepsis["subject_id"], errors="coerce").astype("Int64")
    sepsis["sepsis_onset_hour"] = pd.to_numeric(sepsis["sepsis_onset_hour"], errors="coerce")
    sepsis = sepsis.dropna(subset=["stay_id", "sepsis_onset_hour"]).copy()
    sepsis["stay_id"] = sepsis["stay_id"].astype("int64")
    sepsis["subject_id"] = sepsis["subject_id"].astype("int64")
    sepsis["sepsis_onset_hour"] = sepsis["sepsis_onset_hour"].astype("int64")
    sepsis = sepsis.sort_values(["stay_id", "sepsis_onset_hour"], kind="mergesort")
    sepsis = sepsis.groupby("stay_id", as_index=False).first()

    cohort = pd.read_csv(args.cohort_csv, usecols=["stay_id"])
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")

    sepsis_cohort = sepsis[sepsis["stay_id"].isin(set(cohort["stay_id"].tolist()))].copy()
    sepsis.to_csv(out_onset, index=False)
    sepsis_cohort.to_csv(out_cohort, index=False)

    n = int(len(sepsis_cohort))
    prevalence = float(n / max(int(len(cohort)), 1))
    print(f"Sepsis stays in main cohort: {n}")
    print(f"Sepsis prevalence: {prevalence:.1%}")
    print("Onset hour stats:")
    print(sepsis_cohort["sepsis_onset_hour"].describe())

    if n < int(args.min_stays):
        raise AssertionError(f"STOP: Only {n} sepsis stays (< {args.min_stays}).")

    summary = {
        "sepsis_table": args.sepsis3_table,
        "icustays_table": args.icustays_table,
        "onset_column": onset_col,
        "n_rows_onset_raw": int(len(sepsis)),
        "n_rows_in_cohort": n,
        "cohort_size": int(len(cohort)),
        "prevalence": prevalence,
        "onset_describe": sepsis_cohort["sepsis_onset_hour"].describe().to_dict(),
        "outputs": {"onset_csv": str(out_onset), "cohort_csv": str(out_cohort)},
    }
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved: {out_summary}")


if __name__ == "__main__":
    main()

