"""
Extract additional structured features from MIMIC-IV BigQuery to "land" a few
Condition-Graph fields that are present in the clinical schema but missing from
the released hourly timeseries table.

Outputs (CSV):
  - data/processed/bq_features/bilirubin_total_hourly.csv   (stay_id, hour, bilirubin_total)
  - data/processed/bq_features/vasopressors_hourly.csv      (stay_id, hour, vasopressors)
  - data/processed/bq_features/rrt_hourly.csv               (stay_id, hour, rrt)
  - data/processed/bq_features/ckd_static.csv               (stay_id, ckd)
  - data/processed/bq_features/bq_feature_extract_meta.json

Notes:
  - This script is intended to run on CREATE (or any environment with BigQuery
    network access + credentials). It does not create BigQuery tables.
  - Billing/execution project defaults to "timely-bench-mimic".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

# Allow importing `config` when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import MERGE_OUTPUT_DIR, PROCESSED_DIR


@dataclass
class ExtractMeta:
    generated_at: str
    billing_project: str
    dataset_hosp: str
    dataset_icu: str
    n_stays: int
    hours: int
    bilirubin_itemids: List[int]
    vasopressor_itemids: List[int]
    rrt_itemids: List[int]
    icd_ckd_rules: dict


DEFAULT_BILLING_PROJECT = "timely-bench-mimic"
DEFAULT_DATASET_HOSP = "physionet-data.mimiciv_3_1_hosp"
DEFAULT_DATASET_ICU = "physionet-data.mimiciv_3_1_icu"


def _chunks(seq: Sequence[int], chunk_size: int) -> Iterable[List[int]]:
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i : i + chunk_size])


def _load_stay_ids(cohort_csv: Path) -> List[int]:
    df = pd.read_csv(cohort_csv, usecols=["stay_id"])
    stay_ids = (
        pd.to_numeric(df["stay_id"], errors="coerce")
        .dropna()
        .astype("int64")
        .drop_duplicates()
        .tolist()
    )
    return stay_ids


def _bq_client(billing_project: str):
    from google.cloud import bigquery

    return bigquery.Client(project=billing_project)


def _query_itemids_bilirubin_total(client, dataset_hosp: str) -> List[int]:
    """
    Find lab itemids for total bilirubin.

    We prefer labels containing both 'bilirubin' and 'total'.
    """
    q = f"""
    SELECT itemid, label
    FROM `{dataset_hosp}.d_labitems`
    WHERE LOWER(label) LIKE '%bilirubin%'
    """
    rows = list(client.query(q).result())
    cand = []
    for r in rows:
        label = str(getattr(r, "label", "")).lower()
        if "total" in label:
            cand.append(int(r.itemid))
    # Conservative fallback: MIMIC-III/IV commonly uses 50885 for total bilirubin.
    if not cand:
        cand = [50885]
    return sorted(set(cand))


def _query_itemids_vasopressors(client, dataset_icu: str) -> List[int]:
    """
    Identify vasopressor/inotrope itemids from ICU d_items linked to inputevents.
    """
    names = [
        "norepinephrine",
        "epinephrine",
        "vasopressin",
        "phenylephrine",
        "dopamine",
        "dobutamine",
    ]
    cond = " OR ".join([f"LOWER(label) LIKE '%{n}%'" for n in names])
    q = f"""
    SELECT itemid, label
    FROM `{dataset_icu}.d_items`
    WHERE linksto = 'inputevents' AND ({cond})
    """
    rows = list(client.query(q).result())
    itemids = [int(r.itemid) for r in rows]
    return sorted(set(itemids))


def _query_itemids_rrt(client, dataset_icu: str) -> List[int]:
    """
    Identify renal replacement therapy related itemids from ICU d_items linked to procedureevents.

    This is deliberately broad; downstream we only create a binary indicator.
    """
    keywords = [
        "dialysis",
        "rrt",
        "crrt",
        "cvvh",
        "hemofiltration",
        "hemodialysis",
        "renal replacement",
    ]
    cond = " OR ".join([f"LOWER(label) LIKE '%{k}%'" for k in keywords])
    q = f"""
    SELECT itemid, label
    FROM `{dataset_icu}.d_items`
    WHERE linksto = 'procedureevents' AND ({cond})
    """
    rows = list(client.query(q).result())
    itemids = [int(r.itemid) for r in rows]
    return sorted(set(itemids))


def _extract_bilirubin_hourly(
    client,
    dataset_hosp: str,
    dataset_icu: str,
    stay_ids: Sequence[int],
    bilirubin_itemids: Sequence[int],
    hours: int,
):
    from google.cloud import bigquery

    # Use per-hour "last value" to preserve extremes at the window aggregation stage.
    q = f"""
    WITH stays AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      s.stay_id,
      CAST(FLOOR(DATETIME_DIFF(l.charttime, s.intime, MINUTE) / 60.0) AS INT64) AS hour,
      ARRAY_AGG(l.valuenum ORDER BY l.charttime DESC LIMIT 1)[OFFSET(0)] AS bilirubin_total
    FROM stays s
    JOIN `{dataset_hosp}.labevents` l
      ON l.subject_id = s.subject_id AND l.hadm_id = s.hadm_id
    WHERE l.itemid IN UNNEST(@itemids)
      AND l.valuenum IS NOT NULL
      AND l.charttime >= s.intime
      AND l.charttime < DATETIME_ADD(s.intime, INTERVAL @hours HOUR)
    GROUP BY s.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("stay_ids", "INT64", list(stay_ids)),
            bigquery.ArrayQueryParameter("itemids", "INT64", list(bilirubin_itemids)),
            bigquery.ScalarQueryParameter("hours", "INT64", int(hours)),
        ]
    )
    df = client.query(q, job_config=job_config).to_dataframe()
    if not df.empty:
        df["stay_id"] = df["stay_id"].astype("int64")
        df["hour"] = df["hour"].astype("int64")
    return df


def _extract_binary_hourly_from_events(
    *,
    client,
    table_fq: str,
    dataset_icu: str,
    stay_ids: Sequence[int],
    itemids: Sequence[int],
    hours: int,
    out_col: str,
):
    """
    Extract a per-hour binary indicator from an event table with (stay_id, starttime, endtime, itemid).
    """
    from google.cloud import bigquery

    if not itemids:
        return pd.DataFrame(columns=["stay_id", "hour", out_col])

    q = f"""
    WITH stays AS (
      SELECT stay_id, intime
      FROM `{dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    ),
    events AS (
      SELECT
        e.stay_id,
        CAST(FLOOR(DATETIME_DIFF(e.starttime, s.intime, MINUTE) / 60.0) AS INT64) AS start_hour,
        CAST(FLOOR(DATETIME_DIFF(COALESCE(e.endtime, e.starttime), s.intime, MINUTE) / 60.0) AS INT64) AS end_hour
      FROM `{table_fq}` e
      JOIN stays s ON s.stay_id = e.stay_id
      WHERE e.itemid IN UNNEST(@itemids)
    ),
    expanded AS (
      SELECT
        stay_id,
        hour
      FROM (
        SELECT
          stay_id,
          GREATEST(start_hour, 0) AS sh,
          LEAST(GREATEST(end_hour, start_hour), @hours - 1) AS eh
        FROM events
        WHERE end_hour >= 0 AND start_hour <= (@hours - 1)
      ),
      UNNEST(GENERATE_ARRAY(sh, eh)) AS hour
    )
    SELECT stay_id, hour, 1 AS {out_col}
    FROM expanded
    GROUP BY stay_id, hour
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("stay_ids", "INT64", list(stay_ids)),
            bigquery.ArrayQueryParameter("itemids", "INT64", list(itemids)),
            bigquery.ScalarQueryParameter("hours", "INT64", int(hours)),
        ]
    )
    df = client.query(q, job_config=job_config).to_dataframe()
    if not df.empty:
        df["stay_id"] = df["stay_id"].astype("int64")
        df["hour"] = df["hour"].astype("int64")
        df[out_col] = df[out_col].astype("int64")
    return df


def _extract_ckd_static(client, dataset_hosp: str, dataset_icu: str, stay_ids: Sequence[int]) -> pd.DataFrame:
    """
    Build a static CKD indicator from diagnosis codes linked to the ICU stay's hadm_id.

    This is an "external_static" comorbidity signal (not a time-series variable).
    """
    from google.cloud import bigquery

    rules = {
        "icd10_prefix": ["N18", "N19", "Z49"],
        "icd10_exact": ["Z992"],
        "icd9_prefix": ["585", "V451", "V56"],
        "icd9_exact": ["586"],
    }
    q = f"""
    WITH stays AS (
      SELECT stay_id, subject_id, hadm_id
      FROM `{dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      s.stay_id,
      MAX(
        IF(
          (d.icd_version = 10 AND (
             STARTS_WITH(d.icd_code, 'N18') OR STARTS_WITH(d.icd_code, 'N19') OR STARTS_WITH(d.icd_code, 'Z49') OR d.icd_code = 'Z992'
          ))
          OR
          (d.icd_version = 9 AND (
             STARTS_WITH(d.icd_code, '585') OR d.icd_code = '586' OR STARTS_WITH(d.icd_code, 'V451') OR STARTS_WITH(d.icd_code, 'V56')
          )),
          1,
          0
        )
      ) AS ckd
    FROM stays s
    JOIN `{dataset_hosp}.diagnoses_icd` d
      ON d.subject_id = s.subject_id AND d.hadm_id = s.hadm_id
    GROUP BY s.stay_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", list(stay_ids))]
    )
    df = client.query(q, job_config=job_config).to_dataframe()
    if df.empty:
        df = pd.DataFrame({"stay_id": list(stay_ids), "ckd": 0})
    else:
        df["stay_id"] = df["stay_id"].astype("int64")
        df["ckd"] = pd.to_numeric(df["ckd"], errors="coerce").fillna(0).astype("int64")
        # Ensure full coverage
        base = pd.DataFrame({"stay_id": list(stay_ids)})
        df = base.merge(df, on="stay_id", how="left")
        df["ckd"] = df["ckd"].fillna(0).astype("int64")
    df = df.sort_values("stay_id")
    df.attrs["icd_rules"] = rules
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--billing-project",
        default=DEFAULT_BILLING_PROJECT,
        help="BigQuery billing/execution project (must have jobs.create).",
    )
    parser.add_argument("--dataset-hosp", default=DEFAULT_DATASET_HOSP)
    parser.add_argument("--dataset-icu", default=DEFAULT_DATASET_ICU)
    parser.add_argument(
        "--cohort-csv",
        type=Path,
        default=MERGE_OUTPUT_DIR / "cohort_final.csv",
        help="Local cohort CSV used to enumerate stay_id.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DIR / "bq_features",
        help="Output directory under data/processed/.",
    )
    parser.add_argument("--hours", type=int, default=24, help="Observation horizon in hours (default: 24).")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    stay_ids = _load_stay_ids(args.cohort_csv)
    if not stay_ids:
        raise RuntimeError(f"No stay_id found in cohort: {args.cohort_csv}")

    client = _bq_client(args.billing_project)

    bilirubin_itemids = _query_itemids_bilirubin_total(client, args.dataset_hosp)
    vaso_itemids = _query_itemids_vasopressors(client, args.dataset_icu)
    rrt_itemids = _query_itemids_rrt(client, args.dataset_icu)

    # Extract
    df_bili = _extract_bilirubin_hourly(
        client=client,
        dataset_hosp=args.dataset_hosp,
        dataset_icu=args.dataset_icu,
        stay_ids=stay_ids,
        bilirubin_itemids=bilirubin_itemids,
        hours=args.hours,
    )
    df_vaso = _extract_binary_hourly_from_events(
        client=client,
        table_fq=f"{args.dataset_icu}.inputevents",
        dataset_icu=args.dataset_icu,
        stay_ids=stay_ids,
        itemids=vaso_itemids,
        hours=args.hours,
        out_col="vasopressors",
    )
    df_rrt = _extract_binary_hourly_from_events(
        client=client,
        table_fq=f"{args.dataset_icu}.procedureevents",
        dataset_icu=args.dataset_icu,
        stay_ids=stay_ids,
        itemids=rrt_itemids,
        hours=args.hours,
        out_col="rrt",
    )
    df_ckd = _extract_ckd_static(client, args.dataset_hosp, args.dataset_icu, stay_ids)

    # Write outputs
    df_bili.to_csv(args.out_dir / "bilirubin_total_hourly.csv", index=False)
    df_vaso.to_csv(args.out_dir / "vasopressors_hourly.csv", index=False)
    df_rrt.to_csv(args.out_dir / "rrt_hourly.csv", index=False)
    df_ckd.to_csv(args.out_dir / "ckd_static.csv", index=False)

    meta = ExtractMeta(
        generated_at=datetime.now().isoformat(timespec="seconds"),
        billing_project=args.billing_project,
        dataset_hosp=args.dataset_hosp,
        dataset_icu=args.dataset_icu,
        n_stays=len(stay_ids),
        hours=int(args.hours),
        bilirubin_itemids=list(bilirubin_itemids),
        vasopressor_itemids=list(vaso_itemids),
        rrt_itemids=list(rrt_itemids),
        icd_ckd_rules=df_ckd.attrs.get("icd_rules", {}),
    )
    (args.out_dir / "bq_feature_extract_meta.json").write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Wrote: {args.out_dir}")


if __name__ == "__main__":
    main()
