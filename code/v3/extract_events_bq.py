#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    ROOT_DIR,
    DEFAULT_V3_COHORT_FILE,
    V3_EVENTS_DIR,
    ensure_v3_directories,
)
from v3.bq_utils import make_bq_client, quota_project_of  # type: ignore
from v3.io_utils import chunk_dir_path, relativize_value, write_table  # type: ignore
from v3.mappings import VALIDATED_ITEMIDS  # type: ignore

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract TIMELY-Bench v3 event tables from MIMIC-IV BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument("--dataset-hosp", default="physionet-data.mimiciv_3_1_hosp")
    p.add_argument("--dataset-icu", default="physionet-data.mimiciv_3_1_icu")
    p.add_argument("--cohort-csv", default=str(DEFAULT_V3_COHORT_FILE))
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--out-dir", default=str(V3_EVENTS_DIR))
    p.add_argument("--extract", choices=["all", "medications", "procedures", "diagnoses"], default="all")
    return p.parse_args()


def _client(project: str):
    return make_bq_client(project)


def _load_cohort(path: Path, stay_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [col for col in ["stay_id", "subject_id", "hadm_id", "intime"] if col in df.columns]
    df = df[cols].copy()
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df["subject_id"] = pd.to_numeric(df["subject_id"], errors="coerce").astype("Int64")
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id", "subject_id", "hadm_id"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")
    df["subject_id"] = df["subject_id"].astype("int64")
    df["hadm_id"] = df["hadm_id"].astype("int64")
    if stay_limit is not None:
        df = df.head(int(stay_limit)).copy()
    return df


def _query_dataframe(client, sql: str, params: list) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(query_parameters=params)
    return client.query(sql, job_config=cfg).to_dataframe()


def _iter_cohort_batches(cohort: pd.DataFrame, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(cohort), batch_size):
        yield cohort.iloc[start : start + batch_size].copy()


def _extract_medications(client, cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      p.subject_id,
      p.hadm_id,
      p.starttime,
      p.stoptime,
      p.drug,
      p.route,
      p.dose_val_rx,
      p.dose_unit_rx,
      'prescriptions' AS source
    FROM co
    JOIN `{args.dataset_hosp}.prescriptions` p
      ON p.subject_id = co.subject_id
     AND p.hadm_id = co.hadm_id
    WHERE TIMESTAMP(p.starttime) < TIMESTAMP_ADD(TIMESTAMP(co.intime), INTERVAL @hours HOUR)
      AND TIMESTAMP(COALESCE(p.stoptime, p.starttime)) >= TIMESTAMP(co.intime)
    """
    df = _query_dataframe(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )
    if df.empty:
        return df
    merged = df.merge(cohort[["stay_id", "subject_id", "hadm_id", "intime"]], on=["stay_id", "subject_id", "hadm_id"], how="inner")
    merged["intime"] = pd.to_datetime(merged["intime"], errors="coerce")
    merged["starttime"] = pd.to_datetime(merged["starttime"], errors="coerce")
    merged["stoptime"] = pd.to_datetime(merged["stoptime"], errors="coerce")
    merged["event_start_hour"] = ((merged["starttime"] - merged["intime"]).dt.total_seconds() / 3600.0).astype("float64")
    merged["event_end_hour"] = ((merged["stoptime"] - merged["intime"]).dt.total_seconds() / 3600.0).astype("float64")
    merged = merged[(merged["event_start_hour"] < args.hours) & (merged["event_end_hour"].fillna(merged["event_start_hour"]) >= 0)].copy()
    merged["event_name"] = merged["drug"].astype(str)
    merged["event_type"] = "medication"
    return merged[
        [
            "stay_id",
            "subject_id",
            "hadm_id",
            "event_start_hour",
            "event_end_hour",
            "event_name",
            "route",
            "dose_val_rx",
            "dose_unit_rx",
            "source",
        ]
    ]


def _extract_procedures(client, cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    itemids = sorted(
        VALIDATED_ITEMIDS["intubation"]
        + VALIDATED_ITEMIDS["extubation"]
        + VALIDATED_ITEMIDS["unplanned_extubation"]
    )
    sql = f"""
    SELECT
      stay_id,
      itemid,
      starttime,
      endtime,
      value,
      valueuom
    FROM `{args.dataset_icu}.procedureevents`
    WHERE stay_id IN UNNEST(@stay_ids)
      AND itemid IN UNNEST(@itemids)
    """
    df = _query_dataframe(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ArrayQueryParameter("itemids", "INT64", itemids),
        ],
    )
    if df.empty:
        return df
    merged = df.merge(cohort[["stay_id", "intime"]], on="stay_id", how="inner")
    merged["intime"] = pd.to_datetime(merged["intime"], errors="coerce")
    merged["starttime"] = pd.to_datetime(merged["starttime"], errors="coerce")
    merged["endtime"] = pd.to_datetime(merged["endtime"], errors="coerce")
    merged["event_start_hour"] = ((merged["starttime"] - merged["intime"]).dt.total_seconds() / 3600.0).astype("float64")
    merged["event_end_hour"] = ((merged["endtime"] - merged["intime"]).dt.total_seconds() / 3600.0).astype("float64")
    merged = merged[(merged["event_start_hour"] < args.hours) & (merged["event_end_hour"].fillna(merged["event_start_hour"]) >= 0)].copy()
    labels = {
        224385: "intubation",
        227194: "extubation",
        225468: "unplanned_extubation_patient",
        225477: "unplanned_extubation_nonpatient",
    }
    merged["event_name"] = merged["itemid"].map(labels).fillna("procedure")
    merged["event_type"] = "procedure"
    merged["source"] = "procedureevents"
    return merged[
        [
            "stay_id",
            "event_start_hour",
            "event_end_hour",
            "event_name",
            "itemid",
            "value",
            "valueuom",
            "source",
        ]
    ]


def _extract_diagnoses(client, cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, subject_id, hadm_id
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      d.subject_id,
      d.hadm_id,
      seq_num,
      icd_code,
      icd_version
    FROM co
    JOIN `{args.dataset_hosp}.diagnoses_icd` d
      ON d.subject_id = co.subject_id
     AND d.hadm_id = co.hadm_id
    """
    df = _query_dataframe(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
        ],
    )
    if df.empty:
        return df
    return df.merge(cohort[["stay_id", "subject_id", "hadm_id"]], on=["stay_id", "subject_id", "hadm_id"], how="inner")


def _run_batched_extract(client, cohort: pd.DataFrame, args: argparse.Namespace, kind: str) -> pd.DataFrame:
    batches = list(_iter_cohort_batches(cohort, args.batch_size))
    frames: list[pd.DataFrame] = []
    for idx, batch in enumerate(batches, start=1):
        print(f"[{kind}] batch {idx}/{len(batches)} n_stays={batch['stay_id'].nunique()}", flush=True)
        if kind == "medications":
            part = _extract_medications(client, batch, args)
        elif kind == "procedures":
            part = _extract_procedures(client, batch, args)
        elif kind == "diagnoses":
            part = _extract_diagnoses(client, batch, args)
        else:
            raise ValueError(f"Unsupported extract kind: {kind}")
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _run_batched_extract_to_parts(
    client,
    cohort: pd.DataFrame,
    args: argparse.Namespace,
    kind: str,
    out_path: Path,
) -> dict:
    batches = list(_iter_cohort_batches(cohort, args.batch_size))
    parts_dir = chunk_dir_path(out_path)
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_parts = 0
    for idx, batch in enumerate(batches, start=1):
        print(f"[{kind}] batch {idx}/{len(batches)} n_stays={batch['stay_id'].nunique()}", flush=True)
        if kind == "medications":
            part = _extract_medications(client, batch, args)
        elif kind == "procedures":
            part = _extract_procedures(client, batch, args)
        elif kind == "diagnoses":
            part = _extract_diagnoses(client, batch, args)
        else:
            raise ValueError(f"Unsupported extract kind: {kind}")

        if part.empty:
            continue
        n_parts += 1
        n_rows += int(len(part))
        part_path = parts_dir / f"part_{n_parts:05d}.parquet"
        written = write_table(part, part_path, index=False)
        print(f"[{kind}] wrote {written} rows={len(part)}", flush=True)

    if n_parts == 0:
        empty_written = write_table(pd.DataFrame(), out_path, index=False)
        return {
            "path": str(empty_written),
            "n_rows": 0,
            "n_parts": 0,
        }
    return {
        "path": str(out_path),
        "n_rows": n_rows,
        "n_parts": n_parts,
    }


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _client(args.billing_project)
    cohort = _load_cohort(Path(args.cohort_csv), args.stay_limit)
    outputs: dict[str, str] = {}

    if args.extract in {"all", "medications"}:
        meds_path = out_dir / "medication_events_bq.parquet"
        meds_meta = _run_batched_extract_to_parts(client, cohort, args, "medications", meds_path)
        outputs["medications"] = meds_meta

    if args.extract in {"all", "procedures"}:
        proc_path = out_dir / "procedure_events_bq.parquet"
        procs_meta = _run_batched_extract_to_parts(client, cohort, args, "procedures", proc_path)
        outputs["procedures"] = procs_meta

    if args.extract in {"all", "diagnoses"}:
        dx_path = out_dir / "diagnoses_icd_bq.parquet"
        dx_meta = _run_batched_extract_to_parts(client, cohort, args, "diagnoses", dx_path)
        outputs["diagnoses"] = dx_meta

    meta = {
        "billing_project": args.billing_project,
        "quota_project": quota_project_of(client),
        "dataset_hosp": args.dataset_hosp,
        "dataset_icu": args.dataset_icu,
        "hours": int(args.hours),
        "n_stays": int(cohort["stay_id"].nunique()),
        "outputs": relativize_value(outputs, root=ROOT_DIR),
    }
    meta_path = out_dir / "extract_events_bq_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
