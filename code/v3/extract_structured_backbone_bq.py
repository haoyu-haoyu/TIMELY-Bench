#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from functools import reduce
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    DEFAULT_V3_COHORT_FILE,
    ROOT_DIR,
    V3_PROCESSED_DIR,
    V3_RESULTS_DIR,
    V3_SOURCE_TIMESERIES_FILE,
    ensure_v3_directories,
)
from v3.bq_utils import make_bq_client, quota_project_of  # type: ignore
from v3.io_utils import chunk_dir_path, relativize_value, write_table  # type: ignore

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract TIMELY-Bench v3 168h structured backbone from BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument("--dataset-derived", default="physionet-data.mimiciv_3_1_derived")
    p.add_argument("--dataset-hosp", default="physionet-data.mimiciv_3_1_hosp")
    p.add_argument("--dataset-icu", default="physionet-data.mimiciv_3_1_icu")
    p.add_argument("--cohort-csv", default=str(DEFAULT_V3_COHORT_FILE))
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--out", default=str(V3_SOURCE_TIMESERIES_FILE))
    p.add_argument("--meta-json", default=str(V3_RESULTS_DIR / "structured_backbone_hourly_v3_meta.json"))
    return p.parse_args()


def _client(project: str):
    return make_bq_client(project)


def _query_df(client, sql: str, params: list) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(query_parameters=params)
    return client.query(sql, job_config=cfg).to_dataframe()


def _load_cohort(path: Path, stay_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [col for col in ["stay_id", "subject_id", "hadm_id", "intime"] if col in df.columns]
    df = df[cols].copy()
    for col in ("stay_id", "subject_id", "hadm_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id", "subject_id", "hadm_id"]).copy()
    for col in ("stay_id", "subject_id", "hadm_id"):
        df[col] = df[col].astype("int64")
    if stay_limit is not None:
        df = df.head(int(stay_limit)).copy()
    return df.sort_values(["stay_id"], kind="mergesort").reset_index(drop=True)


def _iter_cohort_batches(cohort: pd.DataFrame, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(cohort), batch_size):
        yield cohort.iloc[start : start + batch_size].copy()


def _normalize_hourly(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.dropna(subset=["stay_id", "hour"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")
    df["hour"] = df["hour"].astype("int64")
    df = df[df["hour"] >= 0].copy()
    return df.reset_index(drop=True)


def _merge_hourly_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    usable = [df for df in frames if df is not None and not df.empty]
    if not usable:
        return pd.DataFrame(columns=["stay_id", "hour"])
    return reduce(lambda left, right: left.merge(right, on=["stay_id", "hour"], how="outer"), usable)


def _extract_vitals(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(v.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(v.heart_rate) AS heart_rate,
      AVG(v.sbp) AS sbp,
      AVG(v.dbp) AS dbp,
      AVG(v.mbp) AS mbp,
      AVG(v.resp_rate) AS resp_rate,
      AVG(v.temperature) AS temperature,
      AVG(v.spo2) AS spo2,
      AVG(v.glucose) AS glucose_chart
    FROM co
    JOIN `{args.dataset_derived}.vitalsign` v
      ON co.stay_id = v.stay_id
    WHERE v.charttime >= co.intime
      AND v.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_chemistry(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(c.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(c.albumin) AS albumin,
      AVG(c.bun) AS bun,
      AVG(c.creatinine) AS creatinine,
      AVG(c.glucose) AS glucose_lab,
      AVG(c.sodium) AS sodium,
      AVG(c.potassium) AS potassium,
      AVG(c.bicarbonate) AS bicarbonate,
      AVG(c.chloride) AS chloride,
      AVG(c.aniongap) AS aniongap
    FROM co
    JOIN `{args.dataset_derived}.chemistry` c
      ON co.hadm_id = c.hadm_id
    WHERE c.charttime >= co.intime
      AND c.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
                bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_cbc(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(c.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(c.wbc) AS wbc,
      AVG(c.hemoglobin) AS hemoglobin,
      AVG(c.hematocrit) AS hematocrit,
      AVG(c.platelet) AS platelet
    FROM co
    JOIN `{args.dataset_derived}.complete_blood_count` c
      ON co.hadm_id = c.hadm_id
    WHERE c.charttime >= co.intime
      AND c.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
                bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_gcs(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(g.charttime, co.intime, HOUR) AS INT64) AS hour,
      MIN(g.gcs) AS gcs_min,
      AVG(g.gcs) AS gcs_total,
      AVG(g.gcs_motor) AS gcs_motor,
      AVG(g.gcs_verbal) AS gcs_verbal,
      AVG(g.gcs_eyes) AS gcs_eye
    FROM co
    JOIN `{args.dataset_derived}.gcs` g
      ON co.stay_id = g.stay_id
    WHERE g.charttime >= co.intime
      AND g.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_urine_output(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(u.charttime, co.intime, HOUR) AS INT64) AS hour,
      SUM(u.urineoutput) AS urineoutput
    FROM co
    JOIN `{args.dataset_derived}.urine_output` u
      ON co.stay_id = u.stay_id
    WHERE u.charttime >= co.intime
      AND u.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_bg(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(bg.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(bg.lactate) AS lactate,
      AVG(bg.ph) AS ph,
      AVG(bg.po2) AS pao2,
      AVG(bg.pco2) AS paco2,
      AVG(bg.pao2fio2ratio) AS pao2_fio2_ratio
    FROM co
    JOIN `{args.dataset_derived}.bg` bg
      ON co.hadm_id = bg.hadm_id
    WHERE bg.charttime >= co.intime
      AND bg.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
                bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_sofa(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    SELECT
      stay_id,
      CAST(hr AS INT64) AS hour,
      MAX(sofa_24hours) AS sofa_total,
      MAX(respiration_24hours) AS sofa_respiration
    FROM `{args.dataset_derived}.sofa`
    WHERE stay_id IN UNNEST(@stay_ids)
      AND hr BETWEEN 0 AND (@hours - 1)
    GROUP BY stay_id, hour
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_bilirubin(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH bilirubin_items AS (
      SELECT itemid
      FROM `{args.dataset_hosp}.d_labitems`
      WHERE LOWER(label) LIKE '%bilirubin%'
        AND LOWER(label) LIKE '%total%'
    ),
    co AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(le.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(le.valuenum) AS bilirubin_total
    FROM co
    JOIN `{args.dataset_hosp}.labevents` le
      ON co.subject_id = le.subject_id
     AND co.hadm_id = le.hadm_id
    JOIN bilirubin_items bi
      ON le.itemid = bi.itemid
    WHERE le.valuenum IS NOT NULL
      AND le.charttime >= co.intime
      AND le.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
                bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_vasopressors(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH drug_items AS (
      SELECT
        itemid,
        CASE
          WHEN LOWER(label) LIKE '%norepinephrine%' THEN 'norepinephrine'
          WHEN LOWER(label) LIKE '%phenylephrine%' THEN 'phenylephrine'
          WHEN LOWER(label) LIKE '%vasopressin%' THEN 'vasopressin'
          WHEN LOWER(label) LIKE '%epinephrine%' THEN 'epinephrine'
          WHEN LOWER(label) LIKE '%dopamine%' THEN 'dopamine'
          ELSE NULL
        END AS drug
      FROM `{args.dataset_icu}.d_items`
      WHERE linksto = 'inputevents'
    ),
    co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    ),
    spans AS (
      SELECT
        co.stay_id,
        GREATEST(CAST(TIMESTAMP_DIFF(ie.starttime, co.intime, HOUR) AS INT64), 0) AS sh,
        LEAST(
          GREATEST(CAST(TIMESTAMP_DIFF(COALESCE(ie.endtime, ie.starttime), co.intime, HOUR) AS INT64),
                   CAST(TIMESTAMP_DIFF(ie.starttime, co.intime, HOUR) AS INT64)),
          @hours - 1
        ) AS eh,
        di.drug,
        COALESCE(ie.rate, 0.0) AS raw_rate
      FROM co
      JOIN `{args.dataset_icu}.inputevents` ie
        ON ie.stay_id = co.stay_id
      JOIN drug_items di
        ON ie.itemid = di.itemid
      WHERE di.drug IS NOT NULL
        AND ie.starttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
        AND COALESCE(ie.endtime, ie.starttime) >= co.intime
    )
    SELECT
      stay_id,
      hour,
      1 AS vasopressors_active,
      1 AS vasopressors,
      SUM(
        CASE
          WHEN drug = 'norepinephrine' THEN raw_rate
          WHEN drug = 'epinephrine' THEN raw_rate
          WHEN drug = 'phenylephrine' THEN raw_rate / 10.0
          WHEN drug = 'dopamine' THEN raw_rate / 100.0
          WHEN drug = 'vasopressin' THEN raw_rate * 2.5
          ELSE 0.0
        END
      ) AS vasopressor_dose_norepi_equiv
    FROM spans, UNNEST(GENERATE_ARRAY(sh, eh)) AS hour
    GROUP BY stay_id, hour
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_rrt(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH rrt_items AS (
      SELECT itemid
      FROM `{args.dataset_icu}.d_items`
      WHERE linksto = 'procedureevents'
        AND (
          LOWER(label) LIKE '%dialysis%'
          OR LOWER(label) LIKE '%rrt%'
          OR LOWER(label) LIKE '%crrt%'
          OR LOWER(label) LIKE '%cvvh%'
          OR LOWER(label) LIKE '%hemofiltration%'
          OR LOWER(label) LIKE '%hemodialysis%'
          OR LOWER(label) LIKE '%renal replacement%'
        )
    ),
    co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    ),
    spans AS (
      SELECT
        co.stay_id,
        GREATEST(CAST(TIMESTAMP_DIFF(pe.starttime, co.intime, HOUR) AS INT64), 0) AS sh,
        LEAST(
          GREATEST(CAST(TIMESTAMP_DIFF(COALESCE(pe.endtime, pe.starttime), co.intime, HOUR) AS INT64),
                   CAST(TIMESTAMP_DIFF(pe.starttime, co.intime, HOUR) AS INT64)),
          @hours - 1
        ) AS eh
      FROM co
      JOIN `{args.dataset_icu}.procedureevents` pe
        ON pe.stay_id = co.stay_id
      JOIN rrt_items ri
        ON pe.itemid = ri.itemid
      WHERE pe.starttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
        AND COALESCE(pe.endtime, pe.starttime) >= co.intime
    )
    SELECT
      stay_id,
      hour,
      1 AS rrt_active,
      1 AS rrt
    FROM spans, UNNEST(GENERATE_ARRAY(sh, eh)) AS hour
    GROUP BY stay_id, hour
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_sedatives(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH sed_items AS (
      SELECT
        itemid,
        CASE
          WHEN LOWER(label) LIKE '%propofol%' THEN 'propofol'
          WHEN LOWER(label) LIKE '%midazolam%' THEN 'midazolam'
          WHEN LOWER(label) LIKE '%fentanyl%' THEN 'fentanyl'
          ELSE NULL
        END AS sed
      FROM `{args.dataset_icu}.d_items`
      WHERE linksto = 'inputevents'
    ),
    co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(ie.starttime, co.intime, HOUR) AS INT64) AS hour,
      SUM(CASE WHEN si.sed = 'propofol' THEN COALESCE(ie.rate, 0.0) ELSE 0.0 END) AS propofol_rate,
      SUM(CASE WHEN si.sed = 'midazolam' THEN COALESCE(ie.rate, 0.0) ELSE 0.0 END) AS midazolam_rate,
      SUM(CASE WHEN si.sed = 'fentanyl' THEN COALESCE(ie.rate, 0.0) ELSE 0.0 END) AS fentanyl_rate
    FROM co
    JOIN `{args.dataset_icu}.inputevents` ie
      ON ie.stay_id = co.stay_id
    JOIN sed_items si
      ON ie.itemid = si.itemid
    WHERE si.sed IS NOT NULL
      AND ie.starttime >= co.intime
      AND ie.starttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_fluid_input(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(ie.starttime, co.intime, HOUR) AS INT64) AS hour,
      SUM(
        CASE
          WHEN LOWER(COALESCE(ie.amountuom, '')) IN ('ml', 'milliliters', 'milliliter')
            THEN COALESCE(ie.amount, 0.0)
          ELSE NULL
        END
      ) AS fluid_input_hourly
    FROM co
    JOIN `{args.dataset_icu}.inputevents` ie
      ON ie.stay_id = co.stay_id
    WHERE ie.amount IS NOT NULL
      AND ie.amount > 0
      AND ie.starttime >= co.intime
      AND ie.starttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _normalize_hourly(
        _query_df(
            client,
            sql,
            [
                bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
                bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
            ],
        )
    )


def _extract_batch_backbone(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frames = [
        _extract_vitals(client, batch, args),
        _extract_chemistry(client, batch, args),
        _extract_cbc(client, batch, args),
        _extract_gcs(client, batch, args),
        _extract_urine_output(client, batch, args),
        _extract_bg(client, batch, args),
        _extract_sofa(client, batch, args),
        _extract_bilirubin(client, batch, args),
        _extract_vasopressors(client, batch, args),
        _extract_rrt(client, batch, args),
        _extract_sedatives(client, batch, args),
        _extract_fluid_input(client, batch, args),
    ]
    merged = _merge_hourly_frames(frames)
    if merged.empty:
        return merged
    fluid_in = pd.to_numeric(merged["fluid_input_hourly"], errors="coerce") if "fluid_input_hourly" in merged.columns else pd.Series(0.0, index=merged.index)
    urine = pd.to_numeric(merged["urineoutput"], errors="coerce") if "urineoutput" in merged.columns else pd.Series(0.0, index=merged.index)
    merged["fluid_balance"] = fluid_in - urine.fillna(0.0)
    merged = merged.sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)
    return merged


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    client = _client(args.billing_project)
    cohort = _load_cohort(Path(args.cohort_csv), args.stay_limit)

    out_path = Path(args.out)
    parts_dir = chunk_dir_path(out_path)
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_parts = 0
    columns_seen: set[str] = set()
    for idx, batch in enumerate(_iter_cohort_batches(cohort, args.batch_size), start=1):
        print(f"[structured_backbone] batch {idx} n_stays={batch['stay_id'].nunique()}", flush=True)
        part = _extract_batch_backbone(client, batch, args)
        if part.empty:
            continue
        n_parts += 1
        n_rows += int(len(part))
        columns_seen.update(part.columns.tolist())
        part_path = parts_dir / f"part_{n_parts:05d}.parquet"
        written = write_table(part, part_path, index=False)
        print(f"[structured_backbone] wrote {written} rows={len(part)}", flush=True)

    if n_parts == 0:
        empty_written = write_table(pd.DataFrame(columns=["stay_id", "hour"]), out_path, index=False)
        output_ref = str(empty_written)
    else:
        output_ref = str(out_path)

    meta = {
        "billing_project": args.billing_project,
        "quota_project": quota_project_of(client),
        "dataset_derived": args.dataset_derived,
        "dataset_hosp": args.dataset_hosp,
        "dataset_icu": args.dataset_icu,
        "cohort_csv": args.cohort_csv,
        "hours": int(args.hours),
        "batch_size": int(args.batch_size),
        "output_path": output_ref,
        "n_rows": n_rows,
        "n_parts": n_parts,
        "n_stays": int(cohort["stay_id"].nunique()) if not cohort.empty else 0,
        "columns": sorted(columns_seen),
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(relativize_value(meta, root=ROOT_DIR), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
