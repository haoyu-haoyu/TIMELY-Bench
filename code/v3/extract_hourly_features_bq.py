#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    ROOT_DIR,
    DEFAULT_V3_COHORT_FILE,
    V3_HOURLY_FEATURES_DIR,
    ensure_v3_directories,
)
from v3.bq_utils import make_bq_client, quota_project_of  # type: ignore
from v3.io_utils import relativize_value, write_table  # type: ignore
from v3.mappings import VALIDATED_ITEMIDS  # type: ignore

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract additional hourly TIMELY-Bench v3 features from BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument("--dataset-derived", default="physionet-data.mimiciv_3_1_derived")
    p.add_argument("--dataset-icu", default="physionet-data.mimiciv_3_1_icu")
    p.add_argument("--cohort-csv", default=str(DEFAULT_V3_COHORT_FILE))
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=5000)
    p.add_argument("--out-dir", default=str(V3_HOURLY_FEATURES_DIR))
    return p.parse_args()


def _client(project: str):
    return make_bq_client(project)


def _load_cohort(path: Path, stay_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [col for col in ["stay_id", "subject_id", "hadm_id", "intime"] if col in df.columns]
    df = df[cols].copy()
    for col in ["stay_id", "subject_id", "hadm_id"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id", "subject_id", "hadm_id"]).copy()
    for col in ["stay_id", "subject_id", "hadm_id"]:
        df[col] = df[col].astype("int64")
    if stay_limit is not None:
        df = df.head(int(stay_limit)).copy()
    return df


def _query_df(client, sql: str, params: list) -> pd.DataFrame:
    cfg = bigquery.QueryJobConfig(query_parameters=params)
    return client.query(sql, job_config=cfg).to_dataframe()


def _iter_cohort_batches(cohort: pd.DataFrame, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for start in range(0, len(cohort), batch_size):
        yield cohort.iloc[start : start + batch_size].copy()


def _extract_gcs(client, cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(g.charttime, co.intime, HOUR) AS INT64) AS hour,
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
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )


def _extract_neuro_chartevents(client, cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    itemids = sorted(VALIDATED_ITEMIDS["rass"] + VALIDATED_ITEMIDS["delirium_assessment"] + VALIDATED_ITEMIDS["cam_components"])
    sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    ),
    raw AS (
      SELECT
        co.stay_id,
        CAST(TIMESTAMP_DIFF(ce.charttime, co.intime, HOUR) AS INT64) AS hour,
        ce.itemid,
        ce.valuenum,
        CAST(ce.value AS STRING) AS value_text
      FROM co
      JOIN `{args.dataset_icu}.chartevents` ce
        ON ce.stay_id = co.stay_id
      WHERE ce.itemid IN UNNEST(@itemids)
        AND ce.charttime >= co.intime
        AND ce.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    )
    SELECT
      stay_id,
      hour,
      AVG(IF(itemid = 228096, valuenum, NULL)) AS rass,
      MAX(IF(itemid = 228332 AND LOWER(value_text) = 'positive', 1, 0)) AS delirium_positive,
      MAX(IF(itemid = 228332 AND LOWER(value_text) = 'negative', 1, 0)) AS delirium_negative,
      MAX(IF(itemid = 228332 AND LOWER(value_text) = 'uta', 1, 0)) AS delirium_uta,
      MAX(IF(itemid IN UNNEST(@cam_items), 1, 0)) AS cam_component_recorded
    FROM raw
    GROUP BY stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ArrayQueryParameter("itemids", "INT64", itemids),
            bigquery.ArrayQueryParameter("cam_items", "INT64", VALIDATED_ITEMIDS["cam_components"]),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )


def _extract_restraints(client, cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH restraint_items AS (
      SELECT itemid
      FROM `{args.dataset_icu}.d_items`
      WHERE LOWER(label) LIKE '%restraint%'
    ),
    co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(ce.charttime, co.intime, HOUR) AS INT64) AS hour,
      1 AS restraint_active
    FROM co
    JOIN `{args.dataset_icu}.chartevents` ce
      ON ce.stay_id = co.stay_id
    JOIN restraint_items ri
      ON ce.itemid = ri.itemid
    WHERE ce.charttime >= co.intime
      AND ce.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )


def _extract_resp_derived(client, cohort: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stay_ids = sorted(cohort["stay_id"].drop_duplicates().tolist())
    hadm_ids = sorted(cohort["hadm_id"].drop_duplicates().tolist())
    vent_sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    ),
    spans AS (
      SELECT
        co.stay_id,
        GREATEST(CAST(TIMESTAMP_DIFF(v.starttime, co.intime, HOUR) AS INT64), 0) AS sh,
        LEAST(
          GREATEST(CAST(TIMESTAMP_DIFF(COALESCE(v.endtime, v.starttime), co.intime, HOUR) AS INT64),
                   CAST(TIMESTAMP_DIFF(v.starttime, co.intime, HOUR) AS INT64)),
          @hours - 1
        ) AS eh,
        v.ventilation_status
      FROM co
      JOIN `{args.dataset_derived}.ventilation` v
        ON co.stay_id = v.stay_id
      WHERE v.starttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
        AND COALESCE(v.endtime, v.starttime) >= co.intime
    )
    SELECT stay_id, hour, ANY_VALUE(ventilation_status) AS ventilation_status
    FROM spans, UNNEST(GENERATE_ARRAY(sh, eh)) AS hour
    GROUP BY stay_id, hour
    """
    vent_settings_sql = f"""
    WITH co AS (
      SELECT stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(vs.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(vs.fio2) AS fio2,
      AVG(vs.peep) AS peep,
      AVG(vs.tidal_volume_observed) AS tidal_volume,
      AVG(vs.minute_volume) AS minute_volume,
      AVG(vs.plateau_pressure) AS plateau_pressure
    FROM co
    JOIN `{args.dataset_derived}.ventilator_setting` vs
      ON co.stay_id = vs.stay_id
    WHERE vs.charttime >= co.intime
      AND vs.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    bg_sql = f"""
    WITH co AS (
      SELECT hadm_id, stay_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
        AND hadm_id IN UNNEST(@hadm_ids)
    )
    SELECT
      co.stay_id,
      CAST(TIMESTAMP_DIFF(bg.charttime, co.intime, HOUR) AS INT64) AS hour,
      AVG(bg.po2) AS pao2,
      AVG(bg.pco2) AS paco2,
      AVG(bg.pao2fio2ratio) AS pao2_fio2_ratio,
      AVG(bg.fio2) AS fio2_bg
    FROM co
    JOIN `{args.dataset_derived}.bg` bg
      ON co.hadm_id = bg.hadm_id
    WHERE bg.charttime >= co.intime
      AND bg.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    GROUP BY co.stay_id, hour
    HAVING hour BETWEEN 0 AND (@hours - 1)
    """
    params = [
        bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
        bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
        bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
    ]
    vent = _query_df(client, vent_sql, [params[0], params[2]])
    settings = _query_df(client, vent_settings_sql, [params[0], params[2]])
    bg = _query_df(client, bg_sql, params)
    return vent, settings, bg


def _normalize_binary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in df.columns:
        if col not in {"stay_id", "hour", "ventilation_status"}:
            try:
                if set(df[col].dropna().unique()).issubset({0, 1}):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
            except Exception:
                pass
    return df


def _run_batched_single(client, cohort: pd.DataFrame, args: argparse.Namespace, name: str, fn) -> pd.DataFrame:
    batches = list(_iter_cohort_batches(cohort, args.batch_size))
    frames: list[pd.DataFrame] = []
    for idx, batch in enumerate(batches, start=1):
        print(f"[{name}] batch {idx}/{len(batches)} n_stays={batch['stay_id'].nunique()}", flush=True)
        part = fn(client, batch, args)
        if not part.empty:
            frames.append(part)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _run_batched_resp(client, cohort: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    batches = list(_iter_cohort_batches(cohort, args.batch_size))
    vent_frames: list[pd.DataFrame] = []
    settings_frames: list[pd.DataFrame] = []
    bg_frames: list[pd.DataFrame] = []
    for idx, batch in enumerate(batches, start=1):
        print(f"[resp] batch {idx}/{len(batches)} n_stays={batch['stay_id'].nunique()}", flush=True)
        vent, settings, bg = _extract_resp_derived(client, batch, args)
        if not vent.empty:
            vent_frames.append(vent)
        if not settings.empty:
            settings_frames.append(settings)
        if not bg.empty:
            bg_frames.append(bg)
    return (
        pd.concat(vent_frames, ignore_index=True) if vent_frames else pd.DataFrame(),
        pd.concat(settings_frames, ignore_index=True) if settings_frames else pd.DataFrame(),
        pd.concat(bg_frames, ignore_index=True) if bg_frames else pd.DataFrame(),
    )


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    client = _client(args.billing_project)
    cohort = _load_cohort(Path(args.cohort_csv), args.stay_limit)
    outputs: dict[str, str] = {}

    gcs = _normalize_binary(_run_batched_single(client, cohort, args, "gcs", _extract_gcs))
    path = write_table(gcs, out_dir / "gcs_hourly_v3.parquet", index=False)
    outputs["gcs"] = str(path)
    print(f"Wrote {path}")

    neuro = _normalize_binary(_run_batched_single(client, cohort, args, "neuro", _extract_neuro_chartevents))
    path = write_table(neuro, out_dir / "delirium_neuro_hourly_v3.parquet", index=False)
    outputs["neuro"] = str(path)
    print(f"Wrote {path}")

    restraints = _normalize_binary(_run_batched_single(client, cohort, args, "restraints", _extract_restraints))
    path = write_table(restraints, out_dir / "restraint_hourly_v3.parquet", index=False)
    outputs["restraints"] = str(path)
    print(f"Wrote {path}")

    vent, settings, bg = _run_batched_resp(client, cohort, args)
    path = write_table(_normalize_binary(vent), out_dir / "ventilation_status_hourly_v3.parquet", index=False)
    outputs["ventilation"] = str(path)
    print(f"Wrote {path}")
    path = write_table(settings, out_dir / "ventilator_settings_hourly_v3.parquet", index=False)
    outputs["ventilator_settings"] = str(path)
    print(f"Wrote {path}")
    path = write_table(bg, out_dir / "blood_gas_hourly_v3.parquet", index=False)
    outputs["blood_gas"] = str(path)
    print(f"Wrote {path}")

    meta = {
        "billing_project": args.billing_project,
        "quota_project": quota_project_of(client),
        "dataset_derived": args.dataset_derived,
        "dataset_icu": args.dataset_icu,
        "hours": int(args.hours),
        "n_stays": int(cohort["stay_id"].nunique()),
        "outputs": relativize_value(outputs, root=ROOT_DIR),
    }
    meta_path = out_dir / "extract_hourly_features_bq_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
