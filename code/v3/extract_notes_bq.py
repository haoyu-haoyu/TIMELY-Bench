#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    DEFAULT_V3_COHORT_FILE,
    ROOT_DIR,
    V3_RAW_DATA_DIR,
    ensure_v3_directories,
)
from v3.bq_utils import make_bq_client, quota_project_of  # type: ignore
from v3.io_utils import chunk_dir_path, relativize_value, write_table  # type: ignore

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


NOTE_OUTPUTS = {
    "discharge": "discharge_notes_v3.parquet",
    "nursing": "nursing_notes_168h.parquet",
    "lab_comment": "lab_comments_168h.parquet",
    "radiology": "radiology_notes_168h.parquet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract TIMELY-Bench v3 note sources from BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument("--dataset-hosp", default="physionet-data.mimiciv_3_1_hosp")
    p.add_argument("--dataset-icu", default="physionet-data.mimiciv_3_1_icu")
    p.add_argument("--dataset-note", default="physionet-data.mimiciv_note")
    p.add_argument("--cohort-csv", default=str(DEFAULT_V3_COHORT_FILE))
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1500)
    p.add_argument("--out-dir", default=str(V3_RAW_DATA_DIR))
    p.add_argument("--meta-json", default=str(V3_RAW_DATA_DIR / "extract_notes_bq_meta.json"))
    p.add_argument("--extract", choices=["all", "discharge", "nursing", "lab_comment", "radiology"], default="all")
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


def _extract_discharge(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    subject_ids = sorted(batch["subject_id"].drop_duplicates().tolist())
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE subject_id IN UNNEST(@subject_ids)
        AND hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      d.subject_id,
      d.hadm_id,
      d.note_id,
      d.note_type,
      d.note_seq,
      d.charttime,
      d.storetime,
      CAST(TIMESTAMP_DIFF(d.charttime, co.intime, HOUR) AS INT64) AS hour_offset,
      d.text AS discharge_text,
      LENGTH(d.text) AS text_length
    FROM co
    JOIN `{args.dataset_note}.discharge` d
      ON co.subject_id = d.subject_id
     AND co.hadm_id = d.hadm_id
    WHERE d.text IS NOT NULL
      AND LENGTH(d.text) > 10
    ORDER BY co.stay_id, d.charttime
    """
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids),
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
        ],
    )


def _extract_radiology(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH co AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      r.subject_id,
      r.hadm_id,
      r.note_id,
      r.note_type,
      r.note_seq,
      r.charttime,
      r.storetime,
      CAST(TIMESTAMP_DIFF(r.charttime, co.intime, HOUR) AS INT64) AS hour_offset,
      r.text AS radiology_text
    FROM co
    JOIN `{args.dataset_note}.radiology` r
      ON co.hadm_id = r.hadm_id
    WHERE r.text IS NOT NULL
      AND LENGTH(r.text) > 10
      AND r.charttime >= co.intime
      AND r.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    ORDER BY co.stay_id, r.charttime
    """
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )


def _extract_nursing(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    stay_ids = sorted(batch["stay_id"].drop_duplicates().tolist())
    sql = f"""
    WITH nursing_items AS (
      SELECT itemid, label, category
      FROM `{args.dataset_icu}.d_items`
      WHERE category IN ('Routine Vital Signs', 'Neurological', 'Respiratory', 'Cardiovascular')
         OR LOWER(label) LIKE '%assessment%'
         OR LOWER(label) LIKE '%note%'
         OR LOWER(label) LIKE '%comment%'
    ),
    co AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      co.subject_id,
      co.hadm_id,
      ce.charttime,
      CAST(TIMESTAMP_DIFF(ce.charttime, co.intime, HOUR) AS INT64) AS hour_offset,
      di.label AS item_label,
      di.category,
      CAST(ce.value AS STRING) AS chart_text,
      ce.valuenum
    FROM co
    JOIN `{args.dataset_icu}.chartevents` ce
      ON ce.stay_id = co.stay_id
    JOIN nursing_items di
      ON ce.itemid = di.itemid
    WHERE ce.value IS NOT NULL
      AND LENGTH(CAST(ce.value AS STRING)) > 10
      AND ce.charttime >= co.intime
      AND ce.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    ORDER BY co.stay_id, ce.charttime
    """
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )


def _extract_lab_comments(client, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    subject_ids = sorted(batch["subject_id"].drop_duplicates().tolist())
    hadm_ids = sorted(batch["hadm_id"].drop_duplicates().tolist())
    sql = f"""
    WITH key_lab_items AS (
      SELECT itemid, label
      FROM `{args.dataset_hosp}.d_labitems`
      WHERE LOWER(label) LIKE '%creatinine%'
         OR LOWER(label) LIKE '%potassium%'
         OR LOWER(label) LIKE '%lactate%'
         OR LOWER(label) LIKE '%bilirubin%'
         OR LOWER(label) LIKE '%platelet%'
         OR LOWER(label) LIKE '%wbc%'
         OR LOWER(label) LIKE '%white blood%'
         OR LOWER(label) LIKE '%hemoglobin%'
         OR LOWER(label) LIKE '%bicarbonate%'
         OR LOWER(label) LIKE '%ph%'
    ),
    co AS (
      SELECT stay_id, subject_id, hadm_id, intime
      FROM `{args.dataset_icu}.icustays`
      WHERE subject_id IN UNNEST(@subject_ids)
        AND hadm_id IN UNNEST(@hadm_ids)
        AND stay_id IN UNNEST(@stay_ids)
    )
    SELECT
      co.stay_id,
      co.subject_id,
      co.hadm_id,
      le.charttime,
      CAST(TIMESTAMP_DIFF(le.charttime, co.intime, HOUR) AS INT64) AS hour_offset,
      di.label AS lab_name,
      le.value,
      le.valuenum,
      le.valueuom,
      le.flag,
      le.ref_range_lower,
      le.ref_range_upper,
      le.comments AS lab_comment
    FROM co
    JOIN `{args.dataset_hosp}.labevents` le
      ON co.subject_id = le.subject_id
     AND co.hadm_id = le.hadm_id
    JOIN key_lab_items di
      ON le.itemid = di.itemid
    WHERE le.comments IS NOT NULL
      AND LENGTH(le.comments) > 5
      AND le.charttime >= co.intime
      AND le.charttime < TIMESTAMP_ADD(co.intime, INTERVAL @hours HOUR)
    ORDER BY co.stay_id, le.charttime
    """
    return _query_df(
        client,
        sql,
        [
            bigquery.ArrayQueryParameter("subject_ids", "INT64", subject_ids),
            bigquery.ArrayQueryParameter("hadm_ids", "INT64", hadm_ids),
            bigquery.ArrayQueryParameter("stay_ids", "INT64", sorted(batch["stay_id"].drop_duplicates().tolist())),
            bigquery.ScalarQueryParameter("hours", "INT64", int(args.hours)),
        ],
    )


def _extract_one(client, kind: str, batch: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if kind == "discharge":
        return _extract_discharge(client, batch, args)
    if kind == "radiology":
        return _extract_radiology(client, batch, args)
    if kind == "nursing":
        return _extract_nursing(client, batch, args)
    if kind == "lab_comment":
        return _extract_lab_comments(client, batch, args)
    raise ValueError(f"Unsupported note kind: {kind}")


def _enforce_note_hour_window(part: pd.DataFrame, kind: str, hours: int) -> pd.DataFrame:
    if kind == "discharge" or "hour_offset" not in part.columns:
        return part
    out = part.copy()
    out["hour_offset"] = pd.to_numeric(out["hour_offset"], errors="coerce")
    out = out[out["hour_offset"].between(0, int(hours) - 1, inclusive="both")].copy()
    if "charttime" in out.columns:
        out = out.sort_values(["stay_id", "charttime"], kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    client = _client(args.billing_project)
    cohort = _load_cohort(Path(args.cohort_csv), args.stay_limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wanted = list(NOTE_OUTPUTS) if args.extract == "all" else [args.extract]
    outputs: dict[str, dict] = {}
    for kind in wanted:
        out_path = out_dir / NOTE_OUTPUTS[kind]
        parts_dir = chunk_dir_path(out_path)
        tmp_parent = parts_dir.parent
        tmp_parts_dir = Path(
            tempfile.mkdtemp(prefix=f"{parts_dir.name}.tmp.", dir=str(tmp_parent))
        )

        n_rows = 0
        n_parts = 0
        try:
            for idx, batch in enumerate(_iter_cohort_batches(cohort, args.batch_size), start=1):
                print(f"[{kind}] batch {idx} n_stays={batch['stay_id'].nunique()}", flush=True)
                part = _extract_one(client, kind, batch, args)
                part = _enforce_note_hour_window(part, kind=kind, hours=args.hours)
                if part.empty:
                    continue
                n_parts += 1
                n_rows += int(len(part))
                part_path = tmp_parts_dir / f"part_{n_parts:05d}.parquet"
                written = write_table(part, part_path, index=False)
                print(f"[{kind}] wrote {written} rows={len(part)}", flush=True)
        except Exception:
            shutil.rmtree(tmp_parts_dir, ignore_errors=True)
            raise

        if n_parts == 0:
            empty_written = write_table(pd.DataFrame(), out_path, index=False)
            path_ref = str(empty_written)
            shutil.rmtree(tmp_parts_dir, ignore_errors=True)
        else:
            if parts_dir.exists():
                shutil.rmtree(parts_dir)
            tmp_parts_dir.replace(parts_dir)
            path_ref = str(out_path)
        outputs[kind] = {
            "path": path_ref,
            "n_rows": n_rows,
            "n_parts": n_parts,
        }

    meta = {
        "billing_project": args.billing_project,
        "quota_project": quota_project_of(client),
        "dataset_hosp": args.dataset_hosp,
        "dataset_icu": args.dataset_icu,
        "dataset_note": args.dataset_note,
        "cohort_csv": args.cohort_csv,
        "hours": int(args.hours),
        "batch_size": int(args.batch_size),
        "outputs": outputs,
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict):
                merged_outputs = dict(existing.get("outputs", {}))
                merged_outputs.update(outputs)
                meta["outputs"] = merged_outputs
                if "refilter_summary" in existing and isinstance(existing["refilter_summary"], dict):
                    refilter_summary = dict(existing["refilter_summary"])
                    for kind in outputs:
                        refilter_summary.pop(kind, None)
                    if refilter_summary:
                        meta["refilter_summary"] = refilter_summary
        except Exception:
            pass
    meta_path.write_text(json.dumps(relativize_value(meta, root=ROOT_DIR), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
