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
    V3_PROCESSED_DIR,
    V3_RESULTS_DIR,
    V3_SOURCE_COHORT_FILE,
    ensure_v3_directories,
)
from v3.bq_utils import make_bq_client, quota_project_of  # type: ignore
from v3.io_utils import relativize_value  # type: ignore

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


CONDITION_MAPPING = {
    "sepsis": ["A41", "A40", "R65.2", "038", "99591", "99592"],
    "aki": ["N17", "584"],
    "ards": ["J80", "51882"],
    "shock": ["R57", "7855"],
    "pneumonia": ["J18", "J15", "J13", "J14", "481", "482", "483", "485", "486"],
    "heart_failure": ["I50", "428"],
    "respiratory_failure": ["J96", "51881", "51884", "7991"],
    "stroke": ["I60", "I61", "I62", "I63", "I64", "G45", "430", "431", "432", "433", "434", "435", "436"],
    "delirium": ["F05", "R41.0", "R41.82", "293", "78009", "78097"],
}
COMORBIDITY_PREFIXES = {
    "ckd": ["N18", "585"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract TIMELY-Bench v3 source-of-truth cohort from MIMIC-IV BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument("--dataset-derived", default="physionet-data.mimiciv_3_1_derived")
    p.add_argument("--dataset-hosp", default="physionet-data.mimiciv_3_1_hosp")
    p.add_argument("--dataset-icu", default="physionet-data.mimiciv_3_1_icu")
    p.add_argument("--out-csv", default=str(V3_SOURCE_COHORT_FILE))
    p.add_argument("--meta-json", default=str(V3_RESULTS_DIR / "cohort_v3_meta.json"))
    p.add_argument("--stay-limit", type=int, default=None)
    return p.parse_args()


def _client(project: str):
    return make_bq_client(project)


def _query_df(client, sql: str) -> pd.DataFrame:
    return client.query(sql).to_dataframe()


def _classify_conditions(icd_codes: str | float | None) -> list[str]:
    if icd_codes is None or (isinstance(icd_codes, float) and pd.isna(icd_codes)):
        return []
    codes = [code.strip().upper() for code in str(icd_codes).split(",") if code.strip()]
    conditions: list[str] = []
    for condition, prefixes in CONDITION_MAPPING.items():
        if any(code.startswith(prefix) for code in codes for prefix in prefixes):
            conditions.append(condition)
    return conditions


def _has_prefix(icd_codes: str | float | None, prefixes: list[str]) -> int:
    if icd_codes is None or (isinstance(icd_codes, float) and pd.isna(icd_codes)):
        return 0
    codes = [code.strip().upper() for code in str(icd_codes).split(",") if code.strip()]
    return int(any(code.startswith(prefix) for code in codes for prefix in prefixes))


def _postprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("subject_id", "hadm_id", "stay_id"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["subject_id", "hadm_id", "stay_id"]).copy()
    for col in ("subject_id", "hadm_id", "stay_id"):
        df[col] = df[col].astype("int64")

    for col in ("intime", "outtime", "deathtime", "icu_intime"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "anchor_age" in df.columns:
        df["anchor_age"] = pd.to_numeric(df["anchor_age"], errors="coerce").fillna(0).astype(int)
    else:
        df["anchor_age"] = 0

    if "los_hours" not in df.columns:
        df["los_hours"] = (
            (pd.to_datetime(df["outtime"], errors="coerce") - pd.to_datetime(df["intime"], errors="coerce"))
            .dt.total_seconds()
            .div(3600.0)
        )
    df["los_hours"] = pd.to_numeric(df["los_hours"], errors="coerce").fillna(0).round().astype(int)
    if "los_days" not in df.columns:
        df["los_days"] = pd.to_numeric(df["los_hours"], errors="coerce").div(24.0)
    df["los_days"] = pd.to_numeric(df["los_days"], errors="coerce").fillna(0).round().astype(int)

    for days in (3, 5, 7):
        df[f"prolonged_los_{days}d"] = (df["los_days"] > days).astype(int)

    if "label_mortality" not in df.columns:
        df["label_mortality"] = pd.to_datetime(df["deathtime"], errors="coerce").notna().astype(int)
    df["label_mortality"] = pd.to_numeric(df["label_mortality"], errors="coerce").fillna(0).astype(int)

    df["sepsis3"] = df["sepsis3"].fillna(False).astype(bool)
    df["sepsis3_clinical"] = df["sepsis3"].astype(int)
    df["sepsis_sofa"] = pd.to_numeric(df["sepsis_sofa"], errors="coerce")
    df["sofa_max"] = pd.to_numeric(df["sofa_max"], errors="coerce")
    df["aki_stage_max"] = pd.to_numeric(df["aki_stage_max"], errors="coerce").fillna(0)
    df["aki_clinical"] = (df["aki_stage_max"] >= 1).astype(int)
    df["aki_stage"] = df["aki_stage_max"].fillna(0).astype(int)

    df["diagnoses_text"] = df["diagnoses_text"].fillna("")
    df["icd_codes"] = df["icd_codes"].fillna("")
    df["conditions_from_icd"] = df["icd_codes"].apply(_classify_conditions)
    for condition in CONDITION_MAPPING:
        df[f"has_{condition}"] = df["conditions_from_icd"].apply(lambda xs, c=condition: int(c in xs))

    df["has_sepsis_final"] = ((df["has_sepsis"] == 1) | (df["sepsis3_clinical"] == 1)).astype(int)
    df["has_aki_final"] = ((df["has_aki"] == 1) | (df["aki_clinical"] == 1)).astype(int)
    df["has_stroke_final"] = df["has_stroke"].astype(int)
    df["has_delirium_final"] = df["has_delirium"].astype(int)
    df["num_conditions"] = (
        df["has_sepsis_final"]
        + df["has_aki_final"]
        + df["has_stroke_final"]
        + df["has_delirium_final"]
        + df["has_ards"]
        + df["has_shock"]
    )
    df["conditions_from_icd"] = df["conditions_from_icd"].apply(lambda xs: str(xs))
    df["ckd"] = df["icd_codes"].apply(lambda x: _has_prefix(x, COMORBIDITY_PREFIXES["ckd"]))
    df["readmission_30d"] = pd.to_numeric(df["readmission_30d"], errors="coerce").fillna(0).astype(int)

    ordered = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "deathtime",
        "icu_intime",
        "anchor_age",
        "gender",
        "label_mortality",
        "sepsis3",
        "sepsis_sofa",
        "sofa_max",
        "aki_stage_max",
        "icd_codes",
        "diagnoses_text",
        "has_sepsis",
        "has_aki",
        "has_ards",
        "has_shock",
        "has_pneumonia",
        "has_heart_failure",
        "has_respiratory_failure",
        "has_stroke",
        "has_delirium",
        "sepsis3_clinical",
        "aki_clinical",
        "aki_stage",
        "has_sepsis_final",
        "has_aki_final",
        "has_stroke_final",
        "has_delirium_final",
        "conditions_from_icd",
        "num_conditions",
        "los_hours",
        "los_days",
        "prolonged_los_3d",
        "prolonged_los_5d",
        "prolonged_los_7d",
        "readmission_30d",
        "ckd",
        "primary_icd_code",
        "primary_diagnosis_text",
    ]
    existing = [col for col in ordered if col in df.columns]
    remainder = [col for col in df.columns if col not in existing]
    return df[existing + remainder].sort_values(["stay_id"], kind="mergesort").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    client = _client(args.billing_project)

    sql = f"""
    WITH base AS (
      SELECT
        i.subject_id,
        i.hadm_id,
        i.stay_id,
        i.intime,
        i.outtime,
        a.deathtime,
        i.intime AS icu_intime,
        COALESCE(age.anchor_age, age.age) AS anchor_age,
        p.gender,
        CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END AS label_mortality,
        CAST(ROUND(COALESCE(i.los * 24.0, DATETIME_DIFF(i.outtime, i.intime, HOUR)), 0) AS INT64) AS los_hours,
        CAST(ROUND(COALESCE(i.los, DATETIME_DIFF(i.outtime, i.intime, HOUR) / 24.0), 0) AS INT64) AS los_days
      FROM `{args.dataset_icu}.icustays` i
      JOIN `{args.dataset_hosp}.admissions` a
        ON i.hadm_id = a.hadm_id
      JOIN `{args.dataset_hosp}.patients` p
        ON i.subject_id = p.subject_id
      LEFT JOIN `{args.dataset_derived}.age` age
        ON age.subject_id = i.subject_id
       AND age.hadm_id = i.hadm_id
    ),
    readmission AS (
      SELECT
        hadm_id,
        CASE
          WHEN LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) IS NOT NULL
           AND LEAD(admittime) OVER (PARTITION BY subject_id ORDER BY admittime) <= DATETIME_ADD(dischtime, INTERVAL 30 DAY)
          THEN 1 ELSE 0
        END AS readmission_30d
      FROM `{args.dataset_hosp}.admissions`
    ),
    sofa_max AS (
      SELECT
        stay_id,
        MAX(sofa_24hours) AS sofa_max
      FROM `{args.dataset_derived}.sofa`
      GROUP BY stay_id
    ),
    aki_max AS (
      SELECT
        stay_id,
        MAX(aki_stage) AS aki_stage_max
      FROM `{args.dataset_derived}.kdigo_stages`
      GROUP BY stay_id
    ),
    sepsis AS (
      SELECT
        stay_id,
        MAX(CAST(sepsis3 AS INT64)) = 1 AS sepsis3,
        MAX(sofa_score) AS sepsis_sofa
      FROM `{args.dataset_derived}.sepsis3`
      GROUP BY stay_id
    ),
    diagnoses AS (
      SELECT
        i.stay_id,
        STRING_AGG(CAST(d.icd_code AS STRING), ',' ORDER BY d.seq_num, d.icd_code) AS icd_codes,
        STRING_AGG(COALESCE(di.long_title, CAST(d.icd_code AS STRING)), ' | ' ORDER BY d.seq_num, d.icd_code) AS diagnoses_text,
        MAX(IF(d.seq_num = 1, CAST(d.icd_code AS STRING), NULL)) AS primary_icd_code,
        MAX(IF(d.seq_num = 1, COALESCE(di.long_title, CAST(d.icd_code AS STRING)), NULL)) AS primary_diagnosis_text
      FROM `{args.dataset_icu}.icustays` i
      LEFT JOIN `{args.dataset_hosp}.diagnoses_icd` d
        ON i.hadm_id = d.hadm_id
      LEFT JOIN `{args.dataset_hosp}.d_icd_diagnoses` di
        ON d.icd_code = di.icd_code
       AND d.icd_version = di.icd_version
      GROUP BY i.stay_id
    )
    SELECT
      b.*,
      COALESCE(r.readmission_30d, 0) AS readmission_30d,
      s.sepsis3,
      s.sepsis_sofa,
      sf.sofa_max,
      a.aki_stage_max,
      d.icd_codes,
      d.diagnoses_text,
      d.primary_icd_code,
      d.primary_diagnosis_text
    FROM base b
    LEFT JOIN readmission r
      ON b.hadm_id = r.hadm_id
    LEFT JOIN sepsis s
      ON b.stay_id = s.stay_id
    LEFT JOIN sofa_max sf
      ON b.stay_id = sf.stay_id
    LEFT JOIN aki_max a
      ON b.stay_id = a.stay_id
    LEFT JOIN diagnoses d
      ON b.stay_id = d.stay_id
    ORDER BY b.stay_id
    """
    df = _query_df(client, sql)
    df = _postprocess(df)
    if args.stay_limit is not None:
      df = df.head(int(args.stay_limit)).copy()

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "billing_project": args.billing_project,
        "quota_project": quota_project_of(client),
        "dataset_derived": args.dataset_derived,
        "dataset_hosp": args.dataset_hosp,
        "dataset_icu": args.dataset_icu,
        "output_csv": str(out_csv),
        "n_rows": int(len(df)),
        "n_stays": int(df["stay_id"].nunique()) if not df.empty else 0,
        "n_subjects": int(df["subject_id"].nunique()) if not df.empty else 0,
        "n_hadm": int(df["hadm_id"].nunique()) if not df.empty else 0,
        "positive_rates": {
            "mortality": float(df["label_mortality"].mean()) if len(df) else None,
            "sepsis_final": float(df["has_sepsis_final"].mean()) if len(df) else None,
            "aki_final": float(df["has_aki_final"].mean()) if len(df) else None,
            "stroke_final": float(df["has_stroke_final"].mean()) if len(df) else None,
            "delirium_final": float(df["has_delirium_final"].mean()) if len(df) else None,
            "ckd": float(df["ckd"].mean()) if len(df) else None,
        },
    }
    meta_path = Path(args.meta_json)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(relativize_value(meta, root=ROOT_DIR), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
