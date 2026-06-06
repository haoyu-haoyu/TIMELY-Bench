#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


DATASET_HOSP = "physionet-data.mimiciv_3_1_hosp"
DATASET_ICU = "physionet-data.mimiciv_3_1_icu"
DATASET_DERIVED = "physionet-data.mimiciv_3_1_derived"
DATASET_NOTE = "physionet-data.mimiciv_note"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run stroke/sepsis feasibility audit on BigQuery.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument(
        "--out-dir",
        default="results/v3/feasibility_remote",
        help="Output directory on CREATE for intermediate BQ audit artefacts.",
    )
    return p.parse_args()


def _client(project: str):
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed.")
    return bigquery.Client(project=project)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_df(client, sql: str) -> pd.DataFrame:
    return client.query(sql).result().to_dataframe()


def _note_table_probe(client) -> dict:
    sql = f"""
    SELECT table_name
    FROM `{DATASET_NOTE}.INFORMATION_SCHEMA.TABLES`
    ORDER BY table_name
    """
    try:
        df = _run_df(client, sql)
        return {
            "status": "ok",
            "tables": df["table_name"].astype(str).tolist(),
            "sql": sql.strip(),
        }
    except Exception as e:  # pragma: no cover
        return {
            "status": "error",
            "tables": [],
            "error_type": type(e).__name__,
            "error": str(e),
            "sql": sql.strip(),
        }


STROKE_COHORT_SQL = f"""
WITH stroke_dx AS (
  SELECT
    i.subject_id,
    i.hadm_id,
    i.stay_id,
    i.intime,
    i.outtime,
    i.los,
    MAX(CASE WHEN d.seq_num = 1 THEN 1 ELSE 0 END) AS primary_dx_flag
  FROM `{DATASET_HOSP}.diagnoses_icd` d
  JOIN `{DATASET_ICU}.icustays` i
    ON d.hadm_id = i.hadm_id
  WHERE
    (d.icd_version = 10 AND d.icd_code LIKE 'I63%')
    OR
    (d.icd_version = 9 AND (d.icd_code LIKE '433%' OR d.icd_code LIKE '434%' OR d.icd_code = '436'))
  GROUP BY 1,2,3,4,5,6
)
SELECT
  s.subject_id,
  s.hadm_id,
  s.stay_id,
  s.intime,
  s.outtime,
  s.los,
  s.primary_dx_flag,
  a.hospital_expire_flag,
  p.gender,
  CAST(p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS INT64) AS age
FROM stroke_dx s
JOIN `{DATASET_HOSP}.admissions` a
  ON s.hadm_id = a.hadm_id
JOIN `{DATASET_HOSP}.patients` p
  ON s.subject_id = p.subject_id
"""


AKI_COHORT_SQL = f"""
SELECT DISTINCT
  i.subject_id,
  i.hadm_id,
  i.stay_id
FROM `{DATASET_DERIVED}.kdigo_stages` k
JOIN `{DATASET_ICU}.icustays` i
  USING (stay_id)
WHERE k.aki_stage >= 1
"""


DELIRIUM_COHORT_SQL = f"""
WITH icd_delirium AS (
  SELECT DISTINCT i.subject_id, i.hadm_id, i.stay_id
  FROM `{DATASET_HOSP}.diagnoses_icd` d
  JOIN `{DATASET_ICU}.icustays` i
    ON d.hadm_id = i.hadm_id
  WHERE (d.icd_version = 10 AND d.icd_code LIKE 'F05%')
),
cam_positive AS (
  SELECT DISTINCT i.subject_id, i.hadm_id, i.stay_id
  FROM `{DATASET_ICU}.chartevents` ce
  JOIN `{DATASET_ICU}.icustays` i
    USING (stay_id)
  WHERE ce.itemid = 228332
    AND LOWER(TRIM(CAST(ce.value AS STRING))) = 'positive'
)
SELECT DISTINCT subject_id, hadm_id, stay_id
FROM (
  SELECT * FROM icd_delirium
  UNION DISTINCT
  SELECT * FROM cam_positive
)
"""


SEPSIS_COHORT_SQL = f"""
SELECT
  i.subject_id,
  i.hadm_id,
  i.stay_id,
  MIN(CAST(DATETIME_DIFF(DATETIME(s.sofa_time), DATETIME(i.intime), HOUR) AS INT64)) AS sepsis_onset_hour
FROM `{DATASET_DERIVED}.sepsis3` s
JOIN `{DATASET_ICU}.icustays` i
  USING (stay_id)
WHERE s.sepsis3 = TRUE
  AND s.sofa_time IS NOT NULL
GROUP BY 1,2,3
"""


def _stroke_structured_specs() -> List[dict]:
    return [
        {"name": "heart_rate", "kind": "chartevents", "itemids": [220045]},
        {"name": "systolic_bp", "kind": "chartevents", "itemids": [220050, 220179]},
        {"name": "map", "kind": "chartevents", "itemids": [220052]},
        {"name": "spo2", "kind": "chartevents", "itemids": [220277]},
        {"name": "temperature", "kind": "chartevents", "itemids": [223761, 223762]},
        {"name": "resp_rate", "kind": "chartevents", "itemids": [220210]},
        {"name": "gcs_total", "kind": "derived_gcs", "column": "gcs"},
        {"name": "glucose", "kind": "labevents", "itemids": [50931]},
        {"name": "inr", "kind": "labevents", "itemids": [51237]},
        {"name": "platelets", "kind": "labevents", "itemids": [51265]},
        {"name": "creatinine", "kind": "labevents", "itemids": [50912]},
        {"name": "troponin_t", "kind": "labevents", "itemids": [51003]},
        {"name": "sodium", "kind": "labevents", "itemids": [50983]},
        {"name": "potassium", "kind": "labevents", "itemids": [50971]},
    ]


def _sepsis_structured_specs() -> List[dict]:
    return [
        {"name": "lactate", "kind": "labevents", "itemids": [50813]},
        {"name": "wbc", "kind": "labevents", "itemids": [51301]},
        {"name": "crp", "kind": "labevents", "itemids": [50889]},
        {"name": "procalcitonin", "kind": "lab_search", "pattern": "procalcitonin"},
        {"name": "creatinine", "kind": "labevents", "itemids": [50912]},
        {"name": "bilirubin_total", "kind": "labevents", "itemids": [50885]},
        {"name": "platelets", "kind": "labevents", "itemids": [51265]},
        {"name": "inr", "kind": "labevents", "itemids": [51237]},
        {"name": "bicarbonate", "kind": "labevents", "itemids": [50882]},
        {"name": "ph_arterial", "kind": "labevents", "itemids": [50820]},
        {"name": "bun", "kind": "labevents", "itemids": [51006]},
        {"name": "albumin", "kind": "labevents", "itemids": [50862]},
        {"name": "heart_rate", "kind": "chartevents", "itemids": [220045]},
        {"name": "map", "kind": "chartevents", "itemids": [220052]},
        {"name": "systolic_bp", "kind": "chartevents", "itemids": [220050, 220179]},
        {"name": "temperature", "kind": "chartevents", "itemids": [223761, 223762]},
        {"name": "spo2", "kind": "chartevents", "itemids": [220277]},
        {"name": "resp_rate", "kind": "chartevents", "itemids": [220210]},
        {"name": "norepinephrine", "kind": "inputevents", "itemids": [221906]},
        {"name": "vasopressin", "kind": "inputevents", "itemids": [222315]},
        {"name": "phenylephrine", "kind": "inputevents", "itemids": [221749]},
        {"name": "epinephrine", "kind": "inputevents", "itemids": [221289]},
        {"name": "dopamine", "kind": "inputevents", "itemids": [221662]},
        {"name": "iv_fluid_any", "kind": "inputevents_any"},
        {"name": "mechanical_ventilation", "kind": "derived_ventilation"},
        {
            "name": "urine_output",
            "kind": "outputevents",
            "itemids": [226559, 226560, 226561, 226563, 226564, 226565, 226566, 226567, 226627, 226631, 227489],
        },
        {"name": "sofa_total", "kind": "derived_sofa", "column": "sofa_24hours"},
        {"name": "sofa_respiration", "kind": "derived_sofa", "column": "respiration_24hours"},
        {"name": "sofa_cardiovascular", "kind": "derived_sofa", "column": "cardiovascular_24hours"},
        {"name": "sofa_liver", "kind": "derived_sofa", "column": "liver_24hours"},
        {"name": "sofa_coagulation", "kind": "derived_sofa", "column": "coagulation_24hours"},
        {"name": "sofa_renal", "kind": "derived_sofa", "column": "renal_24hours"},
        {"name": "sofa_cns", "kind": "derived_sofa", "column": "cns_24hours"},
    ]


def _coverage_sql(cohort_sql: str, spec: dict) -> str:
    if spec["kind"] == "chartevents":
        itemids = ",".join(str(x) for x in spec["itemids"])
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_ICU}.chartevents` ce ON ce.stay_id = c.stay_id
          WHERE ce.itemid IN ({itemids})
            AND DATETIME_DIFF(DATETIME(ce.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "labevents":
        itemids = ",".join(str(x) for x in spec["itemids"])
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_HOSP}.labevents` le ON le.hadm_id = c.hadm_id
          WHERE le.itemid IN ({itemids})
            AND DATETIME_DIFF(DATETIME(le.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "derived_gcs":
        col = spec["column"]
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_DERIVED}.gcs` g ON g.stay_id = c.stay_id
          WHERE g.{col} IS NOT NULL
            AND DATETIME_DIFF(DATETIME(g.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "inputevents":
        itemids = ",".join(str(x) for x in spec["itemids"])
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_ICU}.inputevents` ie ON ie.stay_id = c.stay_id
          WHERE ie.itemid IN ({itemids})
            AND DATETIME_DIFF(DATETIME(ie.starttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "inputevents_any":
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_ICU}.inputevents` ie ON ie.stay_id = c.stay_id
          WHERE ie.amount IS NOT NULL
            AND ie.amount > 0
            AND DATETIME_DIFF(DATETIME(ie.starttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "derived_ventilation":
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_DERIVED}.ventilation` v ON v.stay_id = c.stay_id
          WHERE v.ventilation_status IS NOT NULL
            AND DATETIME_DIFF(DATETIME(v.starttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "outputevents":
        itemids = ",".join(str(x) for x in spec["itemids"])
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_ICU}.outputevents` oe ON oe.stay_id = c.stay_id
          WHERE oe.itemid IN ({itemids})
            AND DATETIME_DIFF(DATETIME(oe.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    if spec["kind"] == "derived_sofa":
        col = spec["column"]
        return f"""
        WITH cohort AS ({cohort_sql}),
        obs AS (
          SELECT c.stay_id, COUNT(*) AS n_meas
          FROM cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_DERIVED}.sofa` s ON s.stay_id = c.stay_id
          WHERE s.{col} IS NOT NULL
            AND DATETIME_DIFF(DATETIME(s.starttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT COUNT(*) AS n_stays_with_measurement,
               APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """
    raise ValueError(f"Unsupported spec kind: {spec['kind']}")


def _lab_search(client, pattern: str) -> dict:
    sql = f"""
    SELECT itemid, label
    FROM `{DATASET_HOSP}.d_labitems`
    WHERE LOWER(label) LIKE '%{pattern.lower()}%'
    ORDER BY itemid
    """
    df = _run_df(client, sql)
    return {
        "matches": df.to_dict(orient="records"),
        "sql": sql.strip(),
    }


def _run_coverage(client, cohort_sql: str, denominator: int, specs: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for spec in specs:
        if spec["kind"] == "lab_search":
            result = _lab_search(client, spec["pattern"])
            rows.append(
                {
                    "variable": spec["name"],
                    "n_stays_with_measurement": 0,
                    "pct_coverage": 0.0,
                    "median_measurements": None,
                    "status": "no_standard_item" if not result["matches"] else "search_matches_found",
                    "matches": result["matches"],
                    "sql": result["sql"],
                }
            )
            continue
        sql = _coverage_sql(cohort_sql, spec)
        df = _run_df(client, sql)
        n = int(df.iloc[0]["n_stays_with_measurement"]) if len(df) else 0
        med = None
        if len(df) and pd.notna(df.iloc[0]["median_measurements"]):
            med = float(df.iloc[0]["median_measurements"])
        rows.append(
            {
                "variable": spec["name"],
                "n_stays_with_measurement": n,
                "pct_coverage": (n / denominator) if denominator else 0.0,
                "median_measurements": med,
                "status": "ok",
                "sql": sql.strip(),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    client = _client(args.billing_project)

    stroke_df = _run_df(client, STROKE_COHORT_SQL)
    stroke_df.to_csv(out_dir / "stroke_cohort.csv", index=False)

    aki_df = _run_df(client, AKI_COHORT_SQL)
    aki_df.to_csv(out_dir / "aki_cohort.csv", index=False)

    delirium_df = _run_df(client, DELIRIUM_COHORT_SQL)
    delirium_df.to_csv(out_dir / "delirium_cohort.csv", index=False)

    sepsis_df = _run_df(client, SEPSIS_COHORT_SQL)
    sepsis_df.to_csv(out_dir / "sepsis_cohort.csv", index=False)

    stroke_coverage = _run_coverage(client, STROKE_COHORT_SQL, int(stroke_df["stay_id"].nunique()), _stroke_structured_specs())
    sepsis_coverage = _run_coverage(client, SEPSIS_COHORT_SQL, int(sepsis_df["stay_id"].nunique()), _sepsis_structured_specs())

    stroke_basic = {
        "total_unique_hadm_id": int(stroke_df["hadm_id"].nunique()),
        "total_unique_stay_id": int(stroke_df["stay_id"].nunique()),
        "age_describe": stroke_df["age"].describe().to_dict() if "age" in stroke_df else {},
        "los_describe": stroke_df["los"].describe().to_dict() if "los" in stroke_df else {},
        "mortality_rate": float(pd.to_numeric(stroke_df["hospital_expire_flag"], errors="coerce").fillna(0).mean()),
        "primary_diagnosis_stays": int((stroke_df["primary_dx_flag"] == 1).sum()),
        "secondary_only_stays": int((stroke_df["primary_dx_flag"] == 0).sum()),
        "sql": STROKE_COHORT_SQL.strip(),
    }

    sepsis_basic = {
        "total_unique_hadm_id": int(sepsis_df["hadm_id"].nunique()),
        "total_unique_stay_id": int(sepsis_df["stay_id"].nunique()),
        "sepsis_onset_describe_hours": sepsis_df["sepsis_onset_hour"].describe().to_dict() if "sepsis_onset_hour" in sepsis_df else {},
        "sql": SEPSIS_COHORT_SQL.strip(),
    }

    result = {
        "billing_project": args.billing_project,
        "datasets": {
            "hosp": DATASET_HOSP,
            "icu": DATASET_ICU,
            "derived": DATASET_DERIVED,
            "note": DATASET_NOTE,
        },
        "note_module_probe": _note_table_probe(client),
        "stroke_basic": stroke_basic,
        "sepsis_basic": sepsis_basic,
        "stroke_structured_coverage": stroke_coverage,
        "sepsis_structured_coverage": sepsis_coverage,
        "exports": {
            "stroke_cohort_csv": str((out_dir / "stroke_cohort.csv").as_posix()),
            "aki_cohort_csv": str((out_dir / "aki_cohort.csv").as_posix()),
            "delirium_cohort_csv": str((out_dir / "delirium_cohort.csv").as_posix()),
            "sepsis_cohort_csv": str((out_dir / "sepsis_cohort.csv").as_posix()),
        },
    }
    _write_json(out_dir / "stroke_sepsis_bq_audit.json", result)
    print(json.dumps({"out_dir": str(out_dir), "stroke_stays": stroke_basic["total_unique_stay_id"], "sepsis_stays": sepsis_basic["total_unique_stay_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
