#!/usr/bin/env python3
from __future__ import annotations

import json

from google.cloud import bigquery


DATASET_ICU = "physionet-data.mimiciv_3_1_icu"
DATASET_DERIVED = "physionet-data.mimiciv_3_1_derived"


SEPSIS_CTE = f"""
WITH sepsis_cohort AS (
  SELECT i.subject_id, i.hadm_id, i.stay_id
  FROM `{DATASET_DERIVED}.sepsis3` s
  JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
  WHERE s.sepsis3 = TRUE
    AND s.sofa_time IS NOT NULL
)
"""


def run_df(client: bigquery.Client, sql: str):
    return client.query(sql).result().to_dataframe()


def main() -> None:
    client = bigquery.Client(project="timely-bench-mimic")

    queries = {
        "cohort_size": SEPSIS_CTE
        + """
        SELECT COUNT(DISTINCT stay_id) AS n_stays
        FROM sepsis_cohort
        """,
        "merged_map": SEPSIS_CTE
        + f"""
        , obs AS (
          SELECT
            c.stay_id,
            COUNT(*) AS n_meas,
            COUNTIF(ce.itemid = 220052) AS n_invasive,
            COUNTIF(ce.itemid = 220181) AS n_noninvasive
          FROM sepsis_cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_ICU}.chartevents` ce ON ce.stay_id = c.stay_id
          WHERE ce.itemid IN (220052, 220181)
            AND DATETIME_DIFF(DATETIME(ce.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT
          COUNT(*) AS n_stays_with_map,
          ROUND(COUNT(*) / (SELECT COUNT(DISTINCT stay_id) FROM sepsis_cohort), 6) AS pct_coverage,
          APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements,
          COUNTIF(n_invasive > 0) AS stays_with_invasive,
          COUNTIF(n_noninvasive > 0) AS stays_with_noninvasive
        FROM obs
        """,
        "vasopressor_any": SEPSIS_CTE
        + f"""
        , obs AS (
          SELECT
            c.stay_id,
            COUNT(*) AS n_meas
          FROM sepsis_cohort c
          JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
          JOIN `{DATASET_ICU}.inputevents` ie ON ie.stay_id = c.stay_id
          WHERE ie.itemid IN (221906, 222315, 221749, 221289, 221662)
            AND DATETIME_DIFF(DATETIME(ie.starttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
          GROUP BY c.stay_id
        )
        SELECT
          COUNT(*) AS n_stays_with_any_vaso,
          ROUND(COUNT(*) / (SELECT COUNT(DISTINCT stay_id) FROM sepsis_cohort), 6) AS pct_coverage,
          APPROX_QUANTILES(n_meas, 100)[OFFSET(50)] AS median_measurements
        FROM obs
        """,
        "vasopressor_by_agent": SEPSIS_CTE
        + f"""
        SELECT
          ie.itemid,
          COUNT(DISTINCT c.stay_id) AS n_stays,
          ROUND(COUNT(DISTINCT c.stay_id) / (SELECT COUNT(DISTINCT stay_id) FROM sepsis_cohort), 6) AS pct_coverage
        FROM sepsis_cohort c
        JOIN `{DATASET_ICU}.icustays` i USING (stay_id)
        JOIN `{DATASET_ICU}.inputevents` ie ON ie.stay_id = c.stay_id
        WHERE ie.itemid IN (221906, 222315, 221749, 221289, 221662)
          AND DATETIME_DIFF(DATETIME(ie.starttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND 72
        GROUP BY ie.itemid
        ORDER BY pct_coverage DESC
        """,
    }

    out = {}
    for name, sql in queries.items():
        out[name] = run_df(client, sql).to_dict(orient="records")

    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
