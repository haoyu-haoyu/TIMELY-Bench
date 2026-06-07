Task A/B Phase 0 Feasibility Report

Environment
- BigQuery CLI binaries (bq/gcloud): not installed on HPC PATH
- Python BigQuery client: available and working
- ADC file: /users/k25113331/.config/gcloud/application_default_credentials.json

Q0.1 sepsis3 table exists
- Planned path timely-bench-mimic.mimiciv_3_1_derived.sepsis3: NOT FOUND (dataset missing)
- Fallback path physionet-data.mimiciv_3_1_derived.sepsis3: EXISTS, rows=32899

Q0.2 kdigo_stages schema
- Fallback table physionet-data.mimiciv_3_1_derived.kdigo_stages exists
- Key columns found:
  stage column: aki_stage
  time column: charttime
  stay_id column: stay_id
- Total rows: 5099899

Q0.3 AKI cohort size (stage>=1 within 0-48h)
- Query source: physionet-data.mimiciv_3_1_derived.kdigo_stages
- aki_stage1_stays = 55285
- min_onset_hour = 0
- max_onset_hour = 48

Q0.4 Sepsis cohort size (sofa_time within 0-48h)
- Query source: physionet-data.mimiciv_3_1_derived.sepsis3
- sepsis_stays = 32042

Q0.5 MBP/vasopressor coverage on HPC timeseries
- File used: ${PROJECT_ROOT}/data/processed/timeseries_sorted_72h.csv
- Required columns present: mbp, vasopressors, vasopressor_dose_norepi_equiv
- Sample coverage (n=50000):
  mbp = 0.7045
  vasopressors = 1.0000
  vasopressor_dose_norepi_equiv = 0.04098

Gate Decision
- Phase 0 gate status: PASS (no hard STOP trigger)
- Required plan correction before Phase A:
  Replace all references of your_project.mimiciv_3_1_derived with physionet-data.mimiciv_3_1_derived,
  or parameterize dataset prefix.
