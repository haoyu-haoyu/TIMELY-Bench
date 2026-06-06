#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/cephfs/volumes/hpc_data_prj/bhi_haoyu_benchmarking/9702e4c9-097c-4b21-8276-01dc96440ad1/TIMELY-Bench_Final}"
RESULTS_ROOT="${RESULTS_ROOT:-/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final/results/cres_v3}"
OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/phase65f_frozen_eval}"

cd "${PROJECT_ROOT}"

python3 code/v3/run_phase65f_frozen_eval_v1.py \
  --mode canonicalize \
  --project-root "${PROJECT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --output-dir "${OUTPUT_DIR}"

python3 code/v3/run_phase65f_frozen_eval_v1.py \
  --mode score \
  --project-root "${PROJECT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-path "${PROJECT_ROOT}/data/processed/v3/cres/cres_eval_sample_12k.parquet" \
  --prompts-path "${PROJECT_ROOT}/data/processed/v3/cres/cres_eval_prompts_12k.jsonl"

python3 code/v3/run_phase65f_frozen_eval_v1.py \
  --mode build_judge \
  --project-root "${PROJECT_ROOT}" \
  --results-root "${RESULTS_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --sample-path "${PROJECT_ROOT}/data/processed/v3/cres/cres_eval_sample_12k.parquet" \
  --prompts-path "${PROJECT_ROOT}/data/processed/v3/cres/cres_eval_prompts_12k.jsonl"
