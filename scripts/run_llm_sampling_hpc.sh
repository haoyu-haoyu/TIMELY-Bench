#!/usr/bin/env bash
set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

python3 code/data_processing/build_llm_annotation_set.py \
  --use-pandas \
  --max-rows 0 \
  --max-chunks 50 \
  --chunk-size 50000 \
  --n-per-stratum 50 \
  > logs/llm_sampling_run.log 2>&1
