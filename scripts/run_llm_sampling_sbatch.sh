#!/bin/bash
#SBATCH --job-name=llm_sampling
#SBATCH --output=logs/llm_sampling_%j.out
#SBATCH --error=logs/llm_sampling_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

ROOT_DIR="/scratch/users/k25113331/TIMELY-Bench_Final"
cd "${ROOT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
python3 -u code/data_processing/build_llm_annotation_set.py \
  --use-pandas \
  --max-rows 0 \
  --max-chunks 50 \
  --chunk-size 50000 \
  --n-per-stratum 50
