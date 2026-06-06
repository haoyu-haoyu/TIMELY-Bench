#!/bin/bash
#SBATCH --job-name=phase65c_full_merge
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3/phase65c_full_merge_%j.out
#SBATCH --error=logs/v3/phase65c_full_merge_%j.err

set -euo pipefail

cd /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
mkdir -p logs/v3 results/cres_v3/phase65c_tier1a_full

python3 -u code/v3/run_phase65c_tier1a_full_v3.py \
  --mode summarize \
  --root . \
  --output-dir results/cres_v3/phase65c_tier1a_full \
  --variant-id full_multimodal \
  --num-shards 8
