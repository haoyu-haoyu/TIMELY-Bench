#!/bin/bash
#SBATCH --job-name=phase65c_gemini31pro_full
#SBATCH --partition=cpu
#SBATCH --array=0-19%20
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/v3/phase65c_gemini31pro_full_%A_%a.out
#SBATCH --error=logs/v3/phase65c_gemini31pro_full_%A_%a.err

set -euo pipefail

cd /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
mkdir -p logs/v3 results/cres_v3/phase65c_tier1a_full

export IKUNCODE_BASE_URL="https://www.vibeapi.cn"
export IKUNCODE_API_KEY="$(cat /users/k25113331/.phase65c_vibeapi_gemini.key)"

python3 -u code/v3/run_phase65c_tier1a_full_v3.py \
  --mode run_provider_shard \
  --root . \
  --output-dir results/cres_v3/phase65c_tier1a_full \
  --provider gemini31pro \
  --model-name gemini-3.1-pro \
  --variant-id full_multimodal \
  --num-shards 20 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --max-workers 1 \
  --temperature 0.0 \
  --max-tokens 1200 \
  --timeout-seconds 180 \
  --max-retries 6
