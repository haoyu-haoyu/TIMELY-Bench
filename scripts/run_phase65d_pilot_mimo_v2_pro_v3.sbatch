#!/bin/bash
#SBATCH --job-name=phase65d_gemma4_pilot
#SBATCH --partition=cpu
#SBATCH --array=0-0%1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3/phase65d_gemma4_pilot_%A_%a.out
#SBATCH --error=logs/v3/phase65d_gemma4_pilot_%A_%a.err

set -euo pipefail

cd /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
mkdir -p logs/v3 results/cres_v3/phase65d_tier1b_pilot

ENV_FILE=/users/k25113331/.phase65d_gemma4_26b.env
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

: "${TIER1B_GEMMA4_26B_BASE_URL:?set in env file or shell}"
: "${TIER1B_GEMMA4_26B_API_KEY:?set in env file or shell}"
: "${TIER1B_GEMMA4_26B_API_MODE:=openai_chat}"
: "${TIER1B_GEMMA4_26B_ENDPOINT:=/v1/chat/completions}"
: "${TIER1B_GEMMA4_26B_MODEL_NAME:=arc:lite}"

python3 -u code/v3/run_phase65d_tier1b_v3.py \
  --mode run_provider_shard \
  --root . \
  --provider gemma4_26b \
  --model-name "${TIER1B_GEMMA4_26B_MODEL_NAME}" \
  --manifest-path results/cres_v3/phase65d_tier1b_pilot/phase65d_pilot100_manifest.jsonl \
  --num-shards 1 \
  --shard-index 0 \
  --max-workers 1 \
  --temperature 0.0 \
  --max-tokens 1200 \
  --timeout-seconds 180 \
  --max-retries 6
