#!/bin/bash
#SBATCH --job-name=phase65d_deepseek_chat_full
#SBATCH --partition=cpu
#SBATCH --array=0-39%40
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/v3/phase65d_deepseek_full_%A_%a.out
#SBATCH --error=logs/v3/phase65d_deepseek_full_%A_%a.err

set -euo pipefail

cd /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
mkdir -p logs/v3 results/cres_v3/phase65d_tier1b_full

ENV_FILE="${TIER1B_DEEPSEEK_CHAT_ENV_FILE:-}"
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

: "${TIER1B_DEEPSEEK_CHAT_BASE_URL:?set in env file or shell}"
: "${TIER1B_DEEPSEEK_CHAT_API_KEY:?set in env file or shell}"
: "${TIER1B_DEEPSEEK_CHAT_API_MODE:=openai_chat}"
: "${TIER1B_DEEPSEEK_CHAT_ENDPOINT:=/chat/completions}"
: "${TIER1B_DEEPSEEK_CHAT_MODEL_NAME:=deepseek-chat}"

python3 -u code/v3/run_phase65d_tier1b_v3.py \
  --mode run_provider_shard \
  --root . \
  --provider deepseek_chat \
  --model-name "${TIER1B_DEEPSEEK_CHAT_MODEL_NAME}" \
  --manifest-path results/cres_v3/phase65d_tier1b_full/phase65d_full_manifest_full_multimodal.jsonl \
  --num-shards 40 \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --max-workers 1 \
  --temperature 0.0 \
  --max-tokens 2600 \
  --timeout-seconds 180 \
  --max-retries 6
