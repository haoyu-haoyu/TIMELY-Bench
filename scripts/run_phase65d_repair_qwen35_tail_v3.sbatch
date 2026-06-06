#!/bin/bash
#SBATCH --job-name=phase65d_qwen_tail
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/v3/phase65d_qwen_tail_%j.out
#SBATCH --error=logs/v3/phase65d_qwen_tail_%j.err

set -euo pipefail

cd /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
mkdir -p logs/v3 results/cres_v3/phase65d_tier1b_full

ENV_FILE="${TIER1B_QWEN35_ENV_FILE:-}"
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

: "${TIER1B_QWEN35_BASE_URL:?set in env file or shell}"
: "${TIER1B_QWEN35_API_KEY:?set in env file or shell}"
: "${TIER1B_QWEN35_API_MODE:=openai_chat}"
: "${TIER1B_QWEN35_ENDPOINT:=/chat/completions}"
: "${TIER1B_QWEN35_MODEL_NAME:=qwen3.5-flash}"

python3 -u code/v3/run_phase65d_tier1b_v3.py \
  --mode run_provider_shard \
  --root . \
  --provider qwen35 \
  --model-name "${TIER1B_QWEN35_MODEL_NAME}" \
  --manifest-path results/cres_v3/phase65d_tier1b_full/phase65d_qwen35_repair_manifest_tail4.jsonl \
  --num-shards 1 \
  --shard-index 0 \
  --max-workers 1 \
  --temperature 0.0 \
  --max-tokens 20000 \
  --timeout-seconds 300 \
  --max-retries 6
