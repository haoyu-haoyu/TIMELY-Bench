#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH OUTPUT_DIR [MAX_WORKERS]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="$3"
MAX_WORKERS="${4:-1}"
MAX_TOKENS="${MAX_TOKENS:-2600}"
ROOT=/cephfs/volumes/hpc_data_prj/bhi_haoyu_benchmarking/9702e4c9-097c-4b21-8276-01dc96440ad1/TIMELY-Bench_Final
PORT="${PORT:-8027}"
LAUNCH_MODE="${PHASE65E_OPENBIO_LAUNCH_MODE:-overlap}"
DIRECT_IN_JOB="${PHASE65E_OPENBIO_DIRECT_IN_JOB:-0}"

CLIENT_CMD="
cd '${ROOT}'
mkdir -p '${OUTPUT_DIR}'
export TIER2_OPENBIOLLM70B_BASE_URL='http://127.0.0.1:${PORT}/v1'
export TIER2_OPENBIOLLM70B_API_KEY='EMPTY'
export TIER2_OPENBIOLLM70B_API_MODE='openai_chat'
export TIER2_OPENBIOLLM70B_ENDPOINT='/chat/completions'
export TIER2_OPENBIOLLM70B_MODEL_NAME='openbio-70b'
export TIER2_OPENBIOLLM70B_EXTRA_BODY_JSON='{\"response_format\":{\"type\":\"json_object\"}}'
export TIER2_OPENBIOLLM70B_USE_JSON_SYSTEM_PROMPT='1'
python3 -u code/v3/run_phase65e_tier2_v1.py \
  --mode run_provider_shard \
  --root . \
  --output-dir '${OUTPUT_DIR}' \
  --provider openbiollm70b \
  --model-name openbio-70b \
  --manifest-path '${MANIFEST_PATH}' \
  --num-shards 1 \
  --shard-index 0 \
  --max-workers '${MAX_WORKERS}' \
  --temperature 0.0 \
  --max-tokens '${MAX_TOKENS}' \
  --timeout-seconds 300 \
  --max-retries 6
"

if [ "${DIRECT_IN_JOB}" = "1" ]; then
  exec bash -lc "${CLIENT_CMD}"
fi

if [ "${LAUNCH_MODE}" = "external" ]; then
  exec srun \
    --jobid="${HOLD_JOB_ID}" \
    --external-launcher \
    bash -lc "${CLIENT_CMD}"
fi

exec srun \
  --overlap \
  --jobid="${HOLD_JOB_ID}" \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export=ALL \
  bash -lc "${CLIENT_CMD}"
