#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH OUTPUT_DIR [MAX_WORKERS]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="$3"
MAX_WORKERS="${4:-12}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-300}"
MAX_RETRIES="${MAX_RETRIES:-6}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
ROOT=/cephfs/volumes/hpc_data_prj/bhi_haoyu_benchmarking/9702e4c9-097c-4b21-8276-01dc96440ad1/TIMELY-Bench_Final
PORT="${PORT:-8057}"
LAUNCH_MODE="${PHASE65E_MEDITRON3_LAUNCH_MODE:-overlap}"
DIRECT_IN_JOB="${PHASE65E_MEDITRON3_DIRECT_IN_JOB:-0}"
API_MODE="${TIER2_MEDITRON3_8B_API_MODE:-openai_chat}"
ENDPOINT="${TIER2_MEDITRON3_8B_ENDPOINT:-/chat/completions}"
MODEL_NAME="${TIER2_MEDITRON3_8B_MODEL_NAME:-meditron3-8b}"
EXTRA_BODY_JSON="${TIER2_MEDITRON3_8B_EXTRA_BODY_JSON:-{\"response_format\":{\"type\":\"json_object\"}}}"
USE_JSON_SYSTEM_PROMPT="${TIER2_MEDITRON3_8B_USE_JSON_SYSTEM_PROMPT:-1}"

export TIER2_MEDITRON3_8B_API_MODE="${API_MODE}"
export TIER2_MEDITRON3_8B_ENDPOINT="${ENDPOINT}"
export TIER2_MEDITRON3_8B_MODEL_NAME="${MODEL_NAME}"
export TIER2_MEDITRON3_8B_EXTRA_BODY_JSON="${EXTRA_BODY_JSON}"
export TIER2_MEDITRON3_8B_USE_JSON_SYSTEM_PROMPT="${USE_JSON_SYSTEM_PROMPT}"

CLIENT_CMD="
cd '${ROOT}'
mkdir -p '${OUTPUT_DIR}'
export TIER2_MEDITRON3_8B_BASE_URL='http://127.0.0.1:${PORT}/v1'
export TIER2_MEDITRON3_8B_API_KEY='EMPTY'
python3 -u code/v3/run_phase65e_tier2_v1.py \
  --mode run_provider_shard \
  --root . \
  --output-dir '${OUTPUT_DIR}' \
  --provider meditron3_8b \
  --model-name '${MODEL_NAME}' \
  --manifest-path '${MANIFEST_PATH}' \
  --num-shards '${NUM_SHARDS}' \
  --shard-index '${SHARD_INDEX}' \
  --max-workers '${MAX_WORKERS}' \
  --temperature 0.0 \
  --max-tokens '${MAX_TOKENS}' \
  --timeout-seconds '${TIMEOUT_SECONDS}' \
  --max-retries '${MAX_RETRIES}'
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
