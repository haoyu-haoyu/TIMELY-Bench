#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH [MAX_WORKERS]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
MAX_WORKERS="${3:-16}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
ROOT="${PROJECT_ROOT:-$PWD}"
PORT="${PORT:-8017}"
DIRECT_IN_JOB="${PHASE65D_VLLM_DIRECT_IN_JOB:-0}"
LAUNCH_MODE="${PHASE65D_VLLM_LAUNCH_MODE:-overlap}"

if [ "${DIRECT_IN_JOB}" = "1" ]; then
  cd "${ROOT}"
  export TIER1B_GEMMA4_26B_BASE_URL="http://127.0.0.1:${PORT}/v1"
  export TIER1B_GEMMA4_26B_API_KEY='EMPTY'
  export TIER1B_GEMMA4_26B_API_MODE='openai_chat'
  export TIER1B_GEMMA4_26B_ENDPOINT='/chat/completions'
  export TIER1B_GEMMA4_26B_MODEL_NAME='arc:lite'
  export TIER1B_GEMMA4_26B_EXTRA_BODY_JSON='{"chat_template_kwargs":{"enable_thinking":false},"response_format":{"type":"json_object"}}'
  exec python3 -u code/v3/run_phase65d_tier1b_v3.py \
    --mode run_provider_shard \
    --root . \
    --provider gemma4_26b \
    --model-name arc:lite \
    --manifest-path "${MANIFEST_PATH}" \
    --num-shards 1 \
    --shard-index 0 \
    --max-workers "${MAX_WORKERS}" \
    --temperature 0.0 \
    --max-tokens "${MAX_TOKENS}" \
    --timeout-seconds 300 \
    --max-retries 6
fi

if [ "${LAUNCH_MODE}" = "external" ]; then
  exec srun \
    --jobid="${HOLD_JOB_ID}" \
    --external-launcher \
    bash -lc "
      cd '${ROOT}'
      export TIER1B_GEMMA4_26B_BASE_URL='http://127.0.0.1:${PORT}/v1'
      export TIER1B_GEMMA4_26B_API_KEY='EMPTY'
      export TIER1B_GEMMA4_26B_API_MODE='openai_chat'
      export TIER1B_GEMMA4_26B_ENDPOINT='/chat/completions'
      export TIER1B_GEMMA4_26B_MODEL_NAME='arc:lite'
      export TIER1B_GEMMA4_26B_EXTRA_BODY_JSON='{\"chat_template_kwargs\":{\"enable_thinking\":false},\"response_format\":{\"type\":\"json_object\"}}'
      python3 -u code/v3/run_phase65d_tier1b_v3.py \
        --mode run_provider_shard \
        --root . \
        --provider gemma4_26b \
        --model-name arc:lite \
        --manifest-path '${MANIFEST_PATH}' \
        --num-shards 1 \
        --shard-index 0 \
        --max-workers '${MAX_WORKERS}' \
        --temperature 0.0 \
        --max-tokens '${MAX_TOKENS}' \
        --timeout-seconds 300 \
        --max-retries 6
    "
fi

exec srun \
  --overlap \
  --jobid="${HOLD_JOB_ID}" \
  --ntasks=1 \
  --cpus-per-task=4 \
  --mem=16G \
  --export=ALL \
  bash -lc "
    cd '${ROOT}'
    export TIER1B_GEMMA4_26B_BASE_URL='http://127.0.0.1:${PORT}/v1'
    export TIER1B_GEMMA4_26B_API_KEY='EMPTY'
    export TIER1B_GEMMA4_26B_API_MODE='openai_chat'
    export TIER1B_GEMMA4_26B_ENDPOINT='/chat/completions'
    export TIER1B_GEMMA4_26B_MODEL_NAME='arc:lite'
    export TIER1B_GEMMA4_26B_EXTRA_BODY_JSON='{\"chat_template_kwargs\":{\"enable_thinking\":false},\"response_format\":{\"type\":\"json_object\"}}'
    python3 -u code/v3/run_phase65d_tier1b_v3.py \
      --mode run_provider_shard \
      --root . \
      --provider gemma4_26b \
      --model-name arc:lite \
      --manifest-path '${MANIFEST_PATH}' \
      --num-shards 1 \
      --shard-index 0 \
      --max-workers '${MAX_WORKERS}' \
      --temperature 0.0 \
      --max-tokens '${MAX_TOKENS}' \
      --timeout-seconds 300 \
      --max-retries 6
  "
