#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 HOLD_JOB_ID [BACKEND_MODEL_NAME]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
BACKEND_MODEL_NAME="${2:-google/medgemma-1.5-4b-it}"
ROOT="${PROJECT_ROOT:-$PWD}"
OUT_DIR="${OUT_DIR:-results/cres_v3/phase65e_tier2_full}"
MANIFEST_PATH="${MANIFEST_PATH:-results/cres_v3/phase65d_tier1b_full/phase65d_full_manifest_full_multimodal.jsonl}"
LOG_DIR="${ROOT}/logs/v3"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv_gemma4_h200/bin/python}"
NUM_SHARDS="${NUM_SHARDS:-2}"
SHARD_OFFSET="${SHARD_OFFSET:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1800}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MEDGEMMA_MODEL_NAME="${MEDGEMMA_MODEL_NAME:-medgemma-1.5-4b-it}"
MEDGEMMA_MAX_ROWS="${MEDGEMMA_MAX_ROWS:-0}"
SKIP_SUMMARY="${SKIP_SUMMARY:-0}"

: "${HF_TOKEN:?set HF_TOKEN before launch}"

export HF_HOME="${HF_HOME:-${PROJECT_SCRATCH:-${ROOT}/.cache}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${LOG_DIR}" "${ROOT}/${OUT_DIR}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

run_shard() {
  local local_shard_index="$1"
  local global_shard_index=$((SHARD_OFFSET + local_shard_index))
  local stdout_log="${LOG_DIR}/phase65e_medgemma_local_h200_${HOLD_JOB_ID}_shard${global_shard_index}.out"
  local stderr_log="${LOG_DIR}/phase65e_medgemma_local_h200_${HOLD_JOB_ID}_shard${global_shard_index}.err"
  srun \
    --overlap \
    --jobid="${HOLD_JOB_ID}" \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task=4 \
    --mem=64G \
    --export=ALL \
    bash -lc "
      cd '${ROOT}'
      '${PYTHON_BIN}' -u code/v3/run_phase65d_gemma4_local_h200_v1.py \
        --root . \
        --provider medgemma15_4b \
        --model-name '${MEDGEMMA_MODEL_NAME}' \
        --backend-model-name '${BACKEND_MODEL_NAME}' \
        --manifest-path '${MANIFEST_PATH}' \
        --output-dir '${OUT_DIR}' \
        --num-shards '${NUM_SHARDS}' \
        --shard-index '${global_shard_index}' \
        --temperature 0.0 \
        --max-new-tokens '${MAX_NEW_TOKENS}' \
        --batch-size '${BATCH_SIZE}' \
        --max-retries 2 \
        ${MEDGEMMA_MAX_ROWS:+--max-rows ${MEDGEMMA_MAX_ROWS}}
    " >"${stdout_log}" 2>"${stderr_log}" &
  LAST_PID=$!
}

run_shard 0
pid0="${LAST_PID}"
run_shard 1
pid1="${LAST_PID}"
echo "launched medgemma local_shard0 global_shard=$((SHARD_OFFSET + 0)) pid=${pid0}"
echo "launched medgemma local_shard1 global_shard=$((SHARD_OFFSET + 1)) pid=${pid1}"
wait "${pid0}" "${pid1}"

if [ "${SKIP_SUMMARY}" != "1" ]; then
  bash -lc "
    cd '${ROOT}'
    python3 code/v3/run_phase65e_tier2_v1.py \
      --mode summarize \
      --root . \
      --output-dir '${OUT_DIR}' \
      --providers medgemma15_4b
  "
fi
