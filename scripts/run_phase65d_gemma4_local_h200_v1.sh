#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 HOLD_JOB_ID [BACKEND_MODEL_NAME]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
BACKEND_MODEL_NAME="${2:-google/gemma-4-26B-A4B-it}"
ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
OUT_DIR=results/cres_v3/phase65d_tier1b_full
MANIFEST_PATH=${OUT_DIR}/phase65d_full_manifest_full_multimodal.jsonl
LOG_DIR=${ROOT}/logs/v3
PYTHON_BIN="${PYTHON_BIN:-python3}"

: "${HF_TOKEN:?set HF_TOKEN before launch}"

export HF_HOME="${HF_HOME:-/scratch/prj/bhi_haoyu_benchmarking/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${LOG_DIR}" "${ROOT}/${OUT_DIR}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

run_shard() {
  local shard_index="$1"
  local stdout_log="${LOG_DIR}/phase65d_gemma4_local_h200_${HOLD_JOB_ID}_shard${shard_index}.out"
  local stderr_log="${LOG_DIR}/phase65d_gemma4_local_h200_${HOLD_JOB_ID}_shard${shard_index}.err"
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
        --provider gemma4_26b \
        --model-name arc:lite \
        --backend-model-name '${BACKEND_MODEL_NAME}' \
        --manifest-path '${MANIFEST_PATH}' \
        --output-dir '${OUT_DIR}' \
        --num-shards 2 \
        --shard-index '${shard_index}' \
        --temperature 0.0 \
        --max-new-tokens 1200 \
        --max-retries 2 \
        ${GEMMA4_MAX_ROWS:+--max-rows ${GEMMA4_MAX_ROWS}}
    " >"${stdout_log}" 2>"${stderr_log}" &
  LAST_PID=$!
}

run_shard 0
pid0="${LAST_PID}"
run_shard 1
pid1="${LAST_PID}"
echo "launched shard0 pid=${pid0}"
echo "launched shard1 pid=${pid1}"
wait "${pid0}" "${pid1}"
