#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH OUTPUT_DIR RUN_TAG" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="$3"
RUN_TAG="$4"

ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
LOG_DIR="${ROOT}/logs/v3"
PYTHON_BIN="${PYTHON_BIN:-${ROOT}/.venv_gemma4_h200/bin/python}"
BACKEND_MODEL_NAME="${BACKEND_MODEL_NAME:-google/medgemma-1.5-4b-it}"
MODEL_NAME="${MODEL_NAME:-medgemma-1.5-4b-it}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1800}"
MAX_RETRIES="${MAX_RETRIES:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"

: "${HF_TOKEN:?set HF_TOKEN before launch}"

export HF_HOME="${HF_HOME:-/scratch/prj/bhi_haoyu_benchmarking/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${LOG_DIR}" "${ROOT}/${OUTPUT_DIR}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

STDOUT_LOG="${LOG_DIR}/phase65e_medgemma_single_${RUN_TAG}_${HOLD_JOB_ID}.out"
STDERR_LOG="${LOG_DIR}/phase65e_medgemma_single_${RUN_TAG}_${HOLD_JOB_ID}.err"

echo "[$(date)] medgemma single hold run started hold=${HOLD_JOB_ID} tag=${RUN_TAG}"
echo "[$(date)] manifest=${MANIFEST_PATH}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] num_shards=${NUM_SHARDS} shard_index=${SHARD_INDEX} batch_size=${BATCH_SIZE}"

exec srun \
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
      --model-name '${MODEL_NAME}' \
      --backend-model-name '${BACKEND_MODEL_NAME}' \
      --manifest-path '${MANIFEST_PATH}' \
      --output-dir '${OUTPUT_DIR}' \
      --num-shards '${NUM_SHARDS}' \
      --shard-index '${SHARD_INDEX}' \
      --temperature 0.0 \
      --max-new-tokens '${MAX_NEW_TOKENS}' \
      --batch-size '${BATCH_SIZE}' \
      --max-retries '${MAX_RETRIES}'
  " >"${STDOUT_LOG}" 2>"${STDERR_LOG}"
