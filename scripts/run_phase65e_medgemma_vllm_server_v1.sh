#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 HOLD_JOB_ID" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
ROOT="${PROJECT_ROOT:-$PWD}"
VENV="${VENV:-${ROOT}/.venv_gemma4_vllm}"
PORT="${PORT:-8057}"
MODEL_REPO="${PHASE65E_MEDGEMMA_MODEL_REPO:-google/medgemma-1.5-4b-it}"
SERVED_MODEL_NAME="${PHASE65E_MEDGEMMA_SERVED_MODEL_NAME:-medgemma-1.5-4b-it-vllm}"
TP_SIZE="${PHASE65E_MEDGEMMA_TP_SIZE:-1}"
GPU_COUNT="${PHASE65E_MEDGEMMA_GPU_COUNT:-${TP_SIZE}}"
MAX_MODEL_LEN="${PHASE65E_MEDGEMMA_MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${PHASE65E_MEDGEMMA_GPU_MEMORY_UTILIZATION:-0.92}"
MAX_NUM_SEQS="${PHASE65E_MEDGEMMA_MAX_NUM_SEQS:-32}"
CPUS_PER_TASK="${PHASE65E_MEDGEMMA_SERVER_CPUS_PER_TASK:-4}"
SERVER_MEM="${PHASE65E_MEDGEMMA_SERVER_MEM:-48G}"
LAUNCH_MODE="${PHASE65E_MEDGEMMA_LAUNCH_MODE:-external}"
DIRECT_IN_JOB="${PHASE65E_MEDGEMMA_DIRECT_IN_JOB:-0}"
OFFLINE_MODE="${PHASE65E_MEDGEMMA_OFFLINE:-1}"
COMPILATION_CONFIG="${PHASE65E_MEDGEMMA_COMPILATION_CONFIG:-{\"pass_config\":{\"fuse_allreduce_rms\":false}}}"

export HF_HOME="${HF_HOME:-${PROJECT_SCRATCH:-${ROOT}/.cache}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"
if [ "${OFFLINE_MODE}" = "1" ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

SERVER_CMD="
cd '${ROOT}'
source '${VENV}/bin/activate'
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ALLREDUCE_USE_FLASHINFER=\${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=\${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-auto}
export HF_HOME='${HF_HOME}'
export TRANSFORMERS_CACHE='${TRANSFORMERS_CACHE}'
export HUGGINGFACE_HUB_CACHE='${HUGGINGFACE_HUB_CACHE}'
export HF_HUB_OFFLINE='${HF_HUB_OFFLINE:-0}'
export TRANSFORMERS_OFFLINE='${TRANSFORMERS_OFFLINE:-0}'
vllm serve '${MODEL_REPO}' \
  --served-model-name '${SERVED_MODEL_NAME}' \
  --host 127.0.0.1 \
  --port '${PORT}' \
  --tensor-parallel-size '${TP_SIZE}' \
  --gpu-memory-utilization '${GPU_MEM_UTIL}' \
  --max-model-len '${MAX_MODEL_LEN}' \
  --max-num-seqs '${MAX_NUM_SEQS}' \
  --compilation-config '${COMPILATION_CONFIG}' \
  --generation-config vllm \
  --enable-prefix-caching
"

if [ "${DIRECT_IN_JOB}" = "1" ]; then
  exec bash -lc "${SERVER_CMD}"
fi

if [ "${LAUNCH_MODE}" = "external" ]; then
  exec srun \
    --jobid="${HOLD_JOB_ID}" \
    --external-launcher \
    bash -lc "${SERVER_CMD}"
fi

exec srun \
  --overlap \
  --jobid="${HOLD_JOB_ID}" \
  --gres="gpu:${GPU_COUNT}" \
  --ntasks=1 \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${SERVER_MEM}" \
  --export=ALL \
  bash -lc "${SERVER_CMD}"
