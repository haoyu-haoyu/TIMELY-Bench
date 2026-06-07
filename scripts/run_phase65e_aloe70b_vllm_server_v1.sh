#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 HOLD_JOB_ID" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
ROOT="${PROJECT_ROOT:-$PWD}"
VENV="${VENV:-${ROOT}/.venv_gemma4_vllm}"
PORT="${PORT:-8047}"
MODEL_REPO="${PHASE65E_ALOE70B_MODEL_REPO:-HPAI-BSC/Llama3.1-Aloe-Beta-70B}"
SERVED_MODEL_NAME="${PHASE65E_ALOE70B_SERVED_MODEL_NAME:-llama31-aloe-beta-70b}"
TP_SIZE="${PHASE65E_ALOE70B_TP_SIZE:-2}"
GPU_COUNT="${PHASE65E_ALOE70B_GPU_COUNT:-${TP_SIZE}}"
MAX_MODEL_LEN="${PHASE65E_ALOE70B_MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${PHASE65E_ALOE70B_GPU_MEMORY_UTILIZATION:-0.94}"
LAUNCH_MODE="${PHASE65E_ALOE70B_LAUNCH_MODE:-overlap}"
DIRECT_IN_JOB="${PHASE65E_ALOE70B_DIRECT_IN_JOB:-0}"
COMPILATION_CONFIG="${PHASE65E_ALOE70B_COMPILATION_CONFIG:-{\"pass_config\":{\"fuse_allreduce_rms\":false}}}"

export HF_HOME="${HF_HOME:-${PROJECT_SCRATCH:-${ROOT}/.cache}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

SERVER_CMD="
cd '${ROOT}'
source '${VENV}/bin/activate'
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-0,1}
export VLLM_ALLREDUCE_USE_FLASHINFER=\${VLLM_ALLREDUCE_USE_FLASHINFER:-0}
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=\${VLLM_FLASHINFER_ALLREDUCE_BACKEND:-auto}
vllm serve '${MODEL_REPO}' \
  --served-model-name '${SERVED_MODEL_NAME}' \
  --host 127.0.0.1 \
  --port '${PORT}' \
  --tensor-parallel-size '${TP_SIZE}' \
  --gpu-memory-utilization '${GPU_MEM_UTIL}' \
  --max-model-len '${MAX_MODEL_LEN}' \
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
  --cpus-per-task=8 \
  --mem=128G \
  --export=ALL \
  bash -lc "${SERVER_CMD}"
