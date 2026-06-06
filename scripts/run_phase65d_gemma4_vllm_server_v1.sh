#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 HOLD_JOB_ID" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
VENV="${ROOT}/.venv_gemma4_vllm"
PORT="${PORT:-8017}"
DIRECT_IN_JOB="${PHASE65D_VLLM_DIRECT_IN_JOB:-0}"
LAUNCH_MODE="${PHASE65D_VLLM_LAUNCH_MODE:-overlap}"
TP_SIZE="${PHASE65D_VLLM_TP_SIZE:-2}"
GPU_COUNT="${PHASE65D_VLLM_GPU_COUNT:-${TP_SIZE}}"
CUDA_VISIBLE_DEVICES_OVERRIDE="${PHASE65D_VLLM_CUDA_VISIBLE_DEVICES:-}"
MAX_MODEL_LEN="${PHASE65D_VLLM_MAX_MODEL_LEN:-16384}"

: "${HF_TOKEN:?set HF_TOKEN before launch}"

export HF_HOME="${HF_HOME:-/scratch/prj/bhi_haoyu_benchmarking/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"

if [ "${DIRECT_IN_JOB}" = "1" ]; then
  cd "${ROOT}"
  source "${VENV}/bin/activate"
  if [ -n "${CUDA_VISIBLE_DEVICES_OVERRIDE}" ]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES_OVERRIDE}"
  elif [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    if [ "${GPU_COUNT}" = "1" ]; then
      export CUDA_VISIBLE_DEVICES=0
    else
      export CUDA_VISIBLE_DEVICES=0,1
    fi
  fi
  exec vllm serve google/gemma-4-26B-A4B-it \
    --served-model-name arc:lite \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --generation-config vllm \
    --enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 0, "audio": 0}'
fi

if [ "${LAUNCH_MODE}" = "external" ]; then
  exec srun \
    --jobid="${HOLD_JOB_ID}" \
    --external-launcher \
    bash -lc "
      cd '${ROOT}'
      source '${VENV}/bin/activate'
      if [ -n '${CUDA_VISIBLE_DEVICES_OVERRIDE}' ]; then
        export CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES_OVERRIDE}'
      elif [ -z \"\${CUDA_VISIBLE_DEVICES:-}\" ]; then
        if [ '${GPU_COUNT}' = '1' ]; then
          export CUDA_VISIBLE_DEVICES=0
        else
          export CUDA_VISIBLE_DEVICES=0,1
        fi
      fi
      vllm serve google/gemma-4-26B-A4B-it \
        --served-model-name arc:lite \
        --host 127.0.0.1 \
        --port '${PORT}' \
        --tensor-parallel-size '${TP_SIZE}' \
        --gpu-memory-utilization 0.90 \
        --max-model-len '${MAX_MODEL_LEN}' \
        --generation-config vllm \
        --enable-prefix-caching \
        --limit-mm-per-prompt '{\"image\": 0, \"audio\": 0}'
    "
fi

exec srun \
  --overlap \
  --jobid="${HOLD_JOB_ID}" \
  --ntasks=1 \
  --gres="gpu:${GPU_COUNT}" \
  --cpus-per-task=4 \
  --mem=96G \
  --export=ALL \
  bash -lc "
    cd '${ROOT}'
    source '${VENV}/bin/activate'
    if [ -z \"\${CUDA_VISIBLE_DEVICES:-}\" ]; then
      if [ '${GPU_COUNT}' = '1' ]; then
        export CUDA_VISIBLE_DEVICES=0
      else
        export CUDA_VISIBLE_DEVICES=0,1
      fi
    fi
    vllm serve google/gemma-4-26B-A4B-it \
      --served-model-name arc:lite \
      --host 127.0.0.1 \
      --port '${PORT}' \
      --tensor-parallel-size '${TP_SIZE}' \
      --gpu-memory-utilization 0.90 \
      --max-model-len '${MAX_MODEL_LEN}' \
      --generation-config vllm \
      --enable-prefix-caching \
      --limit-mm-per-prompt '{\"image\": 0, \"audio\": 0}'
  "
