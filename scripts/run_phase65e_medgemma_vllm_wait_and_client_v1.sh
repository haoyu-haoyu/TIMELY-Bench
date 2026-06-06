#!/bin/bash
set -euo pipefail

if [ $# -lt 4 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH OUTPUT_DIR MAX_WORKERS" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="$3"
MAX_WORKERS="$4"
PORT="${PORT:-8057}"
MAX_POLLS="${MAX_POLLS:-240}"
SLEEP_SECONDS="${SLEEP_SECONDS:-5}"
ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final

cd "${ROOT}"

for ((poll=1; poll<=MAX_POLLS; poll++)); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    echo "[$(date)] medgemma vllm ready on poll=${poll}"
    break
  fi
  if [ "${poll}" -eq "${MAX_POLLS}" ]; then
    echo "[$(date)] medgemma vllm not ready after ${MAX_POLLS} polls" >&2
    exit 1
  fi
  sleep "${SLEEP_SECONDS}"
done

exec env PHASE65E_MEDGEMMA_LAUNCH_MODE=external \
  bash scripts/run_phase65e_medgemma_vllm_client_v1.sh \
    "${HOLD_JOB_ID}" \
    "${MANIFEST_PATH}" \
    "${OUTPUT_DIR}" \
    "${MAX_WORKERS}"
