#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH [OUTPUT_DIR]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="${3:-results/cres_v3/phase65e_tier2_medgemma_vllm_pilot100}"

ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
LOG_DIR="${ROOT}/logs/v3"
PORT="${PORT:-8057}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-18000}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-5}"
READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS:-1800}"
STEP_CPUS_PER_TASK="${STEP_CPUS_PER_TASK:-8}"
STEP_MEM="${STEP_MEM:-64G}"
MANIFEST_SUMMARY_PATH="${OUTPUT_DIR}/medgemma15_4b_manifest_summary.json"
INNER_FLAG="${PHASE65E_MEDGEMMA_VLLM_INNER:-0}"
SELF_PATH="${ROOT}/scripts/$(basename "$0")"
LOG_TAG_RAW="${PHASE65E_MEDGEMMA_LOG_TAG:-$(basename "${OUTPUT_DIR}")_p${PORT}}"
LOG_TAG="$(printf '%s' "${LOG_TAG_RAW}" | tr '/: ' '___')"
SERVER_LOG="${LOG_DIR}/phase65e_medgemma_vllm_server_${LOG_TAG}.out"
CLIENT_LOG="${LOG_DIR}/phase65e_medgemma_vllm_client_${LOG_TAG}.out"

if [ "${INNER_FLAG}" != "1" ]; then
  exec srun \
    --overlap \
    --jobid="${HOLD_JOB_ID}" \
    --gres="gpu:1" \
    --ntasks=1 \
    --cpus-per-task="${STEP_CPUS_PER_TASK}" \
    --mem="${STEP_MEM}" \
    --export=ALL,PHASE65E_MEDGEMMA_VLLM_INNER=1,PORT="${PORT}",MAX_WORKERS="${MAX_WORKERS}",MAX_TOKENS="${MAX_TOKENS}",TIMEOUT_SECONDS="${TIMEOUT_SECONDS}",READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS}",READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS}",STEP_CPUS_PER_TASK="${STEP_CPUS_PER_TASK}",STEP_MEM="${STEP_MEM}" \
    bash "${SELF_PATH}" "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}"
fi

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[$(date)] medgemma vllm pilot started on hold ${HOLD_JOB_ID}"
echo "[$(date)] manifest=${MANIFEST_PATH}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] port=${PORT} max_workers=${MAX_WORKERS} max_tokens=${MAX_TOKENS} timeout_seconds=${TIMEOUT_SECONDS}"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PHASE65E_MEDGEMMA_DIRECT_IN_JOB=1 \
PHASE65E_MEDGEMMA_TP_SIZE=1 \
PHASE65E_MEDGEMMA_GPU_COUNT=1 \
PHASE65E_MEDGEMMA_MAX_MODEL_LEN="${PHASE65E_MEDGEMMA_MAX_MODEL_LEN:-32768}" \
PHASE65E_MEDGEMMA_GPU_MEMORY_UTILIZATION="${PHASE65E_MEDGEMMA_GPU_MEMORY_UTILIZATION:-0.92}" \
PHASE65E_MEDGEMMA_MAX_NUM_SEQS="${PHASE65E_MEDGEMMA_MAX_NUM_SEQS:-16}" \
PORT="${PORT}" \
bash scripts/run_phase65e_medgemma_vllm_server_v1.sh "${HOLD_JOB_ID}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
max_attempts=$(( (READINESS_MAX_WAIT_SECONDS + READINESS_POLL_SECONDS - 1) / READINESS_POLL_SECONDS ))
for attempt in $(seq 1 "${max_attempts}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    echo "[$(date)] MedGemma vLLM server ready after ${attempt} polls"
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[$(date)] MedGemma vLLM server exited before readiness"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep "${READINESS_POLL_SECONDS}"
done

if [ "${ready}" != "1" ]; then
  echo "[$(date)] MedGemma vLLM server readiness timeout"
  tail -n 120 "${SERVER_LOG}" || true
  exit 1
fi

PHASE65E_MEDGEMMA_DIRECT_IN_JOB=1 \
PORT="${PORT}" \
MAX_TOKENS="${MAX_TOKENS}" \
TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
bash scripts/run_phase65e_medgemma_vllm_client_v1.sh "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}" "${MAX_WORKERS}" >"${CLIENT_LOG}" 2>&1

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize_manifest_subset \
  --root . \
  --output-dir "${OUTPUT_DIR}" \
  --provider medgemma15_4b \
  --model-name medgemma-1.5-4b-it-vllm \
  --manifest-path "${MANIFEST_PATH}" \
  --summary-path "${MANIFEST_SUMMARY_PATH}"

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize \
  --root . \
  --output-dir "${OUTPUT_DIR}" \
  --providers medgemma15_4b

echo "[$(date)] medgemma vllm pilot completed"
