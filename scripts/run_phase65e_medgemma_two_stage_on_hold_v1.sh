#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH OUTPUT_DIR" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="$3"

ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
LOG_DIR="${ROOT}/logs/v3"
PORT="${PORT:-8057}"
MAX_WORKERS="${MAX_WORKERS:-1}"
STAGE_A_MAX_TOKENS="${STAGE_A_MAX_TOKENS:-900}"
STAGE_B_MAX_TOKENS="${STAGE_B_MAX_TOKENS:-320}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-5}"
READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS:-1800}"
STEP_CPUS_PER_TASK="${STEP_CPUS_PER_TASK:-8}"
STEP_MEM="${STEP_MEM:-64G}"
INNER_FLAG="${PHASE65E_MEDGEMMA_VLLM_INNER:-0}"
SELF_PATH="${ROOT}/scripts/$(basename "$0")"
LOG_TAG_RAW="${PHASE65E_MEDGEMMA_LOG_TAG:-$(basename "${OUTPUT_DIR}")_p${PORT}}"
LOG_TAG="$(printf '%s' "${LOG_TAG_RAW}" | tr '/: ' '___')"
SERVER_LOG="${LOG_DIR}/phase65e_medgemma_vllm_server_${LOG_TAG}.out"
RUN_LOG="${LOG_DIR}/phase65e_medgemma_twostage_${LOG_TAG}.out"
OUTPUT_PATH="${OUTPUT_DIR}/medgemma15_4b_two_stage.jsonl"

if [ "${INNER_FLAG}" != "1" ]; then
  exec srun \
    --overlap \
    --jobid="${HOLD_JOB_ID}" \
    --ntasks=1 \
    --cpus-per-task="${STEP_CPUS_PER_TASK}" \
    --mem="${STEP_MEM}" \
    --export=ALL,PHASE65E_MEDGEMMA_VLLM_INNER=1,PORT="${PORT}",MAX_WORKERS="${MAX_WORKERS}",STAGE_A_MAX_TOKENS="${STAGE_A_MAX_TOKENS}",STAGE_B_MAX_TOKENS="${STAGE_B_MAX_TOKENS}",TIMEOUT_SECONDS="${TIMEOUT_SECONDS}",READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS}",READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS}",STEP_CPUS_PER_TASK="${STEP_CPUS_PER_TASK}",STEP_MEM="${STEP_MEM}" \
    bash "${SELF_PATH}" "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}"
fi

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[$(date)] medgemma two-stage repair started on hold ${HOLD_JOB_ID}"
echo "[$(date)] manifest=${MANIFEST_PATH}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] port=${PORT} workers=${MAX_WORKERS} stage_a_max_tokens=${STAGE_A_MAX_TOKENS} stage_b_max_tokens=${STAGE_B_MAX_TOKENS}"

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
PHASE65E_MEDGEMMA_MAX_MODEL_LEN="${PHASE65E_MEDGEMMA_MAX_MODEL_LEN:-65536}" \
PHASE65E_MEDGEMMA_GPU_MEMORY_UTILIZATION="${PHASE65E_MEDGEMMA_GPU_MEMORY_UTILIZATION:-0.65}" \
PHASE65E_MEDGEMMA_MAX_NUM_SEQS="${PHASE65E_MEDGEMMA_MAX_NUM_SEQS:-1}" \
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

export TIER2_MEDGEMMA15_4B_BASE_URL="http://127.0.0.1:${PORT}/v1"
export TIER2_MEDGEMMA15_4B_API_KEY="EMPTY"
export TIER2_MEDGEMMA15_4B_API_MODE="openai_chat"
export TIER2_MEDGEMMA15_4B_ENDPOINT="/chat/completions"
export TIER2_MEDGEMMA15_4B_MODEL_NAME="medgemma-1.5-4b-it-vllm"
export TIER2_MEDGEMMA15_4B_EXTRA_BODY_JSON='{"response_format":{"type":"json_object"}}'
export TIER2_MEDGEMMA15_4B_USE_JSON_SYSTEM_PROMPT="1"
export TIER2_MEDGEMMA15_4B_JSON_SYSTEM_PROMPT="Return exactly one valid JSON object and nothing else."

python3 -u code/v3/run_phase65e_medgemma_two_stage_repair_v1.py \
  --manifest-path "${MANIFEST_PATH}" \
  --output-path "${OUTPUT_PATH}" \
  --provider medgemma15_4b \
  --model-name medgemma-1.5-4b-it-vllm \
  --temperature 0.0 \
  --stage-a-max-tokens "${STAGE_A_MAX_TOKENS}" \
  --stage-b-max-tokens "${STAGE_B_MAX_TOKENS}" \
  --timeout-seconds "${TIMEOUT_SECONDS}" \
  --max-workers "${MAX_WORKERS}" \
  --max-retries 4 >"${RUN_LOG}" 2>&1

echo "[$(date)] medgemma two-stage repair completed"
