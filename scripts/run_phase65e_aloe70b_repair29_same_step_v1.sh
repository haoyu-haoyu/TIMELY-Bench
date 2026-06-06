#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH [OUTPUT_DIR]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="${3:-results/cres_v3/phase65e_tier2_full_aloe70b}"

ROOT=/cephfs/volumes/hpc_data_prj/bhi_haoyu_benchmarking/9702e4c9-097c-4b21-8276-01dc96440ad1/TIMELY-Bench_Final
LOG_DIR="${ROOT}/logs/v3"
PORT="${PORT:-8048}"
MAX_WORKERS="${MAX_WORKERS:-2}"
MAX_TOKENS="${MAX_TOKENS:-18000}"
MAX_RETRIES="${MAX_RETRIES:-6}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-5}"
READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS:-3600}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"
SERVER_LOG="${LOG_DIR}/phase65e_aloe70b_repair29_server_${HOLD_JOB_ID}.out"
CLIENT_LOG="${LOG_DIR}/phase65e_aloe70b_repair29_client_${HOLD_JOB_ID}.out"
MANIFEST_SUMMARY_PATH="${OUTPUT_DIR}/aloe70b_repair29_manifest_summary.json"
INNER_FLAG="${PHASE65E_ALOE70B_REPAIR29_INNER:-0}"
SELF_PATH="${ROOT}/scripts/$(basename "$0")"

if [ "${INNER_FLAG}" != "1" ]; then
  exec srun \
    --overlap \
    --jobid="${HOLD_JOB_ID}" \
    --gres="gpu:2" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=120G \
    --export=ALL,PHASE65E_ALOE70B_REPAIR29_INNER=1,PORT="${PORT}",MAX_WORKERS="${MAX_WORKERS}",MAX_TOKENS="${MAX_TOKENS}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${SHARD_INDEX}",READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS}",READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS}",TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
    bash "${SELF_PATH}" "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}"
fi

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[$(date)] aloe70b repair29 same-step started on hold ${HOLD_JOB_ID}"
echo "[$(date)] manifest=${MANIFEST_PATH}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] port=${PORT} max_workers=${MAX_WORKERS} max_tokens=${MAX_TOKENS} timeout_seconds=${TIMEOUT_SECONDS} max_retries=${MAX_RETRIES}"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PHASE65E_ALOE70B_DIRECT_IN_JOB=1 \
PHASE65E_ALOE70B_TP_SIZE=2 \
PHASE65E_ALOE70B_GPU_COUNT=2 \
PHASE65E_ALOE70B_MAX_MODEL_LEN="${PHASE65E_ALOE70B_MAX_MODEL_LEN:-32768}" \
PORT="${PORT}" \
bash scripts/run_phase65e_aloe70b_vllm_server_v1.sh "${HOLD_JOB_ID}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
max_attempts=$(( (READINESS_MAX_WAIT_SECONDS + READINESS_POLL_SECONDS - 1) / READINESS_POLL_SECONDS ))
for attempt in $(seq 1 "${max_attempts}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    echo "[$(date)] Aloe70B repair29 server ready after ${attempt} polls"
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[$(date)] Aloe70B repair29 server exited before readiness"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep "${READINESS_POLL_SECONDS}"
done

if [ "${ready}" != "1" ]; then
  echo "[$(date)] Aloe70B repair29 readiness timeout"
  tail -n 120 "${SERVER_LOG}" || true
  exit 1
fi

PHASE65E_ALOE70B_DIRECT_IN_JOB=1 \
PORT="${PORT}" \
MAX_TOKENS="${MAX_TOKENS}" \
TIMEOUT_SECONDS="${TIMEOUT_SECONDS}" \
MAX_RETRIES="${MAX_RETRIES}" \
NUM_SHARDS="${NUM_SHARDS}" \
SHARD_INDEX="${SHARD_INDEX}" \
TIER2_ALOE70B_API_MODE='openai_chat' \
TIER2_ALOE70B_ENDPOINT='/chat/completions' \
TIER2_ALOE70B_MODEL_NAME='llama31-aloe-beta-70b' \
TIER2_ALOE70B_EXTRA_BODY_JSON='{"response_format":{"type":"json_object"}}' \
TIER2_ALOE70B_USE_JSON_SYSTEM_PROMPT='1' \
bash scripts/run_phase65e_aloe70b_vllm_client_v1.sh \
  "${HOLD_JOB_ID}" \
  "${MANIFEST_PATH}" \
  "${OUTPUT_DIR}" \
  "${MAX_WORKERS}" >"${CLIENT_LOG}" 2>&1

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize_manifest_subset \
  --root . \
  --output-dir "${OUTPUT_DIR}" \
  --provider aloe70b \
  --model-name llama31-aloe-beta-70b \
  --manifest-path "${MANIFEST_PATH}" \
  --summary-path "${MANIFEST_SUMMARY_PATH}"

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize \
  --root . \
  --output-dir "${OUTPUT_DIR}" \
  --providers aloe70b

echo "[$(date)] aloe70b repair29 same-step completed"
