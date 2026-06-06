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
PORT="${PORT:-8047}"
MAX_WORKERS="${MAX_WORKERS:-8}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-5}"
READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS:-3600}"
SERVER_LOG="${LOG_DIR}/phase65e_aloe70b_server_${HOLD_JOB_ID}.out"
CLIENT_LOG="${LOG_DIR}/phase65e_aloe70b_client_${HOLD_JOB_ID}.out"
MANIFEST_SUMMARY_PATH="${OUTPUT_DIR}/aloe70b_manifest_summary.json"

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[$(date)] aloe70b full run started on hold ${HOLD_JOB_ID}"
echo "[$(date)] manifest=${MANIFEST_PATH}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] port=${PORT} max_workers=${MAX_WORKERS} max_tokens=${MAX_TOKENS} num_shards=${NUM_SHARDS} shard_index=${SHARD_INDEX}"

probe_server_in_hold() {
  srun \
    --overlap \
    --jobid="${HOLD_JOB_ID}" \
    --ntasks=1 \
    --cpus-per-task=1 \
    --mem=1G \
    --export=ALL \
    bash -lc "curl -fsS 'http://127.0.0.1:${PORT}/v1/models' >/dev/null"
}

PHASE65E_ALOE70B_TP_SIZE=2 \
PHASE65E_ALOE70B_GPU_COUNT=2 \
PHASE65E_ALOE70B_MAX_MODEL_LEN="${PHASE65E_ALOE70B_MAX_MODEL_LEN:-32768}" \
PORT="${PORT}" \
bash scripts/run_phase65e_aloe70b_vllm_server_v1.sh "${HOLD_JOB_ID}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
max_attempts=$(( (READINESS_MAX_WAIT_SECONDS + READINESS_POLL_SECONDS - 1) / READINESS_POLL_SECONDS ))
for attempt in $(seq 1 "${max_attempts}"); do
  if probe_server_in_hold >/dev/null 2>&1; then
    ready=1
    echo "[$(date)] Aloe70B server ready after ${attempt} polls"
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[$(date)] Aloe70B server exited before readiness"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep "${READINESS_POLL_SECONDS}"
done

if [ "${ready}" != "1" ]; then
  echo "[$(date)] Aloe70B server readiness timeout"
  tail -n 120 "${SERVER_LOG}" || true
  exit 1
fi

PORT="${PORT}" \
MAX_TOKENS="${MAX_TOKENS}" \
NUM_SHARDS="${NUM_SHARDS}" \
SHARD_INDEX="${SHARD_INDEX}" \
bash scripts/run_phase65e_aloe70b_vllm_client_v1.sh "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}" "${MAX_WORKERS}" >"${CLIENT_LOG}" 2>&1

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

echo "[$(date)] aloe70b full run completed"
