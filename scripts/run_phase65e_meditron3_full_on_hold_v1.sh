#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "usage: $0 HOLD_JOB_ID MANIFEST_PATH [OUTPUT_DIR]" >&2
  exit 1
fi

HOLD_JOB_ID="$1"
MANIFEST_PATH="$2"
OUTPUT_DIR="${3:-results/cres_v3/phase65e_tier2_full_meditron3_8b}"

ROOT=/cephfs/volumes/hpc_data_prj/bhi_haoyu_benchmarking/9702e4c9-097c-4b21-8276-01dc96440ad1/TIMELY-Bench_Final
LOG_DIR="${ROOT}/logs/v3"
PORT="${PORT:-8057}"
MAX_WORKERS="${MAX_WORKERS:-12}"
MAX_TOKENS="${MAX_TOKENS:-1800}"
NUM_SHARDS="${NUM_SHARDS:-1}"
SHARD_INDEX="${SHARD_INDEX:-0}"
READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS:-5}"
READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS:-1800}"
SERVER_LOG="${LOG_DIR}/phase65e_meditron3_server_${HOLD_JOB_ID}_shard${SHARD_INDEX}.out"
CLIENT_LOG="${LOG_DIR}/phase65e_meditron3_client_${HOLD_JOB_ID}_shard${SHARD_INDEX}.out"
MANIFEST_SUMMARY_PATH="${OUTPUT_DIR}/meditron3_8b_manifest_summary.json"
INNER_FLAG="${PHASE65E_MEDITRON3_INNER:-0}"
SELF_PATH="${ROOT}/scripts/$(basename "$0")"

if [ "${INNER_FLAG}" != "1" ]; then
  exec srun \
    --overlap \
    --jobid="${HOLD_JOB_ID}" \
    --gres="gpu:1" \
    --ntasks=1 \
    --cpus-per-task=8 \
    --mem=64G \
    --export=ALL,PHASE65E_MEDITRON3_INNER=1,PORT="${PORT}",MAX_WORKERS="${MAX_WORKERS}",MAX_TOKENS="${MAX_TOKENS}",NUM_SHARDS="${NUM_SHARDS}",SHARD_INDEX="${SHARD_INDEX}",READINESS_POLL_SECONDS="${READINESS_POLL_SECONDS}",READINESS_MAX_WAIT_SECONDS="${READINESS_MAX_WAIT_SECONDS}" \
    bash "${SELF_PATH}" "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}"
fi

cd "${ROOT}"
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "[$(date)] meditron3 full run started on hold ${HOLD_JOB_ID}"
echo "[$(date)] manifest=${MANIFEST_PATH}"
echo "[$(date)] output_dir=${OUTPUT_DIR}"
echo "[$(date)] port=${PORT} max_workers=${MAX_WORKERS} max_tokens=${MAX_TOKENS} num_shards=${NUM_SHARDS} shard_index=${SHARD_INDEX}"

cleanup() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

PHASE65E_MEDITRON3_DIRECT_IN_JOB=1 \
PHASE65E_MEDITRON3_TP_SIZE=1 \
PHASE65E_MEDITRON3_GPU_COUNT=1 \
PHASE65E_MEDITRON3_MAX_MODEL_LEN="${PHASE65E_MEDITRON3_MAX_MODEL_LEN:-32768}" \
PORT="${PORT}" \
bash scripts/run_phase65e_meditron3_vllm_server_v1.sh "${HOLD_JOB_ID}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

ready=0
max_attempts=$(( (READINESS_MAX_WAIT_SECONDS + READINESS_POLL_SECONDS - 1) / READINESS_POLL_SECONDS ))
for attempt in $(seq 1 "${max_attempts}"); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    echo "[$(date)] Meditron3 server ready after ${attempt} polls"
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[$(date)] Meditron3 server exited before readiness"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep "${READINESS_POLL_SECONDS}"
done

if [ "${ready}" != "1" ]; then
  echo "[$(date)] Meditron3 server readiness timeout"
  tail -n 120 "${SERVER_LOG}" || true
  exit 1
fi

PHASE65E_MEDITRON3_DIRECT_IN_JOB=1 \
PORT="${PORT}" \
MAX_TOKENS="${MAX_TOKENS}" \
NUM_SHARDS="${NUM_SHARDS}" \
SHARD_INDEX="${SHARD_INDEX}" \
bash scripts/run_phase65e_meditron3_vllm_client_v1.sh "${HOLD_JOB_ID}" "${MANIFEST_PATH}" "${OUTPUT_DIR}" "${MAX_WORKERS}" >"${CLIENT_LOG}" 2>&1

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize_manifest_subset \
  --root . \
  --output-dir "${OUTPUT_DIR}" \
  --provider meditron3_8b \
  --model-name meditron3-8b \
  --manifest-path "${MANIFEST_PATH}" \
  --summary-path "${MANIFEST_SUMMARY_PATH}"

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize \
  --root . \
  --output-dir "${OUTPUT_DIR}" \
  --providers meditron3_8b

echo "[$(date)] meditron3 full run completed"
