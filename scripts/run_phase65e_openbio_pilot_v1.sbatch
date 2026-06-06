#!/bin/bash
#SBATCH --job-name=phase65e_openbio_pilot
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v3/phase65e_openbio_pilot_%j.out
#SBATCH --error=logs/v3/phase65e_openbio_pilot_%j.err

set -euo pipefail

ROOT=/cephfs/volumes/hpc_data_prj/bhi_haoyu_benchmarking/9702e4c9-097c-4b21-8276-01dc96440ad1/TIMELY-Bench_Final
OUT_DIR=results/cres_v3/phase65e_tier2_pilot
MANIFEST_PATH="${OUT_DIR}/phase65e_openbio_smoke5_manifest.jsonl"
SUMMARY_PATH="${OUT_DIR}/openbiollm70b_smoke5_summary.json"
SERVER_LOG="logs/v3/phase65e_openbio_server_${SLURM_JOB_ID}.out"
CLIENT_LOG="logs/v3/phase65e_openbio_client_${SLURM_JOB_ID}.out"
PORT="${PORT:-8027}"

cd "${ROOT}"
mkdir -p logs/v3 "${OUT_DIR}"

echo "[$(date)] phase65e_openbio_pilot started on $(hostname)"
echo "[$(date)] CUDA devices visible to batch shell:"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

PHASE65E_OPENBIO_DIRECT_IN_JOB=1 \
PHASE65E_OPENBIO_MAX_MODEL_LEN=8192 \
bash scripts/run_phase65e_openbio_vllm_server_v1.sh "${SLURM_JOB_ID}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

cleanup() {
  kill "${SERVER_PID}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ready=0
for _ in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[$(date)] OpenBio server exited before readiness check passed"
    tail -n 120 "${SERVER_LOG}" || true
    exit 1
  fi
  sleep 5
done

if [ "${ready}" != "1" ]; then
  echo "[$(date)] OpenBio server did not become ready within timeout"
  tail -n 120 "${SERVER_LOG}" || true
  exit 1
fi

echo "[$(date)] OpenBio server is ready"

PHASE65E_OPENBIO_DIRECT_IN_JOB=1 \
MAX_TOKENS=2600 \
bash scripts/run_phase65e_openbio_vllm_client_v1.sh "${SLURM_JOB_ID}" "${MANIFEST_PATH}" "${OUT_DIR}" 1 >"${CLIENT_LOG}" 2>&1

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize_manifest_subset \
  --root . \
  --output-dir "${OUT_DIR}" \
  --provider openbiollm70b \
  --model-name openbio-70b \
  --manifest-path "${MANIFEST_PATH}" \
  --summary-path "${SUMMARY_PATH}"

python3 code/v3/run_phase65e_tier2_v1.py \
  --mode summarize \
  --root . \
  --output-dir "${OUT_DIR}" \
  --providers openbiollm70b

echo "[$(date)] phase65e_openbio_pilot completed"
