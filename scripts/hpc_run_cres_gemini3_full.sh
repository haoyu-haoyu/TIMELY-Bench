#!/bin/bash
#SBATCH --job-name=cres_gemini3
#SBATCH --output=/scratch/users/k25113331/TIMELY-Bench_Final/logs/cres_gemini3_%j.out
#SBATCH --error=/scratch/users/k25113331/TIMELY-Bench_Final/logs/cres_gemini3_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs
source /scratch/users/k25113331/venvs/timer/bin/activate

export OPENAI_BASE_URL="https://api.ikuncode.cc/v1"
export GEMINI_BASE_URL="https://code.newcli.com/gemini"
export GEMINI_USER_AGENT="${GEMINI_USER_AGENT:-GeminiBridge/1.0}"
export GEMINI_USE_CURL="${GEMINI_USE_CURL:-1}"
export GEMINI_RESOLVE_IP="${GEMINI_RESOLVE_IP:-162.159.36.20}"
export GEMINI_DISABLE_THINKING="${GEMINI_DISABLE_THINKING:-1}"
RUN_ID="${RUN_ID:-cres_gemini3_full_${SLURM_JOB_ID}}"
MAX_RETRIES="${MAX_RETRIES:-12}"
RETRY_SLEEP="${RETRY_SLEEP:-1.5}"
RETRY_MAX_SLEEP="${RETRY_MAX_SLEEP:-20}"

if [ -z "${GEMINI_API_KEYS:-}" ] && [ -z "${GEMINI_API_KEY:-}" ] && [ -z "${OPENAI_API_KEYS:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "GEMINI_API_KEYS/GEMINI_API_KEY (or OPENAI_API_KEYS/OPENAI_API_KEY) is required"
  exit 1
fi

extra_args=()
if [ "${RESUME:-0}" = "1" ]; then
  extra_args+=(--resume)
fi

python code/cres/run_cres_model_eval.py \
  --backend gemini_native \
  --model-name gemini-3-flash-preview \
  --run-id "${RUN_ID}" \
  --max-per-task 0 \
  --temperature 0 \
  --max-tokens 64 \
  --max-retries "${MAX_RETRIES}" \
  --retry-sleep "${RETRY_SLEEP}" \
  --retry-max-sleep "${RETRY_MAX_SLEEP}" \
  --request-timeout 120 \
  "${extra_args[@]}"
