#!/bin/bash
#SBATCH --job-name=cres_gpt51
#SBATCH --output=/scratch/users/k25113331/TIMELY-Bench_Final/logs/cres_gpt51_%j.out
#SBATCH --error=/scratch/users/k25113331/TIMELY-Bench_Final/logs/cres_gpt51_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs
source /scratch/users/k25113331/venvs/timer/bin/activate

export OPENAI_BASE_URL="https://api.ikuncode.cc/v1"
: "${OPENAI_API_KEY:?OPENAI_API_KEY is required}"

python code/cres/run_cres_model_eval.py \
  --backend openai \
  --model-name gpt-5.1 \
  --run-id "cres_gpt51_full_${SLURM_JOB_ID}" \
  --max-per-task 0 \
  --temperature 0 \
  --max-tokens 24 \
  --max-retries 8 \
  --retry-sleep 1.5 \
  --request-timeout 120
