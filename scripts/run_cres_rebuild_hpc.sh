#!/bin/bash
#SBATCH --job-name=cres_rebuild
#SBATCH --output=logs/cres_rebuild_%j.out
#SBATCH --error=logs/cres_rebuild_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/cres/build_cres_tasks.py \
  --n-trend 900 \
  --n-grounding 900 \
  --n-diagnostic 900 \
  --n-contrastive 900 \
  --seed 42 \
  --max-rows 0 \
  --min-multimorbidity-ratio 0.3

# Canonical CRES evaluation now requires real prediction files.
# For smoke runs, backend=heuristic is allowed; for release, set:
#   CRES_BACKEND=openai
#   CRES_MODEL_NAME=<model-id>
CRES_BACKEND=${CRES_BACKEND:-heuristic}
CRES_MODEL_NAME=${CRES_MODEL_NAME:-heuristic_debug}

python code/cres/run_cres_model_eval.py \
  --backend "$CRES_BACKEND" \
  --model-name "$CRES_MODEL_NAME" \
  --run-id "cres_${SLURM_JOB_ID}" \
  --write-canonical

mkdir -p final_release/cres
find results/cres -maxdepth 1 -type f -exec cp -f {} final_release/cres/ \;
if [ -d results/cres/model_runs ]; then
  mkdir -p final_release/cres/model_runs
  rsync -a --delete results/cres/model_runs/ final_release/cres/model_runs/
fi
