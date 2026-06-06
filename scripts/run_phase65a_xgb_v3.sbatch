#!/bin/bash
#SBATCH --job-name=phase65a_xgb_v3
#SBATCH --partition=cpu
#SBATCH --output=logs/v3/phase65a_xgb_%j.out
#SBATCH --error=logs/v3/phase65a_xgb_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

set -euo pipefail

ROOT="/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final"
cd "$ROOT"

mkdir -p logs/v3 results/cres_v3/phase65a_baselines

if [ -f /scratch/users/k25113331/venvs/timer/bin/activate ]; then
  source /scratch/users/k25113331/venvs/timer/bin/activate
fi

python3 -u code/v3/run_phase65a_baselines_v3.py \
  --mode xgb \
  --root "$ROOT" \
  --output-dir results/cres_v3/phase65a_baselines \
  --n-jobs "${SLURM_CPUS_PER_TASK}"
