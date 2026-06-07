#!/bin/bash
#SBATCH --job-name=phase65a_merge_v3
#SBATCH --partition=cpu
#SBATCH --output=logs/v3/phase65a_merge_%j.out
#SBATCH --error=logs/v3/phase65a_merge_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"

mkdir -p logs/v3 results/cres_v3/phase65a_baselines

if [ -n "${VENV:-}" ] && [ -f "${VENV}/bin/activate" ]; then
  source "${VENV}/bin/activate"
fi

python3 -u code/v3/run_phase65a_baselines_v3.py \
  --mode merge \
  --root "$ROOT" \
  --output-dir results/cres_v3/phase65a_baselines
