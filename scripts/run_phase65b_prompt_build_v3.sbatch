#!/bin/bash
#SBATCH --job-name=phase65b_prompt_v3
#SBATCH --partition=cpu
#SBATCH --output=logs/v3/phase65b_prompt_%j.out
#SBATCH --error=logs/v3/phase65b_prompt_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

set -euo pipefail

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"

mkdir -p logs/v3 results/cres_v3 data/processed/v3/cres

if [ -n "${VENV:-}" ] && [ -f "${VENV}/bin/activate" ]; then
  source "${VENV}/bin/activate"
fi

python3 -u code/v3/run_phase65b_prompt_build_v3.py \
  --root "$ROOT" \
  --output-dir data/processed/v3/cres \
  --results-dir results/cres_v3
