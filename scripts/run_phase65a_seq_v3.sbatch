#!/bin/bash
#SBATCH --job-name=phase65a_seq_v3
#SBATCH --partition=gpu
#SBATCH --output=logs/v3/phase65a_seq_%j.out
#SBATCH --error=logs/v3/phase65a_seq_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100

set -euo pipefail

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"

mkdir -p logs/v3 results/cres_v3/phase65a_baselines

if [ -n "${VENV:-}" ] && [ -f "${VENV}/bin/activate" ]; then
  source "${VENV}/bin/activate"
fi

ulimit -n 65535 || true

python3 -u code/v3/run_phase65a_baselines_v3.py \
  --mode seq \
  --root "$ROOT" \
  --output-dir results/cres_v3/phase65a_baselines \
  --seq-model temporal_transformer \
  --num-workers 2
