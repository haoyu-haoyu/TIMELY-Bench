#!/bin/bash
# Build the `final_release/` bundle from the latest `results/` artifacts.
# Kept as a standalone job to avoid re-running heavy pipelines unnecessarily.
#
# Run on KCL CREATE via: sbatch scripts/build_final_release_bundle_hpc.sh

#SBATCH --job-name=build_release
#SBATCH --output=logs/build_release_%j.log
#SBATCH --error=logs/build_release_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "${PROJECT_ROOT}"
mkdir -p logs

if [ -n "${VENV:-}" ] && [ -f "${VENV}/bin/activate" ]; then
  source "${VENV}/bin/activate"
fi
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "========================================"
echo "TIMELY-Bench Final Release Bundle"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "========================================"

python -u code/data_processing/build_final_release_bundle.py

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"

