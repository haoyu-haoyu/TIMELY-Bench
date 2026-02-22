#!/bin/bash
# CPU-only: regenerate ML robustness + calibration with the canonical labels.
#
# This should be run on KCL CREATE (Slurm) to avoid local memory issues.

#SBATCH --job-name=ml_robust_calib
#SBATCH --output=logs/ml_robust_calib_%j.log
#SBATCH --error=logs/ml_robust_calib_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code
export PYTHONUNBUFFERED=1

echo "========================================"
echo "TIMELY-Bench ML Robustness + Calibration"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "========================================"

echo ""
echo "[1/2] Robustness (cross-window) ..."
python -u code/evaluation/robustness_analysis.py

echo ""
echo "[2/2] Calibration (cross-window) ..."
python -u code/evaluation/run_calibration_evaluation.py

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"

