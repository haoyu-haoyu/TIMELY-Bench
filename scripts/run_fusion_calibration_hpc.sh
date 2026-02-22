#!/bin/bash
# CPU-only: compute calibration for fusion baselines (Structured/Text/Early/Late),
# including ClinicalBERT variants, using prediction files and tuned alphas.
#
# Run this on KCL CREATE (Slurm) to avoid local numpy/OpenMP sandbox limits.

#SBATCH --job-name=fusion_calib
#SBATCH --output=logs/fusion_calib_%j.log
#SBATCH --error=logs/fusion_calib_%j.err
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
echo "TIMELY-Bench Fusion Calibration"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "========================================"

# Always regenerate Early Fusion prediction CSVs to avoid stale calibration when
# structured/text feature sets change across iterations.
python -u code/evaluation/compute_fusion_calibration.py --window 24h --cohort all --regen-early

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"
