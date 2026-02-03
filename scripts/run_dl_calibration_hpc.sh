#!/bin/bash
#SBATCH --job-name=timely_dl_calib
#SBATCH --output=logs/dl_calibration_%j.log
#SBATCH --error=logs/dl_calibration_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

echo "========================================"
echo "TIMELY-Bench DL Calibration Evaluation"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "========================================"

# Run DL calibration evaluation
python code/evaluation/run_dl_calibration_hpc.py

# Run statistical tests (CPU only)
echo ""
echo "Running statistical tests..."
python code/evaluation/add_statistical_tests.py

echo ""
echo "========================================"
echo "All evaluations completed at $(date)"
echo "========================================"
