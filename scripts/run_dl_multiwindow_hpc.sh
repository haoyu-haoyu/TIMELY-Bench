#!/bin/bash
#SBATCH --job-name=dl_multiwindow
#SBATCH --output=logs/dl_multiwindow_%j.log
#SBATCH --error=logs/dl_multiwindow_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Multi-Window DL Training for Robustness Analysis
# Trains ClinicalGRU and EarlyFusion models across 6h, 12h, 24h windows

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code
export PYTHONUNBUFFERED=1

echo "========================================"
echo "TIMELY-Bench Multi-Window DL Training"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Python: $(python --version)"
echo "========================================"

# Run all window-task combinations
# This trains: ClinicalGRU × (6h, 12h, 24h) × (mortality, prolonged_los)
#              EarlyFusion × (6h, 12h, 24h) × (mortality, prolonged_los)

echo ""
echo "Running all window-task combinations..."
python -u code/baselines/train_dl_multiwindow.py --all

echo ""
echo "========================================"
echo "Multi-window training completed at $(date)"
echo "========================================"

# Check output
echo ""
echo "Results summary:"
if [ -f results/robustness/dl_window_performance.csv ]; then
    echo "DL window performance saved to: results/robustness/dl_window_performance.csv"
    wc -l results/robustness/dl_window_performance.csv
    head -20 results/robustness/dl_window_performance.csv
else
    echo "WARNING: Output file not found!"
fi
