#!/bin/bash
# CPU-only: regenerate missing text-only prediction artifacts (ClinicalBERT LR + MedCAT)
# and upsert their calibration metrics into results/calibration/calibration_summary.csv.
#
# Run on KCL CREATE (Slurm) to avoid local memory / OpenMP SHM limits.

#SBATCH --job-name=textonly_calib
#SBATCH --output=logs/textonly_calib_%j.log
#SBATCH --error=logs/textonly_calib_%j.err
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
echo "TIMELY-Bench TextOnly Missing Calibration"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "Python: $(python --version)"
echo "========================================"

echo ""
echo "[1/3] Exporting ClinicalBERT LR predictions (24h/all)..."
python -u code/baselines/train_text_only_embeddings.py --models lr --export-preds --fast-export

echo ""
echo "[2/3] Exporting MedCAT predictions (24h/all)..."
python -u code/baselines/train_text_only_medcat.py

echo ""
echo "[3/3] Upserting calibration (fusion + text-only + MedCAT)..."
python -u code/evaluation/compute_fusion_calibration.py --window 24h --cohort all --regen-early

echo ""
echo "========================================"
echo "Completed at $(date)"
echo "========================================"
