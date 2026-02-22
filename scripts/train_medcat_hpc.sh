#!/bin/bash
# CPU-only job to run the MedCAT/UMLS concept baseline and refresh standardized summaries.
# Keeping it separate avoids re-running other text baselines unnecessarily.

#SBATCH --job-name=medcat_text
#SBATCH --output=logs/train_medcat_%j.log
#SBATCH --error=logs/train_medcat_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/baselines/train_text_only_medcat.py
python code/utils/standardize_results.py --step text

