#!/bin/bash
#SBATCH --job-name=text_baselines
#SBATCH --output=logs/train_text_%j.log
#SBATCH --error=logs/train_text_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/baselines/train_text_only.py
python code/utils/standardize_results.py --step text
