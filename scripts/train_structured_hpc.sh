#!/bin/bash
#SBATCH --job-name=structured_baselines
#SBATCH --output=logs/train_structured_%j.log
#SBATCH --error=logs/train_structured_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/baselines/run_baselines.py
python code/utils/standardize_results.py --step structured
