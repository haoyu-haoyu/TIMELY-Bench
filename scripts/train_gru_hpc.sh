#!/bin/bash
#SBATCH --job-name=gru_temporal
#SBATCH --output=logs/train_gru_%j.log
#SBATCH --error=logs/train_gru_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/baselines/train_temporal_gru_v2.py
python code/utils/standardize_results.py --step gru
