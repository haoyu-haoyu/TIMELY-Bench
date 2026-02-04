#!/bin/bash
#SBATCH --job-name=smoke_structured
#SBATCH --output=logs/smoke_structured_%j.log
#SBATCH --error=logs/smoke_structured_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/baselines/smoke_structured_mortality.py
