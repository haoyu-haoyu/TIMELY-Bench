#!/bin/bash
#SBATCH --job-name=perm_sanity
#SBATCH --output=logs/perm_sanity_%j.log
#SBATCH --error=logs/perm_sanity_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH=/scratch/users/k25113331/TIMELY-Bench_Final/code

python code/baselines/permutation_sanity_structured.py --window 24h --cohort all --n-permutations 5
