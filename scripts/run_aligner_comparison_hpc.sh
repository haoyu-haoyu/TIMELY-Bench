#!/bin/bash
#SBATCH -J timely_aligner
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/aligner_%j.out
#SBATCH -e logs/aligner_%j.err

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate

python code/baselines/train_aligner_comparison.py

