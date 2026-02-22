#!/bin/bash
#SBATCH -J timely_ablation
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source /scratch/users/k25113331/venvs/timer/bin/activate
cd /scratch/users/k25113331/TIMELY-Bench_Final

# Canonical mode: discharge excluded unless explicitly overridden.
python code/baselines/eval_note_ablation.py

