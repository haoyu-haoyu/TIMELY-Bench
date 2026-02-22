#!/bin/bash
#SBATCH -J timely_note_ablation
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/note_ablation_%j.out
#SBATCH -e logs/note_ablation_%j.err

set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

source /scratch/users/k25113331/venvs/timer/bin/activate

python code/baselines/eval_note_ablation.py

