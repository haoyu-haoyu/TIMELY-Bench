#!/bin/bash
#SBATCH --job-name=temporal_align
#SBATCH --output=logs/temporal_align_%j.out
#SBATCH --error=logs/temporal_align_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

set -euo pipefail

ROOT_DIR="/scratch/users/k25113331/TIMELY-Bench_Final"
cd "${ROOT_DIR}"
mkdir -p logs

export PYTHONUNBUFFERED=1
python3 -u code/data_processing/temporal_textual_alignment.py
