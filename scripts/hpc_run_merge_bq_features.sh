#!/bin/bash
#SBATCH -J timely_bq_merge
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source /scratch/users/k25113331/venvs/timer/bin/activate
cd /scratch/users/k25113331/TIMELY-Bench_Final

export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

python3 -u code/data_processing/extend_timeseries_with_bq_features.py

