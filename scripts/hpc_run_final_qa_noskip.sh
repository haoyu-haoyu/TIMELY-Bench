#!/bin/bash
#SBATCH -J timely_finalqa_noskip
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source /scratch/users/k25113331/venvs/timer/bin/activate
cd /scratch/users/k25113331/TIMELY-Bench_Final

export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"
# Use nrows greater than actual rows to force a full-order check and avoid SKIP.
python code/data_processing/run_final_qa.py --timeseries-rows 5000000
