#!/bin/bash
#SBATCH -J timely_qasmoke
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source /scratch/users/k25113331/venvs/timer/bin/activate
cd /scratch/users/k25113331/TIMELY-Bench_Final
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

python code/data_processing/run_final_qa.py --sample 2000 --timeseries-rows 500000
