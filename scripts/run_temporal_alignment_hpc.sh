#!/usr/bin/env bash
set -euo pipefail

cd /scratch/users/k25113331/TIMELY-Bench_Final
mkdir -p logs

export PYTHONUNBUFFERED=1
echo "$(date) starting temporal_textual_alignment" > logs/temporal_alignment_run.log
nohup python3 -u code/data_processing/temporal_textual_alignment.py >> logs/temporal_alignment_run.log 2>&1 &
echo $! > logs/temporal_alignment_run.pid
