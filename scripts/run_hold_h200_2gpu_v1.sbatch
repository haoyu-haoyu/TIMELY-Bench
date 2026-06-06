#!/bin/bash
#SBATCH --job-name=hold_h200_2gpu
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/v3/hold_h200_2gpu_%j.out
#SBATCH --error=logs/v3/hold_h200_2gpu_%j.err

set -euo pipefail

cd /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
mkdir -p logs/v3

echo "[$(date)] hold_h200_2gpu started on $(hostname)"
echo "[$(date)] CUDA devices:"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

trap 'echo "[$(date)] hold_h200_2gpu exiting"' EXIT

while true; do
  sleep 600
done
