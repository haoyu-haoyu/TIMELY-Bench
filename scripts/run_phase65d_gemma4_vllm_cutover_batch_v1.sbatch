#!/bin/bash
#SBATCH --job-name=phase65d_g4vllm
#SBATCH --partition=gpu
#SBATCH --constraint=h200
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/v3/phase65d_gemma4_vllm_batch_%j.out
#SBATCH --error=logs/v3/phase65d_gemma4_vllm_batch_%j.err

set -euo pipefail

ROOT=/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
cd "${ROOT}"
mkdir -p logs/v3

: "${HF_TOKEN:?set HF_TOKEN before sbatch or with --export=ALL,HF_TOKEN=...}"

echo "[$(date)] phase65d_gemma4_vllm batch started on $(hostname)"
echo "[$(date)] CUDA devices visible to batch shell:"
nvidia-smi --query-gpu=index,name,uuid,memory.total --format=csv,noheader

export PHASE65D_VLLM_DIRECT_IN_JOB=1
bash "${ROOT}/scripts/run_phase65d_gemma4_vllm_cutover_v1.sh" "${SLURM_JOB_ID}"
