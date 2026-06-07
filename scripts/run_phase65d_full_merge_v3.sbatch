#!/bin/bash
#SBATCH --job-name=phase65d_full_merge
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --output=logs/v3/phase65d_full_merge_%j.out
#SBATCH --error=logs/v3/phase65d_full_merge_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "${PROJECT_ROOT}"
mkdir -p logs/v3 results/cres_v3/phase65d_tier1b_full

python3 -u code/v3/run_phase65d_tier1b_v3.py \
  --mode summarize \
  --root . \
  --output-dir results/cres_v3/phase65d_tier1b_full \
  --variant-id full_multimodal \
  --providers deepseek_chat qwen35 gemma4_26b
