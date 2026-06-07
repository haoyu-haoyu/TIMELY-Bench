#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/progression/prog_%j.out
#SBATCH --error=logs/progression/prog_%j.err

set -euo pipefail

if [ "$#" -lt 6 ]; then
  echo "Usage: $0 <task> <modality> <model> <window> <text_method> <output_dir>"
  exit 2
fi

TASK="$1"
MODALITY="$2"
MODEL="$3"
WINDOW="$4"
TEXT_METHOD="$5"
OUTDIR="$6"

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
cd "${PROJECT_ROOT}"
mkdir -p logs/progression "$OUTDIR"

python3 code/baselines/train_progression_baselines.py \
  --task "$TASK" \
  --modality "$MODALITY" \
  --model "$MODEL" \
  --window "$WINDOW" \
  --text_method "$TEXT_METHOD" \
  --output_dir "$OUTDIR" \
  --n-jobs "${SLURM_CPUS_PER_TASK:-8}"
