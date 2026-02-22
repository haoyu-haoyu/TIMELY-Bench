#!/bin/bash
#SBATCH -J timely_medcat_strict
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

source /scratch/users/k25113331/venvs/timer/bin/activate
cd /scratch/users/k25113331/TIMELY-Bench_Final
export PYTHONPATH="$PWD/code:${PYTHONPATH:-}"

# 1) Data-level canonical filtering: remove discharge at source CSV.
python code/data_processing/filter_medcat_note_concepts.py \
  --input data/processed/medcat_full/medcat_note_concepts_24h.csv \
  --output data/processed/medcat_full/medcat_note_concepts_24h.csv \
  --exclude-note-types discharge \
  --report results/qc/medcat_note_concepts_filter_report.json

# 2) Rebuild stay-level concept features from filtered note-level CSV.
python code/data_processing/build_medcat_has_features.py \
  --input data/processed/medcat_full/medcat_note_concepts_24h.csv \
  --output data/processed/medcat_full/medcat_has_concepts_24h.csv \
  --include-all-stays

# 3) Recompute canonical downstream outputs that depend on MedCAT text artifacts.
python code/baselines/eval_note_ablation.py
python code/baselines/train_text_only_medcat.py
