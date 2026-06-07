#!/bin/bash
#SBATCH --job-name=phase6_cres_release_v3
#SBATCH --output=logs/v3/phase6_cres_release_%j.out
#SBATCH --error=logs/v3/phase6_cres_release_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G

set -euo pipefail

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"

mkdir -p logs/v3

python3 -B code/v3/build_phase6_cres_release_v3.py
