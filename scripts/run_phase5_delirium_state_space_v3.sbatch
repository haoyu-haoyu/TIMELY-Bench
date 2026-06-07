#!/bin/bash
#SBATCH --job-name=phase5_delirium_state_space_v3
#SBATCH --output=logs/v3/phase5_delirium_state_space_%j.out
#SBATCH --error=logs/v3/phase5_delirium_state_space_%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

ROOT="${PROJECT_ROOT:-$PWD}"
cd "$ROOT"

mkdir -p logs/v3

python3 -B code/v3/build_phase5_state_space_v3.py --condition delirium
