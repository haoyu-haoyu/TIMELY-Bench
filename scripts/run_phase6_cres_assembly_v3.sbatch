#!/bin/bash
#SBATCH --job-name=phase6_cres_assembly_v3
#SBATCH --output=logs/v3/phase6_cres_assembly_%j.out
#SBATCH --error=logs/v3/phase6_cres_assembly_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

set -euo pipefail

ROOT="/scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final"
cd "$ROOT"

mkdir -p logs/v3

python3 -B code/v3/build_phase6_cres_assembly_v3.py
