#!/bin/bash
#SBATCH --job-name=syndrome_add
#SBATCH --output=syndrome_add_%j.log
#SBATCH --error=syndrome_add_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

echo "=== Add Syndrome Detection ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

if command -v module >/dev/null 2>&1; then
    module load python/3.10 2>/dev/null || module load python/3.9 2>/dev/null || true
fi

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

source medcat_env/bin/activate 2>/dev/null || {
    echo "Using system python..."
}

pip install --quiet pandas tqdm || true

echo ""
echo "Starting syndrome detection..."
python code/data_processing/add_syndrome_detection.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
