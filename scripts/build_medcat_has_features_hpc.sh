#!/bin/bash
#SBATCH --job-name=medcat_has
#SBATCH --output=medcat_has_%j.log
#SBATCH --error=medcat_has_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

echo "=== Build MedCAT has_* Features ==="
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

python code/data_processing/build_medcat_has_features.py --include-all-stays

echo ""
echo "=== Done ==="
echo "Date: $(date)"
