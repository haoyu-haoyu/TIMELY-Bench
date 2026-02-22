#!/bin/bash
#SBATCH --job-name=medcat_base
#SBATCH --output=medcat_base_%j.log
#SBATCH --error=medcat_base_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

echo "=== Baselines with MedCAT Features ==="
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

pip install --quiet pandas numpy scikit-learn tqdm xgboost || true

echo ""
echo "Running tabular baselines (mortality + LOS)..."
python code/baselines/train_tabular_baselines.py

echo ""
echo "Running LOS baselines..."
python code/baselines/train_los_baselines.py

echo ""
echo "Running readmission baselines..."
python code/baselines/train_readmission_baselines.py
python code/utils/standardize_results.py --step readmission

echo ""
echo "Running enhanced feature ablation..."
python code/baselines/train_enhanced_features.py

echo ""
echo "=== Done ==="
echo "Date: $(date)"
