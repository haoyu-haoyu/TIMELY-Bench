#!/bin/bash
#SBATCH --job-name=diff_diag
#SBATCH --output=diff_diag_%j.log
#SBATCH --error=diff_diag_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu

echo "=== Differential Diagnosis Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

if command -v module >/dev/null 2>&1; then
    module load python/3.10 2>/dev/null || module load python/3.9 2>/dev/null || true
fi

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

# 激活环境（如无则使用系统 Python）
source medcat_env/bin/activate 2>/dev/null || {
    echo "Using system python..."
}

# 依赖保障
pip install --quiet scikit-learn pandas numpy tqdm || true

echo ""
echo "Starting differential diagnosis training..."
python code/baselines/train_differential_diagnosis.py

echo ""
echo "=== Training Complete ==="
echo "Date: $(date)"
