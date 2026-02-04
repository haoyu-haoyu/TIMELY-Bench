#!/bin/bash
#SBATCH --job-name=medcat_full
#SBATCH --output=medcat_full_%j.log
#SBATCH --error=medcat_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu

# MedCAT Full Extraction - 处理全部 74,829 episodes (24h)
echo "=== MedCAT Full Extraction ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

if command -v module >/dev/null 2>&1; then
    module load python/3.10 2>/dev/null || module load python/3.9 2>/dev/null || true
fi

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT_DIR}"

# 激活环境
VENV_DIR="${ROOT_DIR}/medcat_env"
source "${VENV_DIR}/bin/activate" 2>/dev/null || {
    echo "Creating virtual environment..."
    PYTHON_BIN="python3"
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        PYTHON_BIN="python"
    fi
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
        echo "Python not found in PATH"
        exit 1
    fi
    "${PYTHON_BIN}" -m venv "${VENV_DIR}"
    source "${VENV_DIR}/bin/activate"
    "${PYTHON_BIN}" -m pip install --upgrade pip
    "${PYTHON_BIN}" -m pip install medcat spacy pandas tqdm
}

# 自动定位模型包
if [[ -z "${MEDCAT_MODEL_PATH:-}" ]]; then
    # 优先选择指定模型，其次选择目录下最大 zip
    preferred="models/medcat/v2_Snomed2025_MIMIC_IV_bbe806e192df009f.zip"
    if [[ -f "${preferred}" ]]; then
        export MEDCAT_MODEL_PATH="${preferred}"
    else
        model_pack=$(ls -S models/medcat/*.zip 2>/dev/null | head -n 1 || true)
        if [[ -n "${model_pack}" ]]; then
            export MEDCAT_MODEL_PATH="${model_pack}"
        fi
    fi
    if [[ -n "${MEDCAT_MODEL_PATH:-}" ]]; then
        echo "Using model pack: ${MEDCAT_MODEL_PATH}"
    else
        echo "未找到模型包，请先下载或设置 MEDCAT_MODEL_PATH"
        exit 1
    fi
fi

# 运行全量提取
echo "Starting extraction for all episodes..."
PYTHON_BIN="python3"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    PYTHON_BIN="python"
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python not found in PATH"
    exit 1
fi

"${PYTHON_BIN}" code/data_processing/extract_concepts_medcat_full.py \
  --sample 0 \
  --window-hours 24 \
  --output-dir data/processed/medcat_full

echo "=== Extraction Complete ==="
echo "Date: $(date)"
