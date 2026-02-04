#!/bin/bash
#SBATCH --job-name=medcat_full_model
#SBATCH --output=medcat_full_model_%j.log
#SBATCH --error=medcat_full_model_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

echo "=== MedCAT Full Model Download ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-${ROOT_DIR}/models/medcat}"
MODEL_URL="${MEDCAT_MODEL_URL:-}"
MODEL_FILE="${MEDCAT_MODEL_FILE:-medcat_full_modelpack.zip}"
export MODEL_DIR
export MODEL_FILE

mkdir -p "${MODEL_DIR}"

if [[ -n "${MEDCAT_MODEL_PATH:-}" ]]; then
  echo "MEDCAT_MODEL_PATH 已设置，跳过下载: ${MEDCAT_MODEL_PATH}"
  exit 0
fi

if [[ -z "${MODEL_URL}" ]]; then
  echo "未设置 MEDCAT_MODEL_URL。请提供全量模型包的下载链接。"
  exit 1
fi

echo "下载模型到: ${MODEL_DIR}/${MODEL_FILE}"
if command -v curl >/dev/null 2>&1; then
  curl -L -o "${MODEL_DIR}/${MODEL_FILE}" "${MODEL_URL}"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${MODEL_DIR}/${MODEL_FILE}" "${MODEL_URL}"
else
  echo "未找到 curl 或 wget"
  exit 1
fi

echo "下载完成，测试模型加载..."
if [[ -f "${ROOT_DIR}/medcat_env/bin/activate" ]]; then
  source "${ROOT_DIR}/medcat_env/bin/activate"
else
  python3 -m venv "${ROOT_DIR}/medcat_env"
  source "${ROOT_DIR}/medcat_env/bin/activate"
  pip install --upgrade pip
  pip install medcat
fi

python - << 'PY'
from medcat.cat import CAT
import os
model_path = os.path.join(os.environ.get("MODEL_DIR", ""), os.environ.get("MODEL_FILE", ""))
print(f"Loading model pack: {model_path}")
cat = CAT.load_model_pack(model_path)
print(f"Loaded CUI count: {len(cat.cdb.cui2names)}")
PY

echo "=== Download Complete ==="
echo "Model path: ${MODEL_DIR}/${MODEL_FILE}"
