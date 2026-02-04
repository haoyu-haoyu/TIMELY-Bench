#!/bin/bash
#SBATCH --job-name=umls_full_zip
#SBATCH --output=umls_full_zip_%j.log
#SBATCH --error=umls_full_zip_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=cpu

set -euo pipefail

echo "=== UMLS Full ZIP Download ==="
echo "Date: $(date)"
echo "Node: $(hostname)"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/models/umls}"
UMLS_URL="${UMLS_DOWNLOAD_URL:-https://download.nlm.nih.gov/umls/kss/2025AB/umls-2025AB-full.zip}"

mkdir -p "${OUT_DIR}"

if [[ -z "${UMLS_API_KEY:-}" ]]; then
  echo "未设置 UMLS_API_KEY 环境变量，无法下载 UMLS 全量压缩包"
  exit 1
fi

echo "获取 UTS 登录票据..."
TGT_LOCATION=$(printf "apikey=%s" "${UMLS_API_KEY}" | \
  curl -s -D - -o /dev/null -X POST --data @- https://utslogin.nlm.nih.gov/cas/v1/api-key | \
  awk -F': ' '/^Location: /{print $2}' | tr -d '\r')

if [[ -z "${TGT_LOCATION}" ]]; then
  echo "无法获取 TGT"
  exit 1
fi

SERVICE_TICKET=$(curl -s -X POST "${TGT_LOCATION}" --data-urlencode "service=${UMLS_URL}")
if [[ -z "${SERVICE_TICKET}" ]]; then
  echo "无法获取 Service Ticket"
  exit 1
fi

echo "开始下载: ${UMLS_URL}"
TARGET="${OUT_DIR}/umls-2025AB-full.zip"
curl -L -C - -o "${TARGET}" "${UMLS_URL}?ticket=${SERVICE_TICKET}"

echo "下载完成: ${TARGET}"
