#!/bin/bash
#SBATCH --job-name=textonly_rerun
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/textonly_rerun_%j.out
#SBATCH --error=logs/textonly_rerun_%j.err

# ============================================
# TIMELY-Bench TextOnly Bug修复后重训练脚本
# 日期: 2026-02-04
# 目的: 修复 text_full 字段Bug后重跑模型
# ============================================

set -euo pipefail

echo "=== TextOnly Bug Fix Rerun ==="
echo "Start time: $(date)"
echo "Node: $(hostname)"

# 项目路径 (使用scratch空间)
PROJECT_DIR="/scratch/users/k25113331/TIMELY-Bench_Final"
cd "$PROJECT_DIR" || exit 1

# 创建日志目录
mkdir -p logs

# 环境设置 - 使用正确的venv
source /scratch/users/k25113331/venvs/timer/bin/activate
export PYTHONPATH="$PROJECT_DIR/code"

echo "Python: $(which python)"
echo "Python version: $(python --version)"

echo ""
echo "=== Step 1: 验证Bug修复 ==="
python -c "
import json, glob
files = sorted(glob.glob('episodes/episodes_enhanced/*.json'))[:10]
total_len = 0
for f in files:
    ep = json.load(open(f))
    notes = ep.get('clinical_text', {}).get('notes', [])
    for n in notes:
        text = n.get('text_full') or n.get('text_relevant') or n.get('text', '')
        total_len += len(text)
print(f'Total text length from 10 episodes: {total_len}')
if total_len == 0:
    print('ERROR: Bug not fixed! text_full still returning empty.')
    exit(1)
else:
    print('PASS: text_full field correctly accessed.')
"

echo ""
echo "=== Step 2: 重训练 TextOnly 模型 ==="
python code/baselines/train_text_only.py 2>&1 | tee logs/textonly_retrain.log

echo ""
echo "=== Step 3: 重训练 Fusion 模型 ==="
python code/baselines/train_fusion.py 2>&1 | tee logs/fusion_retrain.log

echo ""
echo "=== Step 4: 重跑校准评估 ==="
python code/evaluation/run_calibration_evaluation.py 2>&1 | tee logs/calibration_reeval.log

echo ""
echo "=== Step 5: 结果对比 ==="
python -c "
import pandas as pd
import os

# 读取新结果
results_files = [
    'results/tabular_baselines/tabular_results.csv',
    'results/fusion_baselines/fusion_results.csv'
]
for f in results_files:
    if os.path.exists(f):
        df = pd.read_csv(f)
        print(f'\n--- {f} ---')
        print(df.to_string())
    else:
        print(f'WARNING: {f} not found')
"

echo ""
echo "=== 完成 ==="
echo "End time: $(date)"
echo "请检查 logs/ 目录中的日志文件"
