#!/bin/bash
#SBATCH --job-name=cross_modal
#SBATCH --output=cross_modal_%j.log
#SBATCH --error=cross_modal_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Cross-Modal Attention Training on HPC
echo "=== Cross-Modal Attention Training ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"

cd /scratch/users/k25113331/TIMELY-Bench_Final

# 激活环境
module load python/3.9 2>/dev/null || true
source medcat_env/bin/activate 2>/dev/null || {
    echo "Using system python..."
}

# 安装 PyTorch 依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || true
pip install scikit-learn pandas tqdm 2>/dev/null || true

# 运行训练
echo ""
echo "Starting Cross-Modal Attention training..."
python code/baselines/train_cross_modal.py \
    --task mortality \
    --folds 5 \
    --epochs 50 \
    --batch_size 64

echo ""
echo "=== Training Complete ==="
echo "Date: $(date)"
