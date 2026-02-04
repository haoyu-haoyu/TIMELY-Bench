#!/usr/bin/env python3
"""
TIMELY-Bench v2.0 - 一键运行脚本
================================

在任意机器上复现实验的最简单方式。

使用方法:
    python run_all.py              # 运行完整pipeline
    python run_all.py --verify     # 只验证数据
    python run_all.py --baselines  # 只运行基线
    python run_all.py --fusion     # 只运行融合实验
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """获取项目根目录"""
    return Path(__file__).parent.absolute()


def verify_environment():
    """验证运行环境"""
    print("验证运行环境...")

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("[ERROR] Python版本需要 >= 3.8")
        return False
    print(f"   Python {sys.version_info.major}.{sys.version_info.minor}")

    # 检查必要的包
    required = ['pandas', 'numpy', 'sklearn', 'xgboost']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"   [ERROR] 缺少依赖: {', '.join(missing)}")
        print(f"   提示: pip install -r requirements.txt")
        return False
    print(f"   所有依赖已安装")

    return True


def verify_data(root: Path):
    """验证数据完整性"""
    print("\n验证数据完整性...")

    import pandas as pd

    # 检查数据分割
    splits_dir = root / 'data' / 'splits'
    for split in ['train.csv', 'val.csv', 'test.csv']:
        path = splits_dir / split
        if path.exists():
            df = pd.read_csv(path)
            print(f"   {split}: {len(df)} episodes")
        else:
            print(f"   [ERROR] {split}: 文件不存在")
            return False

    # 检查LLM特征
    llm_path = root / 'data' / 'llm_features' / 'llm_features_deepseek.csv'
    if llm_path.exists():
        df = pd.read_csv(llm_path)
        print(f"   LLM features: {len(df)} rows")
    else:
        print(f"   [ERROR] LLM features: 文件不存在")
        return False

    # 检查时序窗口
    for window in ['6h', '12h', '24h']:
        path = root / 'data' / 'processed' / 'data_windows' / f'window_{window}' / 'features_aggregated.csv'
        if path.exists():
            df = pd.read_csv(path)
            print(f"   window_{window}: {len(df)} samples")
        else:
            print(f"   [ERROR] window_{window}: 文件不存在")
            return False

    return True


def run_script(script_path: Path, description: str):
    """运行Python脚本"""
    print(f"\n{description}...")
    print(f"   脚本: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=script_path.parent.parent.parent,  # 项目根目录
        capture_output=False
    )

    if result.returncode == 0:
        print(f"   完成")
        return True
    else:
        print(f"   [ERROR] 失败 (exit code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description='TIMELY-Bench v2.0 Pipeline Runner')
    parser.add_argument('--verify', action='store_true', help='只验证数据')
    parser.add_argument('--baselines', action='store_true', help='只运行XGBoost基线')
    parser.add_argument('--fusion', action='store_true', help='只运行融合实验')
    parser.add_argument('--gru', action='store_true', help='只运行GRU模型')
    parser.add_argument('--delta', action='store_true', help='只运行Delta特征训练')
    args = parser.parse_args()

    print("=" * 60)
    print("TIMELY-Bench v2.0 Pipeline")
    print("=" * 60)

    root = get_project_root()
    os.chdir(root)

    # 验证环境
    if not verify_environment():
        sys.exit(1)

    # 验证数据
    if not verify_data(root):
        sys.exit(1)

    if args.verify:
        print("\n数据验证通过!")
        return

    # 运行实验
    baselines_script = root / 'code' / 'baselines' / 'run_baselines.py'
    fusion_script = root / 'code' / 'baselines' / 'run_fusion_baselines.py'
    gru_script = root / 'code' / 'baselines' / 'run_temporal_gru.py'
    delta_script = root / 'code' / 'baselines' / 'train_with_delta_features.py'

    run_all = not (args.baselines or args.fusion or args.gru or args.delta)

    if args.baselines or run_all:
        if baselines_script.exists():
            run_script(baselines_script, "运行XGBoost基线")

    if args.fusion or run_all:
        if fusion_script.exists():
            run_script(fusion_script, "运行Fusion实验")

    if args.gru or run_all:
        if gru_script.exists():
            run_script(gru_script, "运行GRU模型")

    if args.delta or run_all:
        if delta_script.exists():
            run_script(delta_script, "运行Delta特征训练")

    print("\n" + "=" * 60)
    print("Pipeline完成!")
    print("=" * 60)
    print("\n结果保存在: results/benchmark_results/")


if __name__ == "__main__":
    main()
