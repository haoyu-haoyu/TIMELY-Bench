#!/usr/bin/env python3
"""
TIMELY-Bench v2.0 - 一键运行脚本
================================

在任意机器上复现实验的简单方式。

使用方法:
    python run_all.py                 # 运行核心pipeline
    python run_all.py --verify        # 只验证数据
    python run_all.py --baselines     # 只运行structured基线
    python run_all.py --fusion        # 只运行fusion实验
    python run_all.py --aligner       # 运行D0/6h/12h/24h对齐器比较
    python run_all.py --ablation      # 运行note-category ablation
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def get_project_root():
    """获取项目根目录 (TIMELY-Bench_Final)。"""
    return Path(__file__).resolve().parent.parent


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

    required_files = [
        root / 'data' / 'processed' / 'merge_output' / 'cohort_final.csv',
        root / 'data' / 'splits' / 'predefined_splits.csv',
        root / 'data' / 'processed' / 'data_windows' / 'window_6h' / 'features_aggregated.csv',
        root / 'data' / 'processed' / 'data_windows' / 'window_12h' / 'features_aggregated.csv',
        root / 'data' / 'processed' / 'data_windows' / 'window_24h' / 'features_aggregated.csv',
        root / 'data' / 'processed' / 'data_windows' / 'window_D0' / 'features_aggregated.csv',
    ]
    for path in required_files:
        if not path.exists():
            print(f"   [ERROR] 缺少文件: {path}")
            return False

    print("   cohort_final.csv / predefined_splits.csv 已找到")
    print("   window_6h / 12h / 24h / D0 特征文件已找到")

    return True


def run_script(project_root: Path, script_path: Path, description: str):
    """运行Python脚本"""
    print(f"\n{description}...")
    print(f"   脚本: {script_path}")

    if not script_path.exists():
        print(f"   [ERROR] 脚本不存在: {script_path}")
        return False

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=project_root,
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
    parser.add_argument('--aligner', action='store_true', help='只运行D0/6h/12h/24h对齐器比较')
    parser.add_argument('--ablation', action='store_true', help='只运行note-category ablation')
    parser.add_argument('--delta', action='store_true', help='只运行Delta特征训练')
    args = parser.parse_args()

    print("=" * 60)
    print("TIMELY-Bench v2.0 Pipeline")
    print("=" * 60)

    root = get_project_root()
    os.chdir(root)
    print(f"项目根目录: {root}")

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
    fusion_script = root / 'code' / 'baselines' / 'train_fusion.py'
    gru_script = root / 'code' / 'baselines' / 'train_temporal_gru_v2.py'
    aligner_script = root / 'code' / 'baselines' / 'train_aligner_comparison.py'
    ablation_script = root / 'code' / 'baselines' / 'eval_note_ablation.py'
    delta_script = root / 'code' / 'baselines' / 'train_with_delta_features.py'

    run_all = not (args.baselines or args.fusion or args.gru or args.aligner or args.ablation or args.delta)

    if args.baselines or run_all:
        run_script(root, baselines_script, "运行Structured基线")

    if args.fusion or run_all:
        run_script(root, fusion_script, "运行Fusion实验")

    if args.gru or run_all:
        run_script(root, gru_script, "运行GRU模型")

    if args.aligner or run_all:
        run_script(root, aligner_script, "运行Canonical Aligner比较 (D0/6h/12h/24h)")

    if args.ablation or run_all:
        run_script(root, ablation_script, "运行Note-category Ablation")

    if args.delta or run_all:
        run_script(root, delta_script, "运行Delta特征训练")

    print("\n" + "=" * 60)
    print("Pipeline完成!")
    print("=" * 60)
    print("\n主要结果目录:")
    print("  - results/benchmark_results/")
    print("  - results/fusion_baselines/")
    print("  - results/aligner_comparison/")
    print("  - results/note_ablation/")


if __name__ == "__main__":
    main()
