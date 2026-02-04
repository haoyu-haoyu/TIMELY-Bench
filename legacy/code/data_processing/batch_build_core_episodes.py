"""
批量构建核心Episode数据集
处理 core_episode_selection.csv 中选定的 3000 个 stay_id

功能：
1. 读取已选定的 3000 个 stay_id
2. 检查 episodes_enhanced/ 中是否已存在
3. 如果存在则复制；不存在则调用 builder + enhancer 生成
4. 支持断点续传、进度跟踪、错误记录

设计：
- 使用多进程加速处理
- 显示详细进度条
- 记录失败的 stay_id 到日志
- 输出完整统计信息
"""

import pandas as pd
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from multiprocessing import Pool, Manager, cpu_count
from datetime import datetime
import logging
import sys

# 导入现有的 Builder 和 Enhancer
from episode_builder import EpisodeBuilder, NumpyEncoder
from episode_enhancer import EpisodeEnhancer

# ==========================================
# 配置
# ==========================================

# 脚本所在目录
_SCRIPT_DIR = Path(__file__).parent
# 项目根目录 (TIMELY-Bench_Final)
PROJECT_ROOT = _SCRIPT_DIR.parent.parent

# Episodes目录在项目根目录下
EPISODES_DIR = PROJECT_ROOT / 'episodes'
CORE_SELECTION_CSV = EPISODES_DIR / 'episodes_core' / 'core_episode_selection.csv'
EPISODES_ENHANCED_DIR = EPISODES_DIR / 'episodes_enhanced'
EPISODES_CORE_DIR = EPISODES_DIR / 'episodes_core'
LOG_FILE = EPISODES_CORE_DIR / 'batch_build.log'
FAILED_IDS_FILE = EPISODES_CORE_DIR / 'failed_stay_ids.txt'

# 进程数（留一个核心给系统）
N_WORKERS = max(1, cpu_count() - 1)

# ==========================================
# 日志配置
# ==========================================

def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ==========================================
# 核心处理逻辑
# ==========================================

def check_existing_file(stay_id: int, source_dir: Path, target_dir: Path) -> Optional[Path]:
    """
    检查文件是否已存在

    Returns:
        如果文件存在返回源文件路径，否则返回 None
    """
    filename = f"TIMELY_v2_{stay_id}.json"

    # 1. 先检查目标目录（episodes_core）是否已有
    target_file = target_dir / filename
    if target_file.exists():
        return None  # 已处理，跳过

    # 2. 检查源目录（episodes_enhanced）是否存在
    source_file = source_dir / filename
    if source_file.exists():
        return source_file

    return None


# 全局变量用于进程间共享已加载的数据
_global_builder = None
_global_enhancer = None


def init_worker():
    """初始化工作进程：加载数据到全局变量"""
    global _global_builder, _global_enhancer
    _global_builder = EpisodeBuilder()
    _global_enhancer = EpisodeEnhancer()
    _global_builder.load_all_data()
    _global_enhancer.aligner.load_data()


def process_single_stay_id(args: Tuple, builder=None, enhancer=None, force_rebuild=False) -> Dict:
    """
    处理单个 stay_id

    Args:
        args: (stay_id, source_dir, target_dir) 或 (stay_id, source_dir, target_dir, force_rebuild)
        builder: 可选的预加载builder（顺序模式）
        enhancer: 可选的预加载enhancer（顺序模式）
        force_rebuild: 是否强制重新生成

    Returns:
        结果字典：{'stay_id', 'status', 'message', 'method'}
    """
    global _global_builder, _global_enhancer

    # 支持两种 args 格式
    if len(args) == 4:
        stay_id, source_dir, target_dir, force_rebuild = args
    else:
        stay_id, source_dir, target_dir = args

    result = {
        'stay_id': stay_id,
        'status': 'unknown',
        'message': '',
        'method': ''
    }

    try:
        # 如果force_rebuild，跳过已存在检查，直接构建
        if force_rebuild:
            # 强制重新构建
            pass  # 直接进入构建流程
        else:
            # 检查是否已存在
            existing_file = check_existing_file(stay_id, source_dir, target_dir)

            if existing_file:
                # 方法1：直接复制
                target_file = target_dir / existing_file.name
                shutil.copy2(existing_file, target_file)
                result['status'] = 'success'
                result['method'] = 'copy'
                result['message'] = f'Copied from {existing_file.name}'
                return result

            elif (target_dir / f"TIMELY_v2_{stay_id}.json").exists():
                # 已经处理过了
                result['status'] = 'skipped'
                result['method'] = 'already_exists'
                result['message'] = 'Already exists in target directory'
                return result

        # 方法2：需要构建
        # 使用传入的或全局的 builder/enhancer
        b = builder if builder else _global_builder
        e = enhancer if enhancer else _global_enhancer

        # 构建基础 Episode
        episode = b.build_episode(stay_id)

        if episode is None:
            result['status'] = 'failed'
            result['method'] = 'build'
            result['message'] = 'Failed to build episode (no data found)'
            return result

        # 转换为字典
        episode_dict = episode.to_dict()

        # 增强 Episode
        enhanced_dict = e.enhance_episode(episode_dict)

        # 保存到目标目录
        target_file = target_dir / f"TIMELY_v2_{stay_id}.json"
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_dict, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

        result['status'] = 'success'
        result['method'] = 'build_enhance'
        result['message'] = f'Built and enhanced from scratch{" (force rebuild)" if force_rebuild else ""}'

    except Exception as e:
        result['status'] = 'failed'
        result['message'] = str(e)

    return result


def process_single_stay_id_parallel(args: Tuple) -> Dict:
    """并行模式下的包装函数，使用全局的builder/enhancer"""
    return process_single_stay_id(args)


def process_batch_sequential(stay_ids: List[int], source_dir: Path,
                            target_dir: Path, force_rebuild: bool = False) -> List[Dict]:
    """
    顺序处理（用于调试或单进程模式）
    优化：只加载一次数据，复用builder和enhancer
    """
    results = []

    # 初始化 builder 和 enhancer（顺序模式下共享）
    builder = EpisodeBuilder()
    enhancer = EpisodeEnhancer()

    print("Loading data once (optimized sequential mode)...")
    builder.load_all_data()
    enhancer.aligner.load_data()

    print(f"\n🔄 Processing {len(stay_ids)} episodes (sequential mode)...")
    if force_rebuild:
        print("   Force rebuild mode: will overwrite existing files")

    for stay_id in tqdm(stay_ids, desc="Processing"):
        args = (stay_id, source_dir, target_dir)
        # 传入预加载的builder和enhancer，以及force_rebuild参数
        result = process_single_stay_id(args, builder=builder, enhancer=enhancer, force_rebuild=force_rebuild)
        results.append(result)

    return results


def process_batch_parallel(stay_ids: List[int], source_dir: Path,
                          target_dir: Path, n_workers: int, force_rebuild: bool = False) -> List[Dict]:
    """
    并行处理（多进程）
    优化：每个进程只加载一次数据
    """
    print(f"\n🔄 Processing {len(stay_ids)} episodes with {n_workers} workers...")
    print(f"   Each worker will load data once at initialization...")
    if force_rebuild:
        print("   Force rebuild mode: will overwrite existing files")

    # 准备参数 - 包含force_rebuild
    args_list = [(stay_id, source_dir, target_dir, force_rebuild) for stay_id in stay_ids]

    # 使用进程池，指定初始化函数
    results = []
    with Pool(processes=n_workers, initializer=init_worker) as pool:
        # 使用 imap 以支持进度条
        for result in tqdm(
            pool.imap(process_single_stay_id_parallel, args_list),
            total=len(stay_ids),
            desc="Processing"
        ):
            results.append(result)

    return results


# ==========================================
# 主流程
# ==========================================

def main(use_parallel: bool = True, max_episodes: Optional[int] = None, force_rebuild: bool = False):
    """
    主处理流程

    Args:
        use_parallel: 是否使用多进程并行
        max_episodes: 最大处理数量（用于测试）
        force_rebuild: 是否强制重新生成所有Episode（忽略已存在文件）
    """
    logger = setup_logging()

    print("=" * 80)
    print("批量构建核心Episode数据集")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {N_WORKERS if use_parallel else 1}")
    print()

    # 创建目标目录
    EPISODES_CORE_DIR.mkdir(parents=True, exist_ok=True)

    # 读取选定的 stay_ids
    logger.info(f"Reading stay_ids from {CORE_SELECTION_CSV}")
    if not CORE_SELECTION_CSV.exists():
        logger.error(f"Selection file not found: {CORE_SELECTION_CSV}")
        return

    df = pd.read_csv(CORE_SELECTION_CSV)
    stay_ids = df['stay_id'].tolist()

    if max_episodes:
        stay_ids = stay_ids[:max_episodes]
        logger.info(f"Limited to first {max_episodes} episodes for testing")

    logger.info(f"Found {len(stay_ids)} stay_ids to process")

    # 检查已存在的文件（断点续传）
    existing_files = list(EPISODES_CORE_DIR.glob('TIMELY_v2_*.json'))
    existing_stay_ids = set()
    
    if force_rebuild:
        logger.info(f"Force rebuild enabled - will regenerate all {len(stay_ids)} episodes")
        # 清空已存在文件列表，强制重新生成
        existing_stay_ids = set()
    else:
        for f in existing_files:
            try:
                stay_id = int(f.stem.replace('TIMELY_v2_', ''))
                existing_stay_ids.add(stay_id)
            except:
                pass
        logger.info(f"Found {len(existing_stay_ids)} already processed episodes")

    # 过滤出需要处理的 stay_ids
    stay_ids_to_process = [sid for sid in stay_ids if sid not in existing_stay_ids]

    if not stay_ids_to_process:
        logger.info("All episodes already processed!")
        print_summary({}, len(stay_ids), 0)
        return

    logger.info(f"Need to process {len(stay_ids_to_process)} episodes")

    # 处理
    start_time = datetime.now()

    if use_parallel:
        results = process_batch_parallel(
            stay_ids_to_process,
            EPISODES_ENHANCED_DIR,
            EPISODES_CORE_DIR,
            N_WORKERS,
            force_rebuild=force_rebuild
        )
    else:
        results = process_batch_sequential(
            stay_ids_to_process,
            EPISODES_ENHANCED_DIR,
            EPISODES_CORE_DIR,
            force_rebuild=force_rebuild
        )

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # 统计结果
    stats = {
        'total': len(stay_ids_to_process),
        'success': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'skipped': sum(1 for r in results if r['status'] == 'skipped'),
        'copied': sum(1 for r in results if r['method'] == 'copy'),
        'built': sum(1 for r in results if r['method'] == 'build_enhance'),
        'duration': duration
    }

    # 记录失败的 stay_ids
    failed_results = [r for r in results if r['status'] == 'failed']
    if failed_results:
        with open(FAILED_IDS_FILE, 'w', encoding='utf-8') as f:
            for r in failed_results:
                f.write(f"{r['stay_id']}\t{r['message']}\n")
        logger.warning(f"Failed stay_ids saved to: {FAILED_IDS_FILE}")

    # 打印摘要
    print_summary(stats, len(stay_ids), len(existing_stay_ids))

    # 日志记录
    logger.info("=" * 80)
    logger.info("Processing complete")
    logger.info(f"Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
    logger.info(f"Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
    logger.info("=" * 80)


def print_summary(stats: Dict, total_target: int, already_done: int):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print("处理摘要")
    print("=" * 80)

    print(f"\n[目标统计]")
    print(f"  目标总数: {total_target}")
    print(f"  已完成: {already_done}")
    print(f"  本次处理: {stats.get('total', 0)}")

    if stats:
        print(f"\n[本次处理结果]")
        print(f"  成功: {stats['success']} ({stats['success']/max(stats['total'], 1)*100:.1f}%)")
        print(f"  失败: {stats['failed']} ({stats['failed']/max(stats['total'], 1)*100:.1f}%)")
        print(f"  跳过: {stats['skipped']}")

        print(f"\n[处理方式]")
        print(f"  复制: {stats['copied']}")
        print(f"  构建: {stats['built']}")

        print(f"\n[性能]")
        print(f"  总耗时: {stats['duration']:.1f}秒 ({stats['duration']/60:.1f}分钟)")
        if stats['total'] > 0:
            print(f"  平均速度: {stats['duration']/stats['total']:.2f}秒/episode")

    print(f"\n[输出目录]")
    print(f"  {EPISODES_CORE_DIR}/")

    current_files = len(list(EPISODES_CORE_DIR.glob('TIMELY_v2_*.json')))
    print(f"\n[当前状态]")
    print(f"  已完成: {current_files}/{total_target} episodes")
    print(f"  完成率: {current_files/total_target*100:.1f}%")

    print("=" * 80)


# ==========================================
# 命令行入口
# ==========================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='批量构建核心Episode数据集')
    parser.add_argument('--sequential', action='store_true',
                       help='使用顺序处理（单进程，用于调试）')
    parser.add_argument('--max', type=int, default=None,
                       help='最大处理数量（用于测试）')
    parser.add_argument('--workers', type=int, default=N_WORKERS,
                       help=f'进程数（默认：{N_WORKERS}）')
    parser.add_argument('--force', action='store_true',
                       help='强制重新生成所有Episode（忽略已存在文件）')

    args = parser.parse_args()

    # 更新进程数
    if args.workers:
        N_WORKERS = args.workers

    # 运行
    try:
        main(use_parallel=not args.sequential, max_episodes=args.max, force_rebuild=args.force)
    except KeyboardInterrupt:
        print("\n\n 用户中断，已保存当前进度")
        print("可以重新运行脚本继续处理（支持断点续传）")
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
