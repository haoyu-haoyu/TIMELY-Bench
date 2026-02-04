"""
为核心 3000 个 stay_id 预过滤对齐数据

策略：
1. 分块读取 47GB CSV
2. 只保留 core_episode_selection.csv 中的 stay_id
3. 输出到较小的 CSV 文件（预计 ~2GB）
4. 用筛选后的文件进行 Episode 生成
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    ROOT_DIR, PROCESSED_DIR, TEMPORAL_ALIGNMENT_DIR
)

# 路径配置
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment.csv'
CORE_SELECTION_FILE = ROOT_DIR / 'episodes' / 'episodes_core' / 'core_episode_selection.csv'
OUTPUT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment_core3000.csv'


def filter_core_alignments():
    """筛选核心 3000 个 stay_id 的对齐数据"""
    
    print("=" * 60)
    print("筛选核心对齐数据")
    print("=" * 60)
    
    # 1. 加载核心 stay_id 列表
    if not CORE_SELECTION_FILE.exists():
        print(f"错误: 找不到核心选择文件 {CORE_SELECTION_FILE}")
        return False
    
    core_df = pd.read_csv(CORE_SELECTION_FILE)
    core_stay_ids = set(core_df['stay_id'].astype(int).tolist())
    print(f"核心 stay_id 数量: {len(core_stay_ids)}")
    
    # 2. 检查源文件
    if not ALIGNMENT_FILE.exists():
        print(f"错误: 找不到对齐文件 {ALIGNMENT_FILE}")
        return False
    
    file_size_gb = ALIGNMENT_FILE.stat().st_size / 1e9
    print(f"源文件大小: {file_size_gb:.1f} GB")
    
    # 3. 分块读取并筛选
    chunk_size = 500_000
    total_rows = 0
    filtered_rows = 0
    first_chunk = True
    
    print(f"\n开始分块读取并筛选...")
    
    reader = pd.read_csv(ALIGNMENT_FILE, chunksize=chunk_size, low_memory=False)
    
    for chunk_idx, chunk in enumerate(reader):
        chunk['stay_id'] = chunk['stay_id'].astype(int)
        total_rows += len(chunk)
        
        # 筛选核心 stay_id
        filtered_chunk = chunk[chunk['stay_id'].isin(core_stay_ids)]
        filtered_rows += len(filtered_chunk)
        
        # 写入输出文件
        if len(filtered_chunk) > 0:
            if first_chunk:
                filtered_chunk.to_csv(OUTPUT_FILE, index=False, mode='w')
                first_chunk = False
            else:
                filtered_chunk.to_csv(OUTPUT_FILE, index=False, mode='a', header=False)
        
        if (chunk_idx + 1) % 20 == 0:
            print(f"  已处理 {total_rows / 1e6:.1f}M 行, "
                  f"筛选 {filtered_rows / 1e6:.2f}M 条 ({filtered_rows/total_rows*100:.1f}%)")
    
    # 4. 输出统计
    output_size_gb = OUTPUT_FILE.stat().st_size / 1e9 if OUTPUT_FILE.exists() else 0
    
    print(f"\n" + "=" * 60)
    print(f"筛选完成!")
    print(f"=" * 60)
    print(f"源文件行数: {total_rows:,}")
    print(f"筛选后行数: {filtered_rows:,} ({filtered_rows/total_rows*100:.1f}%)")
    print(f"输出文件大小: {output_size_gb:.2f} GB")
    print(f"输出文件: {OUTPUT_FILE}")
    
    return True


if __name__ == "__main__":
    filter_core_alignments()
