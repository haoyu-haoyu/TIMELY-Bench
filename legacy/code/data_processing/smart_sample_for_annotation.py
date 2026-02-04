"""
智能采样脚本：从核心 3000 stay_id 的对齐数据中采样代表性样本
用于 LLM 标注

采样策略：
1. 只选择 alignment_quality = 'high' 的对齐
2. 每个 stay_id 采样 5 条代表性样本
3. 确保覆盖不同 pattern_name 和 note_type
4. 对每个 pattern_name + note_type 组合去重
"""

import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR, PROCESSED_DIR

# 配置
ALIGNMENT_FILE = TEMPORAL_ALIGNMENT_DIR / 'temporal_textual_alignment_core3000.csv'
OUTPUT_DIR = PROCESSED_DIR / 'pattern_annotations'
OUTPUT_FILE = OUTPUT_DIR / 'samples_for_llm_annotation.csv'

# 采样参数
MAX_SAMPLES_PER_STAY = 5  # 每个 stay_id 最多采样数
MIN_TEXT_LENGTH = 50      # 最小文本长度
ONLY_HIGH_QUALITY = True  # 只采样高质量对齐


def smart_sample_for_annotation(
    alignment_file: Path = ALIGNMENT_FILE,
    output_file: Path = OUTPUT_FILE,
    samples_per_stay: int = MAX_SAMPLES_PER_STAY,
    dry_run: bool = False
):
    """从对齐数据中智能采样用于 LLM 标注"""
    
    print("=" * 60)
    print("智能采样：准备 LLM 标注数据")
    print("=" * 60)
    
    # 1. 加载对齐数据
    print(f"\n加载对齐数据: {alignment_file}")
    if not alignment_file.exists():
        print(f"错误: 文件不存在 {alignment_file}")
        return None
    
    df = pd.read_csv(alignment_file, low_memory=False)
    print(f"  总记录数: {len(df):,}")
    print(f"  stay_id 数: {df['stay_id'].nunique():,}")
    
    # 2. 过滤条件
    print("\n应用过滤条件...")
    
    # 只保留高质量对齐
    if ONLY_HIGH_QUALITY and 'alignment_quality' in df.columns:
        df = df[df['alignment_quality'] == 'high']
        print(f"  高质量对齐: {len(df):,}")
    
    # 过滤文本长度
    if 'note_text_relevant' in df.columns:
        df['text_length'] = df['note_text_relevant'].fillna('').str.len()
        df = df[df['text_length'] >= MIN_TEXT_LENGTH]
        print(f"  文本长度 >= {MIN_TEXT_LENGTH}: {len(df):,}")
    
    # 3. 分层采样
    print("\n分层采样...")
    
    # 按 stay_id 分组
    grouped = df.groupby('stay_id')
    
    sampled_rows = []
    for stay_id, group in grouped:
        # 对每个 stay_id，按 pattern_name 和 note_type 分层
        # 确保多样性
        
        # 策略：每个 (pattern_name, note_type) 组合选 1 条
        if 'pattern_name' in group.columns and 'note_type' in group.columns:
            unique_combinations = group.drop_duplicates(
                subset=['pattern_name', 'note_type'], 
                keep='first'
            )
        else:
            unique_combinations = group.drop_duplicates(keep='first')
        
        # 限制每个 stay_id 的数量
        if len(unique_combinations) > samples_per_stay:
            unique_combinations = unique_combinations.sample(
                n=samples_per_stay, 
                random_state=42
            )
        
        sampled_rows.append(unique_combinations)
    
    # 合并采样结果
    sampled_df = pd.concat(sampled_rows, ignore_index=True)
    
    # 4. 统计
    print(f"\n采样结果:")
    print(f"  总采样数: {len(sampled_df):,}")
    print(f"  stay_id 数: {sampled_df['stay_id'].nunique():,}")
    
    if 'pattern_name' in sampled_df.columns:
        print(f"\n  Pattern 分布:")
        pattern_counts = sampled_df['pattern_name'].value_counts().head(10)
        for pattern, count in pattern_counts.items():
            print(f"    {pattern}: {count}")
    
    if 'note_type' in sampled_df.columns:
        print(f"\n  Note Type 分布:")
        for note_type, count in sampled_df['note_type'].value_counts().items():
            print(f"    {note_type}: {count}")
    
    # 5. 保存
    if not dry_run:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 选择必要的列
        output_cols = [
            'stay_id', 'pattern_name', 'pattern_hour', 'pattern_value',
            'pattern_severity', 'note_type', 'note_id', 'note_hour',
            'alignment_quality', 'note_text_relevant'
        ]
        output_cols = [c for c in output_cols if c in sampled_df.columns]
        
        sampled_df[output_cols].to_csv(output_file, index=False)
        print(f"\n已保存采样数据: {output_file}")
        print(f"  文件大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("\n[Dry Run] 未保存文件")
    
    return sampled_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='智能采样用于 LLM 标注')
    parser.add_argument('--samples-per-stay', type=int, default=MAX_SAMPLES_PER_STAY,
                        help='每个 stay_id 采样数量')
    parser.add_argument('--dry-run', action='store_true',
                        help='只统计不保存')
    args = parser.parse_args()
    
    smart_sample_for_annotation(
        samples_per_stay=args.samples_per_stay,
        dry_run=args.dry_run
    )
