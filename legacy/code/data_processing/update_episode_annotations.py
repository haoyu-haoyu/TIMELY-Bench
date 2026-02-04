"""
Episode 标注后处理脚本
更新已生成 Episode 的 pattern_annotations 部分

策略：
1. 加载最新的智能标注数据 (smart_annotations_v2.csv)
2. 遍历所有 Episode JSON 文件
3. 更新 reasoning.pattern_annotations 中的标注信息
4. 重新计算 n_supportive, n_contradictory 等统计
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Optional
from tqdm import tqdm
from multiprocessing import Pool
import sys

# 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent

ANNOTATIONS_FILE = PROJECT_ROOT / 'data' / 'processed' / 'pattern_annotations' / 'smart_annotations_v2.csv'
EPISODES_ALL_DIR = PROJECT_ROOT / 'episodes' / 'episodes_enhanced'


# 全局标注索引
_annotation_index: Dict = {}


def load_annotations():
    """加载标注数据并创建索引"""
    global _annotation_index
    
    print("加载标注数据...")
    df = pd.read_csv(ANNOTATIONS_FILE)
    print(f"   总标注: {len(df):,} 条")
    
    # 创建索引: (stay_id, pattern_name) -> annotation
    print("创建索引...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
        key = (int(row['stay_id']), str(row['pattern_name']))
        if key not in _annotation_index:
            _annotation_index[key] = []
        _annotation_index[key].append({
            'category': row.get('annotation_category', 'UNRELATED'),
            'confidence': row.get('annotation_confidence', 0.5),
            'reasoning': row.get('annotation_reasoning', ''),
            'source': row.get('annotation_source', 'unknown')
        })
    
    print(f"   索引完成: {len(_annotation_index):,} 个 (stay_id, pattern) 组合")


def update_episode(episode_path: Path) -> Optional[Dict]:
    """更新单个 Episode 的标注"""
    global _annotation_index
    
    try:
        # 读取 Episode
        with open(episode_path, 'r', encoding='utf-8') as f:
            ep = json.load(f)
        
        stay_id = ep.get('stay_id')
        if not stay_id:
            return {'status': 'error', 'message': 'No stay_id'}
        
        # 获取 reasoning
        reasoning = ep.get('reasoning', {})
        annotations = reasoning.get('pattern_annotations', [])
        
        if not annotations:
            return {'status': 'skipped', 'message': 'No annotations'}
        
        # 更新标注
        updated_count = 0
        n_supportive = 0
        n_contradictory = 0
        
        for annot in annotations:
            pattern_name = annot.get('pattern_name', '')
            key = (stay_id, pattern_name)
            
            if key in _annotation_index:
                # 使用最新的标注（取第一个匹配的）
                new_annot = _annotation_index[key][0]
                annot['annotation_category'] = new_annot['category']
                annot['annotation_confidence'] = new_annot['confidence']
                annot['annotation_reasoning'] = new_annot['reasoning']
                annot['annotation_source'] = new_annot['source']
                updated_count += 1
            
            # 统计
            cat = annot.get('annotation_category', 'UNRELATED')
            if cat == 'SUPPORTIVE':
                n_supportive += 1
            elif cat == 'CONTRADICTORY':
                n_contradictory += 1
        
        # 更新统计
        reasoning['n_supportive'] = n_supportive
        reasoning['n_contradictory'] = n_contradictory
        reasoning['n_alignments'] = len(annotations)
        ep['reasoning'] = reasoning
        
        # 保存
        with open(episode_path, 'w', encoding='utf-8') as f:
            json.dump(ep, f, indent=2, ensure_ascii=False)
        
        return {
            'status': 'success',
            'updated': updated_count,
            'supportive': n_supportive,
            'contradictory': n_contradictory
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def main(n_workers: int = 10):
    """主处理流程"""
    print("=" * 60)
    print("Episode 标注后处理")
    print("=" * 60)
    
    # 加载标注
    load_annotations()
    
    # 获取所有 Episode 文件
    print("\n扫描 Episode 文件...")
    episode_files = list(EPISODES_ALL_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
    # 处理
    print(f"\n开始更新标注（使用 {n_workers} workers）...")
    
    results = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(update_episode, episode_files),
            total=len(episode_files),
            desc="Processing"
        ):
            results.append(result)
    
    # 统计
    success = sum(1 for r in results if r and r.get('status') == 'success')
    errors = sum(1 for r in results if r and r.get('status') == 'error')
    skipped = sum(1 for r in results if r and r.get('status') == 'skipped')
    total_supportive = sum(r.get('supportive', 0) for r in results if r)
    total_contradictory = sum(r.get('contradictory', 0) for r in results if r)
    
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功: {success:,}")
    print(f"错误: {errors}")
    print(f"跳过: {skipped}")
    print(f"总 SUPPORTIVE: {total_supportive:,}")
    print(f"总 CONTRADICTORY: {total_contradictory:,}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=10)
    args = parser.parse_args()
    
    main(n_workers=args.workers)
