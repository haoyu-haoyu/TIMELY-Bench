"""
Episode 标注后处理脚本 v2.0
修复问题：
1. NULL 标注 → 默认设为 UNRELATED
2. 过滤冗余数据：只保留 SUPPORTIVE/CONTRADICTORY + 少量 UNRELATED 采样
3. 重新计算统计
"""

import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, Optional, Set
from tqdm import tqdm
from multiprocessing import Pool
import random

# 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent

ANNOTATIONS_FILE = PROJECT_ROOT / 'data' / 'processed' / 'pattern_annotations' / 'smart_annotations_full.csv'
EPISODES_ALL_DIR = PROJECT_ROOT / 'episodes' / 'episodes_enhanced'

# 配置
MAX_UNRELATED_PER_EPISODE = 50  # 每个 Episode 最多保留 50 条 UNRELATED
RANDOM_SEED = 42

# 全局标注索引
_annotation_index: Dict = {}
_covered_stay_ids: Set[int] = set()


def load_annotations():
    """加载标注数据并创建索引"""
    global _annotation_index, _covered_stay_ids
    
    print("加载标注数据...")
    
    if ANNOTATIONS_FILE.exists():
        df = pd.read_csv(ANNOTATIONS_FILE)
        print(f"   smart_annotations_v2.csv: {len(df):,} 条")
        
        # 创建索引: (stay_id, pattern_name) -> annotation
        print("创建索引...")
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Indexing"):
            stay_id = int(row['stay_id'])
            pattern_name = str(row['pattern_name'])
            key = (stay_id, pattern_name)
            
            _covered_stay_ids.add(stay_id)
            
            if key not in _annotation_index:
                _annotation_index[key] = {
                    'category': row.get('annotation_category', 'UNRELATED'),
                    'confidence': row.get('annotation_confidence', 0.5),
                    'reasoning': row.get('annotation_reasoning', ''),
                    'source': row.get('annotation_source', 'unknown')
                }
        
        print(f"   索引完成: {len(_annotation_index):,} 个 (stay_id, pattern) 组合")
        print(f"   覆盖 {len(_covered_stay_ids):,} 个 stay_id")
    else:
        print(f"   标注文件不存在: {ANNOTATIONS_FILE}")


def process_episode(episode_path: Path) -> Optional[Dict]:
    """处理单个 Episode - 修复标注并清理冗余"""
    global _annotation_index, _covered_stay_ids
    
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
        
        # 处理每条标注
        updated_annotations = []
        supportive_list = []
        contradictory_list = []
        unrelated_list = []
        
        for annot in annotations:
            pattern_name = annot.get('pattern_name', '')
            key = (stay_id, pattern_name)
            
            # 尝试从索引获取标注
            if key in _annotation_index:
                new_annot = _annotation_index[key]
                annot['annotation_category'] = new_annot['category']
                annot['annotation_confidence'] = new_annot['confidence']
                annot['annotation_reasoning'] = new_annot['reasoning']
                annot['annotation_source'] = new_annot['source']
            else:
                # 未覆盖的 → 设为 UNRELATED
                if annot.get('annotation_category') is None:
                    annot['annotation_category'] = 'UNRELATED'
                    annot['annotation_confidence'] = 0.3
                    annot['annotation_reasoning'] = 'Default: no annotation available'
                    annot['annotation_source'] = 'default'
            
            # 分类
            cat = annot.get('annotation_category', 'UNRELATED')
            if cat == 'SUPPORTIVE':
                supportive_list.append(annot)
            elif cat == 'CONTRADICTORY':
                contradictory_list.append(annot)
            else:
                unrelated_list.append(annot)
        
        # 过滤冗余：保留所有 SUPPORTIVE/CONTRADICTORY + 采样 UNRELATED
        random.seed(RANDOM_SEED + stay_id)
        if len(unrelated_list) > MAX_UNRELATED_PER_EPISODE:
            unrelated_sample = random.sample(unrelated_list, MAX_UNRELATED_PER_EPISODE)
        else:
            unrelated_sample = unrelated_list
        
        # 合并
        filtered_annotations = supportive_list + contradictory_list + unrelated_sample
        
        # 更新 Episode
        reasoning['pattern_annotations'] = filtered_annotations
        reasoning['n_alignments'] = len(filtered_annotations)
        reasoning['n_supportive'] = len(supportive_list)
        reasoning['n_contradictory'] = len(contradictory_list)
        reasoning['n_unrelated'] = len(unrelated_sample)
        reasoning['original_n_alignments'] = len(annotations)  # 保留原始数量
        ep['reasoning'] = reasoning
        
        # 保存
        with open(episode_path, 'w', encoding='utf-8') as f:
            json.dump(ep, f, indent=2, ensure_ascii=False)
        
        return {
            'status': 'success',
            'supportive': len(supportive_list),
            'contradictory': len(contradictory_list),
            'unrelated': len(unrelated_sample),
            'original': len(annotations),
            'filtered': len(filtered_annotations)
        }
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def main(n_workers: int = 20):
    """主处理流程"""
    print("=" * 60)
    print("Episode 标注后处理 v2.0")
    print("=" * 60)
    print(f"策略:")
    print(f"  1. NULL → UNRELATED (默认)")
    print(f"  2. 保留所有 SUPPORTIVE/CONTRADICTORY")
    print(f"  3. UNRELATED 采样: 最多 {MAX_UNRELATED_PER_EPISODE} 条/Episode")
    print()
    
    # 加载标注
    load_annotations()
    
    # 获取所有 Episode 文件
    print("\n扫描 Episode 文件...")
    episode_files = list(EPISODES_ALL_DIR.glob('TIMELY_v2_*.json'))
    print(f"   找到 {len(episode_files):,} 个 Episode")
    
    # 处理
    print(f"\n开始处理（使用 {n_workers} workers）...")
    
    results = []
    with Pool(processes=n_workers) as pool:
        for result in tqdm(
            pool.imap(process_episode, episode_files),
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
    total_unrelated = sum(r.get('unrelated', 0) for r in results if r)
    total_original = sum(r.get('original', 0) for r in results if r)
    total_filtered = sum(r.get('filtered', 0) for r in results if r)
    
    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"成功: {success:,}")
    print(f"错误: {errors}")
    print(f"跳过: {skipped}")
    print()
    print(f"标注统计:")
    print(f"  原始总对齐: {total_original:,}")
    print(f"  过滤后对齐: {total_filtered:,} ({total_filtered/total_original*100:.1f}%)")
    print(f"  SUPPORTIVE: {total_supportive:,}")
    print(f"  CONTRADICTORY: {total_contradictory:,}")
    print(f"  UNRELATED (采样): {total_unrelated:,}")
    print()
    print(f"数据压缩比: {total_original/total_filtered:.1f}x")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=20)
    args = parser.parse_args()
    
    random.seed(RANDOM_SEED)
    main(n_workers=args.workers)
