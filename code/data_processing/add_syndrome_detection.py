"""
为所有 Episode 添加综合征检测结果
利用现有的 pattern_annotations、aligned_text 和 MedCAT 概念
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))
from syndrome_detector import SyndromeDetector, get_accuracy

# 配置
BASE_DIR = Path(__file__).resolve().parents[2]
EPISODES_DIR = BASE_DIR / 'episodes' / 'episodes_enhanced'
RULES_FILE = BASE_DIR / 'code' / 'config' / 'diagnostic_rules.json'
MEDCAT_FILE = BASE_DIR / 'data' / 'processed' / 'medcat_full' / 'medcat_has_concepts_24h.csv'
OUTPUT_DIR = EPISODES_DIR  # 原地更新


def main():
    print("=" * 60)
    print("综合征检测 - 为 Episodes 添加诊断结果")
    print("=" * 60)
    
    # 加载 MedCAT 概念
    medcat_df = None
    if MEDCAT_FILE.exists():
        medcat_df = pd.read_csv(MEDCAT_FILE)
        print(f"加载 MedCAT 概念: {len(medcat_df):,} 条")
    
    # 初始化检测器
    rules_path = str(RULES_FILE) if RULES_FILE.exists() else None
    detector = SyndromeDetector(rules_path=rules_path, medcat_df=medcat_df, use_condition_labels=False)
    print(f"初始化检测器完成")
    
    # 获取所有 Episode 文件
    episode_files = list(EPISODES_DIR.glob('*.json'))
    print(f"待处理 Episodes: {len(episode_files):,}")
    
    # 统计
    stats = {
        'sepsis': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
        'aki': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0},
        'ards': {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    }
    
    # 处理每个 Episode
    for ep_file in tqdm(episode_files, desc="检测综合征"):
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            
            # 检测综合征
            syndrome_results = detector.detect_all(ep)
            
            # 添加到 episode
            if 'reasoning' not in ep:
                ep['reasoning'] = {}
            ep['reasoning']['syndrome_detection'] = syndrome_results
            
            # 获取真实标签
            has_sepsis = ep.get('labels', {}).get('has_sepsis', False)
            has_aki = ep.get('labels', {}).get('has_aki', False)
            has_ards = ep.get('labels', {}).get('has_ards', False)
            
            # 计算诊断准确性
            sepsis_acc = get_accuracy(syndrome_results['sepsis']['detected'], has_sepsis)
            aki_acc = get_accuracy(syndrome_results['aki']['detected'], has_aki)
            ards_acc = get_accuracy(syndrome_results['ards']['detected'], has_ards)
            
            ep['reasoning']['diagnostic_accuracy'] = {
                'sepsis': sepsis_acc,
                'aki': aki_acc,
                'ards': ards_acc
            }
            
            # 更新统计
            stats['sepsis'][sepsis_acc] += 1
            stats['aki'][aki_acc] += 1
            stats['ards'][ards_acc] += 1
            
            # 保存
            with open(ep_file, 'w') as f:
                json.dump(ep, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"处理 {ep_file.name} 失败: {e}")
            continue
    
    # 输出结果
    print("\n" + "=" * 60)
    print("综合征检测结果汇总")
    print("=" * 60)
    
    for disease, counts in stats.items():
        tp, fp, fn, tn = counts['TP'], counts['FP'], counts['FN'], counts['TN']
        total = tp + fp + fn + tn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{disease.upper()}:")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision: {precision*100:.1f}%")
        print(f"  Recall: {recall*100:.1f}%")
        print(f"  F1-Score: {f1*100:.1f}%")
    
    print("\n完成！")


if __name__ == "__main__":
    main()
