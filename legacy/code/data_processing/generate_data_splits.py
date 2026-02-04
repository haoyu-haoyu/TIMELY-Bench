"""
TIMELY-Bench Data Split Generator
生成预定义的train/val/test数据分割

分割策略:
- 70% train / 15% validation / 15% test
- 按subject_id分组（防止同一患者在不同split中）
- 分层抽样保持疾病/标签分布

输出:
- data_splits/train.csv
- data_splits/val.csv
- data_splits/test.csv
- data_splits/split_summary.json
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# 配置
ROOT_DIR = Path(__file__).parent
CORE_EPISODES_FILE = ROOT_DIR / 'episodes_core' / 'core_episode_selection.csv'
COHORT_FILE = ROOT_DIR / 'merge_output' / 'cohort_final.csv'
OUTPUT_DIR = ROOT_DIR / 'data_splits'

# 分割比例
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def load_data():
    """加载核心数据集"""
    print("Loading core episode data...")

    # 加载核心episodes
    if not CORE_EPISODES_FILE.exists():
        raise FileNotFoundError(f"Core episodes file not found: {CORE_EPISODES_FILE}")

    core_df = pd.read_csv(CORE_EPISODES_FILE)
    print(f"   Loaded {len(core_df)} core episodes")

    # 加载cohort获取subject_id
    if COHORT_FILE.exists():
        cohort_df = pd.read_csv(COHORT_FILE)
        cohort_df = cohort_df[['stay_id', 'subject_id']].drop_duplicates()
        core_df = core_df.merge(cohort_df, on='stay_id', how='left')
        print(f"   Merged subject_id from cohort")
    else:
        # 如果没有cohort文件，使用stay_id作为subject_id
        core_df['subject_id'] = core_df['stay_id']
        print(f"   Using stay_id as subject_id (no cohort file)")

    return core_df


def create_stratification_key(df: pd.DataFrame) -> pd.Series:
    """创建分层抽样的键"""
    # 组合多个标签创建分层键
    # mortality + has_sepsis + has_aki
    keys = (
        df['mortality'].astype(str) + '_' +
        df['has_sepsis'].astype(str) + '_' +
        df['has_aki'].astype(str)
    )
    return keys


def split_by_subject(df: pd.DataFrame) -> tuple:
    """
    按subject_id分割数据
    确保同一患者的所有stays都在同一个split中
    """
    print("\nSplitting data by subject_id...")

    # 获取唯一的subject_ids
    unique_subjects = df['subject_id'].unique()
    n_subjects = len(unique_subjects)
    print(f"   Total unique subjects: {n_subjects}")

    # 为每个subject创建分层键（使用该subject的第一个stay的标签）
    subject_labels = df.groupby('subject_id').first()[['mortality', 'has_sepsis', 'has_aki']]
    subject_strat_key = (
        subject_labels['mortality'].astype(str) + '_' +
        subject_labels['has_sepsis'].astype(str) + '_' +
        subject_labels['has_aki'].astype(str)
    )

    # 第一次分割: train vs (val + test)
    train_subjects, temp_subjects = train_test_split(
        unique_subjects,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=subject_strat_key.loc[unique_subjects]
    )

    # 第二次分割: val vs test
    temp_strat_key = subject_strat_key.loc[temp_subjects]
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        stratify=temp_strat_key
    )

    # 创建split标签
    df['split'] = 'unknown'
    df.loc[df['subject_id'].isin(train_subjects), 'split'] = 'train'
    df.loc[df['subject_id'].isin(val_subjects), 'split'] = 'val'
    df.loc[df['subject_id'].isin(test_subjects), 'split'] = 'test'

    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    print(f"   Train: {len(train_df)} episodes ({len(train_subjects)} subjects)")
    print(f"   Val:   {len(val_df)} episodes ({len(val_subjects)} subjects)")
    print(f"   Test:  {len(test_df)} episodes ({len(test_subjects)} subjects)")

    return train_df, val_df, test_df


def compute_split_statistics(train_df, val_df, test_df) -> dict:
    """计算各split的统计信息"""
    stats = {}

    for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        stats[name] = {
            'n_episodes': len(df),
            'n_subjects': df['subject_id'].nunique(),
            'mortality_rate': round(df['mortality'].mean(), 3),
            'sepsis_rate': round(df['has_sepsis'].mean(), 3),
            'aki_rate': round(df['has_aki'].mean(), 3),
            'ards_rate': round(df['has_ards'].mean(), 3) if 'has_ards' in df.columns else 0,
            'avg_quality_score': round(df['quality_score'].mean(), 3),
            'avg_n_patterns': round(df['n_patterns'].mean(), 1),
            'avg_n_alignments': round(df['n_alignments'].mean(), 1),
        }

    return stats


def save_splits(train_df, val_df, test_df, stats: dict):
    """保存分割结果"""
    print("\nSaving data splits...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 保存CSV文件
    columns_to_save = ['stay_id', 'subject_id', 'split', 'mortality',
                       'has_sepsis', 'has_aki', 'has_ards', 'quality_score',
                       'n_patterns', 'n_alignments', 'has_alignment']

    train_df[columns_to_save].to_csv(OUTPUT_DIR / 'train.csv', index=False)
    val_df[columns_to_save].to_csv(OUTPUT_DIR / 'val.csv', index=False)
    test_df[columns_to_save].to_csv(OUTPUT_DIR / 'test.csv', index=False)

    print(f"   train.csv: {len(train_df)} episodes")
    print(f"   val.csv: {len(val_df)} episodes")
    print(f"   test.csv: {len(test_df)} episodes")

    # 保存完整的stay_id列表（用于快速加载）
    split_ids = {
        'train': train_df['stay_id'].tolist(),
        'val': val_df['stay_id'].tolist(),
        'test': test_df['stay_id'].tolist()
    }
    with open(OUTPUT_DIR / 'split_ids.json', 'w') as f:
        json.dump(split_ids, f, indent=2)
    print(f"   split_ids.json")

    # 保存统计摘要
    summary = {
        'version': '2.0',
        'created': pd.Timestamp.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'split_ratios': {
            'train': TRAIN_RATIO,
            'val': VAL_RATIO,
            'test': TEST_RATIO
        },
        'split_method': 'subject_id_grouped_stratified',
        'stratification_keys': ['mortality', 'has_sepsis', 'has_aki'],
        'statistics': stats
    }

    with open(OUTPUT_DIR / 'split_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"   split_summary.json")


def print_summary(stats: dict):
    """打印分割摘要"""
    print("\n" + "=" * 70)
    print("DATA SPLIT SUMMARY")
    print("=" * 70)

    print("\n[Episode Distribution]")
    print(f"{'Split':<10} {'Episodes':>10} {'Subjects':>10} {'Mortality':>12} {'Sepsis':>10} {'AKI':>10}")
    print("-" * 62)

    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        print(f"{split_name:<10} {s['n_episodes']:>10} {s['n_subjects']:>10} "
              f"{s['mortality_rate']*100:>10.1f}% {s['sepsis_rate']*100:>9.1f}% "
              f"{s['aki_rate']*100:>9.1f}%")

    print("\n[Quality Metrics]")
    print(f"{'Split':<10} {'Avg Quality':>12} {'Avg Patterns':>14} {'Avg Alignments':>16}")
    print("-" * 52)

    for split_name in ['train', 'val', 'test']:
        s = stats[split_name]
        print(f"{split_name:<10} {s['avg_quality_score']:>12.3f} "
              f"{s['avg_n_patterns']:>14.1f} {s['avg_n_alignments']:>16.1f}")


def main():
    print("=" * 70)
    print("TIMELY-Bench Data Split Generator")
    print("=" * 70)

    # 加载数据
    df = load_data()

    # 分割数据
    train_df, val_df, test_df = split_by_subject(df)

    # 计算统计
    stats = compute_split_statistics(train_df, val_df, test_df)

    # 保存结果
    save_splits(train_df, val_df, test_df, stats)

    # 打印摘要
    print_summary(stats)

    print(f"\nData splits saved to: {OUTPUT_DIR}/")
    print("   Use these splits for consistent train/val/test evaluation")


if __name__ == "__main__":
    main()
