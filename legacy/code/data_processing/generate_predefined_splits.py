"""
生成并保存固定的 train/val/test 数据划分

确保可复现性：在发布后任何人可以使用相同的划分
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# 配置
COHORT_FILE = Path(__file__).parent.parent.parent / 'data' / 'processed' / 'cohorts' / 'cohort_with_conditions.csv'
OUTPUT_DIR = Path(__file__).parent.parent.parent / 'data' / 'processed'
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # of train+val


def main():
    print("=" * 60)
    print("生成 Predefined Data Splits")
    print("=" * 60)
    
    # 加载 cohort
    cohort = pd.read_csv(COHORT_FILE)
    print(f"Total episodes: {len(cohort)}")
    
    stay_ids = cohort['stay_id'].values
    subject_ids = cohort['subject_id'].values
    
    # 第一次划分: train+val vs test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss1.split(stay_ids, groups=subject_ids))
    
    # 第二次划分: train vs val
    train_val_stay_ids = stay_ids[train_val_idx]
    train_val_subject_ids = subject_ids[train_val_idx]
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE / (1 - TEST_SIZE), random_state=RANDOM_STATE)
    train_idx_rel, val_idx_rel = next(gss2.split(train_val_stay_ids, groups=train_val_subject_ids))
    
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    
    # 提取 stay_ids
    train_stays = stay_ids[train_idx]
    val_stays = stay_ids[val_idx]
    test_stays = stay_ids[test_idx]
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_stays)} ({len(train_stays)/len(stay_ids)*100:.1f}%)")
    print(f"  Val:   {len(val_stays)} ({len(val_stays)/len(stay_ids)*100:.1f}%)")
    print(f"  Test:  {len(test_stays)} ({len(test_stays)/len(stay_ids)*100:.1f}%)")
    
    # 验证无患者泄漏
    train_subjects = set(subject_ids[train_idx])
    val_subjects = set(subject_ids[val_idx])
    test_subjects = set(subject_ids[test_idx])
    
    assert len(train_subjects & val_subjects) == 0, "Train-Val overlap!"
    assert len(train_subjects & test_subjects) == 0, "Train-Test overlap!"
    assert len(val_subjects & test_subjects) == 0, "Val-Test overlap!"
    print("\nNo patient overlap between splits!")
    
    # 保存
    splits_df = pd.DataFrame({
        'stay_id': np.concatenate([train_stays, val_stays, test_stays]),
        'split': ['train'] * len(train_stays) + ['val'] * len(val_stays) + ['test'] * len(test_stays)
    })
    
    output_path = OUTPUT_DIR / 'predefined_splits.csv'
    splits_df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    
    # 统计
    print("\nSplit distribution:")
    print(splits_df['split'].value_counts())


if __name__ == "__main__":
    main()
