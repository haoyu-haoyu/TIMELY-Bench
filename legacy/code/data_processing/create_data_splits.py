"""
创建训练/验证/测试集划分
按 subject_id 分组避免数据泄露

策略：
1. 按 subject_id 分组（同一患者的多次住院不会分到不同集）
2. 分层采样保证标签分布一致
3. 划分比例：70% train, 15% val, 15% test
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

# 路径配置
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from config import PROCESSED_DIR, SPLITS_DIR, COHORT_FILE

RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def create_splits():
    """创建数据集划分"""
    print("=" * 60)
    print("创建训练/验证/测试集划分")
    print("=" * 60)
    
    # 加载 cohort
    print("\n1. 加载 cohort 数据...")
    cohort = pd.read_csv(COHORT_FILE)
    print(f"   总 stay_id: {len(cohort):,}")
    print(f"   唯一 subject_id: {cohort['subject_id'].nunique():,}")
    
    # 创建 subject 级别的标签（一个 subject 如果有任何一次 mortality 就标为 1）
    print("\n2. 创建 subject 级别标签...")
    subject_labels = cohort.groupby('subject_id').agg({
        'label_mortality': 'max',  # 任意一次死亡
        'prolonged_los_7d': 'max',  # 任意一次长住院
        'stay_id': 'count'
    }).rename(columns={'stay_id': 'n_stays'}).reset_index()
    
    print(f"   Subject 级别 mortality=1: {(subject_labels['label_mortality']==1).sum()}")
    print(f"   Subject 级别 prolonged_los=1: {(subject_labels['prolonged_los_7d']==1).sum()}")
    
    # 按 subject_id 划分（分层采样按 mortality）
    print("\n3. 按 subject_id 分层划分...")
    
    # 第一次划分：分出 train (70%) 和 temp (30%)
    train_subjects, temp_subjects = train_test_split(
        subject_labels,
        test_size=(VAL_RATIO + TEST_RATIO),
        stratify=subject_labels['label_mortality'],
        random_state=RANDOM_STATE
    )
    
    # 第二次划分：从 temp 中分出 val 和 test（各 50%）
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=0.5,
        stratify=temp_subjects['label_mortality'],
        random_state=RANDOM_STATE
    )
    
    print(f"   Train subjects: {len(train_subjects):,}")
    print(f"   Val subjects: {len(val_subjects):,}")
    print(f"   Test subjects: {len(test_subjects):,}")
    
    # 获取每个集的 stay_id
    print("\n4. 映射到 stay_id...")
    train_stay_ids = cohort[cohort['subject_id'].isin(train_subjects['subject_id'])]['stay_id'].tolist()
    val_stay_ids = cohort[cohort['subject_id'].isin(val_subjects['subject_id'])]['stay_id'].tolist()
    test_stay_ids = cohort[cohort['subject_id'].isin(test_subjects['subject_id'])]['stay_id'].tolist()
    
    print(f"   Train stay_ids: {len(train_stay_ids):,} ({len(train_stay_ids)/len(cohort)*100:.1f}%)")
    print(f"   Val stay_ids: {len(val_stay_ids):,} ({len(val_stay_ids)/len(cohort)*100:.1f}%)")
    print(f"   Test stay_ids: {len(test_stay_ids):,} ({len(test_stay_ids)/len(cohort)*100:.1f}%)")
    
    # 验证标签分布
    print("\n5. 验证标签分布...")
    for name, stay_ids in [('Train', train_stay_ids), ('Val', val_stay_ids), ('Test', test_stay_ids)]:
        subset = cohort[cohort['stay_id'].isin(stay_ids)]
        mortality_rate = subset['label_mortality'].mean() * 100
        los_rate = subset['prolonged_los_7d'].mean() * 100
        print(f"   {name}: mortality={mortality_rate:.1f}%, prolonged_los={los_rate:.1f}%")
    
    # 保存划分
    print("\n6. 保存划分文件...")
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 保存完整信息（包含标签）
    train_df = cohort[cohort['stay_id'].isin(train_stay_ids)][['stay_id', 'subject_id', 'label_mortality', 'prolonged_los_7d']]
    val_df = cohort[cohort['stay_id'].isin(val_stay_ids)][['stay_id', 'subject_id', 'label_mortality', 'prolonged_los_7d']]
    test_df = cohort[cohort['stay_id'].isin(test_stay_ids)][['stay_id', 'subject_id', 'label_mortality', 'prolonged_los_7d']]
    
    train_df.to_csv(SPLITS_DIR / 'train.csv', index=False)
    val_df.to_csv(SPLITS_DIR / 'val.csv', index=False)
    test_df.to_csv(SPLITS_DIR / 'test.csv', index=False)
    
    print(f"   保存到: {SPLITS_DIR}/")
    print(f"     - train.csv: {len(train_df):,} stay_ids")
    print(f"     - val.csv: {len(val_df):,} stay_ids")
    print(f"     - test.csv: {len(test_df):,} stay_ids")
    
    # 验证无数据泄露
    print("\n7. 验证无数据泄露...")
    train_subjects_set = set(train_df['subject_id'])
    val_subjects_set = set(val_df['subject_id'])
    test_subjects_set = set(test_df['subject_id'])
    
    train_val_overlap = train_subjects_set & val_subjects_set
    train_test_overlap = train_subjects_set & test_subjects_set
    val_test_overlap = val_subjects_set & test_subjects_set
    
    if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
        print("   ✅ 无数据泄露！所有集之间的 subject_id 完全不重叠")
    else:
        print(f"   ❌ 发现泄露！Train-Val: {len(train_val_overlap)}, Train-Test: {len(train_test_overlap)}, Val-Test: {len(val_test_overlap)}")
    
    print("\n" + "=" * 60)
    print("完成！")
    print("=" * 60)


if __name__ == "__main__":
    create_splits()
