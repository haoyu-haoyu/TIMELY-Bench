"""
合并 LOS 标签到主数据集
将 los_labels.csv 合并到 cohort_with_conditions.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os

from config import RAW_DATA_DIR, MERGE_OUTPUT_DIR

# 路径配置
COHORT_FILE = RAW_DATA_DIR / 'cohort_with_conditions.csv'
LOS_LABELS_FILE = RAW_DATA_DIR / 'los_labels.csv'
OUTPUT_DIR = MERGE_OUTPUT_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / 'cohort_final.csv'


def main():
    # 1. 加载数据
    print("[1] Loading data...")

    df_cohort = pd.read_csv(COHORT_FILE)
    df_los = pd.read_csv(LOS_LABELS_FILE)

    print(f"   Cohort: {len(df_cohort)} rows")
    print(f"   LOS Labels: {len(df_los)} rows")

    # 确保stay_id类型一致
    df_cohort['stay_id'] = df_cohort['stay_id'].astype(int)
    df_los['stay_id'] = df_los['stay_id'].astype(int)

    # ==========================================
    # 2. 合并数据
    # ==========================================
    print("\n[2] Merging LOS labels...")

    # 选择需要的列（避免重复的subject_id, hadm_id）
    los_cols = ['stay_id', 'los_hours', 'los_days',
                'prolonged_los_3d', 'prolonged_los_5d', 'prolonged_los_7d',
                'readmission_30d']

    df_merged = df_cohort.merge(
        df_los[los_cols],
        on='stay_id',
        how='left'
    )

    print(f"   Merged: {len(df_merged)} rows")

    # ==========================================
    # 2.1 修复 readmission vs mortality 冲突
    # ==========================================
    if 'label_mortality' in df_merged.columns and 'readmission_30d' in df_merged.columns:
        mortality_mask = df_merged['label_mortality'] == 1
        conflict_mask = mortality_mask & (df_merged['readmission_30d'] == 1)
        conflict_count = int(conflict_mask.sum())
        if conflict_count > 0:
            print(f"   修复冲突: {conflict_count} 条死亡样本被标记为再入院 -> 设为 NA")
        # 死亡样本 readmission 不适用，设为 NaN 以便后续过滤
        df_merged.loc[mortality_mask, 'readmission_30d'] = np.nan

    # 检查合并情况
    matched = df_merged['los_hours'].notna().sum()
    print(f"   LOS labels matched: {matched} ({matched/len(df_merged)*100:.1f}%)")

    # ==========================================
    # 3. 统计分析
    # ==========================================
    print("\n[3] Task Label Statistics:")
    print("=" * 60)

    # 原有标签
    print("\n[Mortality]")
    mortality_count = df_merged['label_mortality'].sum()
    print(f"   Deaths: {mortality_count} ({mortality_count/len(df_merged)*100:.1f}%)")

    # LOS 分布
    print("\n[Length of Stay Distribution]")
    print(f"   Mean LOS: {df_merged['los_days'].mean():.1f} days")
    print(f"   Median LOS: {df_merged['los_days'].median():.1f} days")
    print(f"   Min LOS: {df_merged['los_days'].min():.0f} days")
    print(f"   Max LOS: {df_merged['los_days'].max():.0f} days")

    # Prolonged LOS 标签分布
    print("\n[Prolonged LOS Labels]")
    for threshold in [3, 5, 7]:
        col = f'prolonged_los_{threshold}d'
        count = df_merged[col].sum()
        print(f"   LOS >= {threshold} days: {count} ({count/len(df_merged)*100:.1f}%)")

    # 30天再入院
    print("\n[30-day Readmission]")
    readmit_count = df_merged['readmission_30d'].sum()
    print(f"   Readmissions: {readmit_count} ({readmit_count/len(df_merged)*100:.1f}%)")

    # ==========================================
    # 4. 按疾病分层统计
    # ==========================================
    print("\n[4] Statistics by Disease Cohort:")
    print("=" * 60)

    def print_cohort_stats(df, name):
        n = len(df)
        mort = df['label_mortality'].sum()
        los_7d = df['prolonged_los_7d'].sum()
        readmit = df['readmission_30d'].sum()

        print(f"\n[{name}] (n={n})")
        print(f"   Mortality: {mort} ({mort/n*100:.1f}%)")
        print(f"   Prolonged LOS (>=7d): {los_7d} ({los_7d/n*100:.1f}%)")
        print(f"   30-day Readmission: {readmit} ({readmit/n*100:.1f}%)")

    # 全体
    print_cohort_stats(df_merged, "All Patients")

    # Sepsis
    if 'has_sepsis_final' in df_merged.columns:
        df_sepsis = df_merged[df_merged['has_sepsis_final'] == 1]
        print_cohort_stats(df_sepsis, "Sepsis Cohort")

    # AKI
    if 'has_aki_final' in df_merged.columns:
        df_aki = df_merged[df_merged['has_aki_final'] == 1]
        print_cohort_stats(df_aki, "AKI Cohort")

    # ARDS
    if 'has_ards' in df_merged.columns:
        df_ards = df_merged[df_merged['has_ards'] == 1]
        if len(df_ards) > 0:
            print_cohort_stats(df_ards, "ARDS Cohort")

    # Sepsis + AKI
    if 'has_sepsis_final' in df_merged.columns and 'has_aki_final' in df_merged.columns:
        df_both = df_merged[(df_merged['has_sepsis_final'] == 1) & (df_merged['has_aki_final'] == 1)]
        print_cohort_stats(df_both, "Sepsis + AKI Cohort")

    # ==========================================
    # 5. 保存最终数据集
    # ==========================================
    print(f"\n[5] Saving to {OUTPUT_FILE}...")

    df_merged.to_csv(OUTPUT_FILE, index=False)
    print(f"   Saved {len(df_merged)} rows with {len(df_merged.columns)} columns")

    # 列出所有可用的标签列
    print("\nAvailable Task Labels:")
    task_cols = ['label_mortality', 'prolonged_los_3d', 'prolonged_los_5d',
                 'prolonged_los_7d', 'readmission_30d']
    for col in task_cols:
        if col in df_merged.columns:
            print(f"   - {col}")

    # ==========================================
    # 6. 创建任务特定的数据集
    # ==========================================
    print("\n[6] Creating task-specific datasets...")

    # 为每个预测任务创建清晰的标签文件
    tasks = {
        'mortality': 'label_mortality',
        'prolonged_los': 'prolonged_los_7d',
        'readmission': 'readmission_30d'
    }

    for task_name, label_col in tasks.items():
        task_df = df_merged[['stay_id', 'subject_id', 'hadm_id', label_col]].copy()
        task_df.columns = ['stay_id', 'subject_id', 'hadm_id', 'label']

        output_path = OUTPUT_DIR / f'labels_{task_name}.csv'
        task_df.to_csv(output_path, index=False)

        positive = task_df['label'].sum()
        print(f"   {task_name}: {positive} positive ({positive/len(task_df)*100:.1f}%) -> {output_path}")

    print("\nMerge LOS Labels Complete!")


if __name__ == "__main__":
    main()
