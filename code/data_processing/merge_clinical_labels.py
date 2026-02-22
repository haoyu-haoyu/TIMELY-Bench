"""
合并 cohort.csv 和 clinical_labels.csv
生成带有疾病标签的完整数据集
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np

from config import RAW_DATA_DIR, PROCESSED_DIR

# ==========================================
# 配置路径
# ==========================================
COHORT_FILE = RAW_DATA_DIR / 'cohort.csv'
CLINICAL_LABELS_FILE = RAW_DATA_DIR / 'clinical_labels.csv'
OUTPUT_DIR = PROCESSED_DIR
OUTPUT_FILE = OUTPUT_DIR / 'cohort_with_conditions.csv'

# ==========================================
# 1. 加载数据
# ==========================================
print("Loading data...")

# Backward-compatible fallback for environments where cohort.csv is under raw/v1_data.
if not COHORT_FILE.exists():
    fallback = RAW_DATA_DIR / 'v1_data' / 'cohort.csv'
    if fallback.exists():
        COHORT_FILE = fallback

df_cohort = pd.read_csv(COHORT_FILE)
df_clinical = pd.read_csv(CLINICAL_LABELS_FILE)

print(f"   Cohort: {len(df_cohort)} rows")
print(f"   Clinical Labels: {len(df_clinical)} rows")

# 确保stay_id类型一致
df_cohort['stay_id'] = df_cohort['stay_id'].astype(int)
df_clinical['stay_id'] = df_clinical['stay_id'].astype(int)

# ==========================================
# 2. 合并数据
# ==========================================
print("\nMerging data...")

# 使用left join，保留cohort中的所有患者
df_merged = df_cohort.merge(
    df_clinical,
    on='stay_id',
    how='left'
)

print(f"   Merged: {len(df_merged)} rows")

# 检查合并情况
matched = df_merged['sepsis3'].notna().sum()
unmatched = df_merged['sepsis3'].isna().sum()
print(f"   Matched: {matched} ({matched/len(df_merged)*100:.1f}%)")
print(f"   Unmatched: {unmatched} ({unmatched/len(df_merged)*100:.1f}%)")

# ==========================================
# 3. 根据ICD码分类疾病
# ==========================================
print("\nClassifying conditions from ICD codes...")

# ICD-10 编码映射
CONDITION_MAPPING = {
    'sepsis': [
        'A41',      # Other sepsis
        'A40',      # Streptococcal sepsis
        'R65.2',    # Severe sepsis
    ],
    'aki': [
        'N17',      # Acute kidney failure
    ],
    'ards': [
        'J80',      # ARDS
        'J96.0',    # Acute respiratory failure
    ],
    'shock': [
        'R57',      # Shock
    ],
    'pneumonia': [
        'J18',      # Pneumonia, unspecified organism
        'J15',      # Bacterial pneumonia
        'J13',      # Pneumonia due to Streptococcus pneumoniae
        'J14',      # Pneumonia due to Haemophilus influenzae
    ],
    'heart_failure': [
        'I50',      # Heart failure
    ],
    'respiratory_failure': [
        'J96',      # Respiratory failure
    ],
    'stroke': [
        'I60',      # Subarachnoid hemorrhage
        'I61',      # Intracerebral hemorrhage
        'I62',      # Other nontraumatic intracranial hemorrhage
        'I63',      # Cerebral infarction
        'I64',      # Stroke, not specified
        'G45',      # Transient cerebral ischemic attacks
    ],
    'delirium': [
        'F05',      # Delirium due to known physiological condition
        'R41.0',    # Disorientation
        'R41.82',   # Altered mental status
    ],
}

def classify_conditions_from_icd(icd_codes_str):
    """根据ICD码判断患者有哪些conditions"""
    if pd.isna(icd_codes_str) or icd_codes_str == '':
        return []
    
    codes = str(icd_codes_str).split(',')
    conditions = []
    
    for condition, prefixes in CONDITION_MAPPING.items():
        for code in codes:
            code = code.strip()
            if any(code.startswith(prefix) for prefix in prefixes):
                if condition not in conditions:
                    conditions.append(condition)
                break
    
    return conditions

# 应用分类
df_merged['conditions_from_icd'] = df_merged['icd_codes'].apply(classify_conditions_from_icd)

# 创建每个condition的二值列（方便后续筛选）
for condition in CONDITION_MAPPING.keys():
    df_merged[f'has_{condition}'] = df_merged['conditions_from_icd'].apply(
        lambda x: 1 if condition in x else 0
    )

# ==========================================
# 4. 结合临床标准标签
# ==========================================
print("\nCombining with clinical criteria labels...")

# Sepsis-3: 来自BigQuery的sepsis3列
df_merged['sepsis3_clinical'] = df_merged['sepsis3'].fillna(0).astype(int)

# AKI: 来自BigQuery的aki_stage_max列
df_merged['aki_clinical'] = (df_merged['aki_stage_max'] >= 1).fillna(0).astype(int)
df_merged['aki_stage'] = df_merged['aki_stage_max'].fillna(0).astype(int)

# 综合标签: ICD码 OR 临床标准
df_merged['has_sepsis_final'] = ((df_merged['has_sepsis'] == 1) | (df_merged['sepsis3_clinical'] == 1)).astype(int)
df_merged['has_aki_final'] = ((df_merged['has_aki'] == 1) | (df_merged['aki_clinical'] == 1)).astype(int)
df_merged['has_stroke_final'] = df_merged['has_stroke'].astype(int)
df_merged['has_delirium_final'] = df_merged['has_delirium'].astype(int)

# ==========================================
# 5. 统计分析
# ==========================================
print("\nCondition Statistics:")
print("=" * 50)

# 各疾病分布
print("\n[By ICD Codes]")
for condition in CONDITION_MAPPING.keys():
    count = df_merged[f'has_{condition}'].sum()
    pct = count / len(df_merged) * 100
    print(f"   {condition}: {count} ({pct:.1f}%)")

print("\n[By Clinical Criteria]")
print(f"   Sepsis-3: {df_merged['sepsis3_clinical'].sum()} ({df_merged['sepsis3_clinical'].mean()*100:.1f}%)")
print(f"   AKI (KDIGO): {df_merged['aki_clinical'].sum()} ({df_merged['aki_clinical'].mean()*100:.1f}%)")

print("\n[Final Labels (ICD OR Clinical)]")
print(f"   Sepsis: {df_merged['has_sepsis_final'].sum()} ({df_merged['has_sepsis_final'].mean()*100:.1f}%)")
print(f"   AKI: {df_merged['has_aki_final'].sum()} ({df_merged['has_aki_final'].mean()*100:.1f}%)")
print(f"   Stroke: {df_merged['has_stroke_final'].sum()} ({df_merged['has_stroke_final'].mean()*100:.1f}%)")
print(f"   Delirium: {df_merged['has_delirium_final'].sum()} ({df_merged['has_delirium_final'].mean()*100:.1f}%)")

# AKI分期分布
print("\n[AKI Stage Distribution]")
aki_dist = df_merged['aki_stage'].value_counts().sort_index()
for stage, count in aki_dist.items():
    print(f"   Stage {int(stage)}: {count}")

# 多病种共存
print("\n[Multimorbidity Analysis]")
df_merged['num_conditions'] = (
    df_merged['has_sepsis_final'] + 
    df_merged['has_aki_final'] + 
    df_merged['has_stroke_final'] +
    df_merged['has_delirium_final'] +
    df_merged['has_ards'] + 
    df_merged['has_shock']
)

for n in range(7):
    count = (df_merged['num_conditions'] == n).sum()
    print(f"   {n} conditions: {count} ({count/len(df_merged)*100:.1f}%)")

# Sepsis + AKI 共存
sepsis_aki = ((df_merged['has_sepsis_final'] == 1) & (df_merged['has_aki_final'] == 1)).sum()
print(f"\n   Sepsis + AKI co-occurrence: {sepsis_aki}")

# ==========================================
# 6. 保存结果
# ==========================================
print(f"\nSaving to {OUTPUT_FILE}...")

# 选择需要保存的列
output_cols = [
    # 原始cohort信息
    'subject_id', 'hadm_id', 'stay_id', 
    'intime', 'outtime', 'deathtime', 'icu_intime',
    'anchor_age', 'gender', 'label_mortality',
    
    # 临床评分
    'sepsis3', 'sepsis_sofa', 'sofa_max', 'aki_stage_max',
    
    # ICD码
    'icd_codes', 'diagnoses_text',
    
    # 疾病标签
    'has_sepsis', 'has_aki', 'has_ards', 'has_shock', 
    'has_pneumonia', 'has_heart_failure', 'has_respiratory_failure', 'has_stroke', 'has_delirium',
    
    # 临床标准标签
    'sepsis3_clinical', 'aki_clinical', 'aki_stage',
    
    # 最终标签
    'has_sepsis_final', 'has_aki_final', 'has_stroke_final', 'has_delirium_final',
    
    # 多病种
    'conditions_from_icd', 'num_conditions'
]

# 只保存存在的列
existing_cols = [c for c in output_cols if c in df_merged.columns]
df_output = df_merged[existing_cols]

df_output.to_csv(OUTPUT_FILE, index=False)
print(f"   Saved {len(df_output)} rows with {len(existing_cols)} columns")

# ==========================================
# 7. 生成按疾病分层的子队列
# ==========================================
print("\nGenerating disease-specific subcohorts...")

# Sepsis队列
df_sepsis = df_merged[df_merged['has_sepsis_final'] == 1]
df_sepsis.to_csv(OUTPUT_DIR / 'cohort_sepsis.csv', index=False)
print(f"   Sepsis cohort: {len(df_sepsis)} patients")

# AKI队列
df_aki = df_merged[df_merged['has_aki_final'] == 1]
df_aki.to_csv(OUTPUT_DIR / 'cohort_aki.csv', index=False)
print(f"   AKI cohort: {len(df_aki)} patients")

# ARDS队列
df_ards = df_merged[df_merged['has_ards'] == 1]
df_ards.to_csv(OUTPUT_DIR / 'cohort_ards.csv', index=False)
print(f"   ARDS cohort: {len(df_ards)} patients")

# Stroke队列
df_stroke = df_merged[df_merged['has_stroke_final'] == 1]
df_stroke.to_csv(OUTPUT_DIR / 'cohort_stroke.csv', index=False)
print(f"   Stroke cohort: {len(df_stroke)} patients")

# Delirium队列
df_delirium = df_merged[df_merged['has_delirium_final'] == 1]
df_delirium.to_csv(OUTPUT_DIR / 'cohort_delirium.csv', index=False)
print(f"   Delirium cohort: {len(df_delirium)} patients")

# Sepsis + AKI 共存队列
df_sepsis_aki = df_merged[(df_merged['has_sepsis_final'] == 1) & (df_merged['has_aki_final'] == 1)]
df_sepsis_aki.to_csv(OUTPUT_DIR / 'cohort_sepsis_aki.csv', index=False)
print(f"   Sepsis+AKI cohort: {len(df_sepsis_aki)} patients")

print("\nDone!")
print("=" * 50)
print("Next steps:")
print("1. Review the statistics above")
print("2. Use cohort_with_conditions.csv for further analysis")
print("3. Use disease-specific cohorts for pattern detection")
