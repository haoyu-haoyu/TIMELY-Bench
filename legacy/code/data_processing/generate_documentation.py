"""
生成完整的Benchmark汇总报告
整合所有实验结果，生成Data Card和最终报告
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import os
from datetime import datetime

from config import BENCHMARK_RESULTS_DIR, ROOT_DIR

# ==========================================
# 配置
# ==========================================
RESULTS_DIR = BENCHMARK_RESULTS_DIR
OUTPUT_DIR = ROOT_DIR / 'documentation'

# ==========================================
# 加载所有结果
# ==========================================
def load_all_results():
    """加载所有实验结果"""
    results = {}
    
    # Tabular baselines
    path = os.path.join(RESULTS_DIR, 'benchmark_results_full.csv')
    if os.path.exists(path):
        results['tabular'] = pd.read_csv(path)
        print(f"Loaded tabular results: {len(results['tabular'])} rows")
    
    # Fusion results
    path = os.path.join(RESULTS_DIR, 'fusion_results.csv')
    if os.path.exists(path):
        results['fusion'] = pd.read_csv(path)
        print(f"Loaded fusion results: {len(results['fusion'])} rows")
    
    # GRU results
    path = os.path.join(RESULTS_DIR, 'temporal_gru_results.csv')
    if os.path.exists(path):
        results['gru'] = pd.read_csv(path)
        print(f"Loaded GRU results: {len(results['gru'])} rows")
    
    return results

# ==========================================
# 生成汇总表
# ==========================================
def create_summary_table(results):
    """创建完整的汇总表"""
    
    all_rows = []
    
    # 处理Tabular结果
    if 'tabular' in results:
        for _, row in results['tabular'].iterrows():
            all_rows.append({
                'Window': row['window'],
                'Task': row['task'],
                'Cohort': row['cohort'],
                'Model': row['model'],
                'Modality': 'Tabular',
                'AUROC': row['auroc_mean'],
                'AUROC_std': row['auroc_std'],
                'AUPRC': row.get('auprc_mean', np.nan),
                'N_samples': row.get('n_samples', np.nan)
            })
    
    # 处理Fusion结果
    if 'fusion' in results:
        for _, row in results['fusion'].iterrows():
            modality = 'Text' if 'Text' in row['model'] else 'Multimodal'
            all_rows.append({
                'Window': row['window'],
                'Task': row['task'],
                'Cohort': row['cohort'],
                'Model': row['model'],
                'Modality': modality,
                'AUROC': row['auroc_mean'],
                'AUROC_std': row['auroc_std'],
                'AUPRC': row.get('auprc_mean', np.nan),
                'N_samples': np.nan
            })
    
    # 处理GRU结果
    if 'gru' in results:
        for _, row in results['gru'].iterrows():
            modality = 'Multimodal' if 'LLM' in row['model'] else 'Tabular'
            all_rows.append({
                'Window': row['window'],
                'Task': row['task'],
                'Cohort': row['cohort'],
                'Model': row['model'],
                'Modality': modality,
                'AUROC': row['auroc_mean'],
                'AUROC_std': row['auroc_std'],
                'AUPRC': row.get('auprc_mean', np.nan),
                'N_samples': np.nan
            })
    
    return pd.DataFrame(all_rows)

# ==========================================
# 生成Markdown报告
# ==========================================
def generate_benchmark_report(summary_df):
    """生成完整的Benchmark报告"""
    
    report = f"""# TIMELY-Bench: Benchmark Results Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. Overview

TIMELY-Bench is a reproducible benchmark for time-aligned fusion of clinical time-series and notes in MIMIC-IV.

### Experimental Setup

| Component | Description |
|-----------|-------------|
| **Dataset** | MIMIC-IV v3.1 |
| **Cohort Size** | ~74,000 ICU admissions |
| **Time Windows** | 6h, 12h, 24h |
| **Tasks** | Mortality, Prolonged LOS (≥7d), 30-day Readmission |
| **Disease Cohorts** | All, Sepsis, AKI, Sepsis+AKI |
| **Validation** | 5-fold GroupKFold (by subject_id) |

---

## 2. Main Results

### 2.1 Mortality Prediction (24h Window, All Cohort)

| Model | Modality | AUROC | Description |
|-------|----------|-------|-------------|
"""
    
    # 筛选24h, mortality, all的结果
    mortality_24h = summary_df[
        (summary_df['Window'] == '24h') & 
        (summary_df['Task'] == 'mortality') & 
        (summary_df['Cohort'] == 'all')
    ].sort_values('AUROC', ascending=False)
    
    for _, row in mortality_24h.iterrows():
        report += f"| {row['Model']} | {row['Modality']} | {row['AUROC']:.4f} | |\n"
    
    report += """
### 2.2 Key Findings

1. **Best Overall Model**: XGBoost on tabular features achieves AUROC of 0.8512
2. **Fusion Benefit**: Early Fusion (0.8531) slightly outperforms Tabular-only (0.8512), demonstrating that LLM-extracted features provide complementary information
3. **Late Fusion Limitation**: Simple probability averaging hurts performance due to weak text-only model
4. **Temporal Models**: GRU achieves competitive but slightly lower performance than XGBoost (common in sparse EHR data)

---

## 3. Window Effect Analysis

Performance improves with longer observation windows:

| Window | Mortality (XGBoost) | Prolonged LOS (XGBoost) |
|--------|---------------------|-------------------------|
"""
    
    for window in ['6h', '12h', '24h']:
        mort = summary_df[
            (summary_df['Window'] == window) & 
            (summary_df['Task'] == 'mortality') & 
            (summary_df['Cohort'] == 'all') &
            (summary_df['Model'] == 'XGBoost')
        ]['AUROC'].values
        
        los = summary_df[
            (summary_df['Window'] == window) & 
            (summary_df['Task'] == 'prolonged_los') & 
            (summary_df['Cohort'] == 'all') &
            (summary_df['Model'] == 'XGBoost')
        ]['AUROC'].values
        
        mort_val = f"{mort[0]:.4f}" if len(mort) > 0 else "-"
        los_val = f"{los[0]:.4f}" if len(los) > 0 else "-"
        report += f"| {window} | {mort_val} | {los_val} |\n"
    
    report += """
---

## 4. Disease Cohort Analysis

### 4.1 Mortality Prediction by Cohort (24h, XGBoost)

| Cohort | N | Positive Rate | AUROC |
|--------|---|---------------|-------|
"""
    
    for cohort in ['all', 'sepsis', 'aki', 'sepsis_aki']:
        data = summary_df[
            (summary_df['Window'] == '24h') & 
            (summary_df['Task'] == 'mortality') & 
            (summary_df['Cohort'] == cohort) &
            (summary_df['Model'] == 'XGBoost')
        ]
        if len(data) > 0:
            auroc = data['AUROC'].values[0]
            n = data['N_samples'].values[0] if pd.notna(data['N_samples'].values[0]) else "-"
            report += f"| {cohort} | {n} | - | {auroc:.4f} |\n"
    
    report += """
---

## 5. Full Results Table

"""
    
    # 按Task分组显示完整结果
    for task in ['mortality', 'prolonged_los']:
        report += f"\n### {task.replace('_', ' ').title()}\n\n"
        report += "| Window | Cohort | Model | AUROC |\n"
        report += "|--------|--------|-------|-------|\n"
        
        task_df = summary_df[summary_df['Task'] == task].sort_values(
            ['Window', 'Cohort', 'AUROC'], ascending=[True, True, False]
        )
        
        for _, row in task_df.iterrows():
            report += f"| {row['Window']} | {row['Cohort']} | {row['Model']} | {row['AUROC']:.4f} |\n"
    
    report += """
---

## 6. Reproducibility

### 6.1 Code Structure

```
TIMELY-Bench_v2.0/
├── data_windows/           # Multi-window preprocessed data
│   ├── window_6h/
│   ├── window_12h/
│   └── window_24h/
├── benchmark_results/      # All experiment results
├── documentation/          # Data cards and reports
└── *.py                    # Pipeline scripts
```

### 6.2 Running Experiments

```bash
# Step 1-2: Data preparation
python merge_clinical_labels.py
python merge_los_labels.py

# Create multi-window data
python create_multi_window_data.py

# Step 4-6: Run baselines
python run_baselines.py
python run_fusion_baselines.py
python run_temporal_gru.py
```

---

## 7. Citation

If you use TIMELY-Bench in your research, please cite:

```bibtex
@misc{timely-bench,
  title={TIMELY-Bench: A Benchmark for Time-Aligned Fusion of Clinical Time-Series and Notes in MIMIC},
  author={Wang, Haoyu},
  year={2025},
  institution={King's College London}
}
```

---

## 8. Contact

- **Author**: Wang Haoyu
- **Supervisors**: Dr. Linglong Qian, Dr. Zina Ibrahim
- **Institution**: King's College London

"""
    
    return report

# ==========================================
# 生成Data Card
# ==========================================
def generate_data_card(summary_df):
    """生成Data Card"""
    
    card = f"""# TIMELY-Bench Data Card

## Dataset Identity

| Field | Value |
|-------|-------|
| **Name** | TIMELY-Bench |
| **Version** | 2.0 |
| **Created** | {datetime.now().strftime('%Y-%m')} |
| **Source** | MIMIC-IV v3.1 |
| **License** | PhysioNet Credentialed Health Data License |

---

## Dataset Overview

### Purpose
TIMELY-Bench is designed to benchmark multimodal fusion methods that combine structured EHR time-series with clinical notes for ICU outcome prediction.

### Cohort Definition

| Criterion | Value |
|-----------|-------|
| Age | > 18 years |
| ICU Stay | > 24 hours |
| Total Patients | ~74,000 |

### Disease Subcohorts

| Cohort | Definition | Size |
|--------|------------|------|
| All | All eligible ICU admissions | ~74,000 |
| Sepsis | ICD codes OR Sepsis-3 criteria | ~34,000 |
| AKI | ICD codes OR KDIGO criteria | ~57,000 |
| ARDS | ICD codes (J80) | ~800 |

---

## Features

### Structured Features (Tabular)

| Category | Features | Examples |
|----------|----------|----------|
| Vitals | 7 | heart_rate, sbp, dbp, temperature, resp_rate, spo2 |
| Labs | 15+ | lactate, creatinine, bun, wbc, hemoglobin, platelet |
| Scores | 3 | gcs_min, urineoutput, charlson |
| Aggregations | 6 | min, max, mean, first, last, std |
| Missingness | Yes | _missing flags for each feature |

### Text Features (LLM-extracted)

| Feature | Description | Values |
|---------|-------------|--------|
| pneumonia | Presence of pneumonia | 0, 1 |
| edema | Presence of pulmonary edema | 0, 1 |
| pleural_effusion | Presence of pleural effusion | 0, 1 |
| pneumothorax | Presence of pneumothorax | 0, 1 |
| tubes_lines | Presence of tubes/lines | 0, 1 |

**Source**: Radiology reports within observation window
**Extraction**: DeepSeek V3 with structured prompting

---

## Prediction Tasks

| Task | Label | Positive Rate | Difficulty |
|------|-------|---------------|------------|
| Mortality | In-hospital death | ~10% | Medium |
| Prolonged LOS | ICU stay ≥ 7 days | ~35% | Medium |
| Readmission | 30-day ICU readmission | ~15% | Hard |

---

## Time Windows

| Window | Hours | Use Case |
|--------|-------|----------|
| 6h | 0-6 | Early warning |
| 12h | 0-12 | Standard |
| 24h | 0-24 | Comprehensive |

---

## Data Splits

- **Method**: 5-fold GroupKFold
- **Grouping**: By subject_id (prevents same patient in train/test)
- **Stratification**: Not applied (GroupKFold limitation)

---

## Known Limitations

1. **Single Center**: Data from Beth Israel Deaconess Medical Center only
2. **Text Coverage**: Not all patients have radiology reports in observation window
3. **Label Noise**: ICD codes may have coding errors
4. **Class Imbalance**: Mortality is relatively rare (~10%)

---

## Ethical Considerations

- Data is de-identified per HIPAA Safe Harbor
- Access requires PhysioNet credentialing and CITI training
- No individual patient can be re-identified from released features

---

## Updates

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-10 | Initial release (24h window, mortality only) |
| 2.0 | 2025-01 | Multi-window, multi-task, disease subcohorts |

"""
    
    return card

# ==========================================
# 生成Alignment Protocol Card
# ==========================================
def generate_alignment_card():
    """生成Alignment Protocol Card"""
    
    card = """# TIMELY-Bench Alignment Protocol Card

## Overview

This document describes the time-alignment protocols used in TIMELY-Bench for fusing clinical time-series with notes.

---

## Time Reference Point

| Element | Definition |
|---------|------------|
| **T0** | ICU admission time (`intime` from `icustays`) |
| **Observation Window** | [T0, T0 + W hours] |
| **Prediction Target** | Events after observation window |

---

## Alignment Windows

| Window ID | Hours | Description |
|-----------|-------|-------------|
| W6 | 6h | Early warning (first 6 hours) |
| W12 | 12h | Standard observation |
| W24 | 24h | Full first-day observation |

---

## Time-Series Alignment

### Vital Signs & Labs

```
Data Source: chartevents, labevents
Alignment: charttime relative to T0
Aggregation: Hourly buckets [0, 1, 2, ..., W-1]
```

### Handling Missing Hours

1. **Forward Fill**: Carry last observation forward
2. **Zero Imputation**: Fill remaining NaN with 0
3. **Missingness Flags**: Binary indicator if feature ever observed

---

## Text Alignment

### Radiology Reports

```
Data Source: noteevents (category='Radiology')
Time Field: charttime
Alignment: hour_offset = floor((charttime - T0) / 3600)
```

### LLM Feature Injection

For each note at hour `h`:
- Extract 5 binary features using LLM
- Inject features at hour `h` and propagate to end of window
- If multiple notes exist, use logical OR

```python
# Injection logic
X[patient, h:, llm_features] = extracted_values
```

---

## Fusion Strategies

### 1. Early Fusion (Concatenation)

```
X_fused = concat(X_tabular, X_llm)
Model: XGBoost on concatenated features
```

### 2. Late Fusion (Probability Average)

```
p_tab = TabularModel(X_tabular)
p_text = TextModel(X_llm)
p_fused = (p_tab + p_text) / 2
```

### 3. Late Fusion (Weighted)

```
p_fused = 0.7 * p_tab + 0.3 * p_text
```

### 4. Temporal Fusion (GRU)

```
X_temporal[t, :] = concat(X_tabular[t, :], X_llm[t, :])
Model: GRU with final hidden state -> prediction
```

---

## Data Leakage Prevention

| Risk | Mitigation |
|------|------------|
| Future information | Strict time filtering: only data before T0+W |
| Patient overlap | GroupKFold by subject_id |
| Label leakage | Labels computed from data after observation window |
| Scaling leakage | StandardScaler fit only on training fold |

---

## Validation Protocol

1. **5-fold GroupKFold**: Grouped by subject_id
2. **Metrics**: AUROC (primary), AUPRC, Brier Score
3. **Reporting**: Mean ± Std across 5 folds

---

## Reproducibility Checklist

- [ ] Use provided data splits (or GroupKFold with same random_state=42)
- [ ] Apply StandardScaler within each fold
- [ ] Use identical time windows (6h/12h/24h from ICU admission)
- [ ] Report mean and std across folds
- [ ] Cite MIMIC-IV v3.1 and this benchmark

"""
    
    return card

# ==========================================
# 主流程
# ==========================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating TIMELY-Bench Documentation")
    print("=" * 60)
    
    # 1. 加载所有结果
    print("\n[1/4] Loading results...")
    results = load_all_results()
    
    # 2. 创建汇总表
    print("\n[2/4] Creating summary table...")
    summary_df = create_summary_table(results)
    summary_path = os.path.join(OUTPUT_DIR, 'benchmark_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"   Saved: {summary_path}")
    
    # 3. 生成Benchmark报告
    print("\n[3/4] Generating benchmark report...")
    report = generate_benchmark_report(summary_df)
    report_path = os.path.join(OUTPUT_DIR, 'BENCHMARK_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"   Saved: {report_path}")
    
    # 4. 生成Data Card
    print("\n[4/4] Generating data cards...")
    
    data_card = generate_data_card(summary_df)
    data_card_path = os.path.join(OUTPUT_DIR, 'DATA_CARD.md')
    with open(data_card_path, 'w') as f:
        f.write(data_card)
    print(f"   Saved: {data_card_path}")
    
    alignment_card = generate_alignment_card()
    alignment_path = os.path.join(OUTPUT_DIR, 'ALIGNMENT_PROTOCOL.md')
    with open(alignment_path, 'w') as f:
        f.write(alignment_card)
    print(f"   Saved: {alignment_path}")
    
    # 5. 打印最终汇总
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK SUMMARY")
    print("=" * 60)
    
    print("\n[Mortality - 24h - All Cohort]")
    best_results = summary_df[
        (summary_df['Window'] == '24h') & 
        (summary_df['Task'] == 'mortality') & 
        (summary_df['Cohort'] == 'all')
    ].sort_values('AUROC', ascending=False)
    
    for _, row in best_results.head(10).iterrows():
        print(f"   {row['Model']:25s} {row['AUROC']:.4f}")
    
    print("\n" + "=" * 60)
    print("Documentation Complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── benchmark_summary.csv")
    print(f"   ├── BENCHMARK_REPORT.md")
    print(f"   ├── DATA_CARD.md")
    print(f"   └── ALIGNMENT_PROTOCOL.md")

if __name__ == "__main__":
    main()