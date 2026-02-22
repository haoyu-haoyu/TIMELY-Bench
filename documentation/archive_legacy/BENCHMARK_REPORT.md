# TIMELY-Bench: Benchmark Results Report

**Generated**: 2025-12-22 13:12:49

> Note (2026-02-06): This report is a **legacy snapshot** from earlier experiment runs and is **not** kept in sync with the final (v2.0) canonical benchmark outputs.
> For canonical numbers used in the paper/poster, use:
> - `results/standardized/results_summary.csv`
> - `results/standardized/results_summary.md`
> - `docs/RESULTS_SUMMARY.md`

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
| Early Fusion (XGB) | Multimodal | 0.8531 | |
| XGBoost | Tabular | 0.8512 | |
| Late Fusion (Wt) | Multimodal | 0.8392 | |
| GRU (Tabular) | Tabular | 0.8385 | |
| GRU (Tab+LLM) | Multimodal | 0.8369 | |
| Late Fusion (Avg) | Multimodal | 0.8231 | |
| LogisticRegression | Tabular | 0.8201 | |
| Text-only (LR) | Text | 0.5858 | |

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
| 6h | 0.7953 | 0.7083 |
| 12h | 0.8242 | 0.7416 |
| 24h | 0.8512 | 0.7738 |

---

## 4. Disease Cohort Analysis

### 4.1 Mortality Prediction by Cohort (24h, XGBoost)

| Cohort | N | Positive Rate | AUROC |
|--------|---|---------------|-------|
| all | 74829.0 | - | 0.8512 |
| sepsis | 34152.0 | - | 0.8000 |
| aki | 57263.0 | - | 0.8303 |
| sepsis_aki | 28876.0 | - | 0.7859 |

---

## 5. Full Results Table


### Mortality

| Window | Cohort | Model | AUROC |
|--------|--------|-------|-------|
| 12h | aki | Early Fusion (XGB) | 0.8058 |
| 12h | aki | XGBoost | 0.8042 |
| 12h | aki | Late Fusion (Wt) | 0.7948 |
| 12h | aki | GRU (Tabular) | 0.7885 |
| 12h | aki | GRU (Tab+LLM) | 0.7876 |
| 12h | aki | Late Fusion (Avg) | 0.7840 |
| 12h | aki | LogisticRegression | 0.7832 |
| 12h | aki | Text-only (LR) | 0.5743 |
| 12h | all | Early Fusion (XGB) | 0.8263 |
| 12h | all | XGBoost | 0.8242 |
| 12h | all | Late Fusion (Wt) | 0.8135 |
| 12h | all | GRU (Tab+LLM) | 0.8055 |
| 12h | all | GRU (Tabular) | 0.8046 |
| 12h | all | LogisticRegression | 0.8001 |
| 12h | all | Late Fusion (Avg) | 0.7979 |
| 12h | all | Text-only (LR) | 0.5858 |
| 12h | sepsis | Early Fusion (XGB) | 0.7761 |
| 12h | sepsis | XGBoost | 0.7733 |
| 12h | sepsis | Late Fusion (Wt) | 0.7586 |
| 12h | sepsis | GRU (Tab+LLM) | 0.7580 |
| 12h | sepsis | LogisticRegression | 0.7579 |
| 12h | sepsis | GRU (Tabular) | 0.7566 |
| 12h | sepsis | Late Fusion (Avg) | 0.7515 |
| 12h | sepsis | Text-only (LR) | 0.5775 |
| 12h | sepsis_aki | XGBoost | 0.7557 |
| 12h | sepsis_aki | LogisticRegression | 0.7456 |
| 24h | aki | Early Fusion (XGB) | 0.8330 |
| 24h | aki | XGBoost | 0.8303 |
| 24h | aki | Late Fusion (Wt) | 0.8208 |
| 24h | aki | GRU (Tabular) | 0.8190 |
| 24h | aki | GRU (Tab+LLM) | 0.8183 |
| 24h | aki | Late Fusion (Avg) | 0.8091 |
| 24h | aki | LogisticRegression | 0.8014 |
| 24h | aki | Text-only (LR) | 0.5743 |
| 24h | all | Early Fusion (XGB) | 0.8531 |
| 24h | all | XGBoost | 0.8512 |
| 24h | all | Late Fusion (Wt) | 0.8392 |
| 24h | all | GRU (Tabular) | 0.8385 |
| 24h | all | GRU (Tab+LLM) | 0.8369 |
| 24h | all | Late Fusion (Avg) | 0.8231 |
| 24h | all | LogisticRegression | 0.8201 |
| 24h | all | Text-only (LR) | 0.5858 |
| 24h | sepsis | Early Fusion (XGB) | 0.8041 |
| 24h | sepsis | XGBoost | 0.8000 |
| 24h | sepsis | Late Fusion (Wt) | 0.7881 |
| 24h | sepsis | GRU (Tab+LLM) | 0.7873 |
| 24h | sepsis | GRU (Tabular) | 0.7872 |
| 24h | sepsis | Late Fusion (Avg) | 0.7791 |
| 24h | sepsis | LogisticRegression | 0.7740 |
| 24h | sepsis | Text-only (LR) | 0.5775 |
| 24h | sepsis_aki | XGBoost | 0.7859 |
| 24h | sepsis_aki | LogisticRegression | 0.7612 |
| 6h | aki | Early Fusion (XGB) | 0.7824 |
| 6h | aki | XGBoost | 0.7770 |
| 6h | aki | Late Fusion (Wt) | 0.7663 |
| 6h | aki | LogisticRegression | 0.7605 |
| 6h | aki | GRU (Tabular) | 0.7584 |
| 6h | aki | GRU (Tab+LLM) | 0.7578 |
| 6h | aki | Late Fusion (Avg) | 0.7574 |
| 6h | aki | Text-only (LR) | 0.5743 |
| 6h | all | Early Fusion (XGB) | 0.8003 |
| 6h | all | XGBoost | 0.7953 |
| 6h | all | Late Fusion (Wt) | 0.7878 |
| 6h | all | GRU (Tab+LLM) | 0.7769 |
| 6h | all | LogisticRegression | 0.7747 |
| 6h | all | Late Fusion (Avg) | 0.7737 |
| 6h | all | GRU (Tabular) | 0.7734 |
| 6h | all | Text-only (LR) | 0.5858 |
| 6h | sepsis | Early Fusion (XGB) | 0.7466 |
| 6h | sepsis | XGBoost | 0.7440 |
| 6h | sepsis | LogisticRegression | 0.7365 |
| 6h | sepsis | GRU (Tabular) | 0.7314 |
| 6h | sepsis | Late Fusion (Wt) | 0.7293 |
| 6h | sepsis | GRU (Tab+LLM) | 0.7285 |
| 6h | sepsis | Late Fusion (Avg) | 0.7238 |
| 6h | sepsis | Text-only (LR) | 0.5775 |
| 6h | sepsis_aki | XGBoost | 0.7287 |
| 6h | sepsis_aki | LogisticRegression | 0.7251 |

### Prolonged Los

| Window | Cohort | Model | AUROC |
|--------|--------|-------|-------|
| 12h | aki | Early Fusion (XGB) | 0.7257 |
| 12h | aki | XGBoost | 0.7180 |
| 12h | aki | Late Fusion (Wt) | 0.7130 |
| 12h | aki | Late Fusion (Avg) | 0.7082 |
| 12h | aki | GRU (Tab+LLM) | 0.7067 |
| 12h | aki | GRU (Tabular) | 0.6958 |
| 12h | aki | LogisticRegression | 0.6915 |
| 12h | aki | Text-only (LR) | 0.5756 |
| 12h | all | Early Fusion (XGB) | 0.7476 |
| 12h | all | XGBoost | 0.7416 |
| 12h | all | Late Fusion (Wt) | 0.7358 |
| 12h | all | GRU (Tab+LLM) | 0.7318 |
| 12h | all | Late Fusion (Avg) | 0.7239 |
| 12h | all | GRU (Tabular) | 0.7224 |
| 12h | all | LogisticRegression | 0.7182 |
| 12h | all | Text-only (LR) | 0.5921 |
| 12h | sepsis | Early Fusion (XGB) | 0.7192 |
| 12h | sepsis | XGBoost | 0.7138 |
| 12h | sepsis | Late Fusion (Wt) | 0.7045 |
| 12h | sepsis | Late Fusion (Avg) | 0.7039 |
| 12h | sepsis | GRU (Tab+LLM) | 0.6997 |
| 12h | sepsis | GRU (Tabular) | 0.6904 |
| 12h | sepsis | LogisticRegression | 0.6834 |
| 12h | sepsis | Text-only (LR) | 0.5766 |
| 12h | sepsis_aki | XGBoost | 0.6958 |
| 12h | sepsis_aki | LogisticRegression | 0.6618 |
| 24h | aki | Early Fusion (XGB) | 0.7542 |
| 24h | aki | XGBoost | 0.7506 |
| 24h | aki | Late Fusion (Wt) | 0.7427 |
| 24h | aki | GRU (Tab+LLM) | 0.7381 |
| 24h | aki | GRU (Tabular) | 0.7372 |
| 24h | aki | Late Fusion (Avg) | 0.7349 |
| 24h | aki | LogisticRegression | 0.7124 |
| 24h | aki | Text-only (LR) | 0.5756 |
| 24h | all | Early Fusion (XGB) | 0.7773 |
| 24h | all | XGBoost | 0.7738 |
| 24h | all | GRU (Tab+LLM) | 0.7644 |
| 24h | all | Late Fusion (Wt) | 0.7641 |
| 24h | all | GRU (Tabular) | 0.7622 |
| 24h | all | Late Fusion (Avg) | 0.7501 |
| 24h | all | LogisticRegression | 0.7409 |
| 24h | all | Text-only (LR) | 0.5921 |
| 24h | sepsis | Early Fusion (XGB) | 0.7479 |
| 24h | sepsis | XGBoost | 0.7457 |
| 24h | sepsis | Late Fusion (Wt) | 0.7356 |
| 24h | sepsis | GRU (Tab+LLM) | 0.7320 |
| 24h | sepsis | Late Fusion (Avg) | 0.7315 |
| 24h | sepsis | GRU (Tabular) | 0.7246 |
| 24h | sepsis | LogisticRegression | 0.7047 |
| 24h | sepsis | Text-only (LR) | 0.5766 |
| 24h | sepsis_aki | XGBoost | 0.7284 |
| 24h | sepsis_aki | LogisticRegression | 0.6819 |
| 6h | aki | Early Fusion (XGB) | 0.7008 |
| 6h | aki | XGBoost | 0.6868 |
| 6h | aki | Late Fusion (Wt) | 0.6855 |
| 6h | aki | Late Fusion (Avg) | 0.6828 |
| 6h | aki | GRU (Tab+LLM) | 0.6743 |
| 6h | aki | GRU (Tabular) | 0.6670 |
| 6h | aki | LogisticRegression | 0.6607 |
| 6h | aki | Text-only (LR) | 0.5756 |
| 6h | all | Early Fusion (XGB) | 0.7218 |
| 6h | all | XGBoost | 0.7083 |
| 6h | all | Late Fusion (Wt) | 0.7080 |
| 6h | all | Late Fusion (Avg) | 0.6984 |
| 6h | all | GRU (Tab+LLM) | 0.6960 |
| 6h | all | GRU (Tabular) | 0.6880 |
| 6h | all | LogisticRegression | 0.6844 |
| 6h | all | Text-only (LR) | 0.5921 |
| 6h | sepsis | Early Fusion (XGB) | 0.6925 |
| 6h | sepsis | XGBoost | 0.6786 |
| 6h | sepsis | Late Fusion (Avg) | 0.6756 |
| 6h | sepsis | Late Fusion (Wt) | 0.6730 |
| 6h | sepsis | GRU (Tab+LLM) | 0.6701 |
| 6h | sepsis | GRU (Tabular) | 0.6584 |
| 6h | sepsis | LogisticRegression | 0.6535 |
| 6h | sepsis | Text-only (LR) | 0.5766 |
| 6h | sepsis_aki | XGBoost | 0.6599 |
| 6h | sepsis_aki | LogisticRegression | 0.6352 |

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

# Step 3: Create multi-window data
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
