# TIMELY-Bench Results Summary (Canonical)

Generated: 2026-02-20

This file summarises the canonical baseline results used for the paper/poster.
For full results, see:
- `results/standardized/results_summary.csv` (canonical per-step outputs)
- `results/robustness/window_performance.csv` and `results/robustness/statistical_tests.json` (cross-window analysis)
- `results/calibration/*` (calibration)

---

## 1. 24h Window (All Cohort) Results

All AUROC/AUPRC values below are holdout test-set metrics from `results/standardized/results_summary.csv`.

### 1.1 Mortality

| Model | AUROC | AUPRC | ECE | Brier |
|------|------:|------:|----:|------:|
| Early Fusion XGBoost (Structured + ClinicalBERT embeddings) | 0.885 | 0.584 | 0.0086 | 0.0740 |
| Late Fusion (tuned $\alpha$, ClinicalBERT) | 0.881 | 0.551 | 0.1078 | 0.0915 |
| Early Fusion XGBoost (Structured + annotation-derived) | 0.873 | 0.557 | 0.0066 | 0.0770 |
| Late Fusion (tuned $\alpha$, annotation-derived) | 0.869 | 0.535 | 0.1813 | 0.1234 |
| XGBoost (Structured) | 0.868 | 0.541 | 0.1974 | 0.1327 |
| Logistic Regression (Structured) | 0.848 | 0.508 | 0.0083 | 0.0823 |
| Clinical GRU (Temporal) | 0.842 | 0.483 | 0.0336 | 0.0871 |
| Logistic Regression (Text-Only, ClinicalBERT embeddings) | 0.832 | 0.444 | --- | --- |
| XGBoost (Text-Only, ClinicalBERT embeddings) | 0.817 | 0.444 | 0.0089 | 0.0881 |
| Logistic Regression (Text-Only, MedCAT concepts) | 0.552 | 0.150 | --- | --- |
| XGBoost (Text-Only, MedCAT concepts) | 0.552 | 0.151 | --- | --- |
| XGBoost (Text-Only, annotation-derived) | 0.755 | 0.327 | 0.0062 | 0.0965 |

### 1.2 Prolonged LOS

| Model | AUROC | AUPRC | ECE | Brier |
|------|------:|------:|----:|------:|
| Early Fusion XGBoost (Structured + ClinicalBERT embeddings) | 0.835 | 0.509 | 0.0135 | 0.1045 |
| Late Fusion (tuned $\alpha$, ClinicalBERT) | 0.834 | 0.506 | 0.1073 | 0.1180 |
| Early Fusion XGBoost (Structured + annotation-derived) | 0.818 | 0.468 | 0.0135 | 0.1094 |
| XGBoost (Structured) | 0.815 | 0.460 | 0.2204 | 0.1658 |
| Logistic Regression (Structured) | 0.797 | 0.422 | 0.0153 | 0.1157 |
| Late Fusion (tuned $\alpha$, annotation-derived) | 0.815 | 0.458 | 0.1755 | 0.1437 |
| XGBoost (Text-Only, ClinicalBERT embeddings) | 0.800 | 0.456 | 0.0108 | 0.1128 |
| Logistic Regression (Text-Only, ClinicalBERT embeddings) | 0.800 | 0.452 | --- | --- |
| XGBoost (Text-Only, MedCAT concepts) | 0.550 | 0.195 | --- | --- |
| Logistic Regression (Text-Only, MedCAT concepts) | 0.549 | 0.192 | --- | --- |
| XGBoost (Text-Only, annotation-derived) | 0.701 | 0.311 | 0.0090 | 0.1276 |

---

## 2. Cross-Window Robustness (Mortality AUROC, Structured Baselines)

| Model | 6h | 12h | 24h | D0 | CV (%) |
|------|---:|----:|----:|---:|-------:|
| XGBoost | 0.805 | 0.839 | 0.868 | 0.811 | 3.05 |
| Logistic Regression | 0.783 | 0.818 | 0.852 | 0.797 | 3.13 |

Statistical tests (AUROC):
Note: statistical tests are computed on charttime windows (6h/12h/24h) and exclude the D0 daily aligner.
- Friedman: $\chi^2$=12.0, p=0.0025 (n=6 model$\times$cohort combinations)
- Pairwise Wilcoxon: p=0.0313 for 6h vs 12h, 12h vs 24h, and 6h vs 24h

---

## 3. Notes / Scope

- Text-only and fusion baselines include both:
  - annotation-derived alignment features (e.g., `supportive_ratio`, `annotation_density`, `n_patterns`) extracted from episode JSONs, and
  - raw text semantic representations via stay-level ClinicalBERT embeddings (first 24h note window; mean pooling across notes).
- A MedCAT/UMLS bag-of-concepts text baseline is included (24h concept presence indicators).
- Canonical aligner comparison includes D0 daily vs 6h/12h/24h under `results/aligner_comparison/aligner_results.csv`.
- Note-category ablation is available under `results/note_ablation/note_ablation_results.csv`.
- Condition graphs and physiology templates are released in `final_release/condition_graphs/` and `final_release/physiology_templates/`.
- Additional exploratory experiments exist under `results/` (e.g., `delta_features_results.csv`), but they are not part of the canonical baseline tables unless explicitly referenced.
