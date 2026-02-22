# Calibration Evaluation (TIMELY-Bench v2.0)

**Canonical file:** `calibration_summary.csv`  
**Last updated:** 2026-02-06

This folder contains reliability (calibration) metrics for selected benchmark models. Calibration is important for clinical risk prediction because the predicted probability should match the observed event rate (e.g., “20% mortality risk” should mean roughly 20% of such patients die).

---

## 1. What Is Evaluated

### Canonical coverage (`calibration_summary.csv` + `calibration_fusion_summary.csv`)

- **Structured baselines (LogisticRegression, XGBoost)**:
  - Windows: 6h / 12h / 24h
  - Cohorts: all / sepsis / aki
  - Tasks: mortality + prolonged_los
- **Additional 24h mortality models in `calibration_summary.csv`**:
  - `ClinicalGRU`
  - `TextOnly_XGBoost` (annotation-derived text features)
  - `EarlyFusion_XGBoost` (structured + annotation-derived text features)
- **Fusion families in `calibration_fusion_summary.csv`**:
  - `LateFusion_XGBoost` (annotation-derived)
  - `LateFusion_XGBoost_ClinicalBERT`
  - `EarlyFusion_XGBoost_ClinicalBERT`
  - `Structured_XGBoost_ClinicalBERT`
  - `TextOnly_XGBoost_ClinicalBERT`

---

## 2. Quick Reference (24h Mortality, All Cohort)

Values below are rounded from canonical outputs (`calibration_summary.csv` and
`calibration_fusion_summary.csv`):

| Model | ECE | Brier | Notes |
|------|----:|------:|------|
| XGBoost (structured) | 0.1974 | 0.1327 | Over-confident on holdout |
| Logistic Regression (structured) | 0.0083 | 0.0823 | Well-calibrated baseline |
| TextOnly_XGBoost | 0.0062 | 0.0965 | Annotation-derived text features |
| EarlyFusion_XGBoost | 0.0066 | 0.0770 | Structured + annotation features |
| LateFusion_XGBoost | 0.1813 | 0.1234 | Weighted-prediction fusion |
| EarlyFusion_XGBoost_ClinicalBERT | 0.0086 | 0.0740 | Structured + embeddings |
| LateFusion_XGBoost_ClinicalBERT | 0.1078 | 0.0915 | Weighted-prediction fusion |
| ClinicalGRU | 0.0336 | 0.0871 | Temporal DL baseline |

---

## 3. File Guide

- `calibration_summary.csv`: core calibration table (structured + selected models).
- `calibration_fusion_summary.csv`: fusion and ClinicalBERT families (24h all cohort).
- `calibration_summary.json`: detailed calibration breakdowns.
- `calibration_dl_summary.json`: DL calibration outputs (if generated via HPC scripts).
- `reliability_diagrams/`: per-model reliability plots.

### Deprecated

- `calibration_complete.csv` is **deprecated**: it contains incorrect positive rates for prolonged LOS (e.g., ~0.48/0.56/0.60), indicating a label mapping bug in that file. Do not use it for reporting.

---

## 4. Metrics

- **ECE (Expected Calibration Error)**: weighted average calibration gap across probability bins (lower is better).
- **MCE (Maximum Calibration Error)**: worst-case bin gap (lower is better).
- **Brier score**: mean squared error of predicted probabilities (lower is better).

---

## 5. Reproduction

From `TIMELY-Bench_Final/`:

```bash
python code/evaluation/run_calibration_evaluation.py
```

If you re-run calibration, treat `calibration_summary.csv` as the source of truth for the report and keep `calibration_complete.csv` out of any “final numbers” tables.
