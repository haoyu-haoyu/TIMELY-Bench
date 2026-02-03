# Calibration Evaluation Results

## Model Coverage

| Model Type | Evaluated | ECE (24h, all) | Notes |
|------------|-----------|----------------|-------|
| XGBoost (Structured) | Done | 0.0081 | Traditional ML baseline |
| Logistic Regression | Done | 0.0094 | Traditional ML baseline |
| ClinicalGRU | Done | 0.0336 | Deep learning temporal model |
| Early Fusion XGBoost | Done | 0.0076 | Structured + text features |
| TextOnly XGBoost | Done | 0.0052 | Text features only |

## Calibration Summary (24h, Mortality, All Cohort)

| Model | ECE | MCE | Brier Score | Interpretation |
|-------|-----|-----|-------------|----------------|
| TextOnly_XGBoost | 0.0052 | 0.0052 | 0.1083 | Excellent |
| EarlyFusion_XGBoost | 0.0076 | 0.0450 | 0.0787 | Excellent |
| XGBoost | 0.0081 | 0.0719 | 0.0790 | Excellent |
| LogisticRegression | 0.0094 | 0.0950 | 0.0835 | Excellent |
| ClinicalGRU | 0.0336 | 0.2426 | 0.0871 | Good |

## Key Findings

1. **All models show good calibration** (ECE < 0.05), suitable for clinical decision support
2. **Early Fusion achieves best overall performance**: lowest Brier score (0.0787) with excellent calibration
3. **ClinicalGRU has higher ECE** (0.0336) compared to XGBoost variants, suggesting some overconfidence
4. **Text-only model has excellent calibration** but poor discrimination (AUROC~0.5)

## Metrics Explanation

- **ECE (Expected Calibration Error)**: Weighted mean absolute difference between predicted probability and actual frequency across probability bins. Lower is better.
- **MCE (Maximum Calibration Error)**: Worst-case calibration error across all probability bins.
- **Brier Score**: Mean squared error of probabilistic predictions. Lower is better. Decomposes into calibration + refinement.

## Interpretation Guidelines

| ECE Range | Interpretation |
|-----------|----------------|
| < 0.02 | Excellent calibration |
| 0.02 - 0.05 | Good calibration |
| 0.05 - 0.10 | Moderate calibration |
| > 0.10 | Poor calibration, consider recalibration |

## File Structure

- `calibration_complete.csv` - All models' calibration metrics (ML + DL combined)
- `calibration_summary.csv` - DL models results (HPC output)
- `calibration_summary.json` - ML models detailed results
- `calibration_dl_summary.json` - DL models detailed results
- `reliability_diagrams/` - Per-model reliability (calibration) curve plots
  - `reliability_ClinicalGRU_mortality_24h_all.png`
  - `reliability_EarlyFusion_XGBoost_mortality_24h_all.png`
  - `reliability_TextOnly_XGBoost_mortality_24h_all.png`
  - `dl_models_comparison.png`

## Reproduction

ML models (local):
```bash
cd TIMELY-Bench_Final
python code/evaluation/run_calibration_evaluation.py
```

DL models (HPC):
```bash
sbatch scripts/run_dl_calibration_hpc.sh
```

## Generated

- ML results: 2026-02-03T02:40:40
- DL results: 2026-02-03T14:38:18
- HPC: KCL CREATE, NVIDIA A100-SXM4-40GB
