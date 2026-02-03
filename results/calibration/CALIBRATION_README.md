# Calibration Evaluation Results

## Model Coverage

| Model Type | Evaluated | Notes |
|------------|-----------|-------|
| XGBoost (Structured) | Done | Traditional ML baseline |
| Logistic Regression | Done | Traditional ML baseline |
| ClinicalGRU | Pending | Requires GPU (HPC) |
| Early Fusion XGBoost | Pending | Requires episodes data (HPC) |
| TextOnly XGBoost | Pending | Requires episodes data (HPC) |

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

- `calibration_summary.csv` - All models' calibration metrics (ML + DL)
- `calibration_summary.json` - ML models detailed results
- `calibration_dl_summary.json` - DL models detailed results (after HPC run)
- `reliability_diagrams/` - Per-model reliability (calibration) curve plots

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
