# TIMELY-Bench

**Clinical Temporal-Text Alignment Benchmark for Multimodal ICU Prediction**

[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-PhysioNet-blue.svg)](https://physionet.org)

English | [中文](README_zh.md)

## Current Status

| Metric | Value |
|--------|-------|
| Total Episodes | **74,829** |
| Unique Patients | ~50,000 |
| Time Windows | 6h, 12h, 24h (+ D0 aligner comparison) |
| Physiological Features | 25 |
| Text representations | annotation-derived stats, ClinicalBERT embeddings (MedCAT concepts optional) |

**Last updated:** February 2026 | **Version:** 2.0 Final

## What it does

TIMELY-Bench is a comprehensive benchmark for evaluating multimodal clinical prediction models that integrate temporal physiological data with clinical text annotations. It:

- Curates benchmark-ready cohorts from MIMIC-IV with transparent alignment protocols
- Provides multi-window temporal feature extraction (6h, 12h, 24h) and D0 aligner protocol comparison
- Implements traditional ML and deep learning baselines
- Includes early fusion approaches combining structured + text features
- Offers calibration evaluation and cross-window robustness analysis

## Key Results (24h Window, All Cohort)

All values below use **holdout test-set** metrics from `results/standardized/results_summary.csv`.

### Mortality Prediction

| Model | AUROC | AUPRC | ECE | Brier |
|-------|-------|-------|-----|-------|
| Early Fusion XGBoost (Structured + ClinicalBERT embeddings) | **0.885** | **0.584** | 0.0086 | 0.0740 |
| Late Fusion (tuned $\alpha$, ClinicalBERT) | 0.881 | 0.551 | 0.1078 | 0.0915 |
| Early Fusion XGBoost (Structured + annotation-derived) | 0.873 | 0.557 | 0.0066 | 0.0770 |
| Late Fusion (tuned $\alpha$, annotation-derived) | 0.869 | 0.535 | 0.1813 | 0.1234 |
| XGBoost (Structured) | 0.868 | 0.541 | 0.1974 | 0.1327 |
| Logistic Regression (Structured) | 0.848 | 0.508 | 0.0083 | 0.0823 |
| Clinical GRU (Temporal) | 0.842 | 0.483 | 0.0336 | 0.0871 |
| Logistic Regression (Text-Only, ClinicalBERT embeddings) | 0.832 | 0.444 | --- | --- |
| XGBoost (Text-Only, ClinicalBERT embeddings) | 0.817 | 0.444 | 0.0089 | 0.0881 |
| XGBoost (Text-Only, annotation-derived) | 0.755 | 0.327 | 0.0062 | 0.0965 |
| Logistic Regression (Text-Only, MedCAT concepts) | 0.552 | 0.150 | --- | --- |
| XGBoost (Text-Only, MedCAT concepts) | 0.552 | 0.151 | --- | --- |

### Prolonged LOS Prediction

| Model | AUROC | AUPRC |
|-------|-------|-------|
| Early Fusion XGBoost (Structured + ClinicalBERT embeddings) | **0.835** | **0.509** |
| Late Fusion (tuned $\alpha$, ClinicalBERT) | 0.834 | 0.506 |
| Early Fusion XGBoost (Structured + annotation-derived) | 0.818 | 0.468 |
| XGBoost (Structured) | 0.815 | 0.460 |
| Logistic Regression (Structured) | 0.797 | 0.422 |
| Late Fusion (tuned $\alpha$, annotation-derived) | 0.815 | 0.458 |
| XGBoost (Text-Only, ClinicalBERT embeddings) | 0.800 | 0.456 |
| Logistic Regression (Text-Only, ClinicalBERT embeddings) | 0.800 | 0.452 |
| XGBoost (Text-Only, annotation-derived) | 0.701 | 0.311 |
| Logistic Regression (Text-Only, MedCAT concepts) | 0.549 | 0.192 |
| XGBoost (Text-Only, MedCAT concepts) | 0.550 | 0.195 |

### Cross-Window Robustness

Mortality AUROC (Structured baselines, all cohort):

| Model | 6h | 12h | 24h | CV (%) |
|------|----|-----|-----|--------|
| XGBoost | 0.805 | 0.839 | 0.868 | 3.05 |
| Logistic Regression | 0.783 | 0.818 | 0.852 | 3.13 |

**Key Finding:** Window choice significantly affects AUROC (Friedman $\chi^2$=12.0, p=0.0025). Longer windows consistently improve performance (Wilcoxon p=0.0313 for each pairwise comparison).

---

## Outputs Included In `final_release/`

- `final_release/` is a lightweight, checksummed bundle of key artefacts (graphs, templates, QC, CRES, and evidence). The full episode JSONs live in `episodes/episodes_enhanced/` and are not duplicated inside `final_release/` due to size.
- `condition_graphs/`: guideline-anchored condition graphs for Sepsis/SIRS, AKI/KDIGO, Delirium/ICU, and Stroke/Neuro (with domain tags like `lab_marker`, `vital_sign`, `symptom`, `medication`, `multimorbidity`).
- `physiology_templates/`: canonical trajectories (physiology templates) describing expected temporal evolution for exemplar conditions.
- `llm_annotations/`: a curated annotation subset (e.g., ~900 items) for quality-control and evaluation.
- `evidence/`, `qc/`, `cres/`: reproducibility artefacts and evaluation scaffolding.

## Important Terminology (Avoid Confusion)

- `Early Fusion (AnnotFeatures)`: structured aggregated features concatenated with annotation-derived text features, trained as one tabular model (`results/fusion_baselines/`).
- `Early Fusion (ClinicalBERT)`: structured aggregated features concatenated with stay-level ClinicalBERT embeddings.
- `EarlyFusion_XGBoost` (in some robustness/calibration scripts): a structured-only label used by legacy naming; it is not multimodal fusion.

## Project Structure

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/                    # Model training scripts
│   │   ├── train_tabular_baselines.py
│   │   ├── train_text_only.py
│   │   ├── train_fusion.py
│   │   └── train_temporal_gru_v2.py
│   ├── evaluation/                   # Evaluation scripts
│   │   ├── run_calibration_evaluation.py
│   │   ├── update_robustness_final.py
│   │   └── merge_dl_robustness.py
│   └── config.py                     # Configuration
├── data/
│   └── processed/
│       ├── data_windows/             # Multi-window features
│       │   ├── window_6h/            # 6-hour features (~208 MB)
│       │   ├── window_12h/           # 12-hour features (~347 MB)
│       │   ├── window_24h/           # 24-hour features (~560 MB)
│       │   └── window_D0/            # D0 calendar-day aligner features
│       └── merge_output/
│           └── cohort_final.csv      # Patient cohort labels
├── results/
│   ├── standardized/                 # Canonical results summary (CSV/JSON)
│   ├── robustness/                   # Cross-window analysis + stats tests
│   ├── calibration/                  # Calibration evaluation
│   ├── fusion_baselines/             # Early/late fusion baselines (+ tuned alpha)
│   └── text_only_baselines/          # Text-only baseline outputs
├── scripts/                          # HPC submission scripts
└── docs/
    ├── RESULTS_SUMMARY.md            # Canonical release-facing metrics
    ├── DATA_CARD.md                  # Dataset documentation
    ├── MODEL_CARD.md                 # Baseline model documentation
    └── ALIGNMENT_PROTOCOL_CARD.md    # Alignment protocol documentation
```

---

## Quick Start

### Prerequisites

```bash
# Create conda environment
conda create -n timely python=3.10
conda activate timely

# Install dependencies
pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn scipy tqdm
```

### Train Models

```bash
cd TIMELY-Bench_Final

# Structured-only baselines
python code/baselines/train_tabular_baselines.py

# Text-only baselines (annotation-derived)
python code/baselines/train_text_only.py

# Fusion baselines (early concat + late weighted)
python code/baselines/train_fusion.py
```

### Run Evaluations

```bash
# Calibration evaluation
python code/evaluation/run_calibration_evaluation.py

# Robustness analysis with statistical tests
python code/evaluation/update_robustness_final.py
```

### HPC Training (Optional)

```bash
# Submit SLURM job
sbatch scripts/run_dl_multiwindow_hpc.sh
```

**HPC Requirements:**
- GPU: NVIDIA A100-SXM4-40GB (or equivalent)
- Memory: 64GB RAM
- Time: ~4-8 hours for full training

---

## Benchmark Tasks

| Task | Definition | Positive Rate |
|------|------------|---------------|
| **In-Hospital Mortality** | Death during hospital stay | ~12.4% |
| **Prolonged LOS** | ICU stay > 7 days | ~15.2% |
| **30-Day Readmission** | Readmission within 30 days | ~8.5% |

---

## Documentation

- [Results Summary](docs/RESULTS_SUMMARY.md) - Canonical benchmark tables (release-facing)
- [Data Card](docs/DATA_CARD.md) - Dataset description and statistics
- [Alignment Protocol Card](docs/ALIGNMENT_PROTOCOL_CARD.md) - Time alignment details
- [Model Card](docs/MODEL_CARD.md) - Baseline model specifications
- [Calibration README](results/calibration/CALIBRATION_README.md) - Calibration evaluation details
- [Legacy archived report](documentation/archive_legacy/FINAL_PROJECT_REPORT.md) - Historical report kept for provenance

---

## Citation

```bibtex
@misc{timely-bench-2026,
  title={TIMELY-Bench: A Benchmark for Clinical Temporal-Text Alignment},
  author={Wang, Haoyu},
  year={2026},
  institution={King's College London}
}
```

---

## License

This project uses MIMIC-IV data, which requires PhysioNet Credentialed Access.

---

## Acknowledgments

- MIMIC-IV Database (PhysioNet)
- King's College London, LOPPN Department
