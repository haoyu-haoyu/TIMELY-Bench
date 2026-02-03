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
| Time Windows | 6h, 12h, 24h |
| Physiological Features | 25 |
| Text-derived Features | 11 |

**Last updated:** February 2026 | **Version:** 2.0 Final

## What it does

TIMELY-Bench is a comprehensive benchmark for evaluating multimodal clinical prediction models that integrate temporal physiological data with clinical text annotations. It:

- Curates benchmark-ready cohorts from MIMIC-IV with transparent alignment protocols
- Provides multi-window temporal feature extraction (6h, 12h, 24h)
- Implements traditional ML and deep learning baselines
- Includes early fusion approaches combining structured + text features
- Offers calibration evaluation and cross-window robustness analysis

## Key Results (24h Window, All Cohort)

### Mortality Prediction

| Model | AUROC | AUPRC | ECE | Brier |
|-------|-------|-------|-----|-------|
| **EarlyFusion_XGBoost** | **0.866** | **0.536** | 0.0076 | 0.0787 |
| XGBoost | 0.865 | 0.535 | 0.0081 | 0.0790 |
| LogisticRegression | 0.839 | 0.487 | 0.0094 | 0.0835 |
| ClinicalGRU | 0.835 | 0.488 | 0.0336 | 0.0871 |

### Prolonged LOS Prediction

| Model | AUROC | AUPRC |
|-------|-------|-------|
| **EarlyFusion_XGBoost** | **0.812** | **0.463** |
| XGBoost | 0.768 | 0.757 |
| ClinicalGRU | 0.751 | 0.372 |
| LogisticRegression | 0.738 | 0.725 |

### Cross-Window Robustness

| Task | Best Model | AUROC CV | Stability |
|------|------------|----------|-----------|
| Mortality | EarlyFusion_XGBoost | **2.72%** | Most Stable |
| Prolonged LOS | ClinicalGRU | **3.99%** | Most Stable |

**Key Finding:** Window choice significantly affects performance (Friedman p<0.001). Longer windows consistently improve prediction accuracy with large effect sizes (Cohen's d > 1.0).

---

## New Features

**LLM-Guided Disease Timelines**
- 74,711 episodes processed with DeepSeek API
- Probabilistic disease progression tracking
- Onset hour prediction and prognosis assessment

**Reasoning Chain**
- Syndrome detection (Sepsis F1: 85.3%, AKI F1: 68.4%)
- Rule-based diagnostic reasoning
- Patient state-space reconstruction (48-hour vectors)

**Enhanced Episode Structure**
- `patient_state_space`: Hourly state vectors
- `reasoning.syndrome_detection`: Clinical criteria detection
- `reasoning.reasoning_chain`: Diagnostic evidence chain
- `reasoning.disease_timeline`: LLM-generated progression

## Project Structure

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/                    # Model training scripts
│   │   ├── run_baselines.py          # Traditional ML (XGBoost, LR)
│   │   ├── train_temporal_gru_v2.py  # ClinicalGRU model
│   │   ├── train_fusion.py           # EarlyFusion model
│   │   ├── train_dl_multiwindow.py   # Multi-window DL training
│   │   └── data_loader.py            # Data loading utilities
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
│       │   └── window_24h/           # 24-hour features (~560 MB)
│       └── merge_output/
│           └── cohort_final.csv      # Patient cohort labels
├── results/
│   ├── robustness/                   # Robustness analysis
│   │   ├── window_performance.csv    # 48 rows (4 models × 3 windows × 2 tasks)
│   │   ├── statistical_tests.json    # Friedman & Wilcoxon tests
│   │   └── *.png                     # Visualizations
│   ├── calibration/                  # Calibration evaluation
│   └── Output_temporal_gru/          # DL model checkpoints
├── scripts/                          # HPC submission scripts
└── documentation/
    └── FINAL_PROJECT_REPORT.md       # Complete project report
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

# Traditional ML baselines (local)
python code/baselines/run_baselines.py

# Deep Learning models (requires GPU)
python code/baselines/train_dl_multiwindow.py --all

# Or train specific window/task
python code/baselines/train_dl_multiwindow.py --window 24h --task mortality
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

- [Final Project Report](documentation/FINAL_PROJECT_REPORT.md) - Complete project report with all metrics
- [Data Card](docs/DATA_CARD.md) - Dataset description and statistics
- [Alignment Protocol Card](docs/ALIGNMENT_PROTOCOL_CARD.md) - Time alignment details
- [Model Card](docs/MODEL_CARD.md) - Baseline model specifications
- [Calibration README](results/calibration/CALIBRATION_README.md) - Calibration evaluation details

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
