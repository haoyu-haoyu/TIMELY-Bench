# TIMELY-Bench

**Clinical Temporal-Text Alignment Benchmark for Multimodal ICU Prediction (MIMIC-IV)**

[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-PhysioNet-blue.svg)](https://physionet.org)

English | [中文](README_zh.md)

## Release Snapshot (v2.0 Note-Centered)

| Item | Value |
|---|---|
| ICU stays | **74,829** |
| Structured features | **42** |
| Time-series horizon | **0-72h** post ICU admission |
| Notes horizon | **0-48h** post ICU admission |
| Total notes | **12,005,731** |
| Tasks | `mortality`, `prolonged_los` |
| Windows | `D0`, `W6`, `W12`, `W24`, `leaked`, `clean` |
| Canonical core experiments | **91 JSONs** |

**Last updated:** March 2026

## What Changed From Legacy Pipeline

- Alignment changed from admission-anchored to **note-centered lookback alignment**.
- Structured features expanded from **25 -> 42**.
- Added explicit leakage controls and decomposition (`leaked` vs `clean`).
- Canonical results are now under `results/note_centered/`.

## Key Results (Phase 4 fixed, canonical 91)

### 1. Structured-only baselines (AUROC)

| Task | Model | D0 | W6 | W12 | W24 |
|---|---:|---:|---:|---:|---:|
| mortality | LR | 0.8775 | 0.8663 | 0.8758 | 0.8839 |
| mortality | XGBoost | 0.9007 | 0.8863 | 0.8960 | 0.9042 |
| prolonged_los | LR | 0.8858 | 0.8641 | 0.8619 | 0.8646 |
| prolonged_los | XGBoost | 0.8972 | 0.8802 | 0.8814 | 0.8817 |

### 2. 2x2 leakage decomposition (Early Fusion XGBoost)

| Task | A full leaked | B struct-only leak | C text-only leak | D clean |
|---|---:|---:|---:|---:|
| mortality | 0.9232 | 0.9231 | 0.9079 | 0.9079 |
| prolonged_los | 0.9368 | 0.9370 | 0.8856 | 0.8860 |

Leakage premium summary:
- Mortality: `A-D = +0.0154`
- Prolonged LOS: `A-D = +0.0508`
- Structural leakage dominates (~99-100% of premium), text leakage is approximately zero in this note-level ClinicalBERT setup.

### 3. Text-only baselines (AUROC)

| Task | Text Type | W24 | leaked | clean |
|---|---|---:|---:|---:|
| mortality | mean | 0.8502 | 0.8502 | 0.8501 |
| mortality | typed | 0.8390 | 0.8390 | 0.8388 |
| prolonged_los | mean | 0.8355 | 0.8355 | 0.8356 |
| prolonged_los | typed | 0.8230 | 0.8230 | 0.8234 |

### 4. Note-type ablation (mortality, W24, early fusion XGBoost)

| Condition | AUROC | Delta vs tabular |
|---|---:|---:|
| No text (tabular only) | 0.9042 | - |
| Nursing only | 0.9079 | +0.0037 |
| Radiology only | 0.9018 | -0.0024 |
| Lab only | 0.9037 | -0.0005 |
| All notes (typed pool) | 0.9073 | +0.0031 |
| All notes (mean pool) | 0.9079 | +0.0036 |

## Where To Find Artifacts

- Core results: `results/note_centered/core_experiments/`
- Tables: `results/note_centered/tables/`
- Figures: `results/note_centered/figures/`
- Analysis notes: `results/note_centered/analysis/analysis_findings.md`
- Data/Model/Protocol cards: `docs/DATA_CARD.md`, `docs/MODEL_CARD.md`, `docs/ALIGNMENT_PROTOCOL_CARD.md`

## Quick Start

### Environment

```bash
conda create -n timely python=3.10
conda activate timely
pip install torch numpy pandas scikit-learn xgboost matplotlib seaborn scipy tqdm
```

### Train baselines

```bash
cd TIMELY-Bench_Final
python code/baselines/train_tabular_baselines.py
python code/baselines/train_text_only.py
python code/baselines/train_text_only_embeddings.py
python code/baselines/train_fusion.py
```

### Generate phase-5 analysis outputs

```bash
python code/analysis/generate_core_tables.py
python code/analysis/compare_old_vs_new.py
python code/analysis/answer_analysis_questions.py
MPLBACKEND=Agg python code/analysis/generate_figures.py
```

## Benchmark Tasks

| Task | Definition |
|---|---|
| In-hospital mortality | Death during hospital stay |
| Prolonged LOS | ICU length of stay > 7 days |

## License and Access

This repository uses MIMIC-IV data and requires PhysioNet credentialed access for raw data extraction.

## Citation

```bibtex
@misc{timely-bench-2026,
  title={TIMELY-Bench: A Benchmark for Clinical Temporal-Text Alignment},
  author={Wang, Haoyu},
  year={2026},
  institution={King's College London}
}
```
