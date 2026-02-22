# TIMELY-Bench Reproducibility Checklist

This checklist ensures that the **canonical benchmark runs** in TIMELY-Bench can be reproduced from the shipped artefacts.

---

## ✅ Data Availability

- [x] **MIMIC-IV v3.1** - Available via PhysioNet (requires credentialed access)
- [x] **Predefined splits** - `data/splits/predefined_splits.csv` (patient-level dev/test + fold_id)
- [x] **Episode JSONs** - Prebuilt in `episodes/episodes_enhanced/` (large)
- [x] **Windowed structured features** - `data/processed/data_windows/window_{6h,12h,24h,D0}/features_aggregated.csv`

---

## ✅ Environment

| Requirement | Version |
|-------------|---------|
| Python | ≥ 3.9 |
| pandas | ≥ 1.4 |
| numpy | ≥ 1.21 |
| scikit-learn | ≥ 1.0 |
| xgboost | ≥ 1.6 |
| torch | ≥ 1.12 |
| transformers | ≥ 4.20 |

### Install
```bash
pip install -r requirements.txt
```

---

## ✅ Code Structure

```
TIMELY-Bench_Final/
├── code/
│   ├── baselines/          # Training scripts
│   ├── data_processing/    # Data preparation
│   └── config.py           # Configuration
├── data/processed/         # Processed data
├── episodes/               # Episode JSONs
├── results/                # Experiment results
└── docs/                   # Data/model/protocol cards
```

---

## ✅ Reproduction Steps

### Core Baselines

```bash
cd TIMELY-Bench_Final

# Structured baselines (XGBoost / Logistic Regression) across windows
python3 code/baselines/run_baselines.py

# Text-only baseline (annotation-derived)
python3 code/baselines/train_text_only.py

# Text-only baselines (raw text semantics)
python3 code/baselines/train_text_only_embeddings.py
python3 code/baselines/train_text_only_medcat.py

# Early/Late fusion (structured + text representations)
python3 code/baselines/train_fusion.py

# ClinicalGRU (mortality)
python3 code/baselines/train_temporal_gru_v2.py

# Canonical results aggregation for reporting (run after each step)
python3 code/utils/standardize_results.py --step structured
python3 code/utils/standardize_results.py --step text
python3 code/utils/standardize_results.py --step fusion
python3 code/utils/standardize_results.py --step gru
```

Notes:
- Training the GRU and running some audits may require HPC/GPU (see `scripts/`).

---

## ✅ Verification

### Data Integrity
```bash
python3 scripts/comprehensive_final_audit.py
```

### Expected Outputs

| Task | Model | Expected AUROC |
|------|-------|----------------|
| Mortality (24h, all) | Structured XGBoost | ~0.865 |
| Mortality (24h, all) | Structured Logistic Regression | ~0.844 |
| Mortality (24h, all) | ClinicalGRU | ~0.839 |
| Prolonged LOS (24h, all) | Structured XGBoost | ~0.809 |

---

## ✅ Random Seeds

All experiments use fixed random seeds for reproducibility:

| Parameter | Value |
|-----------|-------|
| RANDOM_STATE | 42 |
| N_FOLDS | 5 |
| TEST_SIZE | 0.20 |

Defined in `code/config.py`.

---

## ✅ Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 16 GB | 32 GB |
| GPU | Not required (tabular) | NVIDIA 16GB+ (GRU) |
| Storage | 100 GB | 200 GB |

---

## ✅ Files to Check

Before submission, verify these files exist:

| File | Purpose |
|------|---------|
| `data/splits/predefined_splits.csv` | Canonical dev/test split + GroupKFold fold assignment |
| `data/splits/split_summary.json` | Split metadata and rates |
| `data/processed/merge_output/cohort_final.csv` | Cohort labels |
| `results/standardized/results_summary.csv` | Canonical results |
| `results/fusion_baselines/fusion_results_folds.json` | Fusion fold outputs |
| `results/text_only_baselines/text_only_results_folds.json` | Text-only fold outputs |
| `docs/DATA_CARD.md` | Data documentation |
| `docs/MODEL_CARD.md` | Model documentation |
| `docs/ALIGNMENT_PROTOCOL_CARD.md` | Alignment documentation |

---

## ✅ Known Issues

1. **Large file sizes**: episode JSONs and raw alignment matrices are large; use `rsync` for transfers.
2. **Text modality definition**: canonical runs include three text representations (annotation-derived features, ClinicalBERT embeddings, and MedCAT concepts); always verify model labels in `results/standardized/text_results.csv` and `results/standardized/fusion_results.csv`.

---

## Contact

For reproducibility issues, contact the project maintainers.
