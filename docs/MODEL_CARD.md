# Model Card: TIMELY-Bench Core Baselines (v2.0)

This model card describes the baseline models that are used as the **canonical benchmark** in TIMELY-Bench.

Key design choice: TIMELY-Bench includes three complementary ``text'' representations:
- **Annotation-derived alignment features** (counts/ratios from pattern-text alignment + simple note statistics). These are transparent and fast, but they are not raw semantic embeddings.
- **ClinicalBERT embeddings** (stay-level CLS embeddings from raw notes within the first 24 hours, mean pooled). This baseline captures ``what the note says'' semantics.
- **MedCAT/UMLS concepts** (stay-level bag-of-concepts features from concept extraction within the first 24 hours). This provides a structured text representation, but discards most context.

## Tasks

- `mortality`: in-hospital mortality (binary label).
- `prolonged_los`: ICU length-of-stay > 7 days (binary label).

## Inputs

- Structured features: windowed aggregated features in `data/processed/data_windows/window_{6h,12h,24h,D0}/features_aggregated.csv`.
- Text (annotation-derived) features: extracted per episode from `episodes/episodes_enhanced/*.json`:
  - note statistics (`n_notes`, total/avg text length)
  - alignment statistics (`n_alignments`, `n_supportive`, `n_contradictory`, `supportive_ratio`, `annotation_density`)
- Text (ClinicalBERT) embeddings: precomputed stay-level embeddings under `data/processed/text_embeddings/`:
  - `clinical_bert_embeddings.npy`
  - `embedding_stay_ids.csv`
- Text (MedCAT) concepts: stay-level bag-of-concepts features under `data/processed/medcat_full/`:
  - `medcat_has_concepts_24h.csv`

## Canonical Results (24h, All Cohort; Test Set)

These values are generated from `results/standardized/results_summary.csv`.

### Mortality

| Model | AUROC | AUPRC | Notes |
|------|------:|------:|------|
| Structured XGBoost | 0.8677 | 0.5414 | `structured_results.csv` |
| Structured Logistic Regression | 0.8481 | 0.5076 | `structured_results.csv` |
| ClinicalGRU | 0.8419 | 0.4832 | `gru_results.csv` |
| Text-only XGBoost (AnnotFeatures) | 0.7551 | 0.3266 | `text_results.csv` |
| Text-only Logistic Regression (MedCAT) | 0.5519 | 0.1501 | `text_results.csv` |
| Text-only XGBoost (MedCAT) | 0.5520 | 0.1506 | `text_results.csv` |
| Text-only Logistic Regression (ClinicalBERT) | 0.8318 | 0.4439 | `text_results.csv` |
| Text-only XGBoost (ClinicalBERT) | 0.8168 | 0.4437 | `text_results.csv` |
| Early Fusion XGBoost (AnnotFeatures) | 0.8725 | 0.5568 | `fusion_results.csv` |
| Early Fusion XGBoost (ClinicalBERT) | 0.8848 | 0.5844 | `fusion_results.csv` |
| Late Fusion (tuned alpha; AnnotFeatures) | 0.8688 | 0.5354 | `fusion_results_late_xgb.csv` |
| Late Fusion (stacking; AnnotFeatures) | 0.8689 | 0.5348 | `fusion_results_late_xgb.csv` |
| Late Fusion (tuned alpha; ClinicalBERT) | 0.8805 | 0.5508 | `fusion_results_late_xgb.csv` |
| Late Fusion (stacking; ClinicalBERT) | 0.8803 | 0.5524 | `fusion_results_late_xgb.csv` |

### Prolonged LOS

| Model | AUROC | AUPRC | Notes |
|------|------:|------:|------|
| Structured XGBoost | 0.8145 | 0.4604 | `structured_results.csv` |
| Structured Logistic Regression | 0.7966 | 0.4219 | `structured_results.csv` |
| Text-only XGBoost (AnnotFeatures) | 0.7007 | 0.3107 | `text_results.csv` |
| Text-only Logistic Regression (MedCAT) | 0.5491 | 0.1922 | `text_results.csv` |
| Text-only XGBoost (MedCAT) | 0.5495 | 0.1946 | `text_results.csv` |
| Text-only Logistic Regression (ClinicalBERT) | 0.8000 | 0.4521 | `text_results.csv` |
| Text-only XGBoost (ClinicalBERT) | 0.7997 | 0.4559 | `text_results.csv` |
| Early Fusion XGBoost (AnnotFeatures) | 0.8182 | 0.4677 | `fusion_results.csv` |
| Early Fusion XGBoost (ClinicalBERT) | 0.8353 | 0.5089 | `fusion_results.csv` |
| Late Fusion (tuned alpha; AnnotFeatures) | 0.8146 | 0.4579 | `fusion_results_late_xgb.csv` |
| Late Fusion (stacking; AnnotFeatures) | 0.8146 | 0.4582 | `fusion_results_late_xgb.csv` |
| Late Fusion (tuned alpha; ClinicalBERT) | 0.8338 | 0.5062 | `fusion_results_late_xgb.csv` |
| Late Fusion (stacking; ClinicalBERT) | 0.8336 | 0.5063 | `fusion_results_late_xgb.csv` |

## Cross-Window Structured Baselines (Mortality, All Cohort; CV Mean AUROC)

| Model | 6h | 12h | 24h | D0 |
|------|---:|----:|----:|---:|
| XGBoost | 0.8052 | 0.8385 | 0.8679 | 0.8111 |
| Logistic Regression | 0.7833 | 0.8177 | 0.8517 | 0.7969 |

## Late Fusion Definition

- Late fusion includes two implementations:
  - weighted blending: `p_fused = alpha * p_structured + (1-alpha) * p_text`
  - stacking: logistic meta-learner trained on out-of-fold structured/text probabilities
- Tuned alpha (AnnotFeatures) is reported in:
  - `results/standardized/late_fusion_sanity_xgb_24h_all_mortality.json`
  - `results/standardized/late_fusion_sanity_xgb_24h_all_prolonged_los.json`
- Tuned alpha (ClinicalBERT) is reported in:
  - `results/standardized/late_fusion_sanity_xgb_clinicalbert_24h_all_mortality.json`
  - `results/standardized/late_fusion_sanity_xgb_clinicalbert_24h_all_prolonged_los.json`

## Calibration (24h, Mortality, All Cohort)

Computed from:
- `results/calibration/calibration_fusion_summary.csv` (structured/text/fusion XGBoost families)
- `results/calibration/calibration_summary.csv` (structured Logistic Regression)
- `results/calibration/calibration_dl_summary.json` (ClinicalGRU)

| Model | ECE | Brier |
|------|----:|------:|
| Structured XGBoost | 0.1974 | 0.1327 |
| Structured Logistic Regression | 0.0083 | 0.0823 |
| ClinicalGRU | 0.0336 | 0.0871 |
| TextOnly XGBoost (annotation-derived) | 0.0062 | 0.0965 |
| TextOnly XGBoost (ClinicalBERT) | 0.0089 | 0.0881 |
| Early Fusion XGBoost (annotation-derived) | 0.0066 | 0.0770 |
| Early Fusion XGBoost (ClinicalBERT) | 0.0086 | 0.0740 |
| Late Fusion XGBoost (annotation-derived) | 0.1813 | 0.1234 |
| Late Fusion XGBoost (ClinicalBERT) | 0.1078 | 0.0915 |

## Training Protocol (Core Baselines)

- Split: patient-level (grouped by `subject_id`), holdout test size = 0.20, seed = 42.
- CV: 5-fold `GroupKFold` on train/val partition (by `subject_id`).
- Metrics: AUROC, AUPRC; calibration metrics (ECE/Brier, and HL where available).

## Reproduction (Core)

```bash
cd TIMELY-Bench_Final

# Structured baselines (multi-window)
python3 code/baselines/run_baselines.py

# ClinicalGRU (mortality)
python3 code/baselines/train_temporal_gru_v2.py

# Text-only baseline (annotation-derived)
python3 code/baselines/train_text_only.py

# Text-only baseline (ClinicalBERT embeddings)
python3 code/baselines/train_text_only_embeddings.py

# Text-only baseline (MedCAT concepts)
python3 code/baselines/train_text_only_medcat.py

# Early / Late fusion (AnnotFeatures + ClinicalBERT variants)
python3 code/baselines/train_fusion.py

# Canonical aligner comparison (D0/6h/12h/24h; MedCAT baseline)
python3 code/baselines/train_aligner_comparison.py

# Note-category ablation (alignment-derived note-type features)
python3 code/baselines/eval_note_ablation.py

# Canonical aggregation for reporting
python3 code/utils/standardize_results.py --step fusion
```

## Known Risks / Common Confusions

1. Naming: `EarlyFusion_XGBoost` in some robustness/calibration scripts is a **structured-only** label used for multi-window comparisons. The multimodal early-fusion baseline lives in `results/fusion_baselines/`.
2. "Text-only" in this repo can refer to either annotation-derived alignment features or ClinicalBERT embeddings; check the model label and `source_json` in `results/standardized/text_results.csv`.

## Optional / Experimental Scripts (Not Part Of Canonical Tables)

- Delta-feature ablations: `code/baselines/train_with_delta_features.py`
- Concept / embedding feature extraction: `code/data_processing/extract_bert_embeddings.py`, `code/data_processing/extract_concepts_medcat_full.py`
