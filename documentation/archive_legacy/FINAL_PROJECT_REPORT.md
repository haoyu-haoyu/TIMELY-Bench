# TIMELY-Bench Final Project Report (v2.0)

**Project:** TIMELY-Bench - Clinical Temporal-Text Alignment Benchmark  
**Completion date:** 2026-02-06  
**Canonical sources (do not override manually):**
- Dataset stats: `dataset/dataset_stats.json`
- Single-window results: `results/standardized/results_summary.csv`
- Cross-window robustness: `results/robustness/window_performance.csv`, `results/robustness/statistical_tests.json`
- Calibration: `results/calibration/calibration_summary.csv`

---

## 1. Scope and Research Design

TIMELY-Bench is a benchmark for evaluating multimodal ICU prediction under **explicit temporal alignment** between structured time-series measurements (vitals/labs) and clinical text. The benchmark is designed to answer:

- How does **alignment window choice** (6h/12h/24h) affect performance?
- How much predictive signal can be recovered from **temporally aligned text**?
- Which fusion strategy is more robust in this setting: **early fusion** (feature concatenation) vs **late fusion** (probability-level combination)?

The current v2.0 release reports **annotation-derived text features** as the primary text signal in the standardized baselines. Semantic text representations (e.g., ClinicalBERT embeddings, MedCAT concepts) are extracted in `data/processed/` for extensibility but are not the core reported text-only baseline in `results/standardized/`.

---

## 2. Data Assets

### 2.1 Episode-Level JSON

- Episodes: `episodes/episodes_enhanced/TIMELY_v2_*.json`
- Each episode includes:
  - `timeseries`: structured temporal measurements
  - `clinical_text`: notes within the prediction window (note spans + optional LLM-style features)
  - `reasoning`: detected patterns, alignment statistics, and an episode-level simplified condition graph
  - `labels`: outcomes (mortality, prolonged LOS)

### 2.2 Processed Tabular Features (Multi-Window)

- Structured aggregated windows:
  - `data/processed/data_windows/window_6h/features_aggregated.csv`
  - `data/processed/data_windows/window_12h/features_aggregated.csv`
  - `data/processed/data_windows/window_24h/features_aggregated.csv`
- Cohort file: `data/processed/merge_output/cohort_final.csv`

### 2.3 Canonical Release Bundle

`final_release/` is a lightweight, checksummed bundle of key artefacts (graphs, templates, QC, CRES, evidence). Full episodes remain under `episodes/episodes_enhanced/` due to size.

---

## 3. Dataset Statistics (From `dataset/dataset_stats.json`)

| Metric | Value |
|--------|-------|
| Total episodes | 74,829 |
| Episodes with notes | 74,811 |
| Episodes with patterns | 74,812 |
| Mortality positives | 8,930 (11.9%) |
| Prolonged LOS positives | 12,095 (16.2%) |
| Total notes (within-window) | 6,975,132 |
| Total pattern events | 3,760,396 |
| Subgroup: Sepsis | 34,152 |
| Subgroup: AKI | 57,263 |
| Subgroup: ARDS | 822 |

---

## 4. Benchmark Tasks

| Task | Definition |
|------|------------|
| In-hospital mortality | Death during the hospital stay |
| Prolonged ICU LOS | ICU length of stay > 7 days |

---

## 5. Methods Implemented (Standardized Baselines)

### Structured-only (tabular)

- Logistic Regression (`results/standardized/structured_results.csv`)
- XGBoost (`results/standardized/structured_results.csv`)

### Text-only (tabular, annotation-derived)

- Logistic Regression (`results/standardized/text_results.csv`)
- XGBoost (`results/standardized/text_results.csv`)

Text features are extracted from episode JSONs and include note length statistics + pattern/alignment-derived statistics such as `supportive_ratio` and `annotation_density`.

### Text-only (tabular, ClinicalBERT embeddings)

- Logistic Regression (`results/standardized/text_results.csv`)
- XGBoost (`results/standardized/text_results.csv`)

ClinicalBERT embeddings are extracted from raw episode notes within the first 24 hours and aggregated to a stay-level representation via mean pooling.

### Text-only (tabular, MedCAT concepts)

- Logistic Regression (`results/standardized/text_results.csv`)
- XGBoost (`results/standardized/text_results.csv`)

MedCAT concept extraction is applied to 24h note windows to produce a stay-level bag-of-concepts representation (concept presence indicators).

### Multimodal fusion

- Early Fusion (tabular): concatenate structured aggregated features with either (i) annotation-derived text features or (ii) ClinicalBERT embeddings (`results/standardized/fusion_results.csv`)
- Late Fusion (probability-level): weighted average of structured XGBoost and text-only XGBoost probabilities; alpha tuned per fold for both representations (`results/standardized/fusion_results_late_xgb.csv`)

Late fusion tuning shows that the contribution of text depends on representation: annotation-derived features typically yield structured-dominant weights (alpha closer to 1.0), whereas ClinicalBERT embeddings yield more balanced weights (alpha around 0.5--0.6).

### Temporal model (deep learning)

- Clinical GRU baseline for mortality: `results/standardized/gru_results.csv`

---

## 6. Key Results (24h Window, All Cohort)

These values are taken from `results/standardized/` outputs.

### 6.1 Mortality

| Model | AUROC | AUPRC |
|------|------:|------:|
| Early Fusion (structured + ClinicalBERT embeddings) | 0.8787 | 0.5581 |
| Late Fusion (tuned alpha, ClinicalBERT) | 0.8738 | 0.5349 |
| Early Fusion (structured + annotation-derived) | 0.8703 | 0.5507 |
| XGBoost (structured) | 0.8651 | 0.5305 |
| Late Fusion (tuned alpha, annotation-derived) | 0.8638 | 0.5326 |
| Logistic Regression (structured) | 0.8442 | 0.4924 |
| Clinical GRU | 0.8392 | 0.4714 |
| Logistic Regression (text-only, ClinicalBERT embeddings) | 0.8284 | 0.4465 |
| XGBoost (text-only, ClinicalBERT embeddings) | 0.8152 | 0.4384 |
| XGBoost (text-only, annotation-derived) | 0.7551 | 0.3266 |
| XGBoost (text-only, MedCAT concepts) | 0.5635 | 0.1664 |

### 6.2 Prolonged LOS

| Model | AUROC | AUPRC |
|------|------:|------:|
| Early Fusion (structured + ClinicalBERT embeddings) | 0.8251 | 0.4874 |
| Late Fusion (tuned alpha, ClinicalBERT) | 0.8220 | 0.4786 |
| Early Fusion (structured + annotation-derived) | 0.8139 | 0.4616 |
| XGBoost (structured) | 0.8087 | 0.4557 |
| Late Fusion (tuned alpha, annotation-derived) | 0.8091 | 0.4570 |
| Logistic Regression (structured) | 0.7886 | 0.4105 |
| XGBoost (text-only, ClinicalBERT embeddings) | 0.7949 | 0.4380 |
| Logistic Regression (text-only, ClinicalBERT embeddings) | 0.7918 | 0.4363 |
| XGBoost (text-only, annotation-derived) | 0.7007 | 0.3107 |
| XGBoost (text-only, MedCAT concepts) | 0.5448 | 0.1915 |

---

## 7. Calibration (Canonical: `results/calibration/calibration_summary.csv`)

Calibration is reported for structured baselines across windows/cohorts, and for select 24h mortality models.

24h mortality (all cohort) calibration highlights:

| Model | ECE |
|------|----:|
| XGBoost (structured) | 0.0076 |
| Logistic Regression (structured) | 0.0083 |
| TextOnly_XGBoost | 0.0061 |
| EarlyFusion_XGBoost | 0.0062 |
| ClinicalGRU | 0.0336 |

**Important:** `results/calibration/calibration_complete.csv` is **deprecated** (it contains incorrect positive rates for prolonged LOS). Use `calibration_summary.csv` for reporting.

---

## 8. Cross-Window Robustness (Structured Baselines)

Cross-window robustness analysis (6h vs 12h vs 24h) is provided in:
- `results/robustness/window_performance.csv`
- `results/robustness/statistical_tests.json`

Findings (mortality AUROC across 6 model×cohort subjects):
- Friedman test: **chi-square = 12.0**, **p = 0.002479** (significant window effect)
- Pairwise Wilcoxon (6h vs 12h, 12h vs 24h, 6h vs 24h): **p = 0.03125** (each pair)

This supports the benchmark goal that **window choice is a meaningful experimental factor**.

---

## 9. Condition Graphs and Physiology Templates (Supervisor Project Alignment)

TIMELY-Bench includes two complementary temporal knowledge artefacts:

1. **Condition Graphs (canonical, guideline-anchored)**
   - Location: `final_release/condition_graphs/`
   - Schema: `final_release/condition_graphs/condition_graph_schema.json`
   - Node types: `structured_indicator`, `pattern_event`, `text_evidence`, `condition`
   - Node domains include: `lab_marker`, `vital_sign`, `symptom`, `medication`, `multimorbidity`

2. **Physiology Templates (canonical trajectories)**
   - Location: `final_release/physiology_templates/canonical_trajectories.json`
   - Encodes phased trajectories (e.g., recovery vs worsening) with expected ranges/directions.

In addition, each episode JSON contains a **simplified episode-level graph** under `reasoning.condition_graph` (node `level`: `pattern`/`condition`, with per-episode onset hours). This is intended as a lightweight, patient-specific event scaffold; the canonical graphs provide the clinically grounded, domain-typed structure.

---

## 10. Known Risks / Potential Pitfalls

- **Text semantics vs annotation features:** standardized text-only and early-fusion baselines use annotation-derived statistics, not raw semantic embeddings. If a downstream project expects “text-only = embeddings”, add embedding-based baselines on top of the extracted artefacts in `data/processed/text_embeddings/` and `data/processed/medcat_full/`.
- **Late Fusion calibration:** calibration metrics are not currently computed for late fusion in `calibration_summary.csv`.
- **Cross-window coverage:** robustness analysis covers structured baselines across windows; fusion models are only reported for the primary 24h benchmark setting.
- **Exploratory LOS scripts:** files under `results/los_baselines/` are exploratory and may use CV strategies that are not group/subject-safe; do not treat them as canonical.

---

## 11. Reproduction Guide (Recommended on KCL CREATE)

From `TIMELY-Bench_Final/`:

```bash
# Structured baselines (multi-window)
python code/baselines/train_tabular_baselines.py

# Text-only (annotation-derived)
python code/baselines/train_text_only.py

# Fusion baselines (early concat + late weighted)
python code/baselines/train_fusion.py

# Robustness + statistical tests (structured-only cross-window)
python code/evaluation/update_robustness_final.py

# Calibration
python code/evaluation/run_calibration_evaluation.py
```

---

**Report last updated:** 2026-02-06
