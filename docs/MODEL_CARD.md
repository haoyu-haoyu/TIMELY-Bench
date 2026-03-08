# Model Card: TIMELY-Bench v2.0 (Note-Centered Baselines)

## Scope
This card summarizes the Phase 4 fixed baselines (post weighted fix + typed-clean fix) used in Phase 5 analysis.

- Tasks: `mortality`, `prolonged_los`
- Structured feature count: 42
- Time-series horizon: 72h extraction, note-centered windows for modeling
- Text horizon: 48h notes (stay-level note-centered aggregation)
- Splits: `predefined_splits.csv` holdout + 5-fold CV

## Core Baseline Results

### Structured-Only (Table 1)

| Task | Model | D0 | W6 | W12 | W24 |
|---|---|---:|---:|---:|---:|
| mortality | LR | 0.8775 | 0.8663 | 0.8758 | 0.8839 |
| mortality | XGBoost | 0.9007 | 0.8863 | 0.8960 | 0.9042 |
| prolonged_los | LR | 0.8858 | 0.8641 | 0.8619 | 0.8646 |
| prolonged_los | XGBoost | 0.8972 | 0.8802 | 0.8814 | 0.8817 |

### 2x2 Leakage Decomposition (Early Fusion XGBoost, Table 2)

| Task | A full leaked | B struct-only leak | C text-only leak | D clean |
|---|---:|---:|---:|---:|
| mortality | 0.9232 | 0.9231 | 0.9079 | 0.9079 |
| prolonged_los | 0.9368 | 0.9370 | 0.8856 | 0.8860 |

Decomposition summary:
- Mortality premium total: +0.0154
- Prolonged LOS premium total: +0.0508
- Structural leakage dominates (~99%-100% of premium)
- Text leakage contribution is ~0 with note-level ClinicalBERT pooling

### Text-Only Baselines (Table 3)

| Task | Text Type | D0 | W6 | W12 | W24 | leaked | clean |
|---|---|---:|---:|---:|---:|---:|---:|
| mortality | mean | 0.8433 | 0.8326 | 0.8436 | 0.8502 | 0.8502 | 0.8501 |
| mortality | typed | 0.8315 | 0.8158 | 0.8322 | 0.8390 | 0.8390 | 0.8388 |
| prolonged_los | mean | 0.8497 | 0.8431 | 0.8429 | 0.8355 | 0.8355 | 0.8356 |
| prolonged_los | typed | 0.8358 | 0.8281 | 0.8291 | 0.8230 | 0.8230 | 0.8234 |

### Note-Type Ablation (Mortality, W24, Early Fusion XGB, Table 4)

| Condition | AUROC | Delta vs tabular |
|---|---:|---:|
| No text (tabular only) | 0.9042 | - |
| Nursing only | 0.9079 | +0.0037 |
| Radiology only | 0.9018 | -0.0024 |
| Lab only | 0.9037 | -0.0005 |
| All notes (typed pool) | 0.9073 | +0.0031 |
| All notes (mean pool) | 0.9079 | +0.0036 |

## Key Findings
1. Leakage Premium is substantial for structured leakage, especially prolonged LOS (+0.0508).
2. Text-side AFTER filtering contributes near-zero AUROC change in this setup.
3. Text adds limited marginal value over the 42-feature structured baseline.
4. Mean pooling is consistently >= typed pooling in this release.
5. D0 is unexpectedly strong for prolonged LOS in structured models.

## Reproducibility Artifacts
- Core JSONs: `results/note_centered/core_experiments/`
- Tables: `results/note_centered/tables/`
- Figures: `results/note_centered/figures/`
- Analysis markdown: `results/note_centered/analysis/analysis_findings.md`
