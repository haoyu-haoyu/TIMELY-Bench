# Alignment Protocol Card

## Overview

This document describes the time-alignment protocols used in TIMELY-Bench to align clinical time-series data with textual notes.

---

## Alignment Windows

| Window ID | Description | Time Offset | Use Case |
|-----------|-------------|-------------|----------|
| **6h** | 6-hour observation horizon | ICU `intime` + [0h, 6h) | Earliest prediction, minimal data |
| **12h** | 12-hour observation horizon | ICU `intime` + [0h, 12h) | Balanced precision/coverage |
| **24h** | 24-hour observation horizon | ICU `intime` + [0h, 24h) | Primary benchmark window |
| **D0** | Calendar-day (admission day) aligner | ICU `intime` + [0h, hours-to-midnight) | Canonical daily aligner (chartdate-style) |

Note: 6h/12h/24h/D0 are all generated in the canonical feature pipeline (`code/data_processing/create_multi_window_data.py`) and consumed by structured baselines (`code/baselines/run_baselines.py`). D0 is also stress-tested in the dedicated aligner comparison (`code/baselines/train_aligner_comparison.py`).

---

## Alignment Algorithm

```
Structured observation windowing (prediction features):
    For each ICU stay:
        1. Compute hour_offset relative to ICU admission (intime)
        2. Keep time-series rows with 0 <= hour_offset < window_hours (6/12/24), or
           use D0 cutoff 0 <= hour_offset < hours_to_midnight (calendar-day aligner)
        3. Aggregate per-feature statistics (min/max/mean/first/last/std) + missingness + counts

Pattern-text alignment (evidence extraction; causal):
    For each detected pattern event at hour t:
        1. Select notes with hour_offset in [t - 6h, t + 0h] (no lookahead)
        2. Mark as temporally aligned (pattern-hour, note-hour)
        3. Optionally annotate alignment as SUPPORTIVE / CONTRADICTORY / UNRELATED
```

---

## Performance by Window Size (Structured Baselines)

These are the canonical structured-only baselines evaluated on the same patient-level split. Source: `results/standardized/structured_results.csv`.

### Mortality (All cohort; test set)

| Window | Logistic Regression AUROC | XGBoost AUROC |
|--------|---------------------------:|--------------:|
| 6h | 0.7812 | 0.8091 |
| 12h | 0.8141 | 0.8355 |
| 24h | 0.8483 | 0.8693 |
| D0 | 0.7908 | 0.8104 |

**Conclusion**: within fixed-hour windows, longer context provides stronger performance (6h < 12h < 24h). D0 is a calendar-day protocol and is expected to fall between early and longer fixed-hour windows.

### Canonical aligner comparison (MedCAT concept baseline; holdout test AUROC)

| Aligner | Mortality | Prolonged LOS |
|--------|----------:|--------------:|
| 6h | 0.516 | 0.527 |
| 12h | 0.530 | 0.535 |
| 24h | 0.552 | 0.550 |
| D0 | 0.523 | 0.528 |

---

## Pattern Detection

Patterns are detected from time-series data using clinical thresholds:

| Pattern | Detection Rule | Clinical Significance |
|---------|----------------|----------------------|
| Tachycardia | HR > 100 bpm for ≥2 hours | Stress, infection, hypovolemia |
| Hypotension | SBP < 90 mmHg | Shock, sepsis |
| Fever | Temp > 38.0°C | Infection |
| Tachypnea | RR > 20/min | Respiratory distress |
| Hypoxia | SpO2 < 92% | Respiratory failure |

---

## Text-Pattern Alignment

### Annotation Categories

| Category | Definition | Example |
|----------|------------|---------|
| **SUPPORTIVE** | Text confirms or explains the pattern | "Patient developed fever overnight" aligns with temp > 38°C |
| **CONTRADICTORY** | Text contradicts the pattern | "Afebrile" when temp > 38°C detected |
| **UNRELATED** | No semantic relationship | Generic text near pattern time |

### Annotation Sources

| Source | Method | Coverage | Precision |
|--------|--------|----------|-----------|
| **Sparse audited subset** | Manual/LLM-assisted auditing of sampled items | Small curated set | Used for QC |

---

## Implementation Details

### Key Files

| Component | File |
|-----------|------|
| Alignment Generation | `temporal_textual_alignment.py` |
| Pattern Detection | `pattern_detector.py` |
| Smart Rule Matcher | `smart_rule_matcher_full.py` |
| Episode Builder | `episode_builder.py` |

### Data Files

| File | Size | Description |
|------|------|-------------|
| `data/processed/temporal_alignment/temporal_textual_alignment.csv` | ~1.1 GB | Canonical alignment matrix (0-24h, discharge-excluded) |
| `episodes/episodes_enhanced/` | large | 74,829 Episode JSONs |

---

## Reproducibility

To regenerate alignments:

```bash
# 1. Run pattern detection
python code/data_processing/pattern_detector.py

# 2. Generate alignments
python code/data_processing/temporal_textual_alignment.py

# 3. Apply smart annotation rules
python code/data_processing/smart_rule_matcher_full.py

# 4. Build episodes
python code/data_processing/batch_build_all_episodes.py
```

---

## Limitations

1. **Window granularity**: released benchmark windows are (6h, 12h, 24h, D0).
2. **Note timestamp accuracy**: documentation time may lag observation time; this motivates explicit time-window alignment.
3. **Annotation sparsity**: SUPPORTIVE/CONTRADICTORY labels are sparse in the released episodes; most alignments are unlabeled (null category).
