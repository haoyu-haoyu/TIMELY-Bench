# Alignment Protocol Card

## Overview

This document describes the time-alignment protocols used in TIMELY-Bench to align clinical time-series data with textual notes.

---

## Alignment Windows

| Window ID | Description | Time Offset | Use Case |
|-----------|-------------|-------------|----------|
| **D0** | Same calendar day | Same day as pattern | Daily aggregated features |
| **W6** | ±6 hours | charttime ± 6h | High temporal precision |
| **W12** | ±12 hours | charttime ± 12h | Balanced precision/coverage |
| **W24** | ±24 hours | charttime ± 24h | Maximum coverage |

---

## Alignment Algorithm

```
For each (time-series observation, clinical note) pair:
    1. Extract charttime from time-series record
    2. Extract note_time from clinical note
    3. Compute time_delta = |charttime - note_time|
    4. If time_delta <= window_size:
        - Mark as temporally aligned
        - Extract relevant text segments
        - Generate annotation (SUPPORTIVE/CONTRADICTORY/UNRELATED)
```

---

## Performance by Window Size

| Window | Test AUROC | Coverage | n_alignments/episode |
|--------|------------|----------|---------------------|
| **D0** | 0.798 | ~50% | ~650 |
| ±6h | 0.777 | 45% | ~520 |
| ±12h | 0.800 | 72% | ~890 |
| **±24h** | **0.833** | 100% | ~1,350 |

**Conclusion**: ±24h window provides best predictive performance. D0 (daily) offers a balance for aggregated features.

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
| **Rule-based** | Keyword + negation detection | 96% | ~70% |
| **LLM (DeepSeek)** | Prompt-based reasoning | 4% | ~90% |

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
| `temporal_textual_alignment.csv` | 47 GB | Full alignment matrix |
| `smart_annotations_full.csv` | 6.9 GB | Pattern annotations |
| `episodes_all/` | ~50 GB | 74,829 Episode JSONs |

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

1. **Window granularity**: 4 predefined windows tested (D0, ±6h, ±12h, ±24h)
2. **Note timestamp accuracy**: Some notes have imprecise `charttime`
3. **Pattern coverage**: Only 15 common patterns detected
4. **Annotation noise**: Rule-based annotations may have false positives
