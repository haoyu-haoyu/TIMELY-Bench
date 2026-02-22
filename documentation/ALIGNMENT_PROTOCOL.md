# TIMELY-Bench Alignment Protocol (Deprecated)

This file is a **legacy snapshot** and is not kept in sync with the canonical benchmark protocols.

Canonical sources:
- `TIMELY-Bench_Final/docs/ALIGNMENT_PROTOCOL_CARD.md`
- `TIMELY-Bench_Final/final_release/ALIGNMENT_PROTOCOL_CARD.md`

The remainder of this file summarises the current protocol at a high level and replaces older "LLM feature injection"
descriptions that are no longer used in v2.0.

---

## Time Reference Point

| Element | Definition |
|---------|------------|
| **T0** | ICU admission time (`intime` from `icustays`) |
| **Observation Window** | [T0, T0 + W hours] |
| **Prediction Target** | Events after observation window |

---

## Alignment Windows

| Window ID | Hours | Description |
|-----------|-------|-------------|
| 6h | 6h | Observation horizon from ICU admission: [0h, 6h) |
| 12h | 12h | Observation horizon from ICU admission: [0h, 12h) |
| 24h | 24h | Observation horizon from ICU admission: [0h, 24h) |

---

## Time-Series Alignment

### Vital Signs & Labs

```
Data Source: chartevents, labevents
Alignment: charttime relative to T0
Aggregation: Hourly buckets [0, 1, 2, ..., W-1]
```

### Handling Missing Hours

1. **Forward Fill**: Carry last observation forward
2. **Zero Imputation**: Fill remaining NaN with 0
3. **Missingness Flags**: Binary indicator if feature ever observed

---

## Text Alignment

TIMELY-Bench supports two complementary text pathways:

1. **Pattern-note alignment (evidence extraction; causal lookback)**:
   for each detected pattern event at hour `t`, align notes in `[t-6h, t]` (no lookahead) and optionally annotate a
   sparse audited subset as SUPPORTIVE / CONTRADICTORY / UNRELATED.
2. **Semantic text baselines**:
   stay-level ClinicalBERT embeddings computed from raw note text within the first 24 hours (mean pooled across notes).

---

## Fusion Strategies

### 1. Early Fusion (Concatenation)

```
X_fused = concat(X_structured, X_text)
Model: XGBoost on concatenated features (Early Fusion)
```

### 2. Late Fusion (Weighted Probabilities)

```
p_struct = StructuredModel(X_structured)
p_text = TextModel(X_text)
p_fused = alpha * p_struct + (1 - alpha) * p_text
```

In v2.0 canonical results, Late Fusion is tuned on a validation split per task and text representation.

---

## Data Leakage Prevention

| Risk | Mitigation |
|------|------------|
| Future information | Strict time filtering: only data before T0+W |
| Patient overlap | GroupKFold by subject_id |
| Label leakage | Labels computed from data after observation window |
| Scaling leakage | StandardScaler fit only on training fold |

---

## Validation Protocol

1. **5-fold GroupKFold**: Grouped by subject_id
2. **Metrics**: AUROC (primary), AUPRC, Brier Score
3. **Reporting**: Mean ± Std across 5 folds

---

## Reproducibility Checklist

- [ ] Use the patient-level splits in `data/splits/` (or GroupKFold with the same random_state=42)
- [ ] Apply StandardScaler within each fold (no leakage)
- [ ] Use identical time windows (6h/12h/24h from ICU admission)
- [ ] Cite the MIMIC-IV version recorded in episode metadata (`metadata.source_version`, e.g. v3.1)
