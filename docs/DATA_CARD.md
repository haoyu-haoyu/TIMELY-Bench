# TIMELY-Bench Data Card (v2.0)

This data card documents the released artefacts in `TIMELY-Bench_Final/` and the episode-level JSON interface.

## Dataset Overview

| Field | Value |
|-------|-------|
| Name | TIMELY-Bench v2.0 |
| Source | MIMIC-IV v3.1 |
| Access | PhysioNet credentialed access |
| License | PhysioNet credentialed health data license |
| Observation window | First 24 hours of ICU stay (fixed), plus D0 calendar-day aligner |
| Supported structured windows | 6h, 12h, 24h, D0 (`code/config.py`) |

## Cohort Statistics (Current Episodes)

Source of truth:
- Cohort labels: `data/processed/merge_output/cohort_final.csv`
- Episode JSONs: `episodes/episodes_enhanced/`

| Metric | Value |
|--------|------:|
| Total ICU stays / Episodes | 74,829 |
| Unique patients (`subject_id`) | 54,551 |
| Mortality positive rate | 11.93% |
| Prolonged LOS positive rate | 16.16% |
| Sepsis (binary label) | 34,152 |
| AKI (binary label) | 57,263 |
| ARDS (binary label) | 822 |

Episode-derived volume stats (computed from `episodes/episodes_enhanced/*.json`):

| Metric | Value |
|--------|------:|
| Total note objects (within 24h window) | 6,975,132 |
| Avg notes per episode | 93.21 |
| Total detected pattern events | 3,760,396 |
| Avg pattern events per episode | 50.25 |
| Total pattern-text alignment objects | 6,974,406 |
| Avg alignments per episode | 93.20 |
| Total SUPPORTIVE annotations | 9,585 |
| Total CONTRADICTORY annotations | 8,730 |

Important nuance: most alignment objects have `annotation_category = null` (labels are sparse; see `final_release/llm_annotations/` for audited samples).

## Prediction Tasks

1. In-hospital mortality (`label_mortality` in `cohort_final.csv`).
2. Prolonged LOS (`prolonged_los_7d` in `cohort_final.csv`).
3. Optional label present: `readmission_30d` (not used in the canonical paper tables unless explicitly stated).

## Modalities

### 1. Time-series (Structured)

- Stored in each episode: `timeseries.vitals` (hourly) and `timeseries.labs` (event-based).
- Aggregated windows for baselines are released as tabular features:
  - `data/processed/data_windows/window_6h/features_aggregated.csv`
  - `data/processed/data_windows/window_12h/features_aggregated.csv`
  - `data/processed/data_windows/window_24h/features_aggregated.csv`
  - `data/processed/data_windows/window_D0/features_aggregated.csv`

### 2. Clinical text

- Episode field: `clinical_text.notes` (note objects within the 24h window), plus `clinical_text.note_types`, `clinical_text.coverage_hours`.
- Precomputed optional embedding artefacts:
  - `data/processed/text_embeddings/clinical_bert_embeddings.npy`
  - `data/processed/text_embeddings/embedding_stay_ids.csv`

### 3. Alignment artefacts (Pattern-text)

- Episode field: `reasoning.pattern_annotations` and summary counters (`n_alignments`, `n_supportive`, `n_contradictory`).
- Large alignment matrix (not required for training baselines, but used for audit/reconstruction):
  - `data/processed/temporal_alignment/temporal_textual_alignment.csv`

### 4. Knowledge scaffolding

The final release bundle contains clinician-facing scaffolds:
- Condition graphs: `final_release/condition_graphs/`
- Physiology templates (canonical trajectories): `final_release/physiology_templates/`

## Data Splits

Patient-level canonical split file (no `subject_id` overlap between dev/test):

- `data/splits/predefined_splits.csv`
  - `split`: `dev` or `test` (20% holdout test)
  - `fold_id`: 1..5 for GroupKFold within `dev`
- Summary metadata:
  - `data/splits/split_summary.json`

## Episode JSON Interface

Canonical schema and example:

- `documentation/episode_schema.json`
- `documentation/example_episode.json`

Core top-level keys (all episodes):
- `episode_id`, `stay_id`, `patient`, `timeseries`, `clinical_text`, `reasoning`, `labels`, `metadata`

## Ethical Considerations

- Data are de-identified; access requires PhysioNet credentialing.
- TIMELY-Bench artefacts are for research benchmarking only and are not validated for clinical use.
