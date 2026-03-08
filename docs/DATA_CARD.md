# TIMELY-Bench Data Card (v2.0 - Note-Centered)

## Data Extraction (v2.0 - Note-Centered)

### Time-Series
- Source: MIMIC-IV `chartevents`, `labevents`, `inputevents`, mimic-code SOFA SQL
- Temporal range: 0-72h post ICU admission
- Variables: 42 clinical features
  - Vitals (8): `heart_rate`, `sbp`, `dbp`, `mbp`, `resp_rate`, `temperature`, `spo2`, `gcs_min`
  - Labs (17): `glucose_chart`, `albumin`, `bun`, `creatinine`, `glucose_lab`, `sodium`, `potassium`, `bicarbonate`, `chloride`, `aniongap`, `wbc`, `hemoglobin`, `hematocrit`, `platelet`, `lactate`, `ph`, `bilirubin_total`
  - Blood Gas (3): `pao2`, `paco2`, `pao2_fio2_ratio`
  - Ventilator (4): `fio2`, `peep`, `tidal_volume`, `minute_volume`
  - Scores (3): `sofa_total`, `sofa_respiration`, `sofa_5`
  - Interventions (2): `vasopressors` (binary), `rrt` (binary)
  - Medication Dose (4): `vasopressor_dose_norepi_equiv`, `propofol_rate`, `midazolam_rate`, `fentanyl_rate`
  - Other (1): `urineoutput`
- Rows: 5,387,688 (hourly, 74,829 stays x up to 72 hours)

### Clinical Notes
- Source: MIMIC-IV `noteevents` (nursing, radiology) + `labevents` (lab comments)
- Temporal range: 0-48h post ICU admission
- Total notes: 12,005,731
- Discharge notes excluded by default

### Known Limitations
- Vasopressor dose uses simplified norepinephrine-equivalent conversion (not weight-adjusted)
- PaO2/FiO2 ratio coverage: row-level 3.6%, stay-level 47.6%; sparse for non-ventilated patients
- 3 stays (`30635125`, `39438562`, `39443966`) have no notes in 0-48h; retained in structured baselines with zero text vectors
- D0 boundary effect: ~9.6% of notes have D0 window < 2h due to calendar-day truncation

## Alignment Protocol

### Window Definitions

| Window | Structured Range | Note Selection | Type |
|---|---|---|---|
| D0 | `[day_start, T]` | Same calendar day, `chart_hour <= T` | Calendar day, no future |
| W6 | `[T-6, T]` | `chart_hour in [T-6, T]` | Lookback |
| W12 | `[T-12, T]` | `chart_hour in [T-12, T]` | Lookback |
| W24 | `[T-24, T]` | `chart_hour in [T-24, T]` | Lookback |
| leaked | `[T-24, T+24]` | All notes including AFTER sentences | Bidirectional (intentional) |
| clean | Same as W24 | BEFORE + OVERLAP only (AFTER excluded) | Lookback + DocTimeRel |

Where `T` is the anchor note's `chart_hour` (stay-level uses `last_note` strategy).

### Leaked Text = W24 Text (By Design)
Since `T = chart_hour(last_note)`, no note exists after `T` at note-level selection time. Therefore, leaked and W24 note pools are identical on text-side note selection.

Text leakage is modeled via DocTimeRel sentence filtering (`original` vs `weighted_no_after` / `weighted_typed_no_after`), while structural leakage is introduced by the symmetric `±24h` structured window.

## Tasks
- In-hospital mortality (`mortality`)
- Prolonged ICU LOS (`prolonged_los`)

## Split Protocol
- Canonical split source: `data/splits/predefined_splits.csv`
- Holdout test + 5-fold CV on development cohort
- All Phase 4/5 core experiments use the same predefined splits

## Reproducibility Pointers
- Core experiment JSONs: `results/note_centered/core_experiments/`
- Tables: `results/note_centered/tables/`
- Figures: `results/note_centered/figures/`
- Analysis notes: `results/note_centered/analysis/analysis_findings.md`
