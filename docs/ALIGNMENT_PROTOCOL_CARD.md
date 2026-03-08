# Alignment Protocol Card (Note-Centered v2.0)

## 1. Protocol Overview
TIMELY-Bench v2.0 aligns structured time-series and text around each note anchor using note-centered windows, then aggregates to stay-level (anchor strategy: `last_note`).

Core goals:
- Enforce strict lookback semantics for clean evaluation.
- Provide an explicit leaked condition for controlled leakage stress tests.
- Quantify leakage sources with a 2x2 decomposition.

## 2. Window Semantics

| Window | Structured Window | Text Selection | Intended Use |
|---|---|---|---|
| D0 | `[day_start, T]` | same-day notes with `chart_hour <= T` | calendar-day boundary sensitivity |
| W6 | `[T-6, T]` | notes in `[T-6, T]` | short lookback |
| W12 | `[T-12, T]` | notes in `[T-12, T]` | medium lookback |
| W24 | `[T-24, T]` | notes in `[T-24, T]` | canonical clean lookback |
| leaked | `[T-24, T+24]` | all notes incl. AFTER | intentional leakage stress condition |
| clean | same structured as W24 | BEFORE+OVERLAP only | strict no-AFTER text condition |

`T` denotes anchor note chart hour. For stay-level release, anchor is the last note in 0-48h.

## 3. D0 Boundary Truncation
D0 uses calendar-day-up-to-T semantics. If anchor note is close to day start, available window duration can be short.

Phase 5 analysis output (`results/note_centered/analysis/d0_boundary_analysis.csv`) confirms a non-trivial short-window mass in `[0,2h)`, but D0 remains competitive and is strongest for prolonged LOS structured baselines.

## 4. Clean vs Leaked Definitions
- `leaked`: bidirectional structured window (+24h future), text includes AFTER content.
- `clean`: W24 lookback structured window, text excludes AFTER via DocTimeRel weighting.

Important implementation detail:
- Text note pool for leaked and W24 is identical under `last_note` anchor (`T` is last note hour).
- Text leakage signal therefore comes from sentence-level DocTimeRel inclusion/exclusion, not from adding future notes.

## 5. 2x2 Leakage Decomposition
We decompose leakage using early fusion XGBoost:

| Structured \ Text | original (AFTER included) | weighted_no_after (AFTER excluded) |
|---|---:|---:|
| leaked (±24h) | A (full leaked) | B (struct-only leak) |
| W24 lookback | C (text-only leak) | D (clean) |

Premium components:
- `premium_total = A - D`
- `premium_struct = B - D`
- `premium_text = C - D`
- `premium_interaction = premium_total - premium_struct - premium_text`

Observed (Phase 4 fixed runs):
- Mortality: total +0.0154, structural share ~99%
- Prolonged LOS: total +0.0508, structural share ~100%
- Text premium approximately 0 in both tasks with note-level ClinicalBERT pooling

## 6. Key Practical Takeaway
For this benchmark setup, AUROC inflation is dominated by structured temporal leakage rather than DocTimeRel AFTER sentence inclusion. Strict lookback on structured data is therefore the primary anti-leakage control.
