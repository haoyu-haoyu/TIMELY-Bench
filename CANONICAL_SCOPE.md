# Canonical Scope

## Canonical note-centered benchmark

- Root results directory: `results/note_centered/`
- Core experiments: `91`
- Progression experiments: `22`
- Total canonical benchmark experiments: `113`
- Canonical tasks:
  - In-hospital mortality
  - Prolonged ICU length of stay
  - AKI progression
  - Sepsis to septic shock progression

## Supplementary tracks

- `results/cres/`
  - Supplementary clinical reasoning evaluation track
  - Not counted in the 113 canonical benchmark experiments
- `results/llm_annotations/`
  - Annotation assets and annotation-specific audit records
  - Not benchmark result tables

## Project-wide audits

- `results/audit/`
  - Project-wide release and integrity audits
  - May reference canonical note-centered results, CRES, and annotation assets
  - Should not be treated as note-centered-only benchmark summaries unless explicitly labelled as such

## Comparison artifacts

- `results/note_centered/comparisons/`
  - Current comparison outputs that may reference legacy admission-anchored values
  - Retained for analysis, but not part of the canonical experiment inventory

## Archived legacy tracks

- `archive/legacy_consolidated/root_cleanup/`
  - Deprecated legacy tracks that are no longer part of the active project scope
