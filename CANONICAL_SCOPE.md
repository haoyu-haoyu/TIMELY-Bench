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
- LLM annotation assets
  - Annotation prompts and outputs are not stored in the public GitHub export
  - Aggregate annotation summaries may be released when they do not include
    note excerpts or patient-level identifiers

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

- Deprecated legacy tracks are not part of the active project scope and are not
  retained in the public GitHub export unless they are needed for a documented
  reproducibility check.
