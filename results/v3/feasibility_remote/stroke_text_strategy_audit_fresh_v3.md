# Stroke Text Strategy Audit

## Cohort
- Stroke ICU stays: `10653`
- Stroke hospital admissions: `9451`
- Primary-dx stays: `2843`

## Key Findings
- Timestamped note coverage (non-discharge): `99.93%`
- Neuro-note stay coverage: `99.45%`
- Brain radiology in first 24h: `53.8%`
- Discharge summary coverage by hadm: `72.03%`
- HPI + Hospital Course both found in sampled discharge notes: `94.5%`
- Admission-section NIHSS mention rate: `8.0%`

## Decision Matrix
- Approach A: `GO`
- Approach B: `GO`
- Approach C: `GO`
- Recommended approach: `B+C combined`

Reasoning:
Timestamped non-discharge notes/radiology provide a viable temporal layer, and discharge summaries appear sectionable enough to support a separately controlled sectioning experiment.

## Section Parsing
- Sample size: `200`
- HPI found: `99.5%`
- Hospital Course found: `95.0%`
- Five or more sections parsed: `100.0%`

## Important Caveats
- Direct BigQuery note-table enumeration was not rerun during this audit; fresh exported v3 note sources and the prior note-module probe were used instead.
- Stroke discharge summaries remain hindsight-rich documents. Any temporal reasoning benchmark must avoid feeding full discharge summaries into forward-looking tasks.
- Structured temporal coverage in Part 4.2 uses hourly observed bins from `hourly_state_grid_168h` rather than raw chart event counts.
- Current exported timestamped note sources are window-limited: nursing: 0.0 to 167.0h, radiology: 0.0 to 167.0h, lab_comment: 0.0 to 167.0h.
