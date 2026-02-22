# Search Strategy and Query Strings

Updated: 2026-02-07

## Core Query Blocks

Block A (modality):
- "multimodal" OR "multi-modal" OR "time-series and notes" OR "clinical notes and vitals"

Block B (clinical context):
- "ICU" OR "critical care" OR "MIMIC" OR "electronic health records"

Block C (temporal/fusion):
- "temporal alignment" OR "alignment window" OR "early fusion" OR "late fusion" OR "cross-modal"

Block D (prediction):
- "mortality" OR "length of stay" OR "readmission" OR "clinical prediction"

## PubMed Example

```
(
  multimodal OR "multi-modal" OR "time-series and notes" OR "clinical notes and vitals"
)
AND
(
  ICU OR "critical care" OR MIMIC OR "electronic health records"
)
AND
(
  "temporal alignment" OR "alignment window" OR "early fusion" OR "late fusion" OR "cross-modal"
)
AND
(
  mortality OR "length of stay" OR readmission OR "clinical prediction"
)
```

## arXiv/General Search Example

```
("MIMIC" OR "ICU")
AND ("multimodal" OR "time series" OR "clinical notes")
AND ("temporal alignment" OR "fusion")
AND ("mortality" OR "length of stay" OR "readmission")
```

## Operational Notes

- Record search date, database, and exact query copy.
- Save raw export (CSV/BibTeX) before deduplication.
- Keep a changelog when refining query strings.

