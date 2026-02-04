# Nursing Duplicates Recheck Report

**Generated**: 2026-02-01T13:32:05.250070+00:00
**Data Source**: episodes/episodes_enhanced/*.json -> clinical_text.notes[note_type=nursing]

## Statistics

| Metric | Value |
|--------|-------|
| Total Nursing Notes | 6,790,265 |
| Episodes with Nursing | 74,734 |
| Unique Texts | 245 |
| Duplicate Rate | 99.9964% |
| Mean Length | 16.47 chars |
| Median Length | 15 chars |
| Short Entries (<50 chars) | 99.69% |

## Top 10 Most Common Texts

| Text (truncated) | Count | % |
|------------------|-------|---|
| SR (Sinus Rhythm)... | 1,007,429 | 14.8364% |
| Full resistance... | 654,844 | 9.6439% |
| Obeys Commands... | 401,848 | 5.918% |
| Some resistance... | 334,541 | 4.9268% |
| Spontaneously... | 324,114 | 4.7732% |
| Patient Verbalized... | 318,370 | 4.6886% |
| Consistently... | 288,005 | 4.2414% |
| ST (Sinus Tachycardia)... | 274,136 | 4.0372% |
| AF (Atrial Fibrillation)... | 177,328 | 2.6115% |
| No response... | 171,446 | 2.5249% |


## Semantic Interpretation

**Conclusion**: FLOWSHEET_ENTRIES

Nursing notes are primarily short flowsheet/charted event entries (e.g., 'Sinus rhythm', 'IV patent'), not full narrative text.

**Processing Strategy**: Per-stay exact dedup before LLM sampling; report as 'structured-like text' in documentation.

## Sample Nursing Notes

- **TIMELY_v2_30000153** (19 chars): `CMV/ASSIST/AutoFlow...`
- **TIMELY_v2_30000153** (22 chars): `ST (Sinus Tachycardia)...`
- **TIMELY_v2_30000153** (16 chars): `Change in Vitals...`
- **TIMELY_v2_30000153** (14 chars): `Localizes Pain...`
- **TIMELY_v2_30000153** (15 chars): `Arouse to Voice...`
