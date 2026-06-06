# Stroke Phase 1+2 Reaudit

## Subtype reconciliation

- Broad fresh stroke cohort: `10653` stays / `9451` hadm
- Conflict stays marked as mixed: `1649` stays (`15.48%`)

### Priority-based assignment

| Subtype | Stays | HADM | % of broad stays |
|---|---:|---:|---:|
| ischaemic | 5505 | 4858 | 51.68% |
| hemorrhagic_ich | 3687 | 3297 | 34.61% |
| hemorrhagic_sah | 1193 | 1059 | 11.2% |
| tia | 268 | 245 | 2.52% |

### Mixed/conflict-marking assignment

| Subtype | Stays | HADM | % of broad stays |
|---|---:|---:|---:|
| ischaemic | 4743 | 4189 | 44.52% |
| hemorrhagic_ich | 3250 | 2933 | 30.51% |
| mixed | 1649 | 1417 | 15.48% |
| hemorrhagic_sah | 750 | 673 | 7.04% |
| tia | 261 | 239 | 2.45% |

## Ischaemic subset reaudit

- Priority-based ischaemic subset: `5505` stays / `4858` hadm
- Pure ischaemic no-conflict subset: `4743` stays / `4189` hadm
- Mixed conflicts inside priority-based ischaemic subset: `762` stays (`13.84%`)
- Primary diagnosis flag in priority-based ischaemic subset: `51.64%`

### Priority-based ischaemic coverage

- Discharge coverage: `71.51%` by hadm, `71.06%` by stay
- HPI + Hospital Course both found: `92.11%`
- Admission-text NIHSS mention: `15.43%`
- Any incremental neuro nursing coverage: `93.95%`
- >=10 incremental neuro observations: `90.39%`
- Median incremental neuro observations per stay: `95`
- Median first incremental neuro observation hour: `1.0`
- Bilateral strength data coverage: `92.41%`
- Brain radiology in first 24h: `52.03%`
- Any brain radiology: `64.52%`
- Median first brain radiology hour: `8.0`

### Priority-based ischaemic tiering (default: incremental neuro without nursing GCS)

| Tier | Stays | HADM | % of subset stays |
|---|---:|---:|---:|
| A | 3570 | 3211 | 64.85% |
| B | 1406 | 1244 | 25.54% |
| C | 342 | 320 | 6.21% |
| D | 187 | 180 | 3.4% |

- Layer 1 eligible stays: `4976`
- Layer 2 eligible stays: `3912`

### Sensitivity: pure ischaemic no-conflict subset

- Discharge coverage by hadm: `72.79%`
- >=10 incremental neuro observations: `90.64%`
- Brain radiology in first 24h: `51.63%`

### Locked rule

- Nursing GCS rows are retained in raw extraction but excluded from incremental Layer 1 stroke features, tier counts, and model-facing neuro observation counts.

