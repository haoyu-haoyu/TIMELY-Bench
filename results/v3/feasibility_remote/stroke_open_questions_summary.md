# Stroke Open Questions Audit

## Part 1: Cohort Reconciliation
- Total fresh stroke stays: `10653`
- Total fresh stroke hadm: `9451`
- Ischaemic-only stays: `4756`
- Hemorrhagic stays: `3029`
- TIA stays: `193`
- Other cerebrovascular stays: `1239`
- Explanation: The fresh cohort has 4756 stays under the old ischaemic-only code family versus 5925 in the prior audit. Relative to the fresh broad stroke cohort (9217 stays), 4461 stays come from non-ischaemic code families or broader cerebrovascular definitions.

## Part 2: Nursing Content
- Median neuro observations/stay: `213`
- Median distinct neuro categories/stay: `9`
- % stays with neuro obs in first 6h: `97.81%`
- % sampled stays with L-arm strength change: `58.0%`

## Part 3: Anchor Definition
- HPI specific onset mention: `60.0%`
- Wake-up/unknown onset: `13.33%`
- First brain imaging median hour: `7.0`
- First neuro observation median hour: `1.0`
- Recommended anchor: `ICU_admission_or_first_neuro_observation`

## Part 4: Physical Exam Split
- Clear admission/discharge split: `3.36%`
- Recommendation: `exclude_from_layer1`

## Part 6: Structured Cross-Reference
- GCS matched observations equal: `100.0%`
- Incremental value: Nursing-note GCS appears to overlap heavily with structured GCS, but limb-strength observations provide additional neuro information not present in the current structured grid.

