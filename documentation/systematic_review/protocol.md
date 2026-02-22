# A1 Review Protocol

Updated: 2026-02-07

## 1. Objective

Systematically map multimodal clinical prediction studies that combine:
- structured clinical time-series (vitals/labs/meds), and
- unstructured clinical text (notes/reports),

with explicit extraction of:
- dataset choice,
- clinical tasks,
- temporal alignment protocol,
- fusion strategy,
- evaluation metrics (including calibration when available).

## 2. Research Questions

1. Which datasets are used (MIMIC-III/IV, eICU, HiRID, others)?
2. Which outcomes are predicted (mortality, prolonged LOS, readmission, others)?
3. How is temporal alignment defined (charttime/chartdate, fixed windows, adaptive windows)?
4. Which model families are used (tabular, sequence, transformer, fusion)?
5. Which metrics are reported (AUROC/AUPRC/calibration/robustness)?

## 3. Population/Intervention/Comparator/Outcome (PICO-like framing)

- Population: ICU/inpatient episodes in EHR datasets.
- Intervention: multimodal models using both time-series and text.
- Comparator: unimodal baselines or alternative alignment/fusion setups.
- Outcome: predictive discrimination/calibration and robustness.

## 4. Databases and Search Window

- Primary sources:
  - PubMed
  - PMC
  - arXiv
- Initial time window:
  - 2018-01-01 to 2026-02-07
- Language:
  - English

## 5. Screening Workflow

1. Identification: run fixed query strings and export candidate records.
2. Deduplication: DOI/PMID/title matching.
3. Title/abstract screening: apply inclusion/exclusion rules.
4. Full-text eligibility screening.
5. Structured extraction into `study_extraction.csv`.
6. Quality scoring into `quality_assessment.csv`.

## 6. Data Items to Extract

- bibliographic metadata (year, venue, id)
- dataset(s)
- cohort size
- note modality and note types
- temporal unit (charttime/chartdate/other)
- alignment window
- task labels
- model family
- fusion stage (early/late/joint)
- key metrics (AUROC/AUPRC/calibration)

## 7. Risk of Bias / Quality

We score each paper on:
- reproducibility transparency
- external validity (dataset diversity)
- alignment protocol clarity
- calibration reporting
- baseline fairness (strong unimodal comparisons)

Each item uses 0/1/2:
- 0 = not addressed
- 1 = partially addressed
- 2 = clearly addressed

