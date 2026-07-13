# Data access and release scope

This GitHub export contains TIMELY-Bench code, manuscript assets, schemas,
aggregate benchmark summaries, and frozen evaluation metric tables.

It intentionally does not include raw MIMIC-IV tables, note extracts, prompt
manifests containing patient-context text, canonical model response JSONL files,
API credentials, local environment files, or virtual environments.
It also excludes per-instance model prediction dumps, per-instance scoring
tables, and judge long-form rationale tables when those files contain patient
identifiers, note identifiers, prompt identifiers, or clinical-text excerpts.

The source clinical data are available through the PhysioNet credentialed data
access program for MIMIC-IV. Derived benchmark artifacts that contain patient
context should be distributed only through the approved release channel described
in the manuscript and data-use documentation.

## Reproducibility layers

The repository therefore separates reproducibility into two layers:

1. **Public software layer:** source code, SQL, schemas, task definitions,
   aggregate metrics, manuscript artifacts, synthetic fixtures, and validation
   tests. This layer can be inspected without MIMIC-IV access.
2. **Credentialed controlled layer:** patient-level derived tables, filled
   prompts, canonical model responses, per-instance scores, judge rationales,
   and any MIMIC-derived fitted artifacts. This layer is not distributed through
   GitHub and must follow the applicable PhysioNet agreement.

The machine-readable boundary is documented in
`release/PUBLIC_ARTIFACT_INVENTORY.csv`. Synthetic examples under `synthetic/`
are fictional and are not sampled, shifted, or paraphrased from MIMIC-IV.

## What credentialed researchers need to supply

Researchers rebuilding V3 must independently obtain authorization for the
applicable MIMIC-IV modules and configure their own institutional extraction
environment. The repository supplies transformation code and data contracts,
but it does not supply credentials, IAM configuration, billing projects, raw
tables, or a copy of the original CREATE execution environment.

See `PUBLIC_ARTIFACT_POLICY.md` for the repository-level inclusion and exclusion
policy used for this public export.
