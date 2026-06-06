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

See `PUBLIC_ARTIFACT_POLICY.md` for the repository-level inclusion and exclusion
policy used for this public export.
