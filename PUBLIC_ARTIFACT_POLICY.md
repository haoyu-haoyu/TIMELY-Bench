# Public Artifact Policy

This repository is a public, reproducibility-oriented export of TIMELY-Bench.
It intentionally contains code, schemas, paper assets, aggregate metric tables,
release summaries, and provenance metadata.

The following artifact classes are intentionally excluded from GitHub:

- Raw MIMIC-IV tables and any derived patient-level cohort files.
- Prompt JSONL files and canonical model response JSONL files.
- Per-instance prompt scoring tables and judge long-form response rationales.
- CSV prediction dumps containing `stay_id`, `subject_id`, `hadm_id`, or
  `note_id` columns.
- Note-level annotation prompts or annotation outputs containing note excerpts.
- Local logs, Slurm outputs, temporary archives, model weights, API keys, and
  environment files.

These artifacts should remain on controlled storage or be released only through
an approved credentialed data-access channel. Public GitHub releases should use
aggregate metrics, summary JSON/CSV files, manuscript tables, and documented
reconstruction scripts instead of patient-level payloads.

