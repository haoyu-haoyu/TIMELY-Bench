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

## Public allowlist

Subject to privacy and secret review, the public release may contain:

- Source code, SQL, schemas, task definitions, and unfilled prompt templates.
- Portable configuration examples that contain no credentials or internal
  infrastructure identifiers.
- Aggregate metrics, manuscript tables, and sufficiently coarse summary data.
- Fully synthetic fixtures created from rules rather than real records.
- Release metadata, whole-file checksums, tests, and software provenance.

## Controlled-by-default gray areas

The following remain controlled unless PhysioNet or the responsible governance
process explicitly approves release:

- Model responses generated from patient-context prompts, even after direct IDs
  are removed.
- Randomized prompt IDs or row-level predictions that can link records across
  artifacts.
- Small-cell subgroup summaries or examples derived from rare trajectories.
- Model weights, embeddings, calibrators, or other artifacts fitted on MIMIC-IV.
- Logs that may contain clinical text, prompts, credentials, endpoints, or
  absolute institutional paths.

Removing `subject_id`, `hadm_id`, or `stay_id` alone is not sufficient to make a
patient-derived artifact public. Release decisions must consider the content and
linkability of the entire artifact.

## Release gate

Every public release must pass `make public-checks` and the human review in
`docs/PUBLIC_RELEASE_CHECKLIST.md`. Review must cover the complete Git history,
tags, releases, and workflow artifacts—not only the current working tree.
