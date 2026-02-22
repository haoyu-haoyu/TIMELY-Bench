# LLM Annotation Inputs and Outputs

This note clarifies which annotation files are canonical versus debug-only.

## Canonical input (release-grade)

- `llm_annotation_set.csv` (900 rows)
- Canonical declaration:
  - `ANNOTATION_INPUT_CANONICAL.json`

This file is the only dataset that should be treated as the official annotation input for release reporting.

## Debug-only sampling files

- `data/processed/temporal_alignment/llm_annotation_debug_samples.csv`

This file is for quick inspection and prompt debugging only.  
It must not be used as canonical evidence for coverage, benchmarking, or release claims.

## Metadata and outputs

- `ANNOTATION_METADATA*.json`: run metadata
- `annotations_*.jsonl`: annotation outputs
- `llm_annotation_summary.json` / `summary_strata*.json`: summary diagnostics

## Operational rule

If canonical and debug files coexist, always prioritize:

1. `llm_annotation_set.csv`
2. `ANNOTATION_INPUT_CANONICAL.json`

And treat debug sample files as non-authoritative.
