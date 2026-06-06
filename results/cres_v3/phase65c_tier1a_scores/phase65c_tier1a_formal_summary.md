# Phase 6.5C Tier 1A Formal Summary

## Scope

This summary freezes the formal `Tier 1A` result package for the `full_multimodal`
prompt variant on the released `12K` evaluation sample. It covers the two
commercial providers completed in `Phase 6.5C`:

- `gpt54`
- `gemini31pro`

## Run Completeness

- `gpt54`
  - `rows=53070`
  - `ok=53070`
  - `parse_success=53070`
  - `parse_success_rate=1.000000`
  - `avg_latency_seconds=11.40`
  - `usage_total_tokens=264064840`
  - `confidence={"high": 37086, "low": 678, "medium": 15306}`
- `gemini31pro`
  - `rows=53070`
  - `ok=53070`
  - `parse_success=53070`
  - `parse_success_rate=1.000000`
  - `avg_latency_seconds=15.94`
  - `usage_total_tokens=343643633`
  - `confidence={"high": 52635, "low": 53, "medium": 382}`

## Auto-Scoring Scope

- `total_prompt_pairs=47`
- `auto_scored_prompt_pairs=20`
- `deferred_prompt_pairs=27`
- `scored_prompt_rows=41325`
- `per_task_dimension_rows=45`
- `stratified_rows=317`

`Tier 1A` automatic scoring covers the directly groundable subset only:

- event-time prediction
- binary yes/no or worsening-vs-non-worsening
- categorical extraction/diagnostic labels
- exact numeric extraction

Judge-dependent or evidence-attribution heavy dimensions remain deferred.

## Key Results

### Strengths shared by both providers

- `DEL-T1 D1/D2` and `DEL-S1 D1` are near-ceiling under automatic scoring for both
  providers.
- `SEP-T1 D1` and `SEP-S1 D1` show excellent event-presence detection for both
  providers, with `event_presence_auprc=1.0`.
- `AKI-T1 D2` and `AKI-S1 D2` are strong on threshold/onset-style event-time tasks.

### GPT-5.4 highlights

- `DEL-T1 D2`: `binary_accuracy=1.0000`, `positive_tolerance_1h_rate=1.0000`
- `DEL-T1 D1`: `binary_accuracy=1.0000`, `positive_tolerance_1h_rate=0.9956`
- `SEP-T1 D1`: `binary_accuracy=0.9887`, `median_abs_hour_error=2.0`
- `AKI-T1 D2`: `binary_accuracy=0.9692`
- `S-T1 D3`: `binary_accuracy=0.8064`

### Gemini 3.1 Pro highlights

- `DEL-T1 D2`: `binary_accuracy=1.0000`, `positive_tolerance_1h_rate=0.9984`
- `DEL-T1 D1`: `binary_accuracy=1.0000`, `positive_tolerance_1h_rate=0.9758`
- `SEP-T1 D1`: `binary_accuracy=0.9956`, `median_abs_hour_error=3.0`
- `AKI-T1 D2`: `binary_accuracy=0.9920`
- `S-T1 D3`: `binary_accuracy=0.7610`

### Weak or unstable automatic subsets

- `S-T1 D1` is weak for both providers:
  - `gpt54 binary_accuracy=0.1080`
  - `gemini31pro binary_accuracy=0.0279`
- `S-R1 D4`, `S-R2 D4`, and `S-T2/S-T3 D4` remain difficult categorical
  interpretation tasks under direct automatic scoring.
- `S-R4 D4` is weak for both providers under the current binary yes/no mapping.

## Audit Notes

- `gemini31pro` final completeness includes:
  - `manual_tail4_count=4`
  - `manual_parsefix_count=1`
- Representation-branch comparisons are not computed in `Tier 1A` because this
  run uses only the `full_multimodal` prompt variant.
- `D6` evidence attribution remains deferred to later judge/evidence analyses.
- Some task-dimension combinations are scored with conservative coarse mappings
  where only partial automatic ground truth is available, especially
  `S-T1 D3` worsening vs non-worsening.

## Output Files

- `phase65c_tier1a_per_task_dimension_metrics.csv`
- `phase65c_tier1a_stratified_metrics.csv`
- `phase65c_tier1a_audit.json`
- `phase65c_tier1a_scoring_summary.json`
- `phase65c_tier1a_result_digest.md`
