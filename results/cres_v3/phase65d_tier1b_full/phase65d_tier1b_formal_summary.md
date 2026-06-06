# Phase 6.5D Tier 1B Formal Summary

## Scope

This summary freezes the formal `Tier 1B` result package for the
`full_multimodal` prompt variant on the released `12K` CRES evaluation sample.
`Tier 1B` covers the three non-Tier-1A providers finalized in this phase:

- `deepseek_chat`
- `qwen35`
- `gemma4_26b`

The final package is based on `53,070` prompt rows spanning `14` tasks. The
row expansion is expected because each evaluation instance fans out across the
enabled task-dimension prompt set.

## Run Completeness

- `deepseek_chat`
  - `rows=53070`
  - `ok=53070`
  - `parse_success=53070`
  - `parse_success_rate=1.000000`
  - `avg_latency_seconds=12.71`
  - `usage_total_tokens=261196668`
  - `confidence={"high": 37994, "low": 437, "medium": 14639}`
- `qwen35`
  - `rows=53070`
  - `ok=53070`
  - `parse_success=53070`
  - `parse_success_rate=1.000000`
  - `avg_latency_seconds=4.84`
  - `usage_total_tokens=285969716`
  - `confidence={"high": 48149, "low": 758, "medium": 4150, "missing": 13}`
- `gemma4_26b`
  - `rows=53070`
  - `ok=53070`
  - `parse_success=53070`
  - `parse_success_rate=1.000000`
  - `avg_latency_seconds=30.04`
  - `usage_total_tokens=277833535`
  - `confidence={"high": 50383, "low": 149, "medium": 2538}`

## Task Coverage

All three providers reached full parse-complete coverage for the same task mix:

- `AKI-T1=11256`
- `AKI-S1=4690`
- `DEL-T1=11250`
- `DEL-S1=3752`
- `SEP-T1=11250`
- `SEP-S1=1876`
- `S-T1=3752`
- `S-T2=469`
- `S-T3=469`
- `S-T4=938`
- `S-R1=938`
- `S-R2=938`
- `S-R3=554`
- `S-R4=938`

Per-task parse success is `1.0` for every provider and every task in the final
merged summaries.

## Provider Pattern Notes

- `deepseek_chat` shows the broadest confidence spread, with a materially larger
  `medium` mass than the other two providers.
- `qwen35` is the fastest provider in this tier by mean latency, but still
  emits `13` rows with missing confidence values.
- `gemma4_26b` is the slowest provider in this tier and is the most
  overconfident by distribution, with `high` confidence on `94.94%` of parsed
  rows.

## Audit Notes

- The abandoned `deepseek-reasoner` path is not part of the final `Tier 1B`
  package. The frozen DeepSeek provider is `deepseek_chat`.
- `gemma4_26b` required a single final tail repair on prompt
  `AKI-T1::37437370::h49::D1::full_multimodal`.
- That repair was completed with a strict JSON-only one-off call and then
  re-merged into the official full summary.
- The repaired Gemma row is now included in the formal merged output, so the
  official provider summary is truly `53070/53070`.
- `gemma4_26b` also preserves a small reasoning audit branch:
  - `reasoning_rows=502`
  - `reasoning_nonempty_rows=502`

## Output Files

- `deepseek_chat_full_summary.json`
- `qwen35_full_summary.json`
- `gemma4_26b_full_summary.json`
- `phase65d_full_summary.json`
