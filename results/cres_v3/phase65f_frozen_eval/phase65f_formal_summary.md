# Phase 6.5F Formal Summary

## Frozen Set

- comparative providers: `gpt54`, `gemini31pro`, `deepseek_chat`, `qwen35`, `gemma4_26b`, `aloe70b`, `aloe7b`, `meditron3_8b`, `medgemma15_4b_it`
- excluded supplementary provider: `openbiollm70b`
- variant: `full_multimodal`

## Canonicalization

- `gpt54`: rows=53070, ok=53070, parse_success=53070
- `gemini31pro`: rows=53070, ok=53070, parse_success=53070
- `deepseek_chat`: rows=53070, ok=53070, parse_success=53070
- `qwen35`: rows=53070, ok=53070, parse_success=53070
- `gemma4_26b`: rows=53070, ok=53070, parse_success=53070
- `aloe70b`: rows=53070, ok=53070, parse_success=53070
- `aloe7b`: rows=53070, ok=53070, parse_success=53070
- `meditron3_8b`: rows=53070, ok=53070, parse_success=53070
- `medgemma15_4b_it`: rows=53070, ok=53070, parse_success=53070

## Auto-Scoring

- scored_prompt_rows: `166019`
- supported_task_dimensions: `20`
- parity_with_tier1a: `True`

## Judge Packet

- prompt_instances: `500`
- judged_response_rows: `2000`
- contestant_roster_mode: `manual_fixed_vendor_diversity_and_parameter_range_coverage`
- execution_status: `manifest_ready_judge_calls_not_executed`

## Judge Contestants

- `gpt54` [tier1a] (overall_macro_primary_score=0.625744, pair_win_count=9)
- `deepseek_chat` [tier1b] (overall_macro_primary_score=0.634618, pair_win_count=10)
- `aloe70b` [tier2] (overall_macro_primary_score=0.519257, pair_win_count=11)
- `medgemma15_4b_it` [tier2] (overall_macro_primary_score=0.534534, pair_win_count=8)

## Notes

- `openbiollm70b` remains excluded from formal comparative tables.
- `human-LLM agreement` is deferred and not executed in this phase.
- Manual judge contestant roster is fixed to `gpt54`, `deepseek_chat`, `aloe70b`, `medgemma15_4b_it` for vendor diversity and parameter range coverage.
- `GPT-5.4` appears both as a contestant and as a cross-check judge; this overlap is documented, while `Claude Opus 4.6` remains the primary judge.
- Judge manifests are built, but judge API calls are not executed by this script.
