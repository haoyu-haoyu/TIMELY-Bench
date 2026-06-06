# Phase 6.5F Judge Formal Summary

## Coverage

- `claude_opus_4_6` (claude-opus-4-6): `2000/2000` ok, avg_latency_seconds=`8.8539`
- `gpt54` (gpt-5.4): `2000/2000` ok, avg_latency_seconds=`4.6225`
- `gemini31pro` (gemini-3.1-pro-preview): `2000/2000` ok, avg_latency_seconds=`12.9915`

- common_judge_rows: `2000`

## Pairwise Agreement

- `claude_opus_4_6` vs `gemini31pro` on `overall_quality_1to5`: rho=`0.7827` exact_match=`0.2835` mad=`0.9190` n=`2000`
- `claude_opus_4_6` vs `gemini31pro` on `clinical_correctness_1to5`: rho=`0.7595` exact_match=`0.2860` mad=`0.9515` n=`2000`
- `claude_opus_4_6` vs `gpt54` on `overall_quality_1to5`: rho=`0.8138` exact_match=`0.5275` mad=`0.5215` n=`2000`
- `claude_opus_4_6` vs `gpt54` on `clinical_correctness_1to5`: rho=`0.8076` exact_match=`0.5115` mad=`0.5460` n=`2000`
- `gemini31pro` vs `gpt54` on `overall_quality_1to5`: rho=`0.7392` exact_match=`0.4100` mad=`0.8315` n=`2000`
- `gemini31pro` vs `gpt54` on `clinical_correctness_1to5`: rho=`0.7188` exact_match=`0.4115` mad=`0.8655` n=`2000`

## Notes

- `GPT-5.4` is both a contestant and a cross-check judge; this overlap remains documented.
- `Gemini 3.1 Pro` required a repair chain: compact JSON -> pipe6 -> micro pipe6.
- Formal CSV outputs include long judge rows, provider summary, condition summary, and pairwise agreement.
