# TIMELY-Bench Final Audit Summary

**Generated**: 2026-02-20T02:30:14Z
**Closure Fix Run ID**: closure_20260220_release_sync_v2
**Overall Verdict**: PASS

## Canonical Data Anchors

| Label | Path | Exists | Size |
|-------|------|--------|------|
| temporal_textual_alignment | data/processed/temporal_textual_alignment.csv | YES (symlink) | 1.08 GB |
| disease_timelines_full | data/processed/disease_timelines_full.json | YES (symlink) | 27.7 MB |

## CRES Canonical Run (Gemini)

| Field | Value |
|-------|-------|
| Run ID | cres_gemini3_full_32008668 |
| Model | gemini-3-flash-preview |
| Backend | openai-compatible |
| Records | 3600 (900 per task x 4 tasks) |
| Predictions Path | `${PROJECT_ROOT}/final_release/cres/model_runs/cres_gemini3_full_32008668/predictions.jsonl` |
| Evidence Validity Rate | 1.0000 |

## CRES Multi-Model Baselines

- `cres_deepseek_full_20260218_063924` (deepseek-chat, 3600 rows)
- `cres_gpt51_full_32004532` (gpt-5.1, 3600 rows)
- `cres_gemini3_full_32008668` (gemini-3-flash-preview, 3600 rows)

## Checks

- QA Gate: PASS
- Subject Leakage: PASS
- Opt-in Isolation: PASS
