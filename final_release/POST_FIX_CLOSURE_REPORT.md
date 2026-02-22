# POST_FIX_CLOSURE_REPORT

**Date**: 2026-02-20T02:20:00+00:00
**Run ID**: closure_20260220_release_sync
**Verdict**: PASS

## 1. Canonical Anchor Files

### temporal_textual_alignment.csv
- **Symlink**: data/processed/temporal_textual_alignment.csv -> temporal_alignment/...
- **Real Size**: 1.08 GB
- **Status**: EXISTS (symlink verified)

### disease_timelines_full.json
- **Symlink**: data/processed/disease_timelines_full.json -> disease_timelines/...
- **Real Size**: 27.7 MB
- **Status**: EXISTS (symlink verified)

## 2. CRES Canonical Run (Gemini)

| Field | Value |
|-------|-------|
| Run ID | cres_gemini3_full_32008668 |
| Model | gemini-3-flash-preview |
| Backend | openai-compatible |
| Records | 3600 (900 per task x 4 tasks) |
| Predictions Path | `${PROJECT_ROOT}/final_release/cres/model_runs/cres_gemini3_full_32008668/predictions.jsonl` |
| Evidence Validity Rate | 1.0000 |

### Retained Multi-Model Baselines
- `cres_deepseek_full_20260218_063924` (deepseek-chat, 3600 records)
- `cres_gpt51_full_32004532` (gpt-5.1, 3600 records)
- `cres_gemini3_full_32008668` (gemini-3-flash-preview, 3600 records)

## 3. All Checks

- Canonical anchors exist: YES (symlinks)
- CRES canonical run: gemini-3-flash-preview (3600 records)
- Multi-model baselines retained: YES (3 complete runs)
- Stale references fixed: YES
- QA Gate: PASS
- Subject Leakage: PASS
- Opt-in Isolation: PASS

## 4. Evidence Files

All audit evidence in final_release/evidence/:
- FINAL_AUDIT_SUMMARY.json
- FINAL_AUDIT_SUMMARY.md
- final_qa_32045137.json
- final_qa_32045137.md
- anchor_fingerprint_strong.json
- evidence_validity_deepseek_v2_20260127_151413.json
- reference_scan_full.txt
