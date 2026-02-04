# POST_FIX_CLOSURE_REPORT

**Date**: 2026-02-01T21:04:24.499268+00:00
**Run ID**: closure_20260201_191928
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

## 2. DeepSeek Canonical Run

| Field | Value |
|-------|-------|
| Run ID | 20260127_151413 |
| Model | deepseek-chat |
| Records | 900 |
| Quote Valid Rate | 0.8078 |
| JSONL Lines | 900 |
| Audited Lines | 900 |

### Archived Non-Canonical Runs
- annotations_deepseek_20260127_151413_part0001.jsonl (900 records)
- annotations_deepseek_20260127_151413_part0001.jsonl (intermediate run)

## 3. All Checks

- Canonical anchors exist: YES (symlinks)
- DeepSeek canonical run: 900 records
- Non-canonical runs archived: YES
- Stale references fixed: YES
- QA Gate: PASS
- Subject Leakage: PASS
- Opt-in Isolation: PASS

## 4. Evidence Files

All audit evidence in final_release/evidence/:
- FINAL_AUDIT_SUMMARY.json
- FINAL_AUDIT_SUMMARY.md
- anchor_fingerprint_strong.json
- evidence_validity_deepseek_v2_20260127_151413.json
- llm_reference_scan.txt
