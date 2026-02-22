# TIMELY-Bench Final Audit Summary

**Generated**: 2026-02-01T21:04:24.499268+00:00
**Closure Fix Run ID**: closure_20260201_191928
**Overall Verdict**: PASS

## Canonical Data Anchors

| Label | Path | Exists | Size |
|-------|------|--------|------|
| temporal_textual_alignment | data/processed/temporal_textual_alignment.csv | YES (symlink) | 1.08 GB |
| disease_timelines_full | data/processed/disease_timelines_full.json | YES (symlink) | 27.7 MB |

## DeepSeek Canonical Run

| Field | Value |
|-------|-------|
| Run ID | 20260127_151413 |
| Model | deepseek-chat |
| Records | 900 |
| Quote Valid Rate | 0.8078 |
| JSONL Lines | 900 |
| Audited Lines | 900 |

## Checks

- QA Gate: PASS
- Subject Leakage: PASS
- Opt-in Isolation: PASS
