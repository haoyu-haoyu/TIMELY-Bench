# Local Sync Verification Report

**Generated**: 2026-02-01T21:15:00+00:00
**Sync Method**: rsync from CREATE HPC to Local Mac

## SHA256 Hash Comparison

| File | Local Hash | CREATE Hash | Match |
|------|------------|-------------|-------|
| FINAL_AUDIT_SUMMARY.json | a47063a0...51ef5a | a47063a0...51ef5a | ✅ |
| POST_FIX_CLOSURE_REPORT.md | fd9515f0...59a36 | fd9515f0...59a36 | ✅ |
| annotations_deepseek_20260127_151413_part0001.jsonl | 2c06f456...acbb1 | 2c06f456...acbb1 | ✅ |

## Synced Directories

| Directory | Status |
|-----------|--------|
| final_release/ | ✅ Synced |
| results/audit/ | ✅ Synced |
| results/standardized/ | ✅ Synced |
| results/llm_annotations/ | ✅ Synced |
| legacy_archive/ | ✅ Synced |
| docs/ | ✅ Synced |

## Key Files Present on Local

- ✅ results/audit/FINAL_AUDIT_SUMMARY.json (verdict=PASS)
- ✅ results/audit/FINAL_AUDIT_SUMMARY.md
- ✅ final_release/POST_FIX_CLOSURE_REPORT.md
- ✅ final_release/evidence/FINAL_AUDIT_SUMMARY.json
- ✅ final_release/llm_annotations/annotations_deepseek_20260127_151413_part0001.jsonl (900 lines)

## Excluded from Sync (Large Files)

- episodes/episodes_enhanced/ (74829 files, too large)
- data/processed/temporal_alignment/ (1.1 GB file)
- data/processed/disease_timelines/ (28 MB file)
- models/medcat/ (model files)

## Verdict

**LOCAL ↔ CREATE CONSISTENCY: PASS**

All critical audit files are hash-identical between Local and CREATE.
