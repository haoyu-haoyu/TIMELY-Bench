# MASTER DELIVERY AUDIT REPORT (Canonical)

**Generated**: 2026-02-20T18:18:36+00:00
**Canonical Root**: `${PROJECT_ROOT}/final_release`
**Overall Verdict**: **PASS**

This report supersedes the earlier 2026-02-01 delivery snapshot.

## 1) Canonical Quality Gate

- QA JSON: `evidence/final_qa_32045137.json`
- QA Markdown: `evidence/final_qa_32045137.md`
- Verdict: PASS

## 2) Release Integrity

- Manifest: `manifest.json`
- Provenance: `PROVENANCE.json`
- Checksums: `CHECKSUMS.sha256`
- Integrity check status: PASS

## 3) Core Data/Method Artifacts

- Condition graphs: Sepsis/SIRS, AKI/KDIGO, Delirium/ICU, Stroke/Neuro
- Physiology templates: canonical trajectories (including delirium and stroke)
- State-space artifacts: schema + episode trajectories + transition summary
- Evaluation artifacts: calibration, note ablation, aligner comparison, robustness summary

## 4) CRES Delivery Status

Canonical run pointer:
- `cres/run_manifest.json` -> `cres_gemini3_full_32008668`

Retained complete model runs:
- `cres/model_runs/cres_deepseek_full_20260218_063924`
- `cres/model_runs/cres_gpt51_full_32004532`
- `cres/model_runs/cres_gemini3_full_32008668`

Known deferred tightening item:
- Canonical Gemini run currently retains resume metadata (`prompt_shas` has two entries).
- Single-prompt non-resume rerun is deferred by project decision.

## 5) Audit Boundaries

- This canonical report is the only source for release-level claims.
- Historical/legacy audit snapshots are archived under `documentation/archive_legacy/` and `results/audit/archive_legacy/`.
