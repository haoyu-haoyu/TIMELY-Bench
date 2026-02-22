# CANONICAL_ONLY

Status: ACTIVE
Last updated: 2026-02-20

This file defines the only artefacts that should be cited for external reporting, review, or grading.

## 1) Canonical Release Entry

- Release root: `final_release/`
- Manifest: `final_release/manifest.json`
- Provenance: `final_release/PROVENANCE.json`
- Checksums: `final_release/CHECKSUMS.sha256`

## 2) Canonical QA Evidence

- JSON: `final_release/evidence/final_qa_32045137.json`
- Markdown: `final_release/evidence/final_qa_32045137.md`

Any older `final_qa_*` IDs are historical snapshots and must not be used as current release evidence.

## 3) Canonical CRES Run + Multi-Model Baselines

- Canonical top-level run: `final_release/cres/run_manifest.json`
  - Expected run id: `cres_gemini3_full_32008668`
- Retained full model runs:
  - `final_release/cres/model_runs/cres_deepseek_full_20260218_063924`
  - `final_release/cres/model_runs/cres_gpt51_full_32004532`
  - `final_release/cres/model_runs/cres_gemini3_full_32008668`

Known deferred (by decision):
- Canonical Gemini run keeps resume metadata (`resume=true`, `prompt_shas` has 2 hashes).
- Single-prompt non-resume rerun is intentionally deferred.

## 4) Scope Policy

- Use `final_release/` for any claim in paper/poster/report.
- `results/` and `results/audit/` may contain historical, diagnostic, or environment-specific outputs.
- Historical snapshots are archived under `documentation/archive_legacy/`.

## 5) Quick Integrity Checks

```bash
# Release integrity
cd final_release && sha256sum -c CHECKSUMS.sha256

# No plaintext keys in release-facing areas
rg -n "sk-[A-Za-z0-9_-]{10,}" code scripts final_release README*.md docs || true

# Canonical QA and run pointers
ls final_release/evidence/final_qa_32045137.*
cat final_release/cres/run_manifest.json
```
