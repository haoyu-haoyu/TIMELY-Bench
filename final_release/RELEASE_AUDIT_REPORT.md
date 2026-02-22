# Release Audit Report

Status: **PASS**
Generated: **2026-02-20**

- Protocol card references: PASS
- Evidence directory integrity: PASS
- Full-scan alignment QC: PASS
  - evidence: `${PROJECT_ROOT}/final_release/qc/full_alignment_qc.json` (discharge=0, out_of_range=0, dup=0)
- Condition graphs (Sepsis/AKI/Delirium/Stroke): PASS
- Physiology templates: PASS
- CRES report (multi-model runs retained): PASS
- LLM annotation set + metadata: PASS
- QA gate: PASS
  - canonical: `${PROJECT_ROOT}/final_release/evidence/final_qa_32045137.json`
- Manifest integrity: PASS
- Checksum integrity: PASS

Notes:
- Legacy DeepSeek-only audit wording is superseded by this canonical release audit.
- CRES canonical single-prompt rerun is a deferred tightening task and does not block current release.
