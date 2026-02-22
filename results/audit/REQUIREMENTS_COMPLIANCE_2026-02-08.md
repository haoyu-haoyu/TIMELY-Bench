# TIMELY-Bench Compliance Audit (2026-02-08)

This audit compares current repository status against:
- `作业要求.md` (ASMH109 assignment requirements)
- `MHaPS School Grants for Early Career Researchers - Application Form.pdf` (supervisor large-project expectations)

Audit scope:
- Code implementation
- Released artifacts and docs
- Result files currently present in repo

---

## A. Assignment (`作业要求.md`) Compliance

### A1 Systematic review / taxonomy

Status: **PASS (minimum scope), not full-scale review**

Evidence:
- `documentation/systematic_review/protocol.md`
- `documentation/systematic_review/search_queries.md`
- `documentation/systematic_review/inclusion_exclusion.md`
- `documentation/systematic_review/prisma_flow.md`
- `documentation/systematic_review/study_extraction.csv`
- `documentation/systematic_review/quality_assessment.csv`
- `report/taxonomy/alignment_taxonomy.md`
- `report/taxonomy/fusion_taxonomy.md`

Notes:
- The full protocol chain exists (query, PRISMA log, extraction, quality scoring).
- Current screened set is small (`n=5` in PRISMA snapshot), so this is a compact mapping review rather than a broad systematic review.

---

### A2 Cohort/tasks/alignment pipeline

Status: **PASS**

Evidence:
- Cohort and labels: `data/processed/merge_output/cohort_final.csv`
- Canonical subject-level split: `data/splits/predefined_splits.csv`
- Split metadata: `data/splits/split_summary.json`
- Multi-window feature generation (6h/12h/24h/D0): `code/data_processing/create_multi_window_data.py`
- Generated windows:
  - `data/processed/data_windows/window_6h/features_aggregated.csv`
  - `data/processed/data_windows/window_12h/features_aggregated.csv`
  - `data/processed/data_windows/window_24h/features_aggregated.csv`
  - `data/processed/data_windows/window_D0/features_aggregated.csv`
- Alignment source: `data/processed/temporal_alignment/temporal_textual_alignment.csv`

---

### A3 Baselines + unified metrics

Status: **PASS**

Evidence:
- Structured baselines: `code/baselines/run_baselines.py`
- Text-only baselines:
  - Annotation-derived: `code/baselines/train_text_only.py`
  - ClinicalBERT embeddings: `code/baselines/train_text_only_embeddings.py`
  - MedCAT concepts: `code/baselines/train_text_only_medcat.py`
- Fusion baselines: `code/baselines/train_fusion.py`
- Optional temporal model (GRU): `code/baselines/train_temporal_gru_v2.py`
- Unified standardized outputs:
  - `results/standardized/structured_results.csv`
  - `results/standardized/text_results.csv`
  - `results/standardized/fusion_results.csv`
  - `results/standardized/gru_results.csv`
  - `results/standardized/results_summary.csv`

---

### A4 Evaluation (AUROC/AUPRC + calibration + multi-window + ablation)

Status: **PASS**

Evidence:
- AUROC/AUPRC:
  - `results/standardized/results_summary.csv`
- Calibration:
  - `results/calibration/calibration_summary.csv`
  - `results/calibration/calibration_fusion_summary.csv`
  - `results/calibration/calibration_dl_summary.json`
- Multi-window:
  - Structured 6h/12h/24h in `results/benchmark_results/benchmark_results_full.csv`
  - D0 included via canonical aligner comparison in `results/aligner_comparison/aligner_results.csv`
- Ablation by note category:
  - `code/baselines/eval_note_ablation.py`
  - `results/note_ablation/note_ablation_results.csv`

Clarification:
- Current note-category ablation is **alignment-derived note-type feature ablation** (nursing/radiology/lab_comment), not MedCAT concept-category ablation.

---

### A4 Documentation / release artifacts

Status: **PASS**

Evidence:
- Data card: `docs/DATA_CARD.md`
- Model card: `docs/MODEL_CARD.md`
- Alignment protocol card: `docs/ALIGNMENT_PROTOCOL_CARD.md`
- Repro checklist: `docs/REPRODUCIBILITY_CHECKLIST.md`
- Release bundle: `final_release/`

---

## B. Supervisor Project (`MHaPS ... Application Form`) Alignment

### B1 Condition Graphs

Status: **PARTIAL PASS**

What matches:
- Graph schema and artifacts exist:
  - `final_release/condition_graphs/condition_graph_schema.json`
  - `final_release/condition_graphs/sepsis_sirs_graph.json`
  - `final_release/condition_graphs/aki_kdigo_graph.json`
- Domain tags include expected clinical node domains:
  - `lab_marker`, `vital_sign`, `symptom`, `medication`, `multimorbidity`

Gap:
- Large-project summary examples mention conditions such as **AKI, delirium, stroke**.
- Current release contains **AKI + Sepsis** graphs only (no delirium/stroke graphs).

---

### B2 Physiology Templates / canonical trajectories

Status: **PASS**

Evidence:
- `final_release/physiology_templates/canonical_trajectories.json`
- `final_release/physiology_templates/TRAJECTORIES_README.md`

What matches:
- Recovery vs worsening/progression trajectories are explicitly encoded.
- Temporal phase structure is present (multi-phase, hour ranges, expected direction).

---

### B3 CRES evaluation suite

Status: **PARTIAL PASS**

Evidence:
- `final_release/cres/temporal_grounding.jsonl`
- `final_release/cres/trend_threshold.jsonl`
- `final_release/cres/cres_eval_summary.json`

What matches:
- Temporal grounding and trend tasks are implemented.
- Evaluation summaries and manifests exist.

Gap:
- Current CRES task coverage is narrower than the full large-project vision (e.g., broad multimorbidity diagnostic consistency and richer attribution suites are not yet fully expanded).

---

## C. Current Integrity / Rigor Issues

### C1 Fixed in this round

1) Canonical pipeline mismatch in Makefile fixed:
- `scripts/Makefile` now uses:
  - `data: splits windows patterns episodes`
  - `baselines -> code/baselines/run_baselines.py` (canonical multi-window baseline)

2) Documentation wording mismatch fixed:
- `docs/MODEL_CARD.md` note-ablation label corrected to alignment-derived note-type ablation.

---

### C2 Remaining issues to close

1) QA job still running on CREATE:
- Job id: `31711179`
- Script: `scripts/hpc_run_final_qa_noskip.sh`
- Until completion, final PASS/FAIL stamp is pending.

2) Absolute-path portability in release metadata:
- Several `final_release/*` metadata files still embed machine-specific absolute paths.
- Not fatal for model training, but not ideal for external release portability.

3) A1 breadth:
- Systematic review chain is present and reproducible, but corpus size is small (`n=5` snapshot).
- If strict “systematic review” breadth is required by marking rubric, expand search batches and PRISMA counts.

---

## D. Practical conclusion

- Against `作业要求.md`: **overall compliant**, with required modules present and executable.
- Against supervisor large-project vision: **substantially aligned but not fully complete** (mainly condition coverage and CRES breadth).
- Current highest-priority blocker before final sign-off: **wait for CREATE QA job completion** and archive the PASS evidence.
