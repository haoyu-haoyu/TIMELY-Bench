# Requirements Traceability (作业要求 + MHaPS)

Audit timestamp: 2026-02-20

This document maps:
- Project spec in `作业要求.md` (ASMHI09 / TIMELY-Bench benchmark deliverables)
- Supervisor larger-project spec in `MHaPS School Grants for Early Career Researchers - Application Form.pdf`

to concrete implementations and artefacts under `TIMELY-Bench_Final/`.

## 0. Source Documents

- Assignment/project spec: `../作业要求.md`
- MHaPS application (OCR text): `../_mhaps_application_extracted.txt`
- MHaPS application (PDF): `../MHaPS School Grants for Early Career Researchers - Application Form.pdf`

## 1. Traceability: 作业要求.md

### A1) Systematic mapping review / taxonomy

Status: DONE (course minimum scope complete)

Evidence:
- Survey/taxonomy doc: `TIMELY-Bench_Final/documentation/SURVEY_TAXONOMY.md`
- Review protocol: `TIMELY-Bench_Final/documentation/systematic_review/protocol.md`
- Search strategy: `TIMELY-Bench_Final/documentation/systematic_review/search_queries.md`
- Inclusion/exclusion rules: `TIMELY-Bench_Final/documentation/systematic_review/inclusion_exclusion.md`
- PRISMA-style flow log: `TIMELY-Bench_Final/documentation/systematic_review/prisma_flow.md`
- Structured extraction table: `TIMELY-Bench_Final/documentation/systematic_review/study_extraction.csv`
- Quality assessment table: `TIMELY-Bench_Final/documentation/systematic_review/quality_assessment.csv`

Scope note:
- Current extraction table is a starter mapping set (small-n). It satisfies current assignment baseline, but can be expanded later if required for publication-grade breadth.

### A2) Curated cohorts + alignment protocols + task definitions

Status: DONE

Evidence:
- Cohort and labels: `TIMELY-Bench_Final/data/processed/merge_output/cohort_final.csv`
- Windows 6h/12h/24h/D0 features: `TIMELY-Bench_Final/data/processed/data_windows/`
- D0 daily aligner comparison: `TIMELY-Bench_Final/code/baselines/train_aligner_comparison.py`
- D0/6h/12h/24h results: `TIMELY-Bench_Final/results/aligner_comparison/aligner_results.csv`
- Alignment protocol card: `TIMELY-Bench_Final/docs/ALIGNMENT_PROTOCOL_CARD.md`
- Alignment outputs: `TIMELY-Bench_Final/data/processed/temporal_alignment/`

### A3) Baselines (structured / text-only / fusion; optional temporal)

Status: DONE

Canonical implementations:
- Structured-only LR/XGBoost: `TIMELY-Bench_Final/code/baselines/run_baselines.py`
- Text-only (annotation-derived): `TIMELY-Bench_Final/code/baselines/train_text_only.py`
- Text-only (ClinicalBERT embeddings): `TIMELY-Bench_Final/code/baselines/train_text_only_embeddings.py`
- Text-only (MedCAT concept features): `TIMELY-Bench_Final/code/baselines/train_text_only_medcat.py`
- Fusion (Early/Late, including stacking): `TIMELY-Bench_Final/code/baselines/train_fusion.py`
- Temporal baseline (GRU v2): `TIMELY-Bench_Final/code/baselines/train_temporal_gru_v2.py`

Outputs:
- Standardized results: `TIMELY-Bench_Final/results/standardized/`
- Unified benchmark table: `TIMELY-Bench_Final/results/benchmark_results/benchmark_results_full.csv`
- Model card: `TIMELY-Bench_Final/docs/MODEL_CARD.md`

### A4) Unified evaluation (AUROC/AUPRC, calibration, temporal robustness, note-category ablation)

Status: DONE

Evidence:
- Calibration summary: `TIMELY-Bench_Final/results/calibration/calibration_summary.csv`
- Fusion calibration: `TIMELY-Bench_Final/results/calibration/calibration_fusion_summary.csv`
- Robustness + stats: `TIMELY-Bench_Final/results/robustness/window_performance.csv`, `TIMELY-Bench_Final/results/robustness/statistical_tests.json`
- Aligner protocol comparison: `TIMELY-Bench_Final/results/aligner_comparison/aligner_results.csv`
- Note-category ablation: `TIMELY-Bench_Final/results/note_ablation/note_ablation_results.csv`

### Documentation & release artefacts

Status: DONE

Evidence:
- Data card: `TIMELY-Bench_Final/docs/DATA_CARD.md`
- Model card: `TIMELY-Bench_Final/docs/MODEL_CARD.md`
- Alignment protocol card: `TIMELY-Bench_Final/docs/ALIGNMENT_PROTOCOL_CARD.md`
- Episode schema + example: `TIMELY-Bench_Final/documentation/episode_schema.json`, `TIMELY-Bench_Final/documentation/example_episode.json`
- Release bundle: `TIMELY-Bench_Final/final_release/`

## 2. Traceability: MHaPS Application (Supervisor Larger Project)

### Condition Graphs (lab markers, vital signs, symptoms, medications, multimorbidity)

Status: DONE (implemented with explicit domain tags)

Evidence:
- Schema: `TIMELY-Bench_Final/final_release/condition_graphs/condition_graph_schema.json`
- Sepsis/SIRS: `TIMELY-Bench_Final/final_release/condition_graphs/sepsis_sirs_graph.json`
- AKI/KDIGO: `TIMELY-Bench_Final/final_release/condition_graphs/aki_kdigo_graph.json`
- Delirium/ICU: `TIMELY-Bench_Final/final_release/condition_graphs/delirium_icu_graph.json`
- Stroke/Neuro: `TIMELY-Bench_Final/final_release/condition_graphs/stroke_neuro_graph.json`
- Feature mapping audit: `TIMELY-Bench_Final/final_release/condition_graphs/mapping_report.json`

Boundary note:
- `ckd` is currently treated as `external_static` (comorbidity context) rather than a dynamic time-series channel.

### Physiology Templates (canonical temporal trajectories; recovery vs worsening)

Status: DONE

Evidence:
- Canonical trajectories: `TIMELY-Bench_Final/final_release/physiology_templates/canonical_trajectories.json`

Included trajectories cover sepsis, AKI, ARDS/HF context, plus delirium and stroke trajectories required by the larger-project framing.

### Transparent extraction pipeline (MIMIC-MCP style; SQL-verifiable; portable)

Status: PARTIAL

Evidence:
- SQL scripts and extraction utilities: `TIMELY-Bench_Final/sql/`, `TIMELY-Bench_Final/code/data_processing/`
- Provenance + release manifest: `TIMELY-Bench_Final/final_release/PROVENANCE.json`, `TIMELY-Bench_Final/final_release/manifest.json`

Boundary note:
- Pipeline is reproducible and SQL-verifiable, but not packaged under a dedicated `mimic_mcp/` namespace.

### CRES (Clinical Reasoning Evaluation Suite) for temporal grounding + attribution

Status: DONE (task suite + multi-model baselines delivered)

Evidence:
- CRES task files and metadata: `TIMELY-Bench_Final/final_release/cres/`
- Model runs retained in release:
  - `TIMELY-Bench_Final/final_release/cres/model_runs/cres_deepseek_full_20260218_063924`
  - `TIMELY-Bench_Final/final_release/cres/model_runs/cres_gpt51_full_32004532`
  - `TIMELY-Bench_Final/final_release/cres/model_runs/cres_gemini3_full_32008668`

Known deferred item (explicitly postponed):
- Canonical Gemini run currently has resume history (`prompt_shas` contains two hashes). A strict single-prompt rerun is deferred by current project decision.

## 3. Current High-Risk Boundaries (Not Blocking Release)

1. `ckd` mapping is intentionally `external_static` in condition-graph mapping.
2. CRES canonical single-prompt strictness is deferred (resume metadata retained).
3. Systematic review breadth remains starter-scope unless extended for publication.
