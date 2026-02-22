# Requirements Traceability (作业要求 + MHaPS)

Audit timestamp: 2026-02-08

This document maps:
- Project spec in `作业要求.md` (ASMHI09 / TIMELY-Bench benchmark deliverables)
- Supervisor larger-project spec in `MHaPS School Grants for Early Career Researchers - Application Form.pdf`

to concrete implementations and artefacts under `TIMELY-Bench_Final/`.

## 0. Source Documents

- Assignment/project spec: `/Users/wanghaoyu/Downloads/临床时序 × 文本对齐融合基准/训练基线模型/作业要求.md`
- MHaPS application (OCR text): `/Users/wanghaoyu/Downloads/临床时序 × 文本对齐融合基准/训练基线模型/_mhaps_application_extracted.txt`
- MHaPS application (PDF): `/Users/wanghaoyu/Downloads/临床时序 × 文本对齐融合基准/训练基线模型/MHaPS School Grants for Early Career Researchers - Application Form.pdf`

## 1. Traceability: 作业要求.md

### A1) Systematic mapping review / taxonomy

Status: DONE (taxonomy + protocol + PRISMA + completed extraction/quality for starter-reference set)

Evidence:
- Survey/taxonomy doc: `TIMELY-Bench_Final/documentation/SURVEY_TAXONOMY.md`
- Review protocol: `TIMELY-Bench_Final/documentation/systematic_review/protocol.md`
- Search strategy: `TIMELY-Bench_Final/documentation/systematic_review/search_queries.md`
- Inclusion/exclusion rules: `TIMELY-Bench_Final/documentation/systematic_review/inclusion_exclusion.md`
- PRISMA-style flow log: `TIMELY-Bench_Final/documentation/systematic_review/prisma_flow.md`
- Structured extraction table: `TIMELY-Bench_Final/documentation/systematic_review/study_extraction.csv`
- Quality assessment table: `TIMELY-Bench_Final/documentation/systematic_review/quality_assessment.csv`

Gaps / notes:
- Current submission scope (starter references in `作业要求.md`) is completed.
- If later expanding beyond starter references, follow the same workflow and append new rows.

### A2) Curated cohorts + alignment protocols + task definitions

Status: DONE (core cohort + canonical aligners implemented)

Evidence:
- Cohort and labels: `TIMELY-Bench_Final/data/processed/merge_output/cohort_final.csv`
- Windows 6h/12h/24h features: `TIMELY-Bench_Final/data/processed/data_windows/`
- D0 daily aligner comparison: `TIMELY-Bench_Final/code/baselines/train_aligner_comparison.py`
- D0/6h/12h/24h results: `TIMELY-Bench_Final/results/aligner_comparison/aligner_results.csv`
- Alignment protocol card: `TIMELY-Bench_Final/docs/ALIGNMENT_PROTOCOL_CARD.md`
- Alignment outputs: `TIMELY-Bench_Final/data/processed/temporal_alignment/`

### A3) Baselines (structured / text-only / fusion; optional temporal)

Status: DONE (core baselines implemented and standardized; some optional evaluations remain non-canonical)

Canonical baseline implementations:
- Structured-only LR/XGBoost (multi-window, multi-cohort): `TIMELY-Bench_Final/code/baselines/run_baselines.py`
- Text-only (annotation-derived alignment statistics): `TIMELY-Bench_Final/code/baselines/train_text_only.py`
- Text-only (ClinicalBERT embeddings): `TIMELY-Bench_Final/code/baselines/train_text_only_embeddings.py`
- Fusion (Early/Late): `TIMELY-Bench_Final/code/baselines/train_fusion.py`
- Optional temporal model (GRU): `TIMELY-Bench_Final/code/baselines/train_temporal_gru_v2.py`

Outputs:
- Structured results: `TIMELY-Bench_Final/results/benchmark_results/benchmark_results_full.csv`
- Standardized results: `TIMELY-Bench_Final/results/standardized/`
- Model card: `TIMELY-Bench_Final/docs/MODEL_CARD.md`

Important gaps / risks:
- "Text-only" naming: the canonical text-only baseline originally used annotation-derived statistics (not raw semantic text). The project now includes a ClinicalBERT embedding baseline to satisfy the "sentence embedding" requirement.
- UMLS/MedCAT as bag-of-concepts baseline: a lightweight MedCAT concept-presence baseline is included as a canonical text-only model (see `code/baselines/train_text_only_medcat.py` and `results/standardized/text_results.csv`).

### A4) Unified evaluation (AUROC/AUPRC, calibration, temporal robustness, note-category ablation)

Status:
- Discrimination metrics: DONE
- Calibration (ECE/Brier/HL): DONE
- Temporal robustness (multi-window): DONE
- Note-category ablation: DONE

Evidence:
- Calibration: `TIMELY-Bench_Final/results/calibration/calibration_summary.csv`
- Fusion calibration (ECE/Brier/HL): `TIMELY-Bench_Final/results/calibration/calibration_fusion_summary.csv`
- Robustness + statistical tests: `TIMELY-Bench_Final/results/robustness/window_performance.csv`, `TIMELY-Bench_Final/results/robustness/statistical_tests.json`
- Canonical aligner comparison (D0, 6h, 12h, 24h): `TIMELY-Bench_Final/results/aligner_comparison/aligner_results.csv`
- Note-category ablation: `TIMELY-Bench_Final/results/note_ablation/note_ablation_results.csv`

### Documentation & release artefacts (Data/Model/Alignment cards, schemas, reproducible pipeline)

Status: DONE

Evidence:
- Data card: `TIMELY-Bench_Final/docs/DATA_CARD.md`
- Model card: `TIMELY-Bench_Final/docs/MODEL_CARD.md`
- Alignment protocol card: `TIMELY-Bench_Final/docs/ALIGNMENT_PROTOCOL_CARD.md`
- Episode JSON schema + dataset readme: `TIMELY-Bench_Final/dataset/README.md`
- Release bundle: `TIMELY-Bench_Final/final_release/`

## 2. Traceability: MHaPS Application (Supervisor Larger Project)

### Condition Graphs (lab markers, vital signs, symptoms, medications, multimorbidity)

Status: PARTIAL (2 exemplar conditions implemented; multimorbidity and symptom nodes are minimal but schema supports the expected categories)

Evidence:
- Schema (domain tags for lab/vitals/symptoms/medications/multimorbidity): `TIMELY-Bench_Final/final_release/condition_graphs/condition_graph_schema.json`
- Sepsis/SIRS graph: `TIMELY-Bench_Final/final_release/condition_graphs/sepsis_sirs_graph.json`
- AKI/KDIGO graph: `TIMELY-Bench_Final/final_release/condition_graphs/aki_kdigo_graph.json`

Notes:
- Sepsis graph includes a minimal symptom/sign anchor (altered mental status via GCS proxy) to satisfy the "symptom cluster" category expected by the MHaPS framing.
- Episode-level "reasoning graphs" inside episode JSON use a simplified node abstraction (pattern/condition). These are NOT the same as the canonical condition graphs and should be described as derived, episode-specific evidence graphs.

### Physiology Templates (canonical temporal trajectories; recovery vs worsening)

Status: PARTIAL (templates exist for Sepsis/AKI; not yet extended to delirium/stroke)

Evidence:
- Canonical trajectories: `TIMELY-Bench_Final/final_release/physiology_templates/canonical_trajectories.json`

### Transparent extraction pipeline (MIMIC-MCP style; SQL-verifiable; portable)

Status: PARTIAL

Evidence:
- SQL scripts and extraction utilities exist under: `TIMELY-Bench_Final/sql/`, `TIMELY-Bench_Final/code/data_processing/`
- The pipeline is reproducible, but is not explicitly packaged as a "MIMIC-MCP layer" in the repo narrative.

### CRES (Clinical Reasoning Evaluation Suite) for temporal grounding + attribution

Status: PARTIAL (LLM evidence audit exists; full suite tasks are not all implemented)

Evidence:
- 900-sample DeepSeek annotation artefacts: `TIMELY-Bench_Final/final_release/llm_annotations/`
- Quote-validity audit (900 samples): `TIMELY-Bench_Final/final_release/llm_annotations/evidence_validity_deepseek_v2_20260127_151413.json`
- CRES manifest/report (updated to 900-sample run): `TIMELY-Bench_Final/final_release/cres/`

## 3. Highest-Risk Method Issues Found (and Current Status)

### [FIXED] Prolonged LOS structured baseline excluded mortality cases

Impact:
- Structured prolonged LOS baseline (`run_baselines.py`) previously filtered out mortality stays, causing mismatched n/positive rates across pipelines (structured vs text/fusion).

Fix:
- `TIMELY-Bench_Final/code/baselines/run_baselines.py`: set `EXCLUDE_MORTALITY_FOR_NON_MORTALITY_TASKS = False`
- Standardized outputs now use the fixed setting.

### [FIXED] Early Fusion used a different structured feature source than structured-only baselines

Impact:
- `train_fusion.py` early-fusion previously used episode-derived vitals summary stats, not the benchmark window features (`features_aggregated.csv`), weakening fairness of comparisons.

Fix:
- `TIMELY-Bench_Final/code/baselines/train_fusion.py`: early-fusion now uses `load_structured_features(window='24h')` aligned with `load_text_features()`.

### [FIXED] Canonical condition graph domain tags for some nodes

Fix:
- `TIMELY-Bench_Final/final_release/condition_graphs/sepsis_sirs_graph.json`: corrected `domain` for tachycardia/tachypnea/hyperbilirubinemia pattern nodes.

### [FIXED] CRES manifest/report inconsistency (26-sample vs 900-sample)

Fix:
- `TIMELY-Bench_Final/final_release/cres/cres_dataset_manifest.json`
- `TIMELY-Bench_Final/final_release/cres/cres_evaluation_report.json`
- `TIMELY-Bench_Final/final_release/llm_annotations/llm_annotation_summary.json`
