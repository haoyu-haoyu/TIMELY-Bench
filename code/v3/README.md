# V3 Code Entry Points

This directory contains both canonical TIMELY-Bench v3 pipeline modules and provenance utilities from the frozen evaluation campaign.

## Canonical Public Modules

Use these files as the primary implementation references:

| Stage | Files |
|---|---|
| Source extraction | `extract_cohort_bq.py`, `extract_structured_backbone_bq.py`, `extract_notes_bq.py`, `extract_events_bq.py`, `extract_hourly_features_bq.py` |
| 168-hour state construction | `build_feature_dictionary.py`, `build_hourly_state_grid.py`, `build_state_vectors.py`, `build_time_aware_contexts.py` |
| Condition/task construction | `build_aki_tasks_v3.py`, `build_delirium_tasks_v3.py`, `build_sepsis_tasks_v3.py`, `build_stroke_tasks_v3.py`, `build_phase6_cres_assembly_v3.py`, `build_phase6_cres_release_v3.py` |
| Structured baselines | `run_phase65a_baselines_v3.py` |
| Prompt serialization | `run_phase65b_prompt_build_v3.py` |
| Hosted LLM inference | `run_phase65c_tier1a_full_v3.py`, `run_phase65d_tier1b_v3.py` |
| Local/open-weight LLM inference | `run_phase65e_tier2_v1.py` |
| Frozen scoring and judge packet construction | `run_phase65f_frozen_eval_v1.py`, `run_phase65f_judge_execute_v1.py` |

## Non-Canonical Provenance Utilities

The following filename patterns are retained for auditability but are not recommended as starting points for external reproduction:

- `*_pilot_*`
- `probe_*`
- `repair_*`
- `audit_openbio_retry_*`
- `build_phase65d_qwen35_repair_manifest_*`
- `run_phase65e_medgemma_two_stage_repair_*`
- provider-specific local tail-fix helpers

These scripts document how frozen runs were repaired or audited after provider-specific parse failures. The formal public results should be reproduced from the canonical modules and the frozen summaries under `results/cres_v3/phase65f_frozen_eval/`.

## Data Boundary

Most V3 modules expect controlled patient-level files under `data/processed/v3/`. Those files are excluded from public GitHub. To run the full pipeline, reconstruct them from credentialed MIMIC-IV access or use the approved controlled release channel.
