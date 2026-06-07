# Task A/B Progression v2 - Phase B Report

Generated: 2026-03-09 14:45:47 GMT
Project: ${PROJECT_ROOT}
Branch: codex/note-centered-alignment

## 1) Experiment Execution Summary

- Submission file: results/note_centered/progression_tasks/submitted_jobs_20260309_140906.txt
- Expected experiments: 22
- JSON files detected: 22
- SLURM jobs completed (0:0): 22/22
- Longest jobs (Elapsed):
  - Job 32435778: 00:28:06 (COMPLETED, 0:0)
  - Job 32435779: 00:27:45 (COMPLETED, 0:0)
  - Job 32435791: 00:26:10 (COMPLETED, 0:0)
  - Job 32435789: 00:24:40 (COMPLETED, 0:0)

## 2) Checkpoint Validation (Phase B)

- [x] 22个JSON，全部有5个fold_results
- [x] split_source = predefined_splits.csv
- [x] AUROC全部 < 0.99
- [x] Cell A > Cell D（both tasks）
- [x] Cell C - Cell D（premium_text）> 0 for at least one task
- [ ] git commit pending approval

## 3) 2x2 Decomposition Snapshot (Early Fusion XGB)

| task | A (leaked+orig) | B (leaked+clean_text) | C (W24+orig_text) | D (clean) | premium_total A-D | premium_struct B-D | premium_text C-D | interaction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aki_progression | 0.917634 | 0.917234 | 0.870915 | 0.871355 | +0.046279 | +0.045879 | -0.000440 | +0.000840 |
| sepsis_shock | 0.984521 | 0.984445 | 0.944612 | 0.944572 | +0.039950 | +0.039874 | +0.000041 | +0.000035 |

## 4) Result Inventory

- Full summary CSV: `results/audit/task_ab_phaseB_results_summary.csv`
- Total rows in summary CSV: 22
- aki_progression: 11 experiments
- sepsis_shock: 11 experiments
