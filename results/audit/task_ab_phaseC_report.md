# Task A/B Progression v2 - Phase C Report

Generated: 2026-03-09 14:59:53 GMT
Project: /scratch/prj/bhi_haoyu_benchmarking/TIMELY-Bench_Final
Branch: codex/note-centered-alignment

## 1) Phase C Execution
- Command run: `python3 code/analysis/progression_leakage_analysis.py`
- Output CSV: `results/note_centered/progression_tasks/cross_task_leakage_decomposition.csv`
- Analysis narrative updated: `results/note_centered/analysis/analysis_findings.md` (added Q8 section)

## 2) Cross-Task Leakage Decomposition (premium_text = C - D)

| Task | D_clean | premium_struct | premium_text (C-D) | text_share |
|---|---:|---:|---:|---:|
| mortality | 0.9079 | +0.0153 | +0.0000 | 0.0% |
| prolonged_los | 0.8860 | +0.0510 | -0.0004 | 0.0% |
| aki_progression | 0.8714 | +0.0459 | -0.0004 | -1.0% |
| sepsis_shock | 0.9446 | +0.0399 | +0.0000 | 0.1% |

## 3) Checkpoint Status (Phase C)
- [x] `cross_task_leakage_decomposition.csv` generated
- [x] Includes Mortality, Prolonged LOS, AKI progression, Sepsis-shock
- [x] `analysis_findings.md` updated with Q8 interpretation
- [x] Finding aligned with audit: text leakage remains ~0 across tasks; structural leakage dominates
- [ ] git commit pending approval

## 4) Scientific Interpretation (for reviewer discussion)
- Initial hypothesis (text leakage grows with task acuity) is not supported by observed C-D values.
- Mechanistic explanation: note-level ClinicalBERT mean pooling dilutes sparse future-text signals.
- Practical implication: leakage control priority remains structural windows; testing text leakage needs finer-grained sentence/span representations.
