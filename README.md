# TIMELY-Bench

**Anchor-bounded clinical temporal reasoning benchmark for ICU trajectories and text (MIMIC-IV).**

[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/Data-MIMIC--IV%20credentialed-blue.svg)](https://physionet.org)

English | [中文](README_zh.md)

## Repository Scope

This public repository contains two reproducibility tracks:

| Track | Purpose | Main artifacts |
|---|---|---|
| **V2 note-centered leakage experiments** | Earlier temporal-text alignment experiments for mortality and prolonged ICU length of stay. | `results/note_centered/`, `code/baselines/`, `code/analysis/` |
| **V3 TIMELY-Bench / CRES evaluation** | Main manuscript benchmark: four clinical conditions, 168-hour anchor-bounded trajectories, structured baselines, nine frozen LLM providers, and judge evaluation. | `results/v3/`, `results/cres_v3/`, `code/v3/`, `paper/npj_digital_medicine/` |

The manuscript results are primarily based on the **V3 TIMELY-Bench / CRES evaluation**. The V2 note-centered experiments are retained because they document the time-leakage experiments that motivated the later benchmark design.

This GitHub export intentionally excludes patient-level MIMIC-IV files, prompt JSONL, canonical model response JSONL, per-instance scoring tables, and long-form judge rationales. See [DATA_ACCESS.md](DATA_ACCESS.md), [PUBLIC_ARTIFACT_POLICY.md](PUBLIC_ARTIFACT_POLICY.md), and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## V3 Benchmark Snapshot

| Item | Value |
|---|---:|
| Source cohort | MIMIC-IV ICU stays |
| ICU stays in source alignment | 74,829 |
| Time grid | 168 hourly states |
| Clinical conditions | AKI, delirium, sepsis, stroke |
| CRES task families | 14 task definitions across temporal, threshold, trend, diagnostic, contrastive, and attribution dimensions |
| Prompt instances per LLM provider | 53,070 |
| Frozen comparative LLM providers | 9 |
| Structured baseline tasks | Eligible binary CRES tasks |

Key frozen result files:

- `results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_per_task_dimension_metrics.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_condition_heatmap_data.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_stratified_metrics.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_temporal_degradation.csv`
- `results/cres_v3/phase65f_frozen_eval/phase65f_formal_summary.md`

Main paper files:

- `paper/npj_digital_medicine/timely_bench_npj_article.tex`
- `paper/npj_digital_medicine/timely_bench_npj_article.pdf`

## V2 Note-Centered Snapshot

| Item | Value |
|---|---:|
| ICU stays | 74,829 |
| Structured features | 42 |
| Time-series horizon | 0-72h post ICU admission |
| Notes horizon | 0-48h post ICU admission |
| Total notes | 12,005,731 |
| Tasks | `mortality`, `prolonged_los` |
| Windows | `D0`, `W6`, `W12`, `W24`, `leaked`, `clean` |

Key files:

- `results/note_centered/leakage_premium_decomposition.csv`
- `results/note_centered/progression_tasks/cross_task_leakage_decomposition.csv`
- `results/note_centered/tables/`
- `results/note_centered/figures/`

## Quick Start

### 1. Inspect public aggregate results

```bash
git clone https://github.com/haoyu-haoyu/TIMELY-Bench.git
cd TIMELY-Bench
python -m pip install -r requirements.txt
```

No credentialed MIMIC-IV files are required to inspect the aggregate metric tables and manuscript artifacts already present in `results/` and `paper/`.

### 2. Regenerate lightweight V2 analysis tables and figures

```bash
python code/analysis/generate_core_tables.py
python code/analysis/compare_old_vs_new.py
python code/analysis/answer_analysis_questions.py
MPLBACKEND=Agg python code/analysis/generate_figures.py
```

These commands use public aggregate V2 result JSON files under `results/note_centered/`.

### 3. Rebuild V3/CRES from controlled data

Full V3 reconstruction requires credentialed MIMIC-IV access and the controlled patient-level derived artifacts that are not distributed on public GitHub. The CREATE/HPC-oriented entrypoints are:

```bash
bash scripts/run_v3_full_source_refresh_create.sh
bash scripts/run_v3_create_pipeline.sh
sbatch scripts/run_phase6_cres_assembly_v3.sbatch
sbatch scripts/run_phase65f_frozen_eval_create.sh
```

These scripts are **templates**. They assume a Slurm/CREATE-like environment and now use the following environment variables where possible:

- `PROJECT_ROOT`: repository root on the execution system.
- `RESULTS_ROOT`: CRES result root, usually `${PROJECT_ROOT}/results/cres_v3`.
- `VENV`: Python virtual environment for model-serving scripts.
- `HF_HOME`: Hugging Face cache root for local model-serving scripts.

If these variables are not set, scripts default to the current working directory where possible. On a different cluster, set them explicitly before submitting jobs.

## Canonical Entry Points

| Purpose | Entry point |
|---|---|
| V2 aggregate table generation | `code/analysis/generate_core_tables.py` |
| V2 leakage decomposition analysis | `code/analysis/progression_leakage_analysis.py` |
| V3 source refresh from MIMIC-IV/BigQuery | `scripts/run_v3_full_source_refresh_create.sh` |
| V3 168-hour state and representation build | `scripts/run_v3_create_pipeline.sh` |
| V3 CRES assembly | `scripts/run_phase6_cres_assembly_v3.sbatch` |
| V3 frozen scoring and judge packet build | `scripts/run_phase65f_frozen_eval_create.sh` |

Some scripts under `code/v3/` are retained for provenance but are not canonical public entrypoints. See [code/v3/README.md](code/v3/README.md).

## Benchmark Tasks

V2 tasks:

| Task | Definition |
|---|---|
| In-hospital mortality | Death during hospital stay |
| Prolonged LOS | ICU length of stay > 7 days |

V3/CRES task definitions are summarized in `results/cres_v3/cres_schema_v3.md` and operationalized in `code/v3/build_phase6_cres_assembly_v3.py`.

## Data Access

Raw MIMIC-IV data are available through the PhysioNet credentialed data access program. Public GitHub does not include raw tables, note text, prompt JSONL, patient-level files, canonical response JSONL, or judge rationales. Credentialed reconstruction details are described in [DATA_ACCESS.md](DATA_ACCESS.md) and [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Citation

```bibtex
@misc{timely-bench-2026,
  title={TIMELY-Bench: Anchor-Bounded Clinical Temporal Reasoning over ICU Trajectories and Text},
  author={Wang, Haoyu},
  year={2026},
  institution={King's College London}
}
```
