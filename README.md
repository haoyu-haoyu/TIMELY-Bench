# TIMELY-Bench

Clinical temporal-text alignment benchmark for ICU prediction on MIMIC-IV.

English | [中文](README_zh.md)

## Current Scope (v2.0 Note-Centered)

| Item | Value |
|---|---|
| ICU stays | 74,829 |
| Structured features | 42 |
| Main tasks | In-hospital mortality, prolonged LOS |
| Progression tasks | AKI Stage1→2+, Sepsis→Septic Shock |
| Total experiments | 113 (91 + 22) |
| Alignment windows | D0, W6, W12, W24, leaked, clean |

## What is included

- Note-centered alignment pipeline for multimodal fusion.
- 2x2 leakage decomposition (structured leakage vs text leakage).
- Canonical experiment outputs for:
  - 91 core note-centered experiments: `results/note_centered/core_experiments/`
  - 22 progression experiments: `results/note_centered/progression_tasks/`
- Analysis tables/figures and reports under `results/note_centered/`.

## Scope notes

- `results/note_centered/` contains the canonical note-centered benchmark outputs.
- `results/cres/` is a supplementary clinical reasoning track and is not part of the 113 canonical benchmark experiments.
- `results/llm_annotations/` contains annotation assets and annotation-specific audit files.
- `results/audit/` contains project-wide release audits and may reference supplementary tracks in addition to the note-centered benchmark.

## Key findings (current canonical results)

- Leakage premium (A-D):
  - Mortality: +0.0154
  - Prolonged LOS: +0.0508
- Structural leakage contribution: ~99%+ across tasks.
- Text leakage contribution (note-level ClinicalBERT pooling): ~0 across tasks.
- Best clean early-fusion AUROC:
  - Mortality (Cell D): 0.9079
  - Prolonged LOS (Cell D): 0.8860
  - AKI progression (Cell D): 0.8714
  - Sepsis→Shock (Cell D): 0.9446

## Canonical layout

```text
TIMELY-Bench_Final/
├── code/
│   ├── baselines/
│   │   ├── note_centered_common.py
│   │   ├── run_baselines.py
│   │   ├── run_single_experiment.sh
│   │   ├── train_fusion.py
│   │   └── train_progression_baselines.py
│   ├── data_processing/
│   ├── evaluation/
│   └── analysis/
├── data/
├── results/
│   ├── note_centered/
│   │   ├── core_experiments/
│   │   ├── progression_tasks/
│   │   ├── comparisons/
│   │   ├── tables/
│   │   ├── figures/
│   │   └── analysis/
│   ├── cres/
│   ├── llm_annotations/
│   └── audit/
├── docs/
└── archive/legacy_consolidated/
```

## Reproduce canonical runs

```bash
cd TIMELY-Bench_Final

# Example: single experiment
bash code/baselines/run_single_experiment.sh early_fusion xgb mortality W24 original results/note_centered

# Progression experiments (AKI / Sepsis-Shock)
python code/baselines/train_progression_baselines.py --task aki_progression --condition all
python code/baselines/train_progression_baselines.py --task sepsis_shock --condition all
```

## Documentation

- Data card: `docs/DATA_CARD.md`
- Alignment protocol: `docs/ALIGNMENT_PROTOCOL_CARD.md`
- Model card: `docs/MODEL_CARD.md`
- Canonical scope: `CANONICAL_SCOPE.md`
- Phase reports / release audits: `results/audit/`

## Legacy content

Deprecated admission-anchored / medcat / readmission / long-horizon tracks are archived under:

`archive/legacy_consolidated/root_cleanup/`

## License

This project uses MIMIC-IV data and requires PhysioNet credentialed access.
