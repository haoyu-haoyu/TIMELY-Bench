# Reproducibility Guide

This repository is a public export of TIMELY-Bench. It supports multiple reproducibility levels because MIMIC-IV patient data and prompt-level clinical text cannot be redistributed through public GitHub.

## Level 1: Inspect Aggregate Results

**Goal:** Verify the reported public metrics and manuscript artifacts without credentialed clinical data.

Required files are already tracked in this repository:

- `results/note_centered/`
- `results/v3/`
- `results/cres_v3/phase65f_frozen_eval/`
- `paper/npj_digital_medicine/`

Examples:

```bash
python - <<'PY'
import pandas as pd

provider = pd.read_csv("results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv")
print(provider)

heatmap = pd.read_csv("results/cres_v3/phase65f_frozen_eval/phase65f_condition_heatmap_data.csv")
print(heatmap.head())
PY
```

This level does not require MIMIC-IV access.

## Level 2: Regenerate Public Tables and Figures

**Goal:** Rebuild lightweight tables and figures from public aggregate files.

```bash
python -m pip install -r requirements.txt

python code/analysis/generate_core_tables.py
python code/analysis/compare_old_vs_new.py
python code/analysis/answer_analysis_questions.py
MPLBACKEND=Agg python code/analysis/generate_figures.py
```

This level uses aggregate V2 result JSON files and does not require raw patient-level data.

For the V3 manuscript tables and figures, the public export includes the frozen aggregate CSV/JSON summaries used by the paper. Per-instance prompt and response files are excluded because they contain patient-context text.

## Level 3: Rebuild From MIMIC-IV

**Goal:** Reconstruct the V3 benchmark artifacts from source clinical data.

Requirements:

- PhysioNet credentialed access to MIMIC-IV.
- A local or institutional data environment with the MIMIC-IV ICU, hospital, and note modules available.
- BigQuery-compatible access or equivalent local table exports.
- Sufficient storage for patient-level structured timelines, note extracts, and derived prompts.

Canonical CREATE/HPC entrypoints:

```bash
export PROJECT_ROOT=/path/to/TIMELY-Bench
export BQ_BILLING_PROJECT=your-billing-project

bash scripts/run_v3_full_source_refresh_create.sh
bash scripts/run_v3_create_pipeline.sh
sbatch scripts/run_phase6_cres_assembly_v3.sbatch
```

Important controlled artifacts generated at this level include:

- `data/processed/v3/`
- `results/v3/`
- `results/cres_v3/cres_eval_sample_12k.parquet`
- `results/cres_v3/*manifest*.jsonl`
- prompt JSONL files containing patient-context text

These files are intentionally not included in public GitHub.

## Level 4: Rerun LLM Inference and Frozen Scoring

**Goal:** Rerun contestant inference or rebuild frozen scoring outputs.

This level requires:

- The prompt manifest and prompt JSONL files generated in Level 3.
- API access for hosted models or local GPU access for open-weight models.
- Model-specific environment variables and serving configuration.

Canonical scoring entrypoint:

```bash
export PROJECT_ROOT=/path/to/TIMELY-Bench
export RESULTS_ROOT=${PROJECT_ROOT}/results/cres_v3
export OUTPUT_DIR=${RESULTS_ROOT}/phase65f_frozen_eval

bash scripts/run_phase65f_frozen_eval_create.sh
```

LLM inference scripts under `scripts/` are CREATE/Slurm templates. Before reuse, set:

- `PROJECT_ROOT`
- `RESULTS_ROOT`
- model API keys or provider-specific environment files
- `VENV` for local vLLM/HF serving scripts
- `HF_HOME` for Hugging Face cache

The public comparative set is frozen to:

- `gpt54`
- `gemini31pro`
- `deepseek_chat`
- `qwen35`
- `gemma4_26b`
- `aloe70b`
- `aloe7b`
- `meditron3_8b`
- `medgemma15_4b_it`

`openbiollm70b` is retained only as a supplementary artifact and is excluded from formal cross-provider comparisons.

## Public vs Controlled Artifacts

Public GitHub includes:

- Code and schemas.
- Aggregate metrics and summary JSON/CSV files.
- Manuscript source and rendered PDF.
- Public release/audit summaries.

Controlled storage is required for:

- Raw MIMIC-IV tables.
- Derived patient-level cohort files.
- Prompt JSONL containing clinical context.
- Canonical model response JSONL.
- Per-instance scoring tables.
- Long-form judge rationales.

This separation is required to preserve reproducibility while respecting MIMIC-IV data-use constraints.
