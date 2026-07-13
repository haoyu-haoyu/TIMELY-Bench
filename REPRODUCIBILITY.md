# TIMELY-Bench reproducibility guide

TIMELY-Bench uses a tiered reproducibility model. The public GitHub repository is sufficient to inspect aggregate results, verify the consistency of the public release, and run deterministic synthetic tests. It is **not** a complete copy of V3 and cannot, by itself, reproduce patient-level artifacts or the exact frozen model outputs.

The boundary is required because V3 is derived from credentialed MIMIC-IV and MIMIC-IV-Note data. Real patient timelines, instantiated prompts, row-level predictions, and judge rationales remain in controlled storage or an approved credentialed release channel. See [DATA_ACCESS.md](DATA_ACCESS.md) and [PUBLIC_ARTIFACT_POLICY.md](PUBLIC_ARTIFACT_POLICY.md).

## Reproducibility levels

| Level | What is reproduced or verified | MIMIC credentials | Restricted TIMELY-Bench bundle | Model/API access |
|---|---|---:|---:|---:|
| 0 | Public aggregate files and release consistency | No | No | No |
| 1 | Deterministic synthetic fixtures and public CI | No | No | No |
| 2 | Credentialed MIMIC-IV extraction and V3 foundation | Yes | No | No |
| 3 | V3 condition tasks, representations, and CRES manifests | Yes | No | No |
| 4 | New LLM inference, canonicalization, scoring, and judge-packet construction | Yes | No | Yes |
| 5 | Byte-level verification of the historical frozen V3/CRES package | Yes | Yes | No, unless rerunning inference |

Levels are cumulative. Level 5 verifies archived frozen assets; it is not the same as making new API calls. Hosted models and APIs can change after the original run, so a new Level 4 run may be methodologically comparable without being byte-for-byte identical.

## Before starting

Clone and pin the exact source revision used for the reproduction record:

```bash
git clone https://github.com/haoyu-haoyu/TIMELY-Bench.git
cd TIMELY-Bench
git rev-parse HEAD
python3 --version
```

Use Python 3.10 or newer. For Levels 2–4, create an isolated environment and record the resolved packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-v3.txt
python -m pip freeze > reproduction-pip-freeze.txt
```

`requirements-v3.txt` extends the lightweight public analysis stack with the
BigQuery, Arrow, baseline-model, hosted-API, and local-model dependencies used
across Levels 2–4. CUDA drivers and provider-specific serving runtimes remain
execution-site specific and must be recorded separately.

Do not commit `reproduction-pip-freeze.txt`, credentials, environment files, or any generated patient-level data to a public repository.

---

## Level 0 — verify the public aggregate release

### Goal

Confirm that the tracked metadata, Phase 6.5F aggregate tables, and finalized judge summaries are internally consistent. This level verifies published summaries; it does not recalculate them from per-instance predictions.

### Requirements

- A fresh Git checkout.
- Python 3.10+; no MIMIC access or third-party package is needed for the fallback check below.
- Optional: `make`, if using the convenience target.

### Run the public verifier

The release helper performs manifest, policy, schema, and aggregate consistency checks:

```bash
python tools/verify_public_release.py .
```

Expected terminal checkpoint:

```text
PASS: public release verification completed with 0 findings.
```

If the top-level Makefile in the checked-out release provides the convenience target, the equivalent command is:

```bash
make verify-public
```

To independently inspect the central frozen counts using only the Python standard library, run:

```bash
python3 - <<'PY'
import csv
import json
from pathlib import Path

root = Path(".")

cohort = json.loads((root / "results/v3/cohort_v3_meta.json").read_text())
assert cohort["n_stays"] == 94_458
assert cohort["n_subjects"] == 65_366
assert cohort["n_hadm"] == 85_242

prompt_build = json.loads(
    (root / "results/cres_v3/phase65b_prompt_build_summary.json").read_text()
)
assert prompt_build["sample_rows"] == 12_000
assert prompt_build["prompt_rows"] == 265_350
assert prompt_build["variants"]["full_multimodal"] == 53_070

freeze_dir = root / "results/cres_v3/phase65f_frozen_eval"
canonical = json.loads((freeze_dir / "phase65f_canonicalization_summary.json").read_text())
assert canonical["all_rows_match_expected"] is True
assert canonical["all_parse_success_match_expected"] is True
assert len(canonical["providers"]) == 9
for provider, summary in canonical["providers"].items():
    assert summary["unique_prompt_ids"] == 53_070, provider
    assert summary["ok_rows"] == 53_070, provider
    assert summary["parse_success_rows"] == 53_070, provider

scoring = json.loads((freeze_dir / "phase65f_scoring_summary.json").read_text())
assert scoring["auto_scoring"]["scored_prompt_rows"] == 166_019
assert scoring["auto_scoring"]["supported_task_dimensions"] == 20
assert scoring["auto_scoring"]["per_task_dimension_rows"] == 180
assert scoring["auto_scoring"]["provider_rows"] == 9
assert scoring["parity_check_tier1a"]["match_within_rounding"] is True

with (freeze_dir / "phase65f_provider_metrics.csv").open(newline="") as fh:
    provider_rows = list(csv.DictReader(fh))
assert len(provider_rows) == 9
winner = max(provider_rows, key=lambda row: float(row["overall_macro_primary_score"]))
assert winner["provider"] == "gemini31pro"
assert abs(float(winner["overall_macro_primary_score"]) - 0.655200) < 1e-6

judge_dir = root / "results/cres_v3/phase65f_frozen_eval_local_final_sync"
judge = json.loads((judge_dir / "phase65f_judge_formal_summary.json").read_text())
assert judge["coverage"]["expected_judge_rows_per_judge"] == 2_000
assert judge["coverage"]["common_judge_rows"] == 2_000
assert set(judge["coverage"]["rows_per_judge"].values()) == {2_000}

print("PASS: public V3 and Phase 6.5F aggregate checkpoints are consistent.")
PY
```

### Expected public checkpoints

- V3 cohort metadata: 94,458 ICU stays, 65,366 subjects, and 85,242 admissions.
- CRES evaluation sample: 12,000 rows; 53,070 `full_multimodal` prompts.
- Frozen contestant set: nine providers, each summarized as 53,070 successful and parseable responses.
- Auto-scoring: 166,019 scored prompt rows across 20 supported task–dimension pairs.
- Judge packet summary: 500 prompt instances and 2,000 contestant-response rows.
- Final judge summaries: 2,000/2,000 rows for each of Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro.

These counts come from public summary files. The underlying prompt-level JSONL and Parquet files are not present at Level 0.

---

## Level 1 — deterministic synthetic reproduction and CI

### Goal

Exercise the public data contracts without using or transforming any real MIMIC record. The fixtures are generated from fictional identifiers, fictional measurements, and synthetic clinical text.

### Requirements

- Level 0 checkout.
- Python 3.10+.
- No network, MIMIC credential, GPU, or API key.

### Regenerate and compare the synthetic fixtures

```bash
python synthetic/generate.py --check
```

Expected output:

```text
OK: 2 synthetic artifacts match the deterministic generator.
```

The generator is deterministic and checks:

- `synthetic/fixtures/synthetic_cases.json`
- `synthetic/fixtures/golden_summary.json`
- the contract in `synthetic/schema.json`

Run the public test suite:

```bash
python -m unittest discover -s tests/public_release -v
```

Then run the complete public verifier:

```bash
python tools/verify_public_release.py .
```

If exposed by the top-level Makefile, the convenience targets are:

```bash
make reproduce-synthetic
make verify-public
```

### Pass criteria

- The generator reports both checked artifacts as unchanged.
- Every `tests/public_release` test ends in `ok`.
- The verifier ends with zero findings.
- `git status --short` shows no unexpected patient-level or generated payloads.

Level 1 demonstrates that schemas, fixture generation, public summaries, and release-policy checks are executable. It does not simulate the statistical distribution of MIMIC-IV and is not evidence that the clinical cohort itself has been reproduced.

---

## Level 2 — credentialed extraction and V3 foundation

### Goal

Reconstruct the V3 cohort, 168-hour structured backbone, source-note extracts, event tables, hourly state grid, state vectors, and time-aware contexts from authorized source data.

### Requirements

- Current credentialed access to MIMIC-IV v3.1 and the MIMIC-IV-Note tables used by the project.
- Access to the following BigQuery datasets, or equivalent private local exports adapted to the same contracts:
  - `physionet-data.mimiciv_3_1_derived`
  - `physionet-data.mimiciv_3_1_hosp`
  - `physionet-data.mimiciv_3_1_icu`
  - `physionet-data.mimiciv_note`
- A Google Cloud billing/quota project with BigQuery jobs enabled.
- `google-cloud-bigquery`, `db-dtypes`, and `pyarrow` in the environment.
- A Linux/HPC workspace with approximately 300 GiB of free controlled storage for a full run. More headroom is advisable for temporary partitions and logs.
- Permission to process MIMIC-derived data in that environment.

Never run this stage in a public GitHub Actions runner or a publicly synchronized folder.

### Authenticate and configure

For a workstation using Application Default Credentials:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_BILLING_PROJECT
```

Set the project and resource controls:

```bash
export PROJECT_ROOT="$PWD"
export BQ_BILLING_PROJECT="YOUR_BILLING_PROJECT"
export PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python -u"
export SOURCE_BATCH_SIZE=5000
export NOTE_BATCH_SIZE=1500
export BATCH_SIZE=5000
export GRID_CHUNK_SIZE=1000
export CONTEXT_STAY_BATCH_SIZE=250
export CONTEXT_NOTE_READ_CHUNKSIZE=10000
```

`PROJECT_ROOT` must be a clean, controlled reconstruction workspace—not the historical frozen directory. Do not place credentials in shell scripts or commit them to Git.

### Smoke test first

The `--stay-limit` option is intended for pipeline validation. It does not reproduce the full cohort:

```bash
bash scripts/run_v3_full_source_refresh_create.sh --stay-limit 100
```

Confirm that the smoke-test workspace contains cohort metadata, source tables or their partition directories, and context output. Delete the smoke-test workspace or start from a new clean workspace before the full run; otherwise resumable outputs can mix limited and full artifacts.

### Run the full credentialed extraction

```bash
bash scripts/run_v3_full_source_refresh_create.sh
```

This wrapper runs the following real entry points in order:

1. `code/v3/extract_cohort_bq.py`
2. `code/v3/extract_structured_backbone_bq.py`
3. `code/v3/extract_notes_bq.py`
4. `scripts/run_v3_create_pipeline.sh`, which builds the feature dictionary, event and hourly-feature extracts, diagnosis pathways, 168-hour grid, condition labels, state vectors, and time-aware contexts.

Depending on table size and the available Parquet engine, a logical table may be a `.parquet` file, a `.csv` fallback, or a `.parquet.parts/` directory. The V3 I/O helpers recognize all three layouts; do not concatenate partitions manually.

### Level 2 checkpoints

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in [
    Path("results/v3/cohort_v3_meta.json"),
    Path("results/v3/structured_backbone_hourly_v3_meta.json"),
    Path("results/v3/hourly_state_grid_168h_meta.json"),
    Path("data/raw/v3/extract_notes_bq_meta.json"),
    Path("data/processed/v3/contexts/time_aware_patient_contexts_168h_summary.json"),
]:
    if not path.exists():
        raise SystemExit(f"missing checkpoint: {path}")
    print(path)

cohort = json.loads(Path("results/v3/cohort_v3_meta.json").read_text())
backbone = json.loads(Path("results/v3/structured_backbone_hourly_v3_meta.json").read_text())
grid = json.loads(Path("results/v3/hourly_state_grid_168h_meta.json").read_text())
print("cohort stays:", cohort["n_stays"])
print("backbone rows/parts:", backbone["n_rows"], backbone["n_parts"])
print("grid rows/parts:", grid["n_rows"], grid["n_parts"])
PY
```

With the historical source snapshot and full-cohort settings, the public reference metadata report:

- cohort: 94,458 stays;
- structured backbone: 6,583,285 rows in 19 parts;
- hourly grid: 15,868,944 rows in 95 parts, equal to 94,458 × 168 hours.

A mismatch is not automatically a software error: access dates, source-table revisions, credentials, query settings, or a prior `--stay-limit` run can change results. Only Level 5 establishes identity with the archived freeze.

---

## Level 3 — rebuild V3 tasks, representations, and CRES

### Goal

Turn the Level 2 foundation into condition-specific task instances, representation branches A/B1/B2/B3, state-space strata, and the CRES master and branch manifests.

### Requirements

- A completed Level 2 workspace.
- BigQuery credentials remain available because the AKI builder may need credentialed source tables.
- Slurm for the supplied `.sbatch` launchers, or equivalent direct Python execution using the command inside each launcher.
- Approximately 128 GiB RAM for the largest supplied CPU baseline job; downstream data-build jobs request less in the retained Slurm templates.

### Build condition tasks

First construct stroke metadata from the extracted sources:

```bash
python3 code/v3/build_stroke_pipeline_v3.py \
  --cohort-v3 data/processed/v3/cohort_v3.csv \
  --diagnoses-parquet data/processed/v3/events/diagnoses_icd_bq.parquet \
  --nursing-source data/raw/v3/nursing_notes_168h.parquet \
  --radiology-source data/raw/v3/radiology_notes_168h.parquet \
  --discharge-source data/raw/v3/discharge_notes_v3.parquet \
  --out-dir data/processed/v3/stroke \
  --summary-json results/v3/stroke/stroke_pipeline_build_summary.json
```

Then build the 14 V3 tasks: two AKI, two delirium, two sepsis, and eight stroke tasks.

```bash
python3 code/v3/build_aki_tasks_v3.py --billing-project "$BQ_BILLING_PROJECT"
python3 code/v3/build_delirium_tasks_v3.py
python3 code/v3/build_sepsis_tasks_v3.py
python3 code/v3/build_stroke_tasks_v3.py
```

Build the executable clinical knowledge assets and representation branches:

```bash
python3 code/cres_v3/build_executable_knowledge.py
python3 code/v3/build_phase4_b3_representations_v3.py
python3 code/v3/build_phase4_b2_original_v3.py
sbatch scripts/run_phase4d_b1_a_v3.sbatch
```

Wait for Phase 4D to finish successfully before starting the condition state-space jobs:

```bash
sbatch scripts/run_phase5_aki_state_space_v3.sbatch
sbatch scripts/run_phase5_delirium_state_space_v3.sbatch
sbatch scripts/run_phase5_sepsis_state_space_v3.sbatch
```

After all three jobs complete, assemble and release the CRES manifests:

```bash
sbatch scripts/run_phase6_cres_assembly_v3.sbatch
```

After Phase 6 assembly completes successfully:

```bash
sbatch scripts/run_phase6_cres_release_v3.sbatch
```

### Level 3 checkpoints

The central output summaries are:

- `results/v3/{aki,delirium,sepsis,stroke}/*_task_build_summary.json`
- `results/v3/representations/phase4d_B1_A_build_summary.json`
- `results/cres_v3/cres_master_manifest_summary.json`
- `results/cres_v3/cres_release_manifest_summary.json`
- `results/cres_v3/cres_v3_build_summary.json`

Check the CRES master:

```bash
python3 - <<'PY'
import json
from pathlib import Path

p = Path("results/cres_v3/cres_master_manifest_summary.json")
d = json.loads(p.read_text())
print("master rows:", d["manifest_rows"])
print("unique stays:", d["unique_stays"])
print("tasks:", len(d["tasks"]))
assert len(d["tasks"]) == 14
PY
```

The historical full V3 reference contains 4,929,069 master rows and 66,485 unique stays. Those values are a checkpoint for a matched source snapshot, not a guarantee for a later reconstruction.

### Optional V3 baselines

The retained Slurm launchers run the structured and sequence baselines:

```bash
sbatch scripts/run_phase65a_xgb_v3.sbatch
sbatch scripts/run_phase65a_seq_v3.sbatch
```

Run the merge only after both jobs finish successfully:

```bash
sbatch scripts/run_phase65a_merge_v3.sbatch
```

---

## Level 4 — LLM inference, frozen scoring, and judge packet

### Goal

Create a new model-evaluation run from the controlled prompts, or reconstruct scoring from a complete set of controlled response JSONL files.

### Requirements

- Completed Level 3 CRES manifests.
- Controlled storage for prompt and response JSONL files.
- Hosted-model API credentials and approved endpoints, or GPUs and model weights for open-weight models.
- For local models: a compatible vLLM/Hugging Face environment, `VENV`, `HF_HOME`, sufficient GPU memory, and accepted model licenses.
- Provider approval for processing MIMIC-derived prompts. Do not send MIMIC-derived content to a third-party service unless that use is permitted by the applicable data-use terms and institutional controls.

API keys must be exported at job submission time or loaded from an untracked permission-restricted environment file. Never put secrets in a tracked script.

### Build the fixed evaluation sample and prompts

```bash
export PROJECT_ROOT="$PWD"
export VENV="${PROJECT_ROOT}/.venv"
sbatch scripts/run_phase65b_prompt_build_v3.sbatch
```

Expected historical checkpoint in `results/cres_v3/phase65b_prompt_build_summary.json`:

- 12,000 sampled rows from 9,587 stays;
- five prompt variants and 265,350 total prompt rows;
- 53,070 `full_multimodal` prompt rows.

The generated files below contain MIMIC-derived patient context and must remain controlled:

- `data/processed/v3/cres/cres_eval_sample_12k.parquet`
- `data/processed/v3/cres/cres_eval_prompts_12k.jsonl`

### Run hosted contestants

The canonical hosted runners are:

- `code/v3/run_phase65c_tier1a_full_v3.py` for GPT-5.4 and Gemini 3.1 Pro;
- `code/v3/run_phase65d_tier1b_v3.py` for DeepSeek Chat, Qwen 3.5, and Gemma 4 26B.

The retained Slurm templates expose the original sharding and decoding settings:

```bash
export IKUNCODE_API_KEY="YOUR_TIER1A_KEY"
sbatch scripts/run_phase65c_full_gpt54_v3.sbatch
sbatch scripts/run_phase65c_full_gemini31pro_v3.sbatch
```

For Tier 1B, configure the provider-specific variables before submission. For example:

```bash
export TIER1B_DEEPSEEK_CHAT_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"
export TIER1B_DEEPSEEK_CHAT_API_KEY="YOUR_KEY"
export TIER1B_DEEPSEEK_CHAT_MODEL_NAME="deepseek-chat"
sbatch scripts/run_phase65d_full_deepseek_v32_thinking_v3.sbatch
```

Equivalent prefixes are `TIER1B_QWEN35_` and `TIER1B_GEMMA4_26B_`. The corresponding templates are:

```bash
sbatch scripts/run_phase65d_full_qwen35_v3.sbatch
sbatch scripts/run_phase65d_full_mimo_v2_pro_v3.sbatch
```

After all shards are complete, summarize them:

```bash
sbatch scripts/run_phase65c_full_merge_v3.sbatch
sbatch scripts/run_phase65d_full_merge_v3.sbatch
```

Provider endpoints and model aliases in historical templates may no longer resolve to the same underlying model. Record the endpoint operator, exact model/revision, request date, tokenizer, temperature, token limit, retry policy, and raw response status for every reproduction.

### Run open-weight contestants

The retained vLLM server/client pairs cover Aloe 70B, Meditron 3 8B, and MedGemma 1.5 4B. The server scripts read `VENV`, `HF_HOME`, `PROJECT_SCRATCH`, and model-specific port/environment variables:

- `scripts/run_phase65e_aloe70b_vllm_server_v1.sh` and `scripts/run_phase65e_aloe70b_vllm_client_v1.sh`
- `scripts/run_phase65e_meditron3_vllm_server_v1.sh` and `scripts/run_phase65e_meditron3_vllm_client_v1.sh`
- `scripts/run_phase65e_medgemma_vllm_server_v1.sh` and `scripts/run_phase65e_medgemma_vllm_client_v1.sh`

Use `code/v3/run_phase65e_tier2_v1.py` as the canonical OpenAI-compatible client. Provider-specific tail-repair and pilot utilities are retained for provenance but are not the recommended starting point for a new run. Aloe 7B must likewise be served through an approved local/OpenAI-compatible endpoint and registered with the canonical Tier 2 client.

`openbiollm70b` is supplementary and excluded from the formal nine-provider comparison.

### Canonicalize and score responses

Phase 6.5F does **not** call contestant models. It reads existing response directories, writes canonical JSONL files, calculates supported automatic scores, and constructs the judge packet.

Do not rerun it over the archived official freeze directory. It is a writing pipeline and will replace canonical and scoring artifacts. Use a new output directory:

```bash
export PROJECT_ROOT="$PWD"
export RESULTS_ROOT="${PROJECT_ROOT}/results/cres_v3"
export OUTPUT_DIR="${PROJECT_ROOT}/results/cres_v3/reproduction_$(date +%Y%m%d_%H%M%S)"
bash scripts/run_phase65f_frozen_eval_create.sh
```

For a complete matched response set, expected structural checkpoints are:

- nine canonical files in `${OUTPUT_DIR}/canonical/`;
- 53,070 unique, successful, parseable prompt IDs per provider;
- `phase65f_scoring_summary.json` with 166,019 scored rows and 20 supported task–dimension pairs;
- a 500-prompt/2,000-response judge packet;
- `execution_status` equal to `manifest_ready_judge_calls_not_executed` after packet construction.

Scores from newly generated responses are not expected to match the historical aggregate scores exactly.

### Execute a judge

`code/v3/run_phase65f_judge_execute_v1.py` is a resumable Messages-API judge executor. A provider-compatible invocation is:

```bash
export ANTHROPIC_API_KEY="YOUR_APPROVED_KEY"
python3 code/v3/run_phase65f_judge_execute_v1.py \
  --manifest-path "${OUTPUT_DIR}/phase65f_judge500_manifest.jsonl" \
  --rubric-path "${OUTPUT_DIR}/phase65f_judge_rubric.md" \
  --output-dir "${OUTPUT_DIR}" \
  --judge-role primary \
  --judge-provider claude \
  --judge-label claude_opus_4_6 \
  --model "YOUR_EXACT_MODEL_ID" \
  --base-url "YOUR_APPROVED_MESSAGES_API_BASE_URL" \
  --api-key-env ANTHROPIC_API_KEY \
  --temperature 0 \
  --workers 4
```

The executor appends results and skips judge-row IDs already recorded as successful. Inspect its generated summary before treating a run as complete; the existence of an output JSONL alone is insufficient.

### Historical judge provenance

The public final judge summaries must be interpreted with the following provenance:

1. CREATE constructed the frozen scoring artifacts and the 500-prompt/2,000-response judge packet.
2. The original CREATE-side Claude attempt produced 225 HTTP-error rows and zero successful rows because the provider returned Cloudflare 403 / Error 1010 (`browser_signature_banned`).
3. GPT-5.4 and Gemini 3.1 Pro cross-check outputs were not produced during that original CREATE judge run.
4. Judge execution, repair, merging, and final aggregation were completed in a synchronized local analysis workspace.
5. The finalized artifacts reached 2,000/2,000 successful rows for each of Claude Opus 4.6, GPT-5.4, and Gemini 3.1 Pro, and were synchronized back to CREATE on 2026-05-12 for archival clarity.

Therefore, do not describe the final judge archive as evidence that all judge calls succeeded inside CREATE Slurm. The exact wording and source record are in `results/cres_v3/phase65f_frozen_eval/phase65f_judge_local_final_sync_provenance.md`.

---

## Level 5 — exact verification with the restricted frozen bundle

### Goal

Verify that an authorized copy of the historical V3/CRES restricted release is byte-identical to the archived freeze and that its row-level assets agree with the public summaries.

This is the only level intended to support an **exact frozen-package verification** claim. Rebuilding from BigQuery or recalling a hosted model is not byte-level verification.

### Requirements

- All Level 2 access permissions.
- Authorized access to the approved restricted TIMELY-Bench V3 bundle.
- The bundle's release-specific checksum manifest, supplied through the controlled channel.
- Enough controlled storage for the full package.
- A clean verification directory. Never verify by writing into the authoritative archive.

The checksum manifest is intentionally not a map to publicly downloadable patient-level files. Obtain it and the bundle through the approved data-access process described in `DATA_ACCESS.md`.

The historical CREATE freeze did not originally include a complete V3 checksum manifest. Level 5 is therefore gated until the data custodian designates an authoritative restricted release, generates its checksum manifest from a read-only snapshot, and distributes both through the controlled channel. A manifest generated now proves identity to that newly designated restricted release; it cannot retrospectively prove that no file changed before the manifest was created.

### Verify checksums before analysis

```bash
export RESTRICTED_ROOT="/controlled/path/to/TIMELY-Bench-V3-frozen"
cd "$RESTRICTED_ROOT"
test -f restricted_release.sha256
shasum -a 256 -c restricted_release.sha256
```

On Linux, `sha256sum -c restricted_release.sha256` is equivalent if the manifest uses the GNU format. Every entry must report `OK`. A failed checksum is a failed exact verification; do not regenerate or silently replace the affected file.

### Required restricted classes

An exact archive must include, subject to its approved release manifest:

- raw and derived patient-level V3 tables under `data/raw/v3/` and `data/processed/v3/`;
- cohort, hourly-state, task, representation, state-space, and CRES Parquet assets;
- the 12,000-row evaluation sample and instantiated prompt JSONL;
- the nine contestant source-response chains and nine canonical response JSONL files;
- `phase65f_scored_prompts.parquet`;
- judge prompt/response manifests;
- full Claude, GPT-5.4, and Gemini judge output and repair chains;
- per-instance judge scores and long-form rationale where approved;
- the local-final-sync provenance and release checksum manifests.

These classes are deliberately absent from public GitHub even when an aggregate summary with the same stem is public.

### Verify row-level structure without rewriting it

After checksum verification, use a read-only mount or copy. For the canonical responses, the historical package should contain nine files matching `phase65f_frozen_eval/canonical/*_canonical_responses.jsonl`. A streaming structural audit is:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESTRICTED_ROOT"])
canonical_dir = root / "results/cres_v3/phase65f_frozen_eval/canonical"
files = sorted(canonical_dir.glob("*_canonical_responses.jsonl"))
assert len(files) == 9, len(files)

for path in files:
    rows = 0
    prompt_ids = set()
    ok = 0
    parsed = 0
    with path.open(encoding="utf-8", errors="strict") as fh:
        for line in fh:
            row = json.loads(line)
            rows += 1
            prompt_ids.add(str(row["prompt_id"]))
            ok += row.get("status") == "ok"
            parsed += row.get("parse_success") is True
    assert (rows, len(prompt_ids), ok, parsed) == (53_070, 53_070, 53_070, 53_070), path
    print("OK", path.name, rows)
PY
```

Then compare the restricted aggregate outputs with the public release in a separate Git checkout:

```bash
export PUBLIC_ROOT="/path/to/clean/TIMELY-Bench"
diff -u \
  "$PUBLIC_ROOT/results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv" \
  "$RESTRICTED_ROOT/results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv"
diff -u \
  "$PUBLIC_ROOT/results/cres_v3/phase65f_frozen_eval/phase65f_scoring_summary.json" \
  "$RESTRICTED_ROOT/results/cres_v3/phase65f_frozen_eval/phase65f_scoring_summary.json"
```

No output from `diff` means an exact text match. Use the checksum manifest, rather than `diff`, as the authority for binary Parquet and JSONL assets.

### Valid Level 5 statement

After all checks pass, an accurate claim is:

> The public code and aggregate summaries were verified against an authorized checksum-matched copy of the restricted TIMELY-Bench V3 frozen release.

Do not claim that the complete V3 dataset is contained in GitHub or that a fresh hosted-model run is byte-identical to the historical freeze.

---

## Public and controlled artifact boundary

| Artifact class | Public GitHub | Credentialed/restricted channel |
|---|---:|---:|
| Source code, SQL, schemas, task definitions | Yes | Mirrored as needed |
| Synthetic fixtures and public tests | Yes | Not required |
| Cohort/build metadata and aggregate metrics | Yes | Yes |
| Raw MIMIC-IV/MIMIC-IV-Note tables | No | Source provider only |
| Derived cohort and patient-level Parquet | No | Yes |
| Hourly state grids and patient contexts | No | Yes |
| Instantiated prompt/input/row manifests | No | Yes |
| Canonical contestant response JSONL | No | Yes |
| Per-instance frozen score tables | No | Yes |
| Judge packet and row-level judge output/rationale | No | Yes |
| API keys, local environment files, logs, caches | No | No release; regenerate locally |

Removing direct identifiers is not, by itself, sufficient to make a MIMIC-derived artifact suitable for public GitHub. Follow the approved data-use and release process for all patient-level and prompt-level assets.

## Troubleshooting

### BigQuery reports permission or quota errors

- Confirm that the authenticated identity has current MIMIC dataset access.
- Run `gcloud auth application-default set-quota-project "$BQ_BILLING_PROJECT"`.
- Verify that billing and the BigQuery API are enabled for the quota project.
- Check the dataset names printed in `results/v3/cohort_v3_meta.json`; do not silently substitute an older MIMIC version.

### `google-cloud-bigquery` or Parquet support is missing

```bash
python -m pip install google-cloud-bigquery google-cloud-bigquery-storage db-dtypes pyarrow
python -c "from google.cloud import bigquery; import pyarrow; print('imports ok')"
```

If Parquet writing falls back to CSV, the V3 I/O helper can still read the logical table, but file size and performance will differ from the frozen run.

### A full run contains only the smoke-test cohort

A previous `--stay-limit` execution likely left resumable outputs. Start the full run in a new controlled workspace. Do not delete files from the historical frozen archive.

### State-grid or context construction exhausts memory

Reduce one or more batching variables and resubmit:

```bash
export SOURCE_BATCH_SIZE=2000
export BATCH_SIZE=2000
export GRID_CHUNK_SIZE=250
export CONTEXT_STAY_BATCH_SIZE=50
export CONTEXT_NOTE_READ_CHUNKSIZE=2500
```

Smaller batches should preserve logical results but may change physical partitioning. Record the values used.

### A downstream builder reports a missing `.parquet`

Check for all supported representations of the logical table:

```bash
ls -ld path/to/table.parquet path/to/table.csv path/to/table.parquet.parts 2>/dev/null
```

If none exists, the preceding stage did not complete. Do not create an empty placeholder.

### Stroke tier counts or public reference counts differ

The stroke pipeline contains frozen cohort expectations, and several reference counts assume the historical source snapshot. Confirm MIMIC version, full-cohort execution, note availability, code commit, and extraction metadata. Treat an unexplained mismatch as a non-matched reconstruction, not as an exact reproduction.

### Hosted inference returns 401, 403, 429, timeouts, or parse failures

- Verify the provider-specific base URL, endpoint mode, model alias, and key environment variable.
- Start with a small shard or `--limit` before a full submission.
- Preserve raw error status and use the resumable runner; do not overwrite failed rows manually without recording a repair chain.
- A provider may have changed model behavior even when the alias is unchanged.
- Cloudflare 403 / Error 1010 was the documented reason the original CREATE-side Claude judge run failed; use the public provenance record when interpreting the finalized local sync.

### Phase 6.5F says source responses are missing

The public repository contains only canonical summaries, not the response JSONL files that canonicalization reads. Complete Level 4 inference or obtain authorized restricted inputs. Setting `OUTPUT_DIR` does not supply the missing `RESULTS_ROOT` response chains.

### Phase 6.5F would overwrite the official freeze

Stop the job and choose a new `OUTPUT_DIR`. `scripts/run_phase65f_frozen_eval_create.sh` writes canonical responses, scores, and judge manifests; it is not a read-only verifier.

### Judge output exists but coverage is incomplete

Read the generated judge summary and require the expected successful unique judge-row IDs. HTTP-error, parse-error, duplicate, or missing rows are not completed coverage. Keep repair and merge provenance with the result.

### A restricted checksum fails

Obtain a fresh authorized archive or checksum manifest from the controlled release channel. Do not normalize line endings, regenerate Parquet, rewrite JSON formatting, or use the public summaries to repair a supposedly exact package.

## Reproduction reporting checklist

For any published reproduction, record:

- Git commit SHA and whether the worktree was clean;
- Python version, resolved dependency file, operating system, and hardware;
- MIMIC-IV and MIMIC-IV-Note versions and access date;
- BigQuery dataset names, billing project identifier, batch settings, and extraction date;
- whether the run was full-cohort or used `--stay-limit`;
- all random seeds and Slurm resource settings;
- exact model and tokenizer revisions, endpoint operator, request dates, decoding parameters, and retry policy;
- prompt-manifest checksum and response/judge repair-chain provenance, stored only in the controlled environment;
- which reproducibility level was completed and which assets were unavailable.

This reporting supports a precise claim about what was verified without implying that GitHub alone contains or exactly reconstructs the complete V3 freeze.
