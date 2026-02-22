# Strict Requirements Fix Plan (A1 / A4 / D0)

Updated: 2026-02-07

This checklist targets the three confirmed gaps against `作业要求.md`:
- A1 systematic review/taxonomy rigor
- A4 note-category ablation consistency
- Canonical aligner D0 integration into core window pipeline

## 0) Baseline Evidence (Before Fix)

- Assignment requirement source: `../作业要求.md`
- A1 currently taxonomy-focused, not full PRISMA-style workflow:
  - `documentation/SURVEY_TAXONOMY.md`
  - `documentation/REQUIREMENTS_TRACEABILITY.md`
- A4 note ablation uses MedCAT concept file and still evaluates discharge:
  - `code/baselines/eval_note_ablation.py`
  - `results/note_ablation/note_ablation_results.csv`
- Core window pipeline only defines `6h/12h/24h`:
  - `code/config.py`
  - `code/data_processing/create_multi_window_data.py`
- D0 currently exists only in aligner comparison:
  - `code/baselines/train_aligner_comparison.py`

---

## 1) A4 Fix: Note-Category Ablation Consistency (P0)

### 1.1 Code changes
- [x] Remove `discharge` from default ablation experiment set.
- [x] Add explicit guard so ablation cannot silently include `discharge` unless a manual override flag is set.
- [x] Add feature-coverage QA in ablation output:
  - non-zero feature rate per note type
  - mark low-coverage note types in JSON report

### 1.2 Data consistency checks
- [x] Verify alignment file has no discharge rows:
  - `data/processed/temporal_alignment/temporal_textual_alignment.csv`
- [ ] Verify MedCAT concept input for ablation is generated under same policy (no discharge for canonical run).
  - Current status: canonical run excludes discharge at runtime; raw concept CSV still contains discharge rows.

### 1.3 Output updates
- [x] Regenerate:
  - `results/note_ablation/note_ablation_results.csv`
  - `results/note_ablation/note_ablation_results.json`
- [x] Ensure no `only_discharge` rows remain in canonical outputs.

Acceptance:
- No discharge-only ablation rows in canonical report.
- Note-category ablation reflects the same data policy as benchmark tasks.

---

## 2) D0 Core Integration (P0)

### 2.1 Config + feature generation
- [x] Add `D0` to `WINDOWS` in `code/config.py`.
- [x] Extend `code/data_processing/create_multi_window_data.py`:
  - compute per-stay D0 cutoff from `intime` (hours to midnight)
  - generate `data_windows/window_D0/features_aggregated.csv`
  - generate `data_windows/window_D0/features_temporal.npy`
  - generate `data_windows/window_D0/features_temporal_mask.npy`

### 2.2 Pipeline and orchestration
- [x] Update pipeline checks to include D0 files:
  - `scripts/run_all.py`
  - `scripts/Snakefile`
- [x] Keep `train_aligner_comparison.py` as protocol-specific comparison, but ensure wording distinguishes:
  - "core D0 window features" vs "aligner stress-test comparison"

### 2.3 Compatibility checks
- [x] Confirm baseline loops over `WINDOWS` can run with D0 without code break:
  - `code/baselines/run_baselines.py`
  - `code/baselines/run_temporal_gru.py`
  - `code/baselines/train_dl_multiwindow.py`

Acceptance:
- D0 appears in core generated window data folder and passes path checks.
- Scripts no longer treat D0 as an external-only protocol.

---

## 3) A1 Completion Pack (P1)

### 3.1 New reproducible review artifacts
- [x] Add `documentation/systematic_review/protocol.md`
- [x] Add `documentation/systematic_review/search_queries.md`
- [x] Add `documentation/systematic_review/prisma_flow.md`
- [x] Add `documentation/systematic_review/study_extraction.csv`
- [x] Add `documentation/systematic_review/quality_assessment.csv`
- [x] Add `documentation/systematic_review/inclusion_exclusion.md`

### 3.2 Traceability update
- [x] Update `documentation/REQUIREMENTS_TRACEABILITY.md`:
  - A1 status upgraded with explicit evidence links
  - distinguish completed artifacts vs optional stretch

Acceptance:
- A1 is backed by a reproducible workflow and machine-readable extraction table, not only narrative taxonomy.

---

## 4) Validation + Execution (CREATE-first for heavy runs) (P0)

### 4.1 Local lightweight checks
- [x] Static path/format checks
- [ ] Smoke-run scripts with small/sample mode where available

### 4.2 CREATE heavy runs
- [x] Regenerate D0-inclusive windows
- [x] Re-run note-category ablation canonical report
- [ ] Re-run required QA checks

### 4.3 Sync policy
- [x] Sync local -> CREATE before runs
- [x] Sync CREATE -> local for regenerated outputs
- [ ] Record changed artifacts list

Acceptance:
- Local and CREATE contain the same canonical artifacts for A1/A4/D0 deliverables.
