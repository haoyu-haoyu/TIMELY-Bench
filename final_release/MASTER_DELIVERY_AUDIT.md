# MASTER DELIVERY AUDIT REPORT (v2.0 FINAL)

**Generated**: 2026-02-01T14:20:01.078830+00:00
**Canonical Root**: `/scratch/users/k25113331/TIMELY-Bench_Final`
**Audit Version**: 2.0-final (fixed JSONPath, quote_valid source, nursing text field)

---

## Executive Summary

| # | Check | Verdict | Evidence |
|---|-------|---------|----------|
| 1 | Discharge Chain (L1-L4) | **PASS** | [discharge_chain_recheck.json](evidence/discharge_chain_recheck.json) |
| 2 | Episode Schema Coverage | **PASS** | [episode_schema_coverage.json](evidence/episode_schema_coverage.json) |
| 3 | Nursing Duplicates | **DOCUMENTED** | [nursing_duplicates_recheck.json](evidence/nursing_duplicates_recheck.json) |
| 4 | DeepSeek Traceability | **PASS** | [deepseek_chain_recheck.json](evidence/deepseek_chain_recheck.json) |
| 5 | Timeline Coverage | **ACCEPTABLE** | [timeline_coverage_recheck.json](evidence/timeline_coverage_recheck.json) |
| 6 | Anchor File Inventory | **8/8 found** | [delivery_anchor_inventory.json](evidence/delivery_anchor_inventory.json) |

---

## 1. Discharge Chain Verification

| Layer | Scope | Discharge Count | Verdict |
|-------|-------|-----------------|---------|
| L1 | Episodes (74,829 eps, 6,975,132 notes) | **0** | PASS |
| L2 | LLM annotation set | **0** | PASS |
| L3 | DeepSeek outputs | **0** | PASS |
| L4 | final_release mirror | **0** | PASS |

**Code Guardrail** (line 71): `chunk = chunk[chunk["note_type"].astype(str).str.lower() != "discharge"]`

**Note Types Distribution**:
- `nursing`: 6,790,265
- `lab_comment`: 99,531
- `radiology`: 85,336

---

## 2. Episode Schema Coverage (FIXED)

### Issue & Fix
Previous audit reported 14 fields as MISSING (0% coverage) because it checked top-level JSONPath instead of nested actual paths.

**Key Alias Mappings Applied**:

| Documentation Field | Actual JSONPath | Previous Coverage | Fixed Coverage |
|--------------------|-----------------|-------------------|----------------|
| `subject_id` | `patient.subject_id` | 0% (MISSING) | 100% (PRESENT) |
| `hadm_id` | `patient.hadm_id` | 0% (MISSING) | 100% (PRESENT) |
| `structured_data.vitals` | `timeseries.vitals` | 0% (MISSING) | 100% (PRESENT) |
| `structured_data.labs` | `timeseries.labs` | 0% (MISSING) | 100% (PRESENT) |
| `labels.mortality` | `labels.outcome.mortality` | 0% (MISSING) | 100% (PRESENT) |
| `labels.prolonged_los` | `labels.outcome.prolonged_los` | 0% (MISSING) | 100% (PRESENT) |

### Current Results

- **Sample Size**: 300 episodes
- **Fully Present**: 26 fields
- **Partial**: 3 fields
- **Not In Schema (v2.0 Planned)**: 6 fields

**Fields classified as v2.0 Planned / Future Extension**:
- `patient_state_space` — documented as planned, not yet implemented in episodes
- `reasoning_chain` — documented as planned, not yet implemented in episodes
- `disease_timeline` — documented as planned, not yet implemented in episodes
- `syndrome_detection` — documented as planned, not yet implemented in episodes
- `conditions` — documented as planned, not yet implemented in episodes
- `risk_factors` — documented as planned, not yet implemented in episodes

**Schema Reality Check** (sample episode keys):
```
Top-level: episode_id, stay_id, patient, timeseries, clinical_text, reasoning, labels, metadata
patient.*: subject_id, hadm_id, age, gender
timeseries.*: vitals[], labs[], missing_rate, start_hour, end_hour
labels.*: outcome.{mortality, prolonged_los, readmission_30d}, has_sepsis, has_aki, has_ards
reasoning.*: detected_patterns[], pattern_annotations[], condition_graph, consistency_stats
metadata.*: schema_version, source_database, data_quality_score
```

**Verdict**: PASS

---

## 3. Nursing Duplicates Analysis (FIXED)

### Issue & Fix
Previous audit reported `mean length = 0 chars` and empty samples. The script was reading from `note_text` which does not exist; the correct field is `text_full` (or `text_relevant`).

### Fixed Statistics

| Metric | Value |
|--------|-------|
| Total Nursing Notes | 6,790,265 |
| Episodes with Nursing | 74,734 |
| Unique Texts | 245 |
| Exact Duplicate Rate | 99.9964% |
| Mean Length | 16.47 chars |
| Median Length | 15 chars |
| Min/Max Length | 10 / 101 chars |
| Short Entries (<50 chars) | 99.69% |

### Top 10 Most Common Nursing Texts

| # | Text (truncated) | Count | % |
|---|-----------------|-------|---|
| 1 | `SR (Sinus Rhythm)` | 1,007,429 | 14.8364% |
| 2 | `Full resistance` | 654,844 | 9.6439% |
| 3 | `Obeys Commands` | 401,848 | 5.918% |
| 4 | `Some resistance` | 334,541 | 4.9268% |
| 5 | `Spontaneously` | 324,114 | 4.7732% |
| 6 | `Patient Verbalized` | 318,370 | 4.6886% |
| 7 | `Consistently` | 288,005 | 4.2414% |
| 8 | `ST (Sinus Tachycardia)` | 274,136 | 4.0372% |
| 9 | `AF (Atrial Fibrillation)` | 177,328 | 2.6115% |
| 10 | `No response` | 171,446 | 2.5249% |

**Conclusion**: FLOWSHEET_ENTRIES

Nursing notes are primarily short flowsheet/charted event entries (e.g., 'Sinus rhythm', 'IV patent'), not full narrative text.

**Processing Strategy**: Per-stay exact dedup before LLM sampling; report as 'structured-like text' in documentation.

---

## 4. DeepSeek Traceability (FIXED)

### Issue & Fix
Previous audit looked for `quote_valid` field directly in the raw DeepSeek JSONL, which does not contain it.
The `quote_valid` metric is computed by `verify_deepseek_evidence_validity.py` and stored in `evidence_validity_deepseek_v2_*.json`.

### Canonical DeepSeek Run

| Parameter | Value |
|-----------|-------|
| **Run ID** | `20260127_151413` |
| **Model** | `deepseek-chat` |
| **Metadata** | `final_release/llm_annotations/ANNOTATION_METADATA_deepseek_20260127_151413.json` |
| **Metadata SHA256** | `70d23589524e8ff2f7fdca72bba260b0` |
| **Annotations** | `final_release/llm_annotations/annotations_deepseek_20260127_151413_part0001.jsonl` |
| **Annotations SHA256** | `2c06f4564f8ee96b0a1dd14ff7363ffa` |

### Records & Completion

- **Input Rows**: 900
- **Completed**: 900
- **Failed**: 0
- **Avg Latency**: 2.79s

### Quote Validation (from evidence_validity report)

| Note Type | Total | Valid | Rate |
|-----------|-------|-------|------|
| lab_comment | 300 | 255 | 0.8500 |
| nursing | 300 | 300 | 1.0000 |
| radiology | 300 | 172 | 0.5733 |

- **Overall Quote Valid Rate**: **0.8078** (727/900)
- **Enforcement Violations**: 0
- **Enforcement Pass**: True

**Consistency with RELEASE_AUDIT_REPORT**: quote_valid_rate = 0.8078 (release) vs 0.807778 (this audit) → **CONSISTENT**

### Opt-in Isolation

- **Baseline Scripts Checked**: 27
- **Violations in Baselines**: 0
- **Pass**: True

Data-processing scripts that legitimately use LLM data (documented integration points):
  - `code/data_processing/build_final_release_bundle.py`
  - `code/data_processing/summarize_llm_annotation_set.py`
  - `code/data_processing/run_llm_annotation.py`
  - `code/data_processing/run_llm_annotation.py`
  - `code/data_processing/verify_llm_annotation_set.py`

**Verdict**: PASS

---

## 5. Timeline Coverage

**Policy**: Timeline is an OPT-IN extension. Not embedded in episodes by default. Baseline models do not depend on it.

| Metric | Value |
|--------|-------|
| Episode Stay IDs | 74,829 |
| Timeline Stay IDs | 74,720 |
| Missing | 109 (0.1457%) |
| Coverage | 99.8543% |

**Verdict**: ACCEPTABLE

**Rationale**: Coverage > 99% is acceptable for opt-in extension. Timeline is opt-in extension; baseline models do not use timeline data. The 109 missing stays (0.1457%) are within acceptable tolerance.

---

## Evidence Files

All evidence files are in `final_release/evidence/`:

| File | Size | SHA256 (first 16) |
|------|------|-------------------|
| [code_tree_hash.json](evidence/code_tree_hash.json) | 1,747 | `6cad790cb1bb8419` |
| [dedup_impact_report.json](evidence/dedup_impact_report.json) | 2,449 | `8006145e4ab00f3c` |
| [dedup_impact_report.md](evidence/dedup_impact_report.md) | 1,270 | `f4e40acc6c1a49bc` |
| [deepseek_chain_recheck.json](evidence/deepseek_chain_recheck.json) | 3,399 | `fc1d9000db339203` |
| [delivery_anchor_inventory.json](evidence/delivery_anchor_inventory.json) | 2,150 | `0e425e84c7cb63b3` |
| [discharge_audit_report.json](evidence/discharge_audit_report.json) | 711 | `f5448cd3d62b6e72` |
| [discharge_audit_report.md](evidence/discharge_audit_report.md) | 476 | `282d62facee81425` |
| [discharge_chain_recheck.json](evidence/discharge_chain_recheck.json) | 1,023 | `14ca77187d5f1daf` |
| [episode_schema_coverage.json](evidence/episode_schema_coverage.json) | 12,894 | `d87d97b32331e750` |
| [episodes_integrity_report.md](evidence/episodes_integrity_report.md) | 5,167 | `2b2df5b9c31b745d` |
| [evidence_validity_deepseek_v2_20260127_151413.json](evidence/evidence_validity_deepseek_v2_20260127_151413.json) | 359,215 | `fd427c1a5e297cf3` |
| [late_fusion_sanity_xgb.json](evidence/late_fusion_sanity_xgb.json) | 2,983 | `4ea4acca350eaf31` |
| [llm_input_inventory.json](evidence/llm_input_inventory.json) | 1,670 | `97da89b5e89bb544` |
| [nursing_duplicates_recheck.json](evidence/nursing_duplicates_recheck.json) | 5,075 | `629b5f0de6c24108` |
| [nursing_duplicates_recheck.md](evidence/nursing_duplicates_recheck.md) | 1,632 | `0460909d6510dc08` |
| [nursing_duplicates_report.md](evidence/nursing_duplicates_report.md) | 937 | `bea70e93803cabef` |
| [nursing_duplicates_summary.json](evidence/nursing_duplicates_summary.json) | 19,114 | `512c3257c15cceeb` |
| [optin_isolation_check.json](evidence/optin_isolation_check.json) | 2,733 | `d8eaa22e28f59d4b` |
| [path_mapping_check.json](evidence/path_mapping_check.json) | 1,323 | `3d37dbe2276e0e6b` |
| [permutation_structured_mortality.csv](evidence/permutation_structured_mortality.csv) | 1,004 | `be28e4088c9dcef0` |
| [permutation_structured_mortality.json](evidence/permutation_structured_mortality.json) | 2,736 | `66ea076051584202` |
| [qa_31341087.err](evidence/qa_31341087.err) | 0 | `e3b0c44298fc1c14` |
| [qa_31341087.log](evidence/qa_31341087.log) | 436 | `30160ea46cf6d524` |
| [rerun_release_audit.json](evidence/rerun_release_audit.json) | 746 | `b747816d618e7558` |
| [results_summary.csv](evidence/results_summary.csv) | 10,287 | `9fffa4e35ccb4722` |
| [results_summary.md](evidence/results_summary.md) | 7,543 | `8149a2cacd588f20` |
| [split_inventory.json](evidence/split_inventory.json) | 447 | `d01f9bb1a2577fdc` |
| [subject_leakage_full.json](evidence/subject_leakage_full.json) | 6,836 | `ca21281882d27d94` |
| [subject_multiplicity.json](evidence/subject_multiplicity.json) | 1,877 | `33de57c366e81c81` |
| [summary_strata_deepseek_20260127_151413.json](evidence/summary_strata_deepseek_20260127_151413.json) | 1,997 | `b164102ff64a77ff` |
| [timeline_coverage.json](evidence/timeline_coverage.json) | 899 | `4d7096b13f3d03fc` |
| [timeline_coverage_recheck.json](evidence/timeline_coverage_recheck.json) | 1,579 | `222d7b3091b8d7f9` |

---

*Generated by COMPREHENSIVE AUDIT FIX v2.0-final on 2026-02-01 14:20:01 UTC*