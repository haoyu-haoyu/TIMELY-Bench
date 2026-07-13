# V3 Data and Results

This document describes the frozen TIMELY-Bench V3 cohort, task catalogue,
representation branches, and Phase 6.5F results. It is intentionally limited to
aggregate statistics and public provenance. It does **not** contain patient-level
MIMIC-IV data, instantiated patient prompts, model response text, or judge
rationales.

## At a glance

| Component | Frozen V3 scope | Public source |
|---|---:|---|
| ICU stays | 94,458 | [Cohort metadata](../results/v3/cohort_v3_meta.json) |
| Subjects | 65,366 | [Cohort metadata](../results/v3/cohort_v3_meta.json) |
| Hospital admissions | 85,242 | [Cohort metadata](../results/v3/cohort_v3_meta.json) |
| Observation window | 168 hours | [Hourly-grid metadata](../results/v3/hourly_state_grid_168h_meta.json) |
| Clinical conditions | 4: AKI, delirium, sepsis, stroke | [CRES schema](../results/cres_v3/cres_schema_v3.md) |
| Tasks | 14 | [CRES master-manifest summary](../results/cres_v3/cres_master_manifest_summary.json) |
| CRES task instances | 4,929,069 | [CRES master-manifest summary](../results/cres_v3/cres_master_manifest_summary.json) |
| Unique stays represented in CRES | 66,485 | [CRES master-manifest summary](../results/cres_v3/cres_master_manifest_summary.json) |
| Frozen comparative providers | 9 | [Frozen provider registry](../results/cres_v3/phase65f_frozen_eval/phase65f_frozen_provider_registry.json) |
| Canonical responses | 53,070 per provider; 477,630 total | [Phase 6.5F formal summary](../results/cres_v3/phase65f_frozen_eval/phase65f_formal_summary.md) |
| Automatically scored rows | 166,019 | [Scoring summary](../results/cres_v3/phase65f_frozen_eval/phase65f_scoring_summary.json) |
| LLM-as-Judge packet | 500 prompt instances; 2,000 contestant responses | [Judge packet summary](../results/cres_v3/phase65f_frozen_eval/phase65f_judge500_summary.json) |
| Final judge coverage | 2,000/2,000 rows for each of 3 judges | [Final judge summary](../results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_formal_summary.json) |

The 94,458 stays describe the full V3 ICU cohort. The 66,485 CRES stays are the
union of stays eligible for at least one benchmark task; condition-specific stay
counts overlap and therefore must not be added together.

## Cohort and temporal backbone

V3 was derived from MIMIC-IV 3.1 under credentialed access. The public repository
records the build contract and aggregate counts, while the source and derived
patient-level tables remain in controlled storage.

| Artifact | Rows | Stays | Parts | Description |
|---|---:|---:|---:|---|
| Cohort | 94,458 | 94,458 | — | One row per ICU stay; 65,366 subjects and 85,242 admissions |
| Structured hourly backbone | 6,583,285 | 94,458 | 19 | Observed structured measurements over the first 168 hours |
| Complete hourly state grid | 15,868,944 | 94,458 | 95 | A complete 168-row grid per stay (`94,458 × 168`) |

Sources: [cohort](../results/v3/cohort_v3_meta.json),
[structured backbone](../results/v3/structured_backbone_hourly_v3_meta.json), and
[hourly state grid](../results/v3/hourly_state_grid_168h_meta.json).

The cohort-level screening labels have the following aggregate prevalence. These
rates describe fields in the full cohort table; they are not interchangeable with
the anchor-level task prevalences shown below.

| Cohort field | Positive rate |
|---|---:|
| AKI | 71.86% |
| Sepsis | 42.61% |
| Chronic kidney disease | 21.86% |
| In-hospital mortality | 12.01% |
| Stroke | 11.28% |
| Delirium | 7.59% |

All rates are calculated in the frozen [cohort metadata](../results/v3/cohort_v3_meta.json).

## Fourteen-task catalogue

Temporal tasks use an anchor hour and only information available at or before
that anchor. Stroke retrospective tasks use full-stay context and are explicitly
separated from temporal evaluation. `Instances` counts task rows; `stays` counts
distinct eligible ICU stays for that task.

| Task | Mode / layer | Target | Instances | Stays | Positive rate | Representation profile |
|---|---|---|---:|---:|---:|---|
| `AKI-T1` | Temporal / single | KDIGO stage 2+ within `(T, T+24]` | 692,013 | 40,915 | 8.46% | A+B1+B2+B3 |
| `AKI-S1` | Temporal / single | RRT proxy within `(T, T+72]` | 1,764,883 | 48,461 | 0.98% | A+B1+B2+B3 |
| `DEL-T1` | Temporal / single | Persistent delirium within `(T, T+24]` | 756,251 | 21,628 | 21.10% | A+B1+B2+B3 |
| `DEL-S1` | Temporal / single | Resolution via a sustained 24-hour non-positive window | 756,251 | 21,628 | 22.03% | A+B1+B2+B3 |
| `SEP-T1` | Temporal / single | Septic shock within `(T, T+12]` | 871,557 | 26,517 | 1.66% | A+B1+B2+B3 |
| `SEP-S1` | Temporal / single | Lactate clearance >10% within `(T, T+6]` | 27,003 | 11,715 | 21.11% | A+B1+B2+B3 |
| `S-T1` | Temporal / Layer 1 | Neurological-strength worsening | 36,519 | 4,792 | 14.25% | A+B1+B2 |
| `S-T2` | Temporal / Layer 1 | Affected side | 4,952 | 4,952 | Categorical | A+B1+B2 |
| `S-T3` | Temporal / Layer 1 | Deficit/imaging consistency | 2,651 | 2,651 | Categorical | A+B1+B2 |
| `S-T4` | Temporal / Layer 1 | Neurological sequence signature | 4,976 | 4,976 | Categorical | A+B1+B2 |
| `S-R1` | Retrospective / Layer 2 | Stroke mechanism | 3,912 | 3,912 | Categorical | B2 only |
| `S-R2` | Retrospective / Layer 2 | Treatment-strategy appropriateness | 3,912 | 3,912 | Categorical | B2 only |
| `S-R3` | Retrospective / Layer 2 | Peak documented NIHSS | 277 | 277 | Numeric | B2 only |
| `S-R4` | Retrospective / Layer 2 | Any stroke complication | 3,912 | 3,912 | 8.79% | B2 only |

Task counts and target fields are frozen in the [CRES master-manifest summary](../results/cres_v3/cres_master_manifest_summary.json)
and [CRES schema](../results/cres_v3/cres_schema_v3.md). Condition-specific label
definitions and prevalence are recorded in the [AKI](../results/v3/aki/aki_task_build_summary.json),
[delirium](../results/v3/delirium/delirium_task_build_summary.json),
[sepsis](../results/v3/sepsis/sepsis_task_build_summary.json), and
[stroke](../results/v3/stroke/stroke_task_build_summary.json) build summaries.

### CRES assembly by condition

| Condition | Task instances | Unique stays |
|---|---:|---:|
| AKI | 2,456,896 | 49,920 |
| Delirium | 1,512,502 | 21,628 |
| Sepsis | 898,560 | 29,713 |
| Stroke | 61,111 | 5,318 |
| **Union / total** | **4,929,069** | **66,485** |

The row total is additive; the final stay count is a union. See the
[CRES build summary](../results/cres_v3/cres_v3_build_summary.json).

## Representation branches

V3 separates the same clinical history into four representation families so
that structured, sequential, textual, and state-space contributions can be
studied without changing the task target.

| Branch | Public definition | Temporal availability |
|---|---|---|
| **A** | Anchor-level statistical summaries across 53 features: last, whole-history mean/min/max, recent-24-hour mean/min/max, and missingness summaries | AKI, delirium, sepsis, and temporal stroke |
| **B1** | Hourly structured sequence bank with explicit missingness masks and a task-aware anchor index | AKI, delirium, sepsis, and temporal stroke |
| **B2 original** | Time-aware original-context bank plus task-aware index; temporal contexts exclude future information, while stroke retrospective contexts may include discharge context | All temporal tasks and retrospective stroke |
| **B3** | State-vector bank plus anchor index; inherits the hourly grid/state-vector values and explicit missingness masks | AKI, delirium, and sepsis |

The resulting profiles are:

- `full_ab1b2b3`: A+B1+B2+B3 for AKI, delirium, and sepsis;
- `stroke_temporal_ab1b2`: A+B1+B2 for temporal stroke tasks;
- `stroke_retrospective_b2_only`: B2 only for retrospective stroke tasks.

The implementation-level layouts and counts are in the public summaries for
[A/B1](../results/v3/representations/phase4d_B1_A_build_summary.json),
[B2](../results/v3/representations/phase4c_B2_original_build_summary.json), and
[B3](../results/v3/representations/phase4b_B3_build_summary.json). The release
branch coverage is summarized in the [CRES release manifest](../results/cres_v3/cres_release_manifest_summary.json).

## Prompt and scoring design

The Phase 6.5 prompt build sampled 12,000 task instances representing 9,587
unique stays. Five controlled input variants produced 265,350 prompts:
`full_multimodal`, `structured_only`, `text_only`, `no_temporal_markers`, and
`shuffled_timeline`, with 53,070 prompts per variant. The frozen nine-provider
comparison uses only `full_multimodal`. These aggregate counts are in the
[prompt-build summary](../results/cres_v3/phase65b_prompt_build_summary.json).

Six reasoning dimensions are used where applicable:

| Dimension | Question family |
|---|---|
| `D1` | Earliest target-event timing |
| `D2` | Threshold crossing, timing, or exact numeric extraction |
| `D3` | Improving/stable/worsening trajectory |
| `D4` | Clinical explanation or categorical phenotype |
| `D5` | Typicality or onset-confidence classification |
| `D6` | Evidence attribution |

Direct ground truth supports automatic scoring for 20 task–dimension pairs.
Other pairs are deliberately deferred to judge-based evaluation rather than
being forced into a weak proxy. The frozen scorer produced 166,019 scored rows,
180 provider/task/dimension metric rows, 36 provider/condition rows, 750
stratified rows, and 331 temporal rows. Its Tier-1a parity check covered 40 rows,
with maximum absolute difference `1.11e-16` and a passing within-rounding match.
See the [scoring summary](../results/cres_v3/phase65f_frozen_eval/phase65f_scoring_summary.json)
and [supported/deferred pair inventory](../results/cres_v3/phase65f_frozen_eval/phase65f_deferred_pairs.json).

## Phase 6.5F frozen automatic results

Every provider has exactly 53,070 canonical rows, all with `status=ok` and a
successful parse. `Overall macro primary score` is the unweighted macro-average
of the primary metric across the 20 automatically supported task–dimension
pairs. It should be interpreted alongside the condition and per-pair tables,
not as a clinical-performance claim.

| Rank | Provider | Tier | Model recorded in registry | Overall macro primary score |
|---:|---|---|---|---:|
| 1 | Gemini 3.1 Pro | Tier 1a | `gemini-3.1-pro` | **0.655200** |
| 2 | Gemma 4 26B | Tier 1b | `arc:lite` | **0.645760** |
| 3 | DeepSeek Chat | Tier 1b | `deepseek-chat` | **0.634618** |
| 4 | Qwen 3.5 | Tier 1b | `qwen3.5-flash` | **0.633547** |
| 5 | GPT-5.4 | Tier 1a | `gpt-5.4` | **0.625744** |
| 6 | MedGemma 1.5 4B IT | Tier 2 | `medgemma-1.5-4b-it` | **0.534534** |
| 7 | Aloe 70B | Tier 2 | `llama31-aloe-beta-70b` | **0.519257** |
| 8 | Meditron 3 8B | Tier 2 | `meditron3-8b` | **0.510010** |
| 9 | Aloe 7B | Tier 2 | `qwen25-aloe-beta-7b` | **0.488846** |

Source: [provider metrics](../results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv).
`openbiollm70b` was excluded from the formal comparative table and is not one of
the nine frozen providers.

### Condition-level score by tier

| Tier | Providers | AKI | Delirium | Sepsis | Stroke |
|---|---:|---:|---:|---:|---:|
| Tier 1a | 2 | 0.787402 | 1.000000 | 0.775872 | 0.346118 |
| Tier 1b | 3 | 0.713438 | 0.996445 | 0.816375 | 0.367184 |
| Tier 2 | 4 | 0.423713 | 0.951200 | 0.647133 | 0.337817 |

These are tier means within each condition, not an overall tier ranking. The
automatic scoring is near ceiling for delirium and substantially lower for
stroke across all three tiers. Full provider×condition values are available in
the [condition heatmap data](../results/cres_v3/phase65f_frozen_eval/phase65f_condition_heatmap_data.csv),
and the table above comes from the [tier comparison](../results/cres_v3/phase65f_frozen_eval/phase65f_tier_comparison.csv).

## LLM-as-Judge evaluation

The fixed judge packet contains 500 prompt instances, balanced at 125 per
condition. Within each condition, 75 were sampled from automatically deferred
cases and 50 from prediction-disagreement cases. Four contestants were fixed for
vendor and parameter-range coverage: GPT-5.4, DeepSeek Chat, Aloe 70B, and
MedGemma 1.5 4B IT. This produces 2,000 contestant responses per judge.

Claude Opus 4.6 was the primary judge; GPT-5.4 and Gemini 3.1 Pro were
cross-check judges. Final coverage is complete:

| Judge | Role | Completed rows | Repair applied |
|---|---|---:|---|
| Claude Opus 4.6 | Primary | 2,000/2,000 | No |
| GPT-5.4 | Cross-check | 2,000/2,000 | Yes |
| Gemini 3.1 Pro | Cross-check | 2,000/2,000 | Yes |

All three judges produced the same ordering by mean overall quality:

| Contestant | Claude Opus 4.6 | GPT-5.4 judge | Gemini 3.1 Pro judge |
|---|---:|---:|---:|
| GPT-5.4 | 4.198 | 4.390 | 4.604 |
| DeepSeek Chat | 3.732 | 3.660 | 4.290 |
| Aloe 70B | 2.796 | 2.802 | 2.872 |
| MedGemma 1.5 4B IT | 1.960 | 1.860 | 1.620 |

Scores use a 1–5 rubric. The aggregated ratings are in the [judge provider
summary](../results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_provider_summary.csv).
GPT-5.4 appears both as a contestant and as one cross-check judge; this overlap
must be disclosed when reporting the result. Claude remains the primary judge.

### Pairwise judge agreement

| Judge pair | Field | Spearman ρ | Exact match | Mean absolute difference | Common rows |
|---|---|---:|---:|---:|---:|
| Claude–Gemini | Overall quality | 0.7827 | 28.35% | 0.9190 | 2,000 |
| Claude–Gemini | Clinical correctness | 0.7595 | 28.60% | 0.9515 | 2,000 |
| Claude–GPT-5.4 | Overall quality | 0.8138 | 52.75% | 0.5215 | 2,000 |
| Claude–GPT-5.4 | Clinical correctness | 0.8076 | 51.15% | 0.5460 | 2,000 |
| Gemini–GPT-5.4 | Overall quality | 0.7392 | 41.00% | 0.8315 | 2,000 |
| Gemini–GPT-5.4 | Clinical correctness | 0.7188 | 41.15% | 0.8655 | 2,000 |

Source: [pairwise agreement CSV](../results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_pairwise_agreement.csv)
and [formal judge summary](../results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_formal_summary.md).
Correlation and exact agreement answer different questions; the 1–5-scale judges
can rank responses similarly while differing in score calibration.

## Provenance of the finalized judge results

The frozen automatic-scoring artifacts and judge packet were constructed on
CREATE. The initial CREATE-side Claude call stopped after provider-side
Cloudflare 403 / Error 1010 failures and produced no successful judge rows;
GPT-5.4 and Gemini cross-check outputs were not produced in that original run.
Final judge execution, repair, merge, and aggregation were then completed in a
synchronized local analysis workspace. The completed artifacts were copied back
to CREATE on 2026-05-12 for archival clarity, with 2,000/2,000 successful rows
for each judge.

Accordingly, the precise provenance statement is:

> The judge packet was constructed from frozen CREATE artifacts; external judge
> execution and final judge aggregation were completed in the synchronized
> analysis workspace.

The archival sync must not be described as evidence that all judge API calls ran
successfully in CREATE Slurm. See the [final-sync provenance record](../results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_local_final_sync_provenance.md).

## Public and controlled artifact boundary

The repository is a reproducibility-oriented public export, not a public copy of
MIMIC-IV or of the complete V3 frozen data payload.

| Public in this repository | Restricted to controlled/credentialed storage |
|---|---|
| Cohort/task/representation schemas and build code | Raw MIMIC-IV and MIMIC-IV-Note tables |
| Aggregate cohort and task build summaries | Derived cohort, hourly-grid, context, task, and representation Parquet/JSONL files |
| Prompt templates, sampling rules, and aggregate prompt counts | Instantiated prompts or manifests containing patient context |
| Frozen provider registry and aggregate automatic metrics | Canonical response JSONL and per-instance scoring tables |
| Judge rubric, aggregate provider/condition scores, agreement, and provenance | Per-row judge outputs, rationales, repair manifests, and identifiers |

Identifiers or clinical-text excerpts must not be reconstructed into public
artifacts. Researchers with the required MIMIC credentials can use the public
code and contracts in their own approved environment; access to MIMIC-derived
TIMELY-Bench payloads must follow the approved credentialed release channel.
See [Data access](../DATA_ACCESS.md) and the [Public artifact policy](../PUBLIC_ARTIFACT_POLICY.md).

## Reporting checklist

When citing V3 results, report all of the following:

1. The V3 cohort size: 94,458 ICU stays, 65,366 subjects, and 85,242 admissions.
2. The 168-hour observation window and the exact subset of the 14 tasks used.
3. The input variant (`full_multimodal` for the frozen nine-provider table).
4. Whether a result is automatic, judge-based, or deferred for lack of direct
   ground truth.
5. The provider/model registry label and the frozen tier.
6. For judge results, Claude Opus 4.6 as primary judge, the GPT-5.4
   contestant/judge overlap, all repair activity, and the synchronized-local
   execution provenance.
7. That patient-level data, prompts, responses, and row-level scores are not
   distributed through GitHub.
