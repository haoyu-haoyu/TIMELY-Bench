# CRES v3 Schema

This schema freezes the Phase 6A assembly scope for TIMELY-Bench v3.

- Scope: assembly only
- Out of scope: baseline model evaluation runs

## Representation Profiles

- `full_ab1b2b3`: `A|B1|B2_original|B3`
- `stroke_temporal_ab1b2`: `A|B1|B2_original`
- `stroke_retrospective_b2_only`: `B2_original`

## Manifest Fields

- `instance_id` (`string`): Unique CRES instance identifier.
- `condition` (`string`): Condition family: aki, delirium, sepsis, stroke.
- `task_id` (`string`): Condition-specific task identifier.
- `task_family` (`string`): High-level task family used in CRES reporting.
- `task_mode` (`string`): temporal or retrospective.
- `layer` (`string`): single_layer, Layer1, or Layer2.
- `stay_id` (`int64`): ICU stay identifier.
- `subject_id` (`Int64`): Patient identifier from cohort_v3.
- `hadm_id` (`Int64`): Hospital admission identifier.
- `anchor_hour` (`Float64`): Prediction anchor hour when applicable.
- `horizon_hours` (`Int64`): Prediction horizon when applicable.
- `primary_label_key` (`string`): Column name used as primary task target.
- `primary_label_type` (`string`): binary, categorical, or numeric.
- `primary_label_binary` (`Int64`): Binary target value if task is binary.
- `primary_label_numeric` (`Float64`): Numeric target value if task is numeric.
- `primary_label_text` (`string`): Categorical/text target value if task is categorical.
- `representation_profile` (`string`): Named availability profile for A/B1/B2/B3 branches.
- `available_representations` (`string`): Pipe-delimited list of representations available to the instance.
- `has_A` (`bool`): A baseline summary available.
- `has_B1` (`bool`): B1 hourly sequence available.
- `has_B2_original` (`bool`): B2 original context available.
- `has_B3` (`bool`): B3 state-space representation available.
- `phase5_available` (`bool`): Whether Phase 5 state-space is in scope for this instance.
- `trajectory_tier` (`string`): Phase 5 trajectory tier for AKI/Delirium/Sepsis.
- `mean_template_executable_support_score` (`Float64`): Mean executable template support score for the stay trajectory.
- `n_supported_atypical_flags` (`Int64`): Count of supported atypical variants flagged for the stay.
- `left_censored` (`Int64`): Delirium-specific left-censoring flag.
- `onset_confidence` (`string`): Sepsis onset confidence: high or low.
- `shock_before_sepsis_onset` (`Int64`): Sepsis metadata stratifier.
- `shock_at_sepsis_onset` (`Int64`): Sepsis metadata stratifier.
- `shock_onset_hour` (`Float64`): First shock-after-sepsis-onset hour when applicable.
- `stroke_layer` (`string`): Stroke Layer1/Layer2 reporting field.
- `stroke_tier` (`string`): Stroke tier A/B/C used in CRES reporting.
- `stroke_subtype_priority` (`string`): Priority-based stroke subtype assignment.
- `stroke_subtype_mixed` (`string`): Mixed-aware stroke subtype assignment.
- `stroke_sensitivity_subset` (`bool`): Whether instance belongs to pure-ischaemic sensitivity subset. Currently false in master manifest.
- `task_source_file` (`string`): Relative path to the source task parquet.

## Task Catalog

- `AKI-T1`: condition=`aki`, mode=`temporal`, layer=`single_layer`, primary_label=`label`, representations=`full_ab1b2b3`
- `AKI-S1`: condition=`aki`, mode=`temporal`, layer=`single_layer`, primary_label=`label`, representations=`full_ab1b2b3`
- `DEL-T1`: condition=`delirium`, mode=`temporal`, layer=`single_layer`, primary_label=`label`, representations=`full_ab1b2b3`
- `DEL-S1`: condition=`delirium`, mode=`temporal`, layer=`single_layer`, primary_label=`label`, representations=`full_ab1b2b3`
- `SEP-T1`: condition=`sepsis`, mode=`temporal`, layer=`single_layer`, primary_label=`label`, representations=`full_ab1b2b3`
- `SEP-S1`: condition=`sepsis`, mode=`temporal`, layer=`single_layer`, primary_label=`label`, representations=`full_ab1b2b3`
- `S-T1`: condition=`stroke`, mode=`temporal`, layer=`Layer1`, primary_label=`label_strength_worsening`, representations=`stroke_temporal_ab1b2`
- `S-T2`: condition=`stroke`, mode=`temporal`, layer=`Layer1`, primary_label=`label_affected_side`, representations=`stroke_temporal_ab1b2`
- `S-T3`: condition=`stroke`, mode=`temporal`, layer=`Layer1`, primary_label=`label_consistency`, representations=`stroke_temporal_ab1b2`
- `S-T4`: condition=`stroke`, mode=`temporal`, layer=`Layer1`, primary_label=`label_sequence_signature`, representations=`stroke_temporal_ab1b2`
- `S-R1`: condition=`stroke`, mode=`retrospective`, layer=`Layer2`, primary_label=`label_mechanism`, representations=`stroke_retrospective_b2_only`
- `S-R2`: condition=`stroke`, mode=`retrospective`, layer=`Layer2`, primary_label=`label_strategy`, representations=`stroke_retrospective_b2_only`
- `S-R3`: condition=`stroke`, mode=`retrospective`, layer=`Layer2`, primary_label=`label_nihss_peak`, representations=`stroke_retrospective_b2_only`
- `S-R4`: condition=`stroke`, mode=`retrospective`, layer=`Layer2`, primary_label=`label_any_complication`, representations=`stroke_retrospective_b2_only`
