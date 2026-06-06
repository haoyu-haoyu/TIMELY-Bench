# TIMELY-Bench v3 Feature Dictionary

- Total features: **111**

## Validation Summary

- `Direct-existing`: 55
- `Direct-existing + BQ extraction logic exists`: 1
- `Direct-existing + validated search logic`: 1
- `Direct-existing / partially audited`: 1
- `Direct-validated`: 26
- `Direct-validated as hs-CRP`: 1
- `Direct-validated coverage basis`: 1
- `Direct-validated feasibility`: 1
- `Direct-validated low coverage`: 2
- `Direct-validated source feasibility`: 1
- `Needs explicit schema`: 1
- `Needs final audit`: 2
- `Needs final source audit`: 1
- `Needs rule design`: 4
- `Needs task design`: 1
- `Planned`: 3
- `Planned for B2`: 1
- `Planned source-confirmed`: 5
- `Refuted`: 1
- `Refuted for ICU use`: 1
- `existing logic, needs window extension`: 1

## static_context

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| age | admissions/patients or cohort table | Derived | Direct-existing | static context |
| sex | patients | Direct | Direct-existing | static context |
| icu_intime | icustays.intime | Direct | Direct-existing | relative timing anchor |
| icu_outtime | icustays.outtime | Direct | Direct-existing | censoring |
| hospital_death_status | admissions/discharge metadata | Direct | Direct-existing | mortality baseline |
| icu_los_hours | icustays | Derived | Direct-existing | LOS baseline |
| ckd | diagnoses_icd | Direct | Direct-existing | AKI comorbidity |
| diabetes | diagnoses_icd | Direct | Direct-existing | AKI/sepsis risk |
| hypertension | diagnoses_icd | Direct | Direct-existing | cardiovascular context |
| cognitive_impairment_or_dementia | diagnoses_icd | Direct | Planned source-confirmed | delirium risk |
| copd | diagnoses_icd | Direct | Planned source-confirmed | ARF context |
| atrial_fibrillation | diagnoses_icd | Direct | Planned source-confirmed | cardiovascular context |
| liver_disease | diagnoses_icd | Direct | Planned source-confirmed | SOFA/liver context |
| prior_stroke_or_stroke_family | diagnoses_icd | Direct | Direct-validated | stroke-proxy cohort |
| elixhauser_score | diagnoses_icd | Derived | Planned | general severity/comorbidity |

## hourly_vitals

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| heart_rate | chartevents | Direct | Direct-existing | all conditions |
| sbp | chartevents | Direct | Direct-existing | hemodynamics |
| dbp | chartevents | Direct | Direct-existing | hemodynamics |
| map_noninvasive | chartevents | Direct | Direct-validated | fallback MAP |
| map_invasive | chartevents | Direct | Direct-validated | preferred MAP |
| map_merged | chartevents | Derived | Direct-validated | merged MAP |
| resp_rate | chartevents | Direct | Direct-existing | ARF/sepsis |
| spo2 | chartevents | Direct | Direct-existing | ARF |
| temperature_f | chartevents | Direct | Direct-validated | temperature normalization |
| temperature_c_raw | chartevents | Direct | Direct-validated | temperature normalization |
| temperature_c | chartevents | Derived | Direct-validated | all conditions |
| weight | chartevents | Direct | Direct-existing | context/dosing |
| urine_output_items | outputevents | Direct multi-item | Direct-validated | AKI/fluid balance |
| urineoutput | outputevents aggregate | Derived | Direct-validated | AKI/sepsis |

## neurologic

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| gcs_eye | chartevents | Direct | Direct-existing | neuro |
| gcs_verbal | chartevents | Direct | Direct-existing | neuro |
| gcs_motor | chartevents | Direct | Direct-validated | neuro |
| gcs_total | chartevents | Derived | Needs final audit | neuro/delirium/stroke-proxy |
| rass | chartevents | Direct | Direct-validated | delirium/stroke-proxy |
| delirium_assessment | chartevents | Direct | Direct-validated | primary delirium signal |
| cam_component_items | chartevents | Direct | Direct-validated | delirium support |
| restraint_events | chartevents | Direct | Direct-validated | neuro-behavioural context |
| delirium_note_keywords | nursing/charted text rows | Proxy / inferred | Direct-validated low coverage | supporting evidence only |

## labs

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| creatinine | labevents | Direct | Direct-existing | AKI |
| bun | labevents | Direct | Direct-existing | AKI |
| sodium | labevents | Direct | Direct-existing | metabolic |
| potassium | labevents | Direct | Direct-existing | AKI |
| bicarbonate | labevents | Direct | Direct-existing | acid-base |
| chloride | labevents | Direct | Direct-existing | metabolic |
| aniongap | labevents | Direct | Direct-existing | acid-base |
| calcium | labevents | Direct | Direct-existing | metabolic |
| phosphate | labevents | Direct | Direct-existing | renal/metabolic |
| magnesium | labevents | Direct | Direct-existing | metabolic |
| wbc | labevents | Direct | Direct-existing | infection |
| hemoglobin | labevents | Direct | Direct-existing | general |
| hematocrit | labevents | Direct | Direct-existing | general |
| platelet | labevents | Direct | Direct-existing | coagulation |
| lactate | labevents | Direct | Direct-existing / partially audited | sepsis/shock |
| albumin | labevents | Direct | Direct-validated | medium-coverage lab |
| bilirubin_total | labevents | Direct | Direct-existing + BQ extraction logic exists | SOFA liver |
| inr | labevents | Direct | Direct-existing | coagulation |
| pt | labevents | Direct | Direct-existing | coagulation |
| serum_glucose | labevents | Direct | Direct-validated | core glucose source |
| bedside_glucose | chartevents | Direct | Direct-validated | complementary glucose source |
| glucose_merged | labevents + chartevents | Derived | Direct-validated | all conditions |
| troponin_t | labevents | Direct | Direct-validated | cardio extension |
| troponin_i | labevents | Direct | Refuted for ICU use | exclude |
| crp | labevents | Direct | Direct-validated low coverage | optional only |
| hs_crp | labevents | Direct | Direct-validated as hs-CRP | optional only |
| procalcitonin | d_labitems | N/A | Refuted | exclude |

## blood_gas

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| ph | blood gas / labevents | Direct | Direct-existing | acid-base |
| pao2 | bg table | Direct/Derived-table | Direct-existing | ARF |
| paco2 | bg table | Direct/Derived-table | Direct-existing | ARF |
| pao2_fio2_ratio | derived.bg.pao2fio2ratio | Derived | Direct-validated | ARF |

## respiratory

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| fio2 | ventilator_setting / chartevents | Direct | Direct-existing | ARF |
| peep | ventilator_setting / chartevents | Direct | Direct-existing | ARF |
| tidal_volume | ventilator_setting | Direct | Direct-existing | ARF |
| tidal_volume_set | chartevents / ventilator settings | Direct | Planned source-confirmed | ARF |
| minute_volume | ventilator_setting | Direct | Direct-existing | ARF |
| plateau_pressure | chartevents | Direct | Direct-validated | ARF |
| ventilation_status | derived.ventilation | Derived | Direct-validated | ARF/stroke-proxy |

## medications

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| vasopressors_active | inputevents | Direct | Direct-validated | shock/AKI |
| vasopressor_dose_norepi_equiv | inputevents conversion | Derived | Direct-existing | hemodynamic burden |
| propofol_rate | inputevents / emar | Direct | Direct-existing | delirium/ventilation |
| midazolam_rate | inputevents / emar | Direct | Direct-existing | delirium/ventilation |
| fentanyl_rate | inputevents / emar | Direct | Direct-existing | analgesia/sedation |
| dexmedetomidine_active | prescriptions/emar/inputevents | Direct | Needs final source audit | delirium |
| antibiotic_active | prescriptions + emar | Derived event logic | Direct-validated coverage basis | sepsis |
| nephrotoxic_drug_active | prescriptions + emar | Derived event logic | Direct-validated source feasibility | AKI |
| fluid_input_hourly | inputevents | Derived | Direct-existing | fluid balance |
| fluid_balance | input minus output | Derived | Direct-existing | AKI/shock |

## procedures

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| rrt_active | procedureevents | Direct/Derived flag | Direct-existing + validated search logic | AKI |
| intubation_event | procedureevents | Direct | Direct-validated | ARF/stroke-proxy |
| extubation_event | procedureevents | Direct | Direct-validated | ARF |
| unplanned_extubation_event | procedureevents | Direct | Direct-validated | ARF |
| tracheostomy | procedureevents / ventilation | Direct | Needs final audit | optional respiratory feature |

## derived_scores

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| sofa_total | derived.sofa | Derived | Direct-existing | severity |
| sofa_subcomponents | derived.sofa | Derived | Direct-existing | organ failure |
| kdigo_stage | derived.kdigo_stages | Derived | Direct-existing | AKI |
| qsofa | RR + SBP + GCS | Derived | Planned | sepsis severity |
| apache_ii | multiple components | Derived | Needs rule design | optional severity score |

## pathway_objects

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| delirium_onset | first Positive 228332 with censoring rule | Proxy / inferred | Needs rule design | delirium task |
| delirium_resolution | sustained non-positive period | Proxy / inferred | Needs rule design | delirium task |
| sepsis_onset | derived.sepsis3 | Derived | Direct-existing | sepsis task |
| shock_onset | project rule + reference rule | Proxy / inferred | Needs rule design | sepsis/shock task |
| arf_onset | PF < 300 and/or invasive ventilation start | Derived / proxy | Direct-validated feasibility | ARF task |
| stroke_proxy_deterioration_markers | GCS / RASS / vent / restraint / sedation pattern | Proxy / inferred | Needs task design | stroke-proxy reasoning |
| diagnosis_pathway_events | threshold crossings + procedure time + note mention + derived onset tables | Derived / proxy | Needs explicit schema | condition-aware pathways |

## text_objects

| Feature | Source | Acquisition mode | Validation status | Use |
|---|---|---|---|---|
| raw_notes_0_168h | noteevents + lab comments | Direct | existing logic, needs window extension | B2 |
| note_timestamp | charttime / relative hour | Direct | Direct-existing | all text alignment |
| note_type | note metadata | Direct | Direct-existing | typed reasoning |
| sentence_level_doctime_rel_tags | LLM/classifier pipeline | Derived | Direct-existing | clean text |
| clean_note_text | raw note minus AFTER sentences | Derived | Planned for B2 | leakage control |
| original_note_text | raw note unchanged | Direct | Direct-existing | original/leaked B2 |
| note_level_embedding | ClinicalBERT or equivalent | Derived | Direct-existing | optional baseline / retrieval |
| note_sequence_object | ordered note list with timestamps and type | Derived | Planned | B2 / CRES |

## Field Definitions

- `unit`: canonical output unit where applicable.
- `normalization_rule`: explicit merge, conversion, or derivation rule used in v3.
- `missingness_expectation`: expected sparsity profile before modeling or imputation.
