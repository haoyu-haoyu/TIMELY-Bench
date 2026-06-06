from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .mappings import VALIDATED_ITEMIDS


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    domain: str
    source: str
    acquisition_mode: str
    validation_status: str
    use: str
    itemids: tuple[int, ...] = ()
    notes: str = ""
    backbone: bool = False
    unit: str = ""
    normalization_rule: str = ""
    missingness_expectation: str = ""


FEATURE_UNITS: dict[str, str] = {
    "anchor_age": "years",
    "icu_los_hours": "hours",
    "heart_rate": "beats/min",
    "sbp": "mmHg",
    "dbp": "mmHg",
    "map_noninvasive": "mmHg",
    "map_invasive": "mmHg",
    "map_merged": "mmHg",
    "resp_rate": "breaths/min",
    "spo2": "percent",
    "temperature_f": "degF",
    "temperature_c_raw": "degC",
    "temperature_c": "degC",
    "weight": "kg",
    "urineoutput": "mL/hour",
    "gcs_eye": "score",
    "gcs_verbal": "score",
    "gcs_motor": "score",
    "gcs_total": "score",
    "rass": "score",
    "creatinine": "mg/dL",
    "bun": "mg/dL",
    "sodium": "mmol/L",
    "potassium": "mmol/L",
    "bicarbonate": "mmol/L",
    "chloride": "mmol/L",
    "aniongap": "mmol/L",
    "calcium": "mg/dL",
    "phosphate": "mg/dL",
    "magnesium": "mg/dL",
    "wbc": "K/uL",
    "hemoglobin": "g/dL",
    "hematocrit": "percent",
    "platelet": "K/uL",
    "lactate": "mmol/L",
    "albumin": "g/dL",
    "bilirubin_total": "mg/dL",
    "inr": "ratio",
    "pt": "seconds",
    "serum_glucose": "mg/dL",
    "bedside_glucose": "mg/dL",
    "glucose_merged": "mg/dL",
    "ph": "unitless",
    "pao2": "mmHg",
    "paco2": "mmHg",
    "pao2_fio2_ratio": "ratio",
    "troponin_t": "ng/L",
    "troponin_i": "ng/mL",
    "crp": "mg/L",
    "hs_crp": "mg/L",
    "fio2": "fraction",
    "peep": "cmH2O",
    "tidal_volume": "mL",
    "tidal_volume_set": "mL",
    "minute_volume": "L/min",
    "plateau_pressure": "cmH2O",
    "vasopressor_dose_norepi_equiv": "mcg/kg/min-equivalent",
    "fluid_input_hourly": "mL/hour",
    "fluid_balance": "mL/hour",
}

FEATURE_NORMALIZATION_RULES: dict[str, str] = {
    "map_merged": "Prefer invasive MAP when available, otherwise non-invasive MAP.",
    "temperature_c": "Convert Fahrenheit readings to Celsius; keep Celsius readings unchanged.",
    "urine_output_items": "Aggregate validated urine-output itemids to hourly totals.",
    "urineoutput": "Hourly aggregate from validated outputevents itemids.",
    "gcs_total": "Sum validated GCS eye, verbal, and motor components unless a validated independent total is selected.",
    "glucose_merged": "Prefer serum glucose; fallback to bedside glucose when serum is missing.",
    "pao2_fio2_ratio": "Use derived PF ratio when available from blood-gas table.",
    "vasopressor_dose_norepi_equiv": "Convert agent-specific vasopressor doses to norepinephrine-equivalent burden.",
    "fluid_input_hourly": "Aggregate infusion volumes within hour bins.",
    "fluid_balance": "Hourly fluid input minus hourly urine output.",
    "delirium_assessment": "Map chartevents values to Positive/Negative/UTA flags.",
    "rass": "Use validated Richmond Agitation-Sedation Scale itemid 228096.",
    "raw_notes_0_168h": "Preserve timestamped note order; no pooling in the mainline representation.",
    "clean_note_text": "Sentence-level clean policy removes heuristic AFTER sentences from note text.",
    "note_sequence_object": "Store notes as ordered per-stay sequences with timestamps and note type.",
    "diagnosis_pathway_events": "Convert diagnosis-related signals into time-stamped pathway events with source/confidence/proxy semantics.",
}

DOMAIN_MISSINGNESS = {
    "static_context": "low",
    "hourly_vitals": "low_to_medium",
    "neurologic": "medium",
    "labs": "medium_to_high",
    "blood_gas": "medium_to_high",
    "respiratory": "medium",
    "medications": "event_driven",
    "procedures": "event_driven",
    "derived_scores": "depends_on_inputs",
    "pathway_objects": "rule_defined",
    "text_objects": "medium_to_high",
}


def _fs(**kwargs: Any) -> FeatureSpec:
    name = str(kwargs["name"])
    domain = str(kwargs["domain"])
    kwargs.setdefault("unit", FEATURE_UNITS.get(name, ""))
    kwargs.setdefault("normalization_rule", FEATURE_NORMALIZATION_RULES.get(name, "native units or source-native encoding"))
    kwargs.setdefault("missingness_expectation", DOMAIN_MISSINGNESS.get(domain, "variable"))
    return FeatureSpec(**kwargs)


FEATURE_SPECS: list[FeatureSpec] = [
    # Static context
    _fs(name="age", domain="static_context", source="admissions/patients or cohort table", acquisition_mode="Derived", validation_status="Direct-existing", use="static context", notes="Age at ICU admission.", backbone=True),
    _fs(name="sex", domain="static_context", source="patients", acquisition_mode="Direct", validation_status="Direct-existing", use="static context", backbone=True),
    _fs(name="icu_intime", domain="static_context", source="icustays.intime", acquisition_mode="Direct", validation_status="Direct-existing", use="relative timing anchor", backbone=True),
    _fs(name="icu_outtime", domain="static_context", source="icustays.outtime", acquisition_mode="Direct", validation_status="Direct-existing", use="censoring", backbone=True),
    _fs(name="hospital_death_status", domain="static_context", source="admissions/discharge metadata", acquisition_mode="Direct", validation_status="Direct-existing", use="mortality baseline", backbone=True),
    _fs(name="icu_los_hours", domain="static_context", source="icustays", acquisition_mode="Derived", validation_status="Direct-existing", use="LOS baseline", backbone=True),
    _fs(name="ckd", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Direct-existing", use="AKI comorbidity", backbone=True),
    _fs(name="diabetes", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Direct-existing", use="AKI/sepsis risk", backbone=True),
    _fs(name="hypertension", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Direct-existing", use="cardiovascular context", backbone=True),
    _fs(name="cognitive_impairment_or_dementia", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Planned source-confirmed", use="delirium risk"),
    _fs(name="copd", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Planned source-confirmed", use="ARF context"),
    _fs(name="atrial_fibrillation", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Planned source-confirmed", use="cardiovascular context"),
    _fs(name="liver_disease", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Planned source-confirmed", use="SOFA/liver context"),
    _fs(name="prior_stroke_or_stroke_family", domain="static_context", source="diagnoses_icd", acquisition_mode="Direct", validation_status="Direct-validated", use="stroke-proxy cohort", notes="Use ICD families I60-I64/G45 and legacy prefixes."),
    _fs(name="elixhauser_score", domain="static_context", source="diagnoses_icd", acquisition_mode="Derived", validation_status="Planned", use="general severity/comorbidity"),
    # Vitals and bedside
    _fs(name="heart_rate", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="all conditions", backbone=True),
    _fs(name="sbp", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="hemodynamics", backbone=True),
    _fs(name="dbp", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="hemodynamics", backbone=True),
    _fs(name="map_noninvasive", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="fallback MAP", itemids=tuple(VALIDATED_ITEMIDS["map_noninvasive"])),
    _fs(name="map_invasive", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="preferred MAP", itemids=tuple(VALIDATED_ITEMIDS["map_invasive"])),
    _fs(name="map_merged", domain="hourly_vitals", source="chartevents", acquisition_mode="Derived", validation_status="Direct-validated", use="merged MAP", notes="Invasive priority, NIBP fallback.", backbone=True),
    _fs(name="resp_rate", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="ARF/sepsis", backbone=True),
    _fs(name="spo2", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="temperature_f", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="temperature normalization", itemids=tuple(VALIDATED_ITEMIDS["temperature_f"])),
    _fs(name="temperature_c_raw", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="temperature normalization", itemids=tuple(VALIDATED_ITEMIDS["temperature_c"])),
    _fs(name="temperature_c", domain="hourly_vitals", source="chartevents", acquisition_mode="Derived", validation_status="Direct-validated", use="all conditions", notes="Normalized to Celsius.", backbone=True),
    _fs(name="weight", domain="hourly_vitals", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="context/dosing"),
    _fs(name="urine_output_items", domain="hourly_vitals", source="outputevents", acquisition_mode="Direct multi-item", validation_status="Direct-validated", use="AKI/fluid balance", itemids=tuple(VALIDATED_ITEMIDS["urine_output"])),
    _fs(name="urineoutput", domain="hourly_vitals", source="outputevents aggregate", acquisition_mode="Derived", validation_status="Direct-validated", use="AKI/sepsis", notes="Hourly urine output aggregate.", backbone=True),
    # Neuro / delirium
    _fs(name="gcs_eye", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="neuro", itemids=tuple(VALIDATED_ITEMIDS["gcs_eye"]), backbone=True),
    _fs(name="gcs_verbal", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="neuro", itemids=tuple(VALIDATED_ITEMIDS["gcs_verbal"]), backbone=True),
    _fs(name="gcs_motor", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="neuro", itemids=tuple(VALIDATED_ITEMIDS["gcs_motor"]), backbone=True),
    _fs(name="gcs_total", domain="neurologic", source="chartevents", acquisition_mode="Derived", validation_status="Needs final audit", use="neuro/delirium/stroke-proxy", notes="Prefer sum of corrected components unless an independent validated total is selected.", backbone=True),
    _fs(name="rass", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="delirium/stroke-proxy", itemids=tuple(VALIDATED_ITEMIDS["rass"]), backbone=True),
    _fs(name="delirium_assessment", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="primary delirium signal", itemids=tuple(VALIDATED_ITEMIDS["delirium_assessment"]), notes="Values: Negative / Positive / UTA.", backbone=True),
    _fs(name="cam_component_items", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="delirium support", itemids=tuple(VALIDATED_ITEMIDS["cam_components"]), notes="CAM support only; 228300 is too sparse to serve as primary signal."),
    _fs(name="restraint_events", domain="neurologic", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="neuro-behavioural context", notes="Restraint-related items validated on CREATE."),
    _fs(name="delirium_note_keywords", domain="neurologic", source="nursing/charted text rows", acquisition_mode="Proxy / inferred", validation_status="Direct-validated low coverage", use="supporting evidence only", notes="Low frequency; do not use as primary label source."),
    # Labs and gas
    _fs(name="creatinine", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="AKI", backbone=True),
    _fs(name="bun", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="AKI", backbone=True),
    _fs(name="sodium", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="metabolic", backbone=True),
    _fs(name="potassium", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="AKI", backbone=True),
    _fs(name="bicarbonate", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="acid-base", backbone=True),
    _fs(name="chloride", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="metabolic", backbone=True),
    _fs(name="aniongap", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="acid-base", backbone=True),
    _fs(name="calcium", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="metabolic"),
    _fs(name="phosphate", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="renal/metabolic"),
    _fs(name="magnesium", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="metabolic"),
    _fs(name="wbc", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="infection", backbone=True),
    _fs(name="hemoglobin", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="general", backbone=True),
    _fs(name="hematocrit", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="general", backbone=True),
    _fs(name="platelet", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="coagulation", backbone=True),
    _fs(name="lactate", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing / partially audited", use="sepsis/shock", backbone=True),
    _fs(name="albumin", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-validated", use="medium-coverage lab", itemids=tuple(VALIDATED_ITEMIDS["albumin"])),
    _fs(name="bilirubin_total", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing + BQ extraction logic exists", use="SOFA liver", backbone=True),
    _fs(name="inr", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="coagulation"),
    _fs(name="pt", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="coagulation"),
    _fs(name="serum_glucose", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-validated", use="core glucose source", itemids=tuple(VALIDATED_ITEMIDS["serum_glucose"])),
    _fs(name="bedside_glucose", domain="labs", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="complementary glucose source", itemids=tuple(VALIDATED_ITEMIDS["bedside_glucose"])),
    _fs(name="glucose_merged", domain="labs", source="labevents + chartevents", acquisition_mode="Derived", validation_status="Direct-validated", use="all conditions", notes="Serum priority, bedside fallback.", backbone=True),
    _fs(name="ph", domain="blood_gas", source="blood gas / labevents", acquisition_mode="Direct", validation_status="Direct-existing", use="acid-base", backbone=True),
    _fs(name="pao2", domain="blood_gas", source="bg table", acquisition_mode="Direct/Derived-table", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="paco2", domain="blood_gas", source="bg table", acquisition_mode="Direct/Derived-table", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="pao2_fio2_ratio", domain="blood_gas", source="derived.bg.pao2fio2ratio", acquisition_mode="Derived", validation_status="Direct-validated", use="ARF", backbone=True),
    _fs(name="troponin_t", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-validated", use="cardio extension", itemids=tuple(VALIDATED_ITEMIDS["troponin_t"])),
    _fs(name="troponin_i", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Refuted for ICU use", use="exclude", notes="No meaningful ICU coverage in v3.1."),
    _fs(name="crp", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-validated low coverage", use="optional only", itemids=tuple(VALIDATED_ITEMIDS["crp"])),
    _fs(name="hs_crp", domain="labs", source="labevents", acquisition_mode="Direct", validation_status="Direct-validated as hs-CRP", use="optional only", itemids=tuple(VALIDATED_ITEMIDS["hs_crp"])),
    _fs(name="procalcitonin", domain="labs", source="d_labitems", acquisition_mode="N/A", validation_status="Refuted", use="exclude", notes="No confirmed standard item in MIMIC-IV v3.1."),
    # Ventilation
    _fs(name="fio2", domain="respiratory", source="ventilator_setting / chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="peep", domain="respiratory", source="ventilator_setting / chartevents", acquisition_mode="Direct", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="tidal_volume", domain="respiratory", source="ventilator_setting", acquisition_mode="Direct", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="tidal_volume_set", domain="respiratory", source="chartevents / ventilator settings", acquisition_mode="Direct", validation_status="Planned source-confirmed", use="ARF"),
    _fs(name="minute_volume", domain="respiratory", source="ventilator_setting", acquisition_mode="Direct", validation_status="Direct-existing", use="ARF", backbone=True),
    _fs(name="plateau_pressure", domain="respiratory", source="chartevents", acquisition_mode="Direct", validation_status="Direct-validated", use="ARF", itemids=tuple(VALIDATED_ITEMIDS["plateau_pressure"]), backbone=True),
    _fs(name="ventilation_status", domain="respiratory", source="derived.ventilation", acquisition_mode="Derived", validation_status="Direct-validated", use="ARF/stroke-proxy", backbone=True),
    # Medication and interventions
    _fs(name="vasopressors_active", domain="medications", source="inputevents", acquisition_mode="Direct", validation_status="Direct-validated", use="shock/AKI", backbone=True),
    _fs(name="vasopressor_dose_norepi_equiv", domain="medications", source="inputevents conversion", acquisition_mode="Derived", validation_status="Direct-existing", use="hemodynamic burden", backbone=True),
    _fs(name="propofol_rate", domain="medications", source="inputevents / emar", acquisition_mode="Direct", validation_status="Direct-existing", use="delirium/ventilation", backbone=True),
    _fs(name="midazolam_rate", domain="medications", source="inputevents / emar", acquisition_mode="Direct", validation_status="Direct-existing", use="delirium/ventilation", backbone=True),
    _fs(name="fentanyl_rate", domain="medications", source="inputevents / emar", acquisition_mode="Direct", validation_status="Direct-existing", use="analgesia/sedation", backbone=True),
    _fs(name="dexmedetomidine_active", domain="medications", source="prescriptions/emar/inputevents", acquisition_mode="Direct", validation_status="Needs final source audit", use="delirium"),
    _fs(name="antibiotic_active", domain="medications", source="prescriptions + emar", acquisition_mode="Derived event logic", validation_status="Direct-validated coverage basis", use="sepsis"),
    _fs(name="nephrotoxic_drug_active", domain="medications", source="prescriptions + emar", acquisition_mode="Derived event logic", validation_status="Direct-validated source feasibility", use="AKI"),
    _fs(name="fluid_input_hourly", domain="medications", source="inputevents", acquisition_mode="Derived", validation_status="Direct-existing", use="fluid balance", backbone=True),
    _fs(name="fluid_balance", domain="medications", source="input minus output", acquisition_mode="Derived", validation_status="Direct-existing", use="AKI/shock", backbone=True),
    _fs(name="rrt_active", domain="procedures", source="procedureevents", acquisition_mode="Direct/Derived flag", validation_status="Direct-existing + validated search logic", use="AKI", backbone=True),
    _fs(name="intubation_event", domain="procedures", source="procedureevents", acquisition_mode="Direct", validation_status="Direct-validated", use="ARF/stroke-proxy", itemids=tuple(VALIDATED_ITEMIDS["intubation"])),
    _fs(name="extubation_event", domain="procedures", source="procedureevents", acquisition_mode="Direct", validation_status="Direct-validated", use="ARF", itemids=tuple(VALIDATED_ITEMIDS["extubation"])),
    _fs(name="unplanned_extubation_event", domain="procedures", source="procedureevents", acquisition_mode="Direct", validation_status="Direct-validated", use="ARF", itemids=tuple(VALIDATED_ITEMIDS["unplanned_extubation"])),
    _fs(name="tracheostomy", domain="procedures", source="procedureevents / ventilation", acquisition_mode="Direct", validation_status="Needs final audit", use="optional respiratory feature"),
    # Derived scores and pathways
    _fs(name="sofa_total", domain="derived_scores", source="derived.sofa", acquisition_mode="Derived", validation_status="Direct-existing", use="severity", backbone=True),
    _fs(name="sofa_subcomponents", domain="derived_scores", source="derived.sofa", acquisition_mode="Derived", validation_status="Direct-existing", use="organ failure"),
    _fs(name="kdigo_stage", domain="derived_scores", source="derived.kdigo_stages", acquisition_mode="Derived", validation_status="Direct-existing", use="AKI", backbone=True),
    _fs(name="qsofa", domain="derived_scores", source="RR + SBP + GCS", acquisition_mode="Derived", validation_status="Planned", use="sepsis severity"),
    _fs(name="apache_ii", domain="derived_scores", source="multiple components", acquisition_mode="Derived", validation_status="Needs rule design", use="optional severity score"),
    _fs(name="delirium_onset", domain="pathway_objects", source="first Positive 228332 with censoring rule", acquisition_mode="Proxy / inferred", validation_status="Needs rule design", use="delirium task"),
    _fs(name="delirium_resolution", domain="pathway_objects", source="sustained non-positive period", acquisition_mode="Proxy / inferred", validation_status="Needs rule design", use="delirium task"),
    _fs(name="sepsis_onset", domain="pathway_objects", source="derived.sepsis3", acquisition_mode="Derived", validation_status="Direct-existing", use="sepsis task"),
    _fs(name="shock_onset", domain="pathway_objects", source="project rule + reference rule", acquisition_mode="Proxy / inferred", validation_status="Needs rule design", use="sepsis/shock task"),
    _fs(name="arf_onset", domain="pathway_objects", source="PF < 300 and/or invasive ventilation start", acquisition_mode="Derived / proxy", validation_status="Direct-validated feasibility", use="ARF task"),
    _fs(name="stroke_proxy_deterioration_markers", domain="pathway_objects", source="GCS / RASS / vent / restraint / sedation pattern", acquisition_mode="Proxy / inferred", validation_status="Needs task design", use="stroke-proxy reasoning"),
    _fs(name="diagnosis_pathway_events", domain="pathway_objects", source="threshold crossings + procedure time + note mention + derived onset tables", acquisition_mode="Derived / proxy", validation_status="Needs explicit schema", use="condition-aware pathways", backbone=True),
    # Text and multimodal
    _fs(name="raw_notes_0_168h", domain="text_objects", source="noteevents + lab comments", acquisition_mode="Direct", validation_status="existing logic, needs window extension", use="B2"),
    _fs(name="note_timestamp", domain="text_objects", source="charttime / relative hour", acquisition_mode="Direct", validation_status="Direct-existing", use="all text alignment", backbone=True),
    _fs(name="note_type", domain="text_objects", source="note metadata", acquisition_mode="Direct", validation_status="Direct-existing", use="typed reasoning", backbone=True),
    _fs(name="sentence_level_doctime_rel_tags", domain="text_objects", source="LLM/classifier pipeline", acquisition_mode="Derived", validation_status="Direct-existing", use="clean text"),
    _fs(name="clean_note_text", domain="text_objects", source="raw note minus AFTER sentences", acquisition_mode="Derived", validation_status="Planned for B2", use="leakage control"),
    _fs(name="original_note_text", domain="text_objects", source="raw note unchanged", acquisition_mode="Direct", validation_status="Direct-existing", use="original/leaked B2"),
    _fs(name="note_level_embedding", domain="text_objects", source="ClinicalBERT or equivalent", acquisition_mode="Derived", validation_status="Direct-existing", use="optional baseline / retrieval"),
    _fs(name="note_sequence_object", domain="text_objects", source="ordered note list with timestamps and type", acquisition_mode="Derived", validation_status="Planned", use="B2 / CRES", backbone=True),
]


def feature_records() -> list[dict[str, Any]]:
    return [asdict(spec) for spec in FEATURE_SPECS]


def grouped_feature_records() -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for spec in FEATURE_SPECS:
        grouped.setdefault(spec.domain, []).append(asdict(spec))
    return grouped
