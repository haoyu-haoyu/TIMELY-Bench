from __future__ import annotations

import re
from typing import Iterable


SUPPORT_LEVELS = ("core_executable", "proxy_executable", "reference_only")
IMPLEMENTATION_MODES = (
    "direct_feature",
    "derived_feature",
    "event_rule",
    "note_proxy",
    "not_implemented",
)

CORE_REQUIREMENTS = {
    "aki": {
        "creatinine",
        "urineoutput",
        "kdigo_stage",
        "map_merged",
        "nephrotoxic_drug_active",
        "rrt_active",
    },
    "delirium": {
        "delirium_assessment",
        "rass",
        "gcs_total",
        "sedation_burden",
        "restraint_active",
    },
    "stroke_proxy": {
        "gcs_total",
        "rass",
        "map_merged",
        "ventilation_status",
        "sedation_burden",
        "restraint_active",
        "stroke_cohort_membership",
    },
}


REFERENCE_ONLY_TERMS = (
    "nihss",
    "mrs",
    "cta",
    "ct perfusion",
    "perfusion",
    "mri",
    "mra",
    "nephrocheck",
    "ngal",
    "kim-1",
    "kim1",
    "igfbp7",
    "timp-2",
    "cystatin c",
    "folate",
    "b12",
    "tsh",
    "cortisol",
    "ammonia",
    "lipids",
    "hba1c",
)

NOTE_PROXY_TERMS = (
    "hallucination",
    "delusion",
    "sleep-wake",
    "sleep wake",
    "encephalopathy",
    "nausea",
    "vomiting",
    "aphasia",
    "dysarthria",
    "neglect",
    "hemianopia",
)


NODE_ID_OVERRIDES: dict[str, dict[str, tuple[str, str, str, str]]] = {
    "aki": {
        "scr": ("creatinine", "core_executable", "direct_feature", "Primary AKI functional marker."),
        "bun": ("bun", "core_executable", "direct_feature", "Renal burden marker."),
        "potassium": ("potassium", "core_executable", "direct_feature", "Electrolyte complication signal."),
        "bicarbonate": ("bicarbonate", "core_executable", "direct_feature", "Metabolic acidosis support."),
        "phosphate": ("phosphate", "proxy_executable", "direct_feature", "Secondary renal/metabolic marker."),
        "lactate": ("lactate", "proxy_executable", "direct_feature", "Perfusion/sepsis context."),
        "calcium": ("calcium", "proxy_executable", "direct_feature", "Secondary metabolic marker."),
        "urine_output": ("urineoutput", "core_executable", "direct_feature", "Core AKI urine signal."),
        "hypotension": ("sbp", "proxy_executable", "direct_feature", "Hypotension proxied by SBP."),
        "map_low": ("map_merged", "core_executable", "derived_feature", "Low MAP derived from merged MAP."),
        "tachycardia": ("heart_rate", "proxy_executable", "direct_feature", "Tachycardia from hourly HR."),
        "fever": ("temperature_c", "proxy_executable", "direct_feature", "Temperature-derived fever."),
        "low_spo2": ("spo2", "proxy_executable", "direct_feature", "Hypoxia proxy from SpO2."),
        "oliguria": ("urineoutput", "core_executable", "derived_feature", "Derived from hourly urine output."),
        "anuria": ("urineoutput", "proxy_executable", "derived_feature", "Derived severe urine reduction."),
        "fluid_overload": ("fluid_balance", "proxy_executable", "derived_feature", "Positive balance proxy."),
        "met_acidosis": ("ph", "proxy_executable", "derived_feature", "Derived from pH/bicarbonate/anion gap."),
        "nephrotoxic_nsaid": ("nephrotoxic_drug_active", "proxy_executable", "event_rule", "Medication class rule on timeline."),
        "nephrotoxic_amino": ("nephrotoxic_drug_active", "proxy_executable", "event_rule", "Medication class rule on timeline."),
        "vancomycin": ("nephrotoxic_drug_active", "proxy_executable", "event_rule", "Medication class rule on timeline."),
        "amphotericin": ("nephrotoxic_drug_active", "proxy_executable", "event_rule", "Medication class rule on timeline."),
        "contrast": ("contrast_exposure", "proxy_executable", "event_rule", "Procedure/medication exposure rule."),
        "vasopressors": ("vasopressors_active", "core_executable", "direct_feature", "Hemodynamic support flag."),
        "diuretics": ("diuretic_exposure", "proxy_executable", "event_rule", "Medication exposure rule."),
        "ace_arb": ("ace_arb_exposure", "proxy_executable", "event_rule", "Medication exposure rule."),
        "fluids_iv": ("fluid_input_hourly", "core_executable", "direct_feature", "IV fluid burden."),
        "rrt": ("rrt_active", "core_executable", "direct_feature", "RRT status."),
        "ckd": ("ckd", "core_executable", "direct_feature", "Static comorbidity."),
        "diabetes": ("diabetes", "proxy_executable", "direct_feature", "Static comorbidity."),
        "hypertension": ("hypertension", "proxy_executable", "direct_feature", "Static comorbidity."),
        "liver_disease": ("liver_disease", "proxy_executable", "direct_feature", "Static comorbidity."),
        "heart_failure": ("heart_failure_context", "reference_only", "not_implemented", "Not in current v3 static context."),
        "cardiac_surgery": ("cardiac_surgery_context", "reference_only", "not_implemented", "No explicit current feature."),
    },
    "delirium": {
        "sodium_d": ("sodium", "proxy_executable", "direct_feature", "Metabolic precipitant."),
        "glucose_d": ("glucose_merged", "proxy_executable", "direct_feature", "Metabolic precipitant."),
        "calcium_d": ("calcium", "proxy_executable", "direct_feature", "Metabolic precipitant."),
        "crp_wbc": ("wbc", "proxy_executable", "derived_feature", "Use WBC backbone and optional CRP."),
        "urea_cr_d": ("creatinine", "proxy_executable", "derived_feature", "Use creatinine+BUN renal burden."),
        "lfts_d": ("bilirubin_total", "proxy_executable", "derived_feature", "Use bilirubin as current liver proxy."),
        "pao2_d": ("pao2", "proxy_executable", "derived_feature", "Use PaO2/PaCO2 blood-gas support."),
        "albumin_d": ("albumin", "proxy_executable", "direct_feature", "Albumin available, medium coverage."),
        "fever_d": ("temperature_c", "proxy_executable", "direct_feature", "Temperature-derived fever."),
        "tachy_d": ("heart_rate", "proxy_executable", "direct_feature", "Autonomic instability proxy."),
        "hypoxia_d": ("spo2", "proxy_executable", "direct_feature", "Hypoxia proxy."),
        "hypotension_d": ("map_merged", "proxy_executable", "direct_feature", "Hemodynamic instability."),
        "rr_abnormal": ("resp_rate", "proxy_executable", "direct_feature", "Abnormal respiratory rate."),
        "inattention": ("delirium_assessment", "proxy_executable", "derived_feature", "Approximate from CAM positivity/components."),
        "fluctuating": ("rass", "proxy_executable", "derived_feature", "RASS/GCS fluctuation proxy."),
        "disorganized": ("delirium_assessment", "proxy_executable", "derived_feature", "Approximate from CAM positivity/components."),
        "altered_loc": ("rass", "core_executable", "derived_feature", "RASS deviation proxy."),
        "hyperactive": ("rass", "proxy_executable", "derived_feature", "Positive RASS range."),
        "hypoactive": ("rass", "proxy_executable", "derived_feature", "Negative RASS range."),
        "mixed_motor": ("rass", "proxy_executable", "derived_feature", "RASS fluctuation proxy."),
        "sleep_wake": ("sleep_wake_note_proxy", "proxy_executable", "note_proxy", "Only notes can support this."),
        "hallucinations": ("hallucination_note_proxy", "proxy_executable", "note_proxy", "Only notes can support this."),
        "agitation": ("rass", "proxy_executable", "derived_feature", "Positive RASS range."),
        "benzo": ("sedation_burden", "core_executable", "derived_feature", "Proxy via midazolam-driven sedation burden."),
        "opioids_d": ("sedation_burden", "proxy_executable", "derived_feature", "Proxy via fentanyl contribution."),
        "dexmedetomidine": ("sedation_burden", "proxy_executable", "event_rule", "Medication timeline pattern."),
        "propofol_d": ("sedation_burden", "core_executable", "derived_feature", "Propofol contributes to sedation burden."),
        "restraints_d": ("restraint_active", "core_executable", "direct_feature", "Restraint status from hourly features."),
        "infection_d": ("diagnosis_pathway::sepsis_present_proxy", "proxy_executable", "event_rule", "Diagnosis pathway proxy."),
        "dementia_d": ("cognitive_impairment_context", "proxy_executable", "event_rule", "Diagnosis pathway/static context proxy."),
    },
    "stroke": {
        "glucose_s": ("glucose_merged", "proxy_executable", "direct_feature", "Metabolic support."),
        "inr_pt": ("inr", "proxy_executable", "derived_feature", "Use INR/PT coagulation proxy."),
        "platelets_s": ("platelet", "proxy_executable", "direct_feature", "Coagulation support."),
        "troponin_s": ("troponin_t", "proxy_executable", "direct_feature", "Cardiac stress proxy."),
        "crp_s": ("crp", "proxy_executable", "direct_feature", "Inflammation, low coverage."),
        "hemoglobin_s": ("hemoglobin", "proxy_executable", "direct_feature", "Anemia burden."),
        "creatinine_s": ("creatinine", "proxy_executable", "direct_feature", "Renal comorbidity burden."),
        "hypertension_s": ("sbp", "proxy_executable", "direct_feature", "Blood pressure support."),
        "tachy_s": ("heart_rate", "proxy_executable", "direct_feature", "Autonomic burden."),
        "spo2_s": ("spo2", "proxy_executable", "direct_feature", "Oxygenation proxy."),
        "bp_post_evt": ("map_merged", "proxy_executable", "direct_feature", "Hemodynamic stability proxy."),
        "gcs_s": ("gcs_total", "core_executable", "direct_feature", "Neurologic proxy backbone."),
        "rass_s": ("rass", "core_executable", "direct_feature", "Level-of-consciousness proxy."),
        "ventilation_s": ("ventilation_status", "core_executable", "direct_feature", "Support burden."),
        "vasopressors_s": ("vasopressors_active", "proxy_executable", "direct_feature", "Support burden."),
        "sedation_s": ("sedation_burden", "core_executable", "derived_feature", "Sedation intensification proxy."),
        "restraint_s": ("restraint_active", "core_executable", "direct_feature", "Safety/behavior proxy."),
        "stroke_cohort": ("stroke_cohort_membership", "core_executable", "event_rule", "Diagnosis pathway membership."),
    },
}


def normalize_text(text: str | None) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("↑", " ").replace("↓", " ")
    text = text.replace("⁺", "").replace("²", "2")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def node_override(condition: str, node_id: str) -> tuple[str, str, str, str] | None:
    return NODE_ID_OVERRIDES.get(condition, {}).get(node_id)


def has_reference_only_term(*texts: str) -> bool:
    blob = normalize_text(" ".join(texts))
    return any(term in blob for term in REFERENCE_ONLY_TERMS)


def has_note_proxy_term(*texts: str) -> bool:
    blob = normalize_text(" ".join(texts))
    return any(term in blob for term in NOTE_PROXY_TERMS)


def _contains(blob: str, terms: Iterable[str]) -> bool:
    return any(term in blob for term in terms)


def heuristic_mapping(condition: str, node_id: str, label: str, source_raw: str) -> tuple[str, str, str, str]:
    blob = normalize_text(f"{node_id} {label} {source_raw}")

    if has_reference_only_term(node_id, label, source_raw):
        return ("reference_only_signal", "reference_only", "not_implemented", "Requires unsupported imaging/research assay or non-MIMIC-standard measurement.")
    if has_note_proxy_term(node_id, label, source_raw):
        return ("note_symptom_proxy", "proxy_executable", "note_proxy", "Only available as note-level proxy in B2.")

    if _contains(blob, ["creatininine"]):  # typo guard
        blob = blob.replace("creatininine", "creatinine")

    rules = [
        (("creatinine", "scr"), ("creatinine", "core_executable", "direct_feature", "Creatinine-backed renal signal.")),
        (("bun", "urea"), ("bun", "proxy_executable", "direct_feature", "BUN renal burden signal.")),
        (("potassium", "hyperkala"), ("potassium", "proxy_executable", "direct_feature", "Electrolyte signal.")),
        (("bicarbonate", "hco3", "acidosis"), ("bicarbonate", "proxy_executable", "derived_feature", "Acid-base proxy.")),
        (("phosphate",), ("phosphate", "proxy_executable", "direct_feature", "Secondary renal/metabolic marker.")),
        (("lactate",), ("lactate", "proxy_executable", "direct_feature", "Perfusion signal.")),
        (("calcium",), ("calcium", "proxy_executable", "direct_feature", "Metabolic signal.")),
        (("urine output", "oliguria", "anuria"), ("urineoutput", "core_executable", "derived_feature", "Hourly urine-output derived signal.")),
        (("map",), ("map_merged", "core_executable", "derived_feature", "Merged MAP signal.")),
        (("hypotension", "sbp"), ("sbp", "proxy_executable", "direct_feature", "Blood-pressure proxy.")),
        (("tachy", "heart rate"), ("heart_rate", "proxy_executable", "direct_feature", "Heart-rate proxy.")),
        (("spo2", "hypoxia"), ("spo2", "proxy_executable", "direct_feature", "Oxygenation proxy.")),
        (("resp rate", "rr "), ("resp_rate", "proxy_executable", "direct_feature", "Respiratory-rate proxy.")),
        (("temperature", "fever", "hypotherm"), ("temperature_c", "proxy_executable", "direct_feature", "Temperature proxy.")),
        (("sodium",), ("sodium", "proxy_executable", "direct_feature", "Electrolyte signal.")),
        (("glucose",), ("glucose_merged", "proxy_executable", "derived_feature", "Merged glucose signal.")),
        (("wbc",), ("wbc", "proxy_executable", "direct_feature", "Inflammation proxy.")),
        (("albumin",), ("albumin", "proxy_executable", "direct_feature", "Nutritional/inflammatory proxy.")),
        (("pao2",), ("pao2", "proxy_executable", "direct_feature", "ABG oxygenation proxy.")),
        (("paco2",), ("paco2", "proxy_executable", "direct_feature", "ABG ventilation proxy.")),
        (("peep",), ("peep", "proxy_executable", "direct_feature", "Ventilator setting.")),
        (("fio2",), ("fio2", "proxy_executable", "direct_feature", "Ventilator setting.")),
        (("plateau",), ("plateau_pressure", "proxy_executable", "direct_feature", "Ventilator mechanics proxy.")),
        (("ventilation", "intubation", "extubation"), ("ventilation_status", "core_executable", "derived_feature", "Respiratory support proxy.")),
        (("rass", "agitation", "consciousness", "loc"), ("rass", "core_executable", "derived_feature", "RASS-backed neurobehavioral proxy.")),
        (("gcs", "attention"), ("gcs_total", "core_executable", "derived_feature", "GCS-backed neurologic proxy.")),
        (("delirium", "cam"), ("delirium_assessment", "core_executable", "derived_feature", "Delirium/CAM assessment proxy.")),
        (("restraint",), ("restraint_active", "core_executable", "direct_feature", "Hourly restraint signal.")),
        (("sedat", "benzodiazep", "opioid", "propofol", "midazolam", "dexmedetomidine"), ("sedation_burden", "proxy_executable", "derived_feature", "Medication burden proxy via sedation features/timeline.")),
        (("vasopressor", "norepinephrine"), ("vasopressors_active", "proxy_executable", "direct_feature", "Vasopressor support.")),
        (("rrt",), ("rrt_active", "core_executable", "direct_feature", "Renal replacement therapy status.")),
        (("fluid",), ("fluid_balance", "proxy_executable", "derived_feature", "Fluid balance proxy.")),
        (("ckd",), ("ckd", "proxy_executable", "direct_feature", "Static comorbidity.")),
        (("diabetes",), ("diabetes", "proxy_executable", "direct_feature", "Static comorbidity.")),
        (("hypertension",), ("hypertension", "proxy_executable", "direct_feature", "Static comorbidity.")),
        (("dementia", "cognitive"), ("cognitive_impairment_context", "proxy_executable", "event_rule", "Static/pathway proxy.")),
        (("stroke", "cva"), ("stroke_cohort_membership", "core_executable", "event_rule", "Stroke cohort/pathway proxy.")),
    ]
    for terms, result in rules:
        if _contains(blob, terms):
            return result
    return ("unmapped_reference", "reference_only", "not_implemented", "No safe executable mapping was found.")

