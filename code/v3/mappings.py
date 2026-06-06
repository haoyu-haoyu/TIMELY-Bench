from __future__ import annotations

from typing import Dict, Iterable


VALIDATED_ITEMIDS = {
    "map_invasive": [220052],
    "map_noninvasive": [220181],
    "temperature_f": [223761],
    "temperature_c": [223762],
    "gcs_eye": [220739],
    "gcs_verbal": [223900],
    "gcs_motor": [223901],
    "rass": [228096],
    "delirium_assessment": [228332],
    "cam_components": [228334, 228337, 229325, 229326],
    "plateau_pressure": [224696],
    "serum_glucose": [50931],
    "bedside_glucose": [220621],
    "troponin_t": [51003],
    "albumin": [50862],
    "crp": [50889],
    "hs_crp": [51652],
    "intubation": [224385],
    "extubation": [227194],
    "unplanned_extubation": [225468, 225477],
    "urine_output": [226559, 226560, 226561, 226563, 226564, 226565, 226566, 226567, 226627, 226631, 227489],
}

RESTRAINT_KEYWORDS = [
    "restraint",
    "restraints ordered",
    "violent restraints",
    "non-violent restraints",
]

COMORBIDITY_PREFIX_RULES: Dict[str, tuple[str, ...]] = {
    "ckd": ("N18", "585"),
    "diabetes": ("E10", "E11", "250"),
    "hypertension": ("I10", "I11", "I12", "I13", "401", "402", "403", "404", "405"),
    "dementia": ("F01", "F02", "F03", "G30", "G31", "290", "331"),
    "copd": ("J44", "491", "492", "496"),
    "atrial_fibrillation": ("I48", "42731", "42732"),
    "liver_disease": ("K70", "K71", "K72", "K73", "K74", "571"),
    "stroke_family": ("I60", "I61", "I62", "I63", "I64", "G45", "430", "431", "432", "433", "434", "435", "436"),
}

CORE_BACKBONE_FEATURES = [
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "map_merged",
    "resp_rate",
    "spo2",
    "temperature_c",
    "glucose_merged",
    "albumin",
    "bun",
    "creatinine",
    "sodium",
    "potassium",
    "bicarbonate",
    "chloride",
    "aniongap",
    "wbc",
    "hemoglobin",
    "hematocrit",
    "platelet",
    "lactate",
    "ph",
    "gcs_eye",
    "gcs_verbal",
    "gcs_motor",
    "gcs_total",
    "rass",
    "delirium_assessment",
    "urineoutput",
    "bilirubin_total",
    "vasopressors_active",
    "vasopressor_dose_norepi_equiv",
    "rrt_active",
    "fio2",
    "peep",
    "tidal_volume",
    "minute_volume",
    "plateau_pressure",
    "pao2",
    "paco2",
    "pao2_fio2_ratio",
    "ventilation_status",
    "sofa_total",
    "sofa_respiration",
    "propofol_rate",
    "midazolam_rate",
    "fentanyl_rate",
    "fluid_input_hourly",
    "fluid_balance",
]


def parse_icd_codes(raw_codes: str | float | None) -> list[str]:
    if raw_codes is None:
        return []
    if isinstance(raw_codes, float):
        if raw_codes != raw_codes:
            return []
        raw_codes = str(raw_codes)
    return [part.strip().upper() for part in str(raw_codes).split(",") if part.strip()]


def match_prefixes(codes: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    codes = [code.upper() for code in codes]
    for name, prefixes in COMORBIDITY_PREFIX_RULES.items():
        out[name] = int(any(code.startswith(prefix) for code in codes for prefix in prefixes))
    return out

