#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stroke_text_strategy_audit import (  # type: ignore
    BRAIN_RAD_FILTER,
    NEURO_NOTE_KEYWORDS,
    analyze_structured_coverage,
    describe_numeric,
    iter_source_chunks,
    normalize_space,
    parse_sections,
    pct,
    safe_ratio,
)


FAMILY_PRIORITY = [
    "ischaemic",
    "hemorrhagic_sah",
    "hemorrhagic_ich",
    "tia",
    "other_cerebrovascular",
]
FAMILY_PRIORITY_INDEX = {name: idx for idx, name in enumerate(FAMILY_PRIORITY)}
FAMILY_DISPLAY = {
    "ischaemic": "Ischaemic stroke",
    "hemorrhagic_sah": "Hemorrhagic stroke / SAH",
    "hemorrhagic_ich": "Hemorrhagic stroke / ICH-or-other-intracranial-hemorrhage",
    "tia": "Transient ischaemic attack",
    "other_cerebrovascular": "Other / unspecified cerebrovascular stroke-coded stay",
    "mixed": "Mixed stroke-family coding",
}
SUBTYPE_PATTERNS: Sequence[tuple[str, Sequence[str]]] = (
    ("hemorrhagic_sah", ("I60", "430")),
    ("hemorrhagic_ich", ("I61", "I62", "431", "432")),
    ("ischaemic", ("I63", "433", "434", "436")),
    ("tia", ("G45", "435")),
    ("other_cerebrovascular", ("I64",)),
)
NEURO_INCREMENTAL_PATTERNS = (
    "strength",
    "speech",
    "orientation",
    "commands response",
)
GCS_PATTERNS = ("gcs", "glasgow")
DEFAULT_OUTPUT_COLUMNS = [
    "stay_id",
    "hadm_id",
    "assignment_source",
    "priority_subtype",
    "mixed_subtype",
    "mixed_conflict",
    "family_count",
    "families_present",
    "priority_basis_seq_num",
    "priority_basis_icd_code",
    "primary_icd_code",
    "primary_icd_family",
    "stroke_code_details",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1+2 stroke subtype reconciliation and ischaemic subset reaudit.")
    p.add_argument("--cohort-v3", required=True)
    p.add_argument("--diagnoses-parquet", required=True)
    p.add_argument("--discharge-source", required=True)
    p.add_argument("--nursing-source", required=True)
    p.add_argument("--radiology-source", required=True)
    p.add_argument("--lab-comments-source", default="")
    p.add_argument("--gcs-hourly", required=True)
    p.add_argument("--hourly-grid-dir", required=True)
    p.add_argument("--out-subtype-json", required=True)
    p.add_argument("--out-ischaemic-json", required=True)
    p.add_argument("--out-markdown", required=True)
    p.add_argument("--out-assignments-csv", required=True)
    return p.parse_args()


def classify_stroke_family(icd_code: object) -> Optional[str]:
    if icd_code is None or (isinstance(icd_code, float) and pd.isna(icd_code)):
        return None
    code = str(icd_code).strip().upper().replace(".", "")
    if not code:
        return None
    for family, prefixes in SUBTYPE_PATTERNS:
        if any(code.startswith(prefix) for prefix in prefixes):
            return family
    return None


def load_broad_stroke_cohort(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    available = header.columns.tolist()
    wanted = [
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "icu_intime",
        "icu_outtime",
        "los",
        "anchor_age",
        "gender",
        "label_mortality",
        "hospital_expire_flag",
        "has_stroke_final",
        "primary_icd_code",
        "icd_codes",
        "diagnoses_text",
    ]
    usecols = [c for c in wanted if c in available]
    frames: List[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=100_000):
        stroke_mask = pd.to_numeric(chunk.get("has_stroke_final"), errors="coerce").fillna(0).astype(int).eq(1)
        sub = chunk[stroke_mask].copy()
        if sub.empty:
            continue
        out = pd.DataFrame()
        out["subject_id"] = pd.to_numeric(sub.get("subject_id"), errors="coerce")
        out["hadm_id"] = pd.to_numeric(sub.get("hadm_id"), errors="coerce")
        out["stay_id"] = pd.to_numeric(sub.get("stay_id"), errors="coerce")
        intime_col = "icu_intime" if "icu_intime" in sub.columns else "intime"
        outtime_col = "icu_outtime" if "icu_outtime" in sub.columns else "outtime"
        out["intime"] = sub.get(intime_col)
        out["outtime"] = sub.get(outtime_col)
        out["los"] = pd.to_numeric(sub.get("los"), errors="coerce") if "los" in sub.columns else pd.Series([math.nan] * len(sub))
        out["hospital_expire_flag"] = pd.to_numeric(
            sub.get("hospital_expire_flag", sub.get("label_mortality")), errors="coerce"
        )
        out["gender"] = sub.get("gender", pd.Series([""] * len(sub)))
        out["age"] = pd.to_numeric(sub.get("anchor_age"), errors="coerce") if "anchor_age" in sub.columns else pd.Series([math.nan] * len(sub))
        out["primary_icd_code"] = sub.get("primary_icd_code", pd.Series([""] * len(sub))).fillna("").astype(str).str.upper().str.replace(".", "", regex=False)
        out["icd_codes"] = sub.get("icd_codes", pd.Series([""] * len(sub))).fillna("").astype(str)
        out["diagnoses_text"] = sub.get("diagnoses_text", pd.Series([""] * len(sub))).fillna("").astype(str)
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id"])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["subject_id", "hadm_id", "stay_id"]).copy()
    for col in ("subject_id", "hadm_id", "stay_id"):
        df[col] = df[col].astype(int)
    df["primary_icd_family"] = df["primary_icd_code"].apply(classify_stroke_family)
    df["primary_dx_flag"] = df["primary_icd_family"].eq("ischaemic").astype(int)
    return df.drop_duplicates(subset=["stay_id"]).sort_values("stay_id", kind="mergesort").reset_index(drop=True)


def load_diagnoses(path: Path, stroke_stays: set[int]) -> pd.DataFrame:
    df = pd.read_parquet(path, columns=["stay_id", "hadm_id", "seq_num", "icd_code", "icd_version"])
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = df["stay_id"].astype(int)
    df = df[df["stay_id"].isin(stroke_stays)].copy()
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    df["seq_num"] = pd.to_numeric(df["seq_num"], errors="coerce")
    df["icd_code"] = df["icd_code"].fillna("").astype(str).str.upper().str.replace(".", "", regex=False)
    df["family"] = df["icd_code"].apply(classify_stroke_family)
    return df


def count_family_rows(family_df: pd.DataFrame) -> List[dict]:
    out: List[dict] = []
    by_code = (
        family_df.groupby(["icd_version", "icd_code", "family"], dropna=False)
        .agg(
            n_stays=("stay_id", "nunique"),
            n_hadm=("hadm_id", "nunique"),
            min_seq_num=("seq_num", "min"),
        )
        .reset_index()
        .sort_values(["family", "n_stays", "icd_code"], ascending=[True, False, True], kind="mergesort")
    )
    for row in by_code.itertuples(index=False):
        out.append(
            {
                "icd_version": None if pd.isna(row.icd_version) else int(row.icd_version),
                "icd_code": row.icd_code,
                "family": row.family,
                "n_stays": int(row.n_stays),
                "n_hadm": int(row.n_hadm),
                "min_seq_num": None if pd.isna(row.min_seq_num) else int(row.min_seq_num),
            }
        )
    return out


def build_assignments(cohort_df: pd.DataFrame, dx_df: pd.DataFrame) -> pd.DataFrame:
    family_df = dx_df.dropna(subset=["family"]).copy()
    family_df = family_df.sort_values(
        ["stay_id", "seq_num", "icd_code"],
        kind="mergesort",
    )
    cohort_index = cohort_df.set_index("stay_id", drop=False)
    rows: List[dict] = []

    for stay_id, stay_dx in family_df.groupby("stay_id", sort=True):
        stay_dx = stay_dx.copy()
        stay_dx["family_priority"] = stay_dx["family"].map(FAMILY_PRIORITY_INDEX)
        stay_dx = stay_dx.sort_values(["seq_num", "family_priority", "icd_code"], kind="mergesort")
        first = stay_dx.iloc[0]
        families = sorted(stay_dx["family"].dropna().unique().tolist(), key=lambda x: FAMILY_PRIORITY_INDEX.get(x, 999))
        family_count = len(families)
        mixed_conflict = family_count > 1
        primary_subtype = str(first["family"])
        mixed_subtype = "mixed" if mixed_conflict else primary_subtype
        stroke_code_details = []
        for fam, fam_df in stay_dx.groupby("family", sort=False):
            codes = []
            for item in fam_df[["icd_code", "seq_num"]].drop_duplicates().itertuples(index=False):
                seq = None if pd.isna(item.seq_num) else int(item.seq_num)
                codes.append({"icd_code": item.icd_code, "seq_num": seq})
            stroke_code_details.append({"family": fam, "codes": codes})
        base = cohort_index.loc[int(stay_id)]
        primary_icd_code = str(base.get("primary_icd_code", "") or "").upper().replace(".", "")
        rows.append(
            {
                "stay_id": int(stay_id),
                "hadm_id": int(base["hadm_id"]),
                "subject_id": int(base["subject_id"]),
                "assignment_source": "diagnoses_parquet",
                "priority_subtype": primary_subtype,
                "mixed_subtype": mixed_subtype,
                "mixed_conflict": bool(mixed_conflict),
                "family_count": int(family_count),
                "families_present": "|".join(families),
                "priority_basis_seq_num": None if pd.isna(first["seq_num"]) else int(first["seq_num"]),
                "priority_basis_icd_code": str(first["icd_code"]),
                "primary_icd_code": primary_icd_code,
                "primary_icd_family": classify_stroke_family(primary_icd_code),
                "stroke_code_details": json.dumps(stroke_code_details, ensure_ascii=False),
            }
        )

    assigned = {int(row["stay_id"]) for row in rows}
    missing_df = cohort_df[~cohort_df["stay_id"].astype(int).isin(assigned)].copy()
    for row in missing_df.itertuples(index=False):
        primary_icd_code = str(getattr(row, "primary_icd_code", "") or "").upper().replace(".", "")
        primary_family = classify_stroke_family(primary_icd_code)
        codes = [
            str(code).strip().upper().replace(".", "")
            for code in str(getattr(row, "icd_codes", "") or "").split(",")
            if str(code).strip()
        ]
        family_to_codes: Dict[str, List[dict]] = {}
        for code in codes:
            fam = classify_stroke_family(code)
            if not fam:
                continue
            family_to_codes.setdefault(fam, []).append({"icd_code": code, "seq_num": None})
        families = sorted(family_to_codes.keys(), key=lambda x: FAMILY_PRIORITY_INDEX.get(x, 999))
        if primary_family and primary_family not in families:
            families = sorted([primary_family] + families, key=lambda x: FAMILY_PRIORITY_INDEX.get(x, 999))
            family_to_codes.setdefault(primary_family, [{"icd_code": primary_icd_code, "seq_num": None}])
        if not families:
            families = ["other_cerebrovascular"]
            if primary_icd_code:
                family_to_codes["other_cerebrovascular"] = [{"icd_code": primary_icd_code, "seq_num": None}]
        primary_subtype = primary_family or families[0]
        mixed_conflict = len(families) > 1
        mixed_subtype = "mixed" if mixed_conflict else primary_subtype
        stroke_code_details = [{"family": fam, "codes": family_to_codes.get(fam, [])} for fam in families]
        rows.append(
            {
                "stay_id": int(row.stay_id),
                "hadm_id": int(row.hadm_id),
                "subject_id": int(row.subject_id),
                "assignment_source": "cohort_icd_fallback",
                "priority_subtype": primary_subtype,
                "mixed_subtype": mixed_subtype,
                "mixed_conflict": bool(mixed_conflict),
                "family_count": int(len(families)),
                "families_present": "|".join(families),
                "priority_basis_seq_num": None,
                "priority_basis_icd_code": primary_icd_code or (codes[0] if codes else ""),
                "primary_icd_code": primary_icd_code,
                "primary_icd_family": primary_family,
                "stroke_code_details": json.dumps(stroke_code_details, ensure_ascii=False),
            }
        )

    out = pd.DataFrame(rows)
    out["primary_dx_flag"] = out["primary_icd_family"].eq("ischaemic").astype(int)
    return out.sort_values(["stay_id"], kind="mergesort").reset_index(drop=True)


def summarize_assignments(assignments: pd.DataFrame, family_rows: List[dict], cohort_df: pd.DataFrame) -> dict:
    broad_n_stays = int(cohort_df["stay_id"].nunique())
    broad_n_hadm = int(cohort_df["hadm_id"].nunique())
    priority_counts = (
        assignments.groupby("priority_subtype")
        .agg(n_stays=("stay_id", "nunique"), n_hadm=("hadm_id", "nunique"))
        .reset_index()
    )
    mixed_counts = (
        assignments.groupby("mixed_subtype")
        .agg(n_stays=("stay_id", "nunique"), n_hadm=("hadm_id", "nunique"))
        .reset_index()
    )
    conflict_df = assignments[assignments["mixed_conflict"]].copy()
    return {
        "broad_stroke_cohort": {"n_stays": broad_n_stays, "n_hadm": broad_n_hadm},
        "family_priority_definition": FAMILY_PRIORITY,
        "family_code_mapping": {
            fam: list(prefixes) for fam, prefixes in SUBTYPE_PATTERNS
        },
        "priority_based_counts": [
            {
                "subtype": row.priority_subtype,
                "display_name": FAMILY_DISPLAY.get(row.priority_subtype, row.priority_subtype),
                "n_stays": int(row.n_stays),
                "n_hadm": int(row.n_hadm),
                "pct_of_broad_stays": pct(int(row.n_stays), broad_n_stays),
            }
            for row in priority_counts.sort_values("n_stays", ascending=False).itertuples(index=False)
        ],
        "mixed_version_counts": [
            {
                "subtype": row.mixed_subtype,
                "display_name": FAMILY_DISPLAY.get(row.mixed_subtype, row.mixed_subtype),
                "n_stays": int(row.n_stays),
                "n_hadm": int(row.n_hadm),
                "pct_of_broad_stays": pct(int(row.n_stays), broad_n_stays),
            }
            for row in mixed_counts.sort_values("n_stays", ascending=False).itertuples(index=False)
        ],
        "conflict_stays": {
            "n_stays": int(conflict_df["stay_id"].nunique()),
            "n_hadm": int(conflict_df["hadm_id"].nunique()),
            "pct_of_broad_stays": pct(int(conflict_df["stay_id"].nunique()), broad_n_stays),
        },
        "assignment_sources": [
            {
                "source": str(source),
                "n_stays": int(count),
                "pct_of_broad_stays": pct(int(count), broad_n_stays),
            }
            for source, count in assignments["assignment_source"].value_counts().items()
        ],
        "icd_family_breakdown": family_rows,
    }


def _is_neuro_incremental_row(category: object, item_label: object) -> bool:
    text = normalize_space(f"{category} {item_label}").lower()
    if "neuro" not in text:
        return False
    if any(p in text for p in GCS_PATTERNS):
        return False
    return any(p in text for p in NEURO_INCREMENTAL_PATTERNS)


def _is_neuro_any_row(category: object, item_label: object, chart_text: object) -> bool:
    text = normalize_space(f"{category} {item_label} {chart_text}").lower()
    return any(kw in text for kw in NEURO_NOTE_KEYWORDS)


def load_best_discharge_rows(discharge_source: Path, hadm_ids: set[int]) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    usecols = ["stay_id", "subject_id", "hadm_id", "note_id", "note_seq", "charttime", "storetime", "hour_offset", "discharge_text", "text_length"]
    for chunk in iter_source_chunks(discharge_source, usecols=usecols, chunksize=10_000):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if chunk.empty:
            continue
        if "text_length" not in chunk.columns:
            chunk["text_length"] = chunk["discharge_text"].fillna("").astype(str).str.len()
        rows.append(chunk)
    if not rows:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(rows, ignore_index=True)
    df["note_seq"] = pd.to_numeric(df["note_seq"], errors="coerce")
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    df["storetime"] = pd.to_datetime(df["storetime"], errors="coerce")
    df = df.sort_values(["hadm_id", "note_seq", "charttime", "storetime"], ascending=[True, False, False, False], kind="mergesort")
    return df.drop_duplicates(subset=["hadm_id"], keep="first").reset_index(drop=True)


def analyze_discharge_full(discharge_df: pd.DataFrame) -> dict:
    if discharge_df.empty:
        return {
            "n_representative_discharge_docs": 0,
            "hpi_found_pct": 0.0,
            "hospital_course_found_pct": 0.0,
            "both_hpi_and_hospital_course_found_pct": 0.0,
            "physical_exam_found_pct": 0.0,
            "pertinent_results_found_pct": 0.0,
            "admission_text_nihss_mention_pct": 0.0,
            "admission_char_count": describe_numeric([]),
        }
    rows = []
    for row in discharge_df.itertuples(index=False):
        parsed = parse_sections(getattr(row, "discharge_text", ""))
        sections = parsed["sections"]
        admission_text = "\n".join(
            sections.get(key, "")
            for key in (
                "chief_complaint",
                "history_of_present_illness",
                "past_medical_history",
                "social_history",
                "family_history",
                "physical_exam",
                "medications_on_admission",
            )
            if sections.get(key)
        )
        admission_lower = admission_text.lower()
        rows.append(
            {
                "hadm_id": int(row.hadm_id),
                "has_hpi": "history_of_present_illness" in sections,
                "has_hospital_course": "hospital_course" in sections,
                "has_physical_exam": "physical_exam" in sections,
                "has_pertinent_results": "pertinent_results" in sections,
                "admission_mentions_nihss": ("nihss" in admission_lower or "nih stroke" in admission_lower),
                "admission_char_count": len(admission_text),
            }
        )
    section_df = pd.DataFrame(rows)
    return {
        "n_representative_discharge_docs": int(len(section_df)),
        "hpi_found_pct": pct(int(section_df["has_hpi"].sum()), len(section_df)),
        "hospital_course_found_pct": pct(int(section_df["has_hospital_course"].sum()), len(section_df)),
        "both_hpi_and_hospital_course_found_pct": pct(
            int((section_df["has_hpi"] & section_df["has_hospital_course"]).sum()), len(section_df)
        ),
        "physical_exam_found_pct": pct(int(section_df["has_physical_exam"].sum()), len(section_df)),
        "pertinent_results_found_pct": pct(int(section_df["has_pertinent_results"].sum()), len(section_df)),
        "admission_text_nihss_mention_pct": pct(int(section_df["admission_mentions_nihss"].sum()), len(section_df)),
        "admission_char_count": describe_numeric(section_df["admission_char_count"].tolist()),
    }


def analyze_nursing_subset(stay_ids: set[int], nursing_source: Path) -> dict:
    any_counts = Counter()
    incremental_counts = Counter()
    first_hour = {}
    bilateral_strength_stays = set()
    strength_categories_per_stay: Dict[int, set[str]] = {}
    category_counts = Counter()
    for chunk in iter_source_chunks(
        nursing_source,
        usecols=["stay_id", "hour_offset", "item_label", "category", "chart_text", "valuenum"],
        chunksize=100_000,
    ):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if chunk.empty:
            continue
        chunk["hour_offset"] = pd.to_numeric(chunk["hour_offset"], errors="coerce")
        for row in chunk.itertuples(index=False):
            stay_id = int(row.stay_id)
            label = normalize_space(getattr(row, "item_label", ""))
            category = normalize_space(getattr(row, "category", ""))
            hour = float(row.hour_offset) if not pd.isna(row.hour_offset) else math.nan
            if _is_neuro_any_row(category, label, getattr(row, "chart_text", "")):
                any_counts[stay_id] += 1
                if not pd.isna(hour):
                    prev = first_hour.get(stay_id)
                    first_hour[stay_id] = hour if prev is None else min(prev, hour)
            if _is_neuro_incremental_row(category, label):
                incremental_counts[stay_id] += 1
                category_counts[label or category] += 1
                lower_label = label.lower()
                if "strength" in lower_label:
                    strength_categories_per_stay.setdefault(stay_id, set()).add(lower_label)
    for stay_id, names in strength_categories_per_stay.items():
        has_left = any("l arm" in x or "l leg" in x or "lu strength" in x or "ll strength" in x for x in names)
        has_right = any("r arm" in x or "r leg" in x or "ru strength" in x or "rl strength" in x for x in names)
        if has_left and has_right:
            bilateral_strength_stays.add(stay_id)

    any_values = [any_counts.get(stay, 0) for stay in stay_ids]
    incremental_values = [incremental_counts.get(stay, 0) for stay in stay_ids]
    first_values = [first_hour.get(stay) for stay in stay_ids if stay in first_hour]
    return {
        "n_stays_with_any_neuro_obs": int(sum(1 for stay in stay_ids if any_counts.get(stay, 0) >= 1)),
        "pct_stays_with_any_neuro_obs": pct(sum(1 for stay in stay_ids if any_counts.get(stay, 0) >= 1), len(stay_ids)),
        "n_stays_with_any_incremental_neuro_obs": int(sum(1 for stay in stay_ids if incremental_counts.get(stay, 0) >= 1)),
        "pct_stays_with_any_incremental_neuro_obs": pct(sum(1 for stay in stay_ids if incremental_counts.get(stay, 0) >= 1), len(stay_ids)),
        "n_stays_with_ge10_incremental_neuro_obs": int(sum(1 for stay in stay_ids if incremental_counts.get(stay, 0) >= 10)),
        "pct_stays_with_ge10_incremental_neuro_obs": pct(sum(1 for stay in stay_ids if incremental_counts.get(stay, 0) >= 10), len(stay_ids)),
        "median_any_neuro_obs_per_stay": round(median(any_values), 2) if any_values else 0.0,
        "median_incremental_neuro_obs_per_stay": round(median(incremental_values), 2) if incremental_values else 0.0,
        "median_first_incremental_neuro_hour": round(median(first_values), 2) if first_values else None,
        "n_stays_with_bilateral_strength_data": int(len(bilateral_strength_stays)),
        "pct_stays_with_bilateral_strength_data": pct(len(bilateral_strength_stays), len(stay_ids)),
        "top_incremental_categories": [
            {"category": cat, "n_rows": int(count)}
            for cat, count in category_counts.most_common(20)
        ],
        "per_stay_any_counts": any_counts,
        "per_stay_incremental_counts": incremental_counts,
    }


def analyze_radiology_subset(stay_ids: set[int], radiology_source: Path) -> dict:
    any_counts = Counter()
    first24_counts = Counter()
    first_hours = {}
    for chunk in iter_source_chunks(
        radiology_source,
        usecols=["stay_id", "hour_offset", "radiology_text"],
        chunksize=50_000,
    ):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stay_ids)]
        if chunk.empty:
            continue
        chunk["hour_offset"] = pd.to_numeric(chunk["hour_offset"], errors="coerce")
        for row in chunk.itertuples(index=False):
            text = str(getattr(row, "radiology_text", "") or "")
            if not BRAIN_RAD_FILTER.search(text):
                continue
            stay_id = int(row.stay_id)
            any_counts[stay_id] += 1
            hour = float(row.hour_offset) if not pd.isna(row.hour_offset) else math.nan
            if not pd.isna(hour):
                prev = first_hours.get(stay_id)
                first_hours[stay_id] = hour if prev is None else min(prev, hour)
                if hour < 24:
                    first24_counts[stay_id] += 1
    first_values = [first_hours.get(stay) for stay in stay_ids if stay in first_hours]
    return {
        "n_stays_with_any_brain_radiology": int(sum(1 for stay in stay_ids if any_counts.get(stay, 0) >= 1)),
        "pct_stays_with_any_brain_radiology": pct(sum(1 for stay in stay_ids if any_counts.get(stay, 0) >= 1), len(stay_ids)),
        "n_stays_with_brain_radiology_first24h": int(sum(1 for stay in stay_ids if first24_counts.get(stay, 0) >= 1)),
        "pct_stays_with_brain_radiology_first24h": pct(sum(1 for stay in stay_ids if first24_counts.get(stay, 0) >= 1), len(stay_ids)),
        "median_first_brain_radiology_hour": round(median(first_values), 2) if first_values else None,
    }


def compute_tiers(base_df: pd.DataFrame, neuro_count_col: str) -> dict:
    df = base_df.copy()
    df["ge10"] = pd.to_numeric(df[neuro_count_col], errors="coerce").fillna(0).astype(int) >= 10
    df["has_discharge_summary"] = df["has_discharge_summary"].fillna(False).astype(bool)
    tier = []
    for row in df.itertuples(index=False):
        if row.has_discharge_summary and row.ge10:
            tier.append("A")
        elif (not row.has_discharge_summary) and row.ge10:
            tier.append("B")
        elif row.has_discharge_summary and (not row.ge10):
            tier.append("C")
        else:
            tier.append("D")
    df["tier"] = tier
    counts = (
        df.groupby("tier")
        .agg(n_stays=("stay_id", "nunique"), n_hadm=("hadm_id", "nunique"))
        .reset_index()
    )
    return {
        "tier_definition_neuro_count_column": neuro_count_col,
        "tiers": [
            {
                "tier": row.tier,
                "n_stays": int(row.n_stays),
                "n_hadm": int(row.n_hadm),
                "pct_of_subset_stays": pct(int(row.n_stays), len(df)),
            }
            for row in counts.sort_values("tier").itertuples(index=False)
        ],
        "layer1_eligible_stays": int(df[df["tier"].isin(["A", "B"])]["stay_id"].nunique()),
        "layer2_eligible_stays": int(df[df["has_discharge_summary"]]["stay_id"].nunique()),
        "per_stay_tiers": df[["stay_id", "tier"]].sort_values("stay_id", kind="mergesort"),
    }


def analyze_ischaemic_subset(
    cohort_df: pd.DataFrame,
    assignments: pd.DataFrame,
    discharge_source: Path,
    nursing_source: Path,
    radiology_source: Path,
    gcs_hourly: Path,
    hourly_grid_dir: Path,
) -> dict:
    merged = cohort_df.merge(assignments, on=["stay_id", "hadm_id", "subject_id"], how="inner", suffixes=("_cohort", "_assign"))
    if "primary_dx_flag_assign" in merged.columns:
        merged["primary_dx_flag_locked"] = pd.to_numeric(merged["primary_dx_flag_assign"], errors="coerce").fillna(0).astype(int)
    elif "primary_dx_flag" in merged.columns:
        merged["primary_dx_flag_locked"] = pd.to_numeric(merged["primary_dx_flag"], errors="coerce").fillna(0).astype(int)
    elif "primary_dx_flag_cohort" in merged.columns:
        merged["primary_dx_flag_locked"] = pd.to_numeric(merged["primary_dx_flag_cohort"], errors="coerce").fillna(0).astype(int)
    else:
        merged["primary_dx_flag_locked"] = 0

    def subset_summary(name: str, subset_df: pd.DataFrame) -> dict:
        stay_ids = set(subset_df["stay_id"].astype(int).tolist())
        hadm_ids = set(subset_df["hadm_id"].astype(int).tolist())
        discharge_df = load_best_discharge_rows(discharge_source, hadm_ids)
        nursing = analyze_nursing_subset(stay_ids, nursing_source)
        radiology = analyze_radiology_subset(stay_ids, radiology_source)
        structured = analyze_structured_coverage(stay_ids, gcs_hourly, hourly_grid_dir)
        stay_has_discharge = subset_df["hadm_id"].isin(set(discharge_df["hadm_id"].astype(int).tolist()))
        base = subset_df[["stay_id", "hadm_id", "primary_dx_flag_locked"]].copy()
        base = base.rename(columns={"primary_dx_flag_locked": "primary_dx_flag"})
        base["has_discharge_summary"] = stay_has_discharge.values
        base["n_any_neuro_obs"] = base["stay_id"].map(nursing["per_stay_any_counts"]).fillna(0).astype(int)
        base["n_incremental_neuro_obs"] = base["stay_id"].map(nursing["per_stay_incremental_counts"]).fillna(0).astype(int)
        tier_incremental = compute_tiers(base, "n_incremental_neuro_obs")
        tier_any = compute_tiers(base, "n_any_neuro_obs")
        discharge_metrics = analyze_discharge_full(discharge_df)
        conflict_stays = int(subset_df["mixed_conflict"].sum())
        return {
            "subset_name": name,
            "n_stays": int(subset_df["stay_id"].nunique()),
            "n_hadm": int(subset_df["hadm_id"].nunique()),
            "n_mixed_conflict_stays": conflict_stays,
            "pct_mixed_conflict_stays": pct(conflict_stays, int(subset_df["stay_id"].nunique())),
            "primary_dx_flag_pct": pct(int(subset_df["primary_dx_flag_locked"].sum()), len(subset_df)),
            "discharge_coverage": {
                "n_hadm_with_discharge": int(discharge_df["hadm_id"].nunique()),
                "pct_hadm_with_discharge": pct(int(discharge_df["hadm_id"].nunique()), int(subset_df["hadm_id"].nunique())),
                "n_stays_with_discharge": int(base["has_discharge_summary"].sum()),
                "pct_stays_with_discharge": pct(int(base["has_discharge_summary"].sum()), int(subset_df["stay_id"].nunique())),
                "sections": discharge_metrics,
            },
            "nursing_neuro": {k: v for k, v in nursing.items() if not k.startswith("per_stay_")},
            "brain_radiology": radiology,
            "structured_coverage": structured,
            "tiering_incremental_no_gcs": {k: v for k, v in tier_incremental.items() if k != "per_stay_tiers"},
            "tiering_any_neuro_including_gcs_like_rows": {k: v for k, v in tier_any.items() if k != "per_stay_tiers"},
        }

    priority_ischaemic = merged[merged["priority_subtype"].eq("ischaemic")].copy()
    pure_ischaemic = merged[merged["mixed_subtype"].eq("ischaemic")].copy()
    return {
        "gcs_dedup_rule": {
            "decision": "Nursing GCS rows are retained in raw extraction but excluded from incremental Layer 1 stroke features, tier counts, and model-facing neuro observation counts.",
            "structured_channel_keeps_gcs": True,
            "nursing_incremental_channel_excludes_gcs": True,
            "tier_counts_default_to_incremental_no_gcs": True,
        },
        "priority_based_ischaemic_subset": subset_summary("priority_based_ischaemic", priority_ischaemic),
        "pure_ischaemic_no_conflict_subset": subset_summary("pure_ischaemic_no_conflict", pure_ischaemic),
    }


def render_markdown(subtype_summary: dict, ischaemic_summary: dict) -> str:
    priority_rows = subtype_summary["priority_based_counts"]
    mixed_rows = subtype_summary["mixed_version_counts"]
    main = ischaemic_summary["priority_based_ischaemic_subset"]
    pure = ischaemic_summary["pure_ischaemic_no_conflict_subset"]

    def render_rows(rows: Sequence[dict], key: str = "subtype") -> List[str]:
        lines = []
        for row in rows:
            lines.append(
                f"| {row[key]} | {row['n_stays']} | {row['n_hadm']} | {row['pct_of_broad_stays']}% |"
            )
        return lines

    def render_tiers(rows: Sequence[dict]) -> List[str]:
        return [f"| {r['tier']} | {r['n_stays']} | {r['n_hadm']} | {r['pct_of_subset_stays']}% |" for r in rows]

    lines = [
        "# Stroke Phase 1+2 Reaudit",
        "",
        "## Subtype reconciliation",
        "",
        f"- Broad fresh stroke cohort: `{subtype_summary['broad_stroke_cohort']['n_stays']}` stays / `{subtype_summary['broad_stroke_cohort']['n_hadm']}` hadm",
        f"- Conflict stays marked as mixed: `{subtype_summary['conflict_stays']['n_stays']}` stays (`{subtype_summary['conflict_stays']['pct_of_broad_stays']}%`)",
        "",
        "### Priority-based assignment",
        "",
        "| Subtype | Stays | HADM | % of broad stays |",
        "|---|---:|---:|---:|",
        *render_rows(priority_rows),
        "",
        "### Mixed/conflict-marking assignment",
        "",
        "| Subtype | Stays | HADM | % of broad stays |",
        "|---|---:|---:|---:|",
        *render_rows(mixed_rows),
        "",
        "## Ischaemic subset reaudit",
        "",
        f"- Priority-based ischaemic subset: `{main['n_stays']}` stays / `{main['n_hadm']}` hadm",
        f"- Pure ischaemic no-conflict subset: `{pure['n_stays']}` stays / `{pure['n_hadm']}` hadm",
        f"- Mixed conflicts inside priority-based ischaemic subset: `{main['n_mixed_conflict_stays']}` stays (`{main['pct_mixed_conflict_stays']}%`)",
        f"- Primary diagnosis flag in priority-based ischaemic subset: `{main['primary_dx_flag_pct']}%`",
        "",
        "### Priority-based ischaemic coverage",
        "",
        f"- Discharge coverage: `{main['discharge_coverage']['pct_hadm_with_discharge']}%` by hadm, `{main['discharge_coverage']['pct_stays_with_discharge']}%` by stay",
        f"- HPI + Hospital Course both found: `{main['discharge_coverage']['sections']['both_hpi_and_hospital_course_found_pct']}%`",
        f"- Admission-text NIHSS mention: `{main['discharge_coverage']['sections']['admission_text_nihss_mention_pct']}%`",
        f"- Any incremental neuro nursing coverage: `{main['nursing_neuro']['pct_stays_with_any_incremental_neuro_obs']}%`",
        f"- >=10 incremental neuro observations: `{main['nursing_neuro']['pct_stays_with_ge10_incremental_neuro_obs']}%`",
        f"- Median incremental neuro observations per stay: `{main['nursing_neuro']['median_incremental_neuro_obs_per_stay']}`",
        f"- Median first incremental neuro observation hour: `{main['nursing_neuro']['median_first_incremental_neuro_hour']}`",
        f"- Bilateral strength data coverage: `{main['nursing_neuro']['pct_stays_with_bilateral_strength_data']}%`",
        f"- Brain radiology in first 24h: `{main['brain_radiology']['pct_stays_with_brain_radiology_first24h']}%`",
        f"- Any brain radiology: `{main['brain_radiology']['pct_stays_with_any_brain_radiology']}%`",
        f"- Median first brain radiology hour: `{main['brain_radiology']['median_first_brain_radiology_hour']}`",
        "",
        "### Priority-based ischaemic tiering (default: incremental neuro without nursing GCS)",
        "",
        "| Tier | Stays | HADM | % of subset stays |",
        "|---|---:|---:|---:|",
        *render_tiers(main["tiering_incremental_no_gcs"]["tiers"]),
        "",
        f"- Layer 1 eligible stays: `{main['tiering_incremental_no_gcs']['layer1_eligible_stays']}`",
        f"- Layer 2 eligible stays: `{main['tiering_incremental_no_gcs']['layer2_eligible_stays']}`",
        "",
        "### Sensitivity: pure ischaemic no-conflict subset",
        "",
        f"- Discharge coverage by hadm: `{pure['discharge_coverage']['pct_hadm_with_discharge']}%`",
        f"- >=10 incremental neuro observations: `{pure['nursing_neuro']['pct_stays_with_ge10_incremental_neuro_obs']}%`",
        f"- Brain radiology in first 24h: `{pure['brain_radiology']['pct_stays_with_brain_radiology_first24h']}%`",
        "",
        "### Locked rule",
        "",
        f"- {ischaemic_summary['gcs_dedup_rule']['decision']}",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    cohort_df = load_broad_stroke_cohort(Path(args.cohort_v3))
    stroke_stays = set(cohort_df["stay_id"].astype(int).tolist())
    dx_df = load_diagnoses(Path(args.diagnoses_parquet), stroke_stays)
    family_rows = count_family_rows(dx_df.dropna(subset=["family"]).copy())
    assignments = build_assignments(cohort_df, dx_df)
    subtype_summary = summarize_assignments(assignments, family_rows, cohort_df)
    ischaemic_summary = analyze_ischaemic_subset(
        cohort_df=cohort_df,
        assignments=assignments,
        discharge_source=Path(args.discharge_source),
        nursing_source=Path(args.nursing_source),
        radiology_source=Path(args.radiology_source),
        gcs_hourly=Path(args.gcs_hourly),
        hourly_grid_dir=Path(args.hourly_grid_dir),
    )

    Path(args.out_subtype_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_ischaemic_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_markdown).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_assignments_csv).parent.mkdir(parents=True, exist_ok=True)

    Path(args.out_subtype_json).write_text(json.dumps(subtype_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_ischaemic_json).write_text(json.dumps(ischaemic_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.out_markdown).write_text(render_markdown(subtype_summary, ischaemic_summary), encoding="utf-8")
    assignments[DEFAULT_OUTPUT_COLUMNS].to_csv(Path(args.out_assignments_csv), index=False)


if __name__ == "__main__":
    main()
