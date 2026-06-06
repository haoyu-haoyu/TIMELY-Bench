#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import pyarrow.parquet as pq


SECTION_ALIASES = {
    "chief_complaint": ["chief complaint"],
    "major_surgical_or_invasive_procedure": [
        "major surgical or invasive procedure",
        "major surgical procedure",
        "major invasive procedure",
    ],
    "history_of_present_illness": [
        "history of present illness",
        "hpi",
        "history of presenting illness",
    ],
    "past_medical_history": ["past medical history", "pmh"],
    "social_history": ["social history"],
    "family_history": ["family history"],
    "physical_exam": ["physical exam", "physical examination", "admission physical exam"],
    "discharge_physical_exam": ["discharge physical exam"],
    "pertinent_results": ["pertinent results", "pertinent result"],
    "hospital_course": ["brief hospital course", "hospital course"],
    "medications_on_admission": ["medications on admission", "admission medications"],
    "discharge_medications": ["discharge medications"],
    "discharge_disposition": ["discharge disposition", "disposition", "facility"],
    "discharge_diagnosis": ["discharge diagnosis", "final diagnoses", "diagnoses"],
    "discharge_condition": ["discharge condition"],
    "discharge_instructions": ["discharge instructions", "instructions"],
    "followup_instructions": ["followup instructions", "follow-up instructions", "follow up instructions"],
    "allergies": ["allergies"],
    "service": ["service"],
}

ALIAS_TO_CANONICAL = {
    alias: canonical
    for canonical, aliases in SECTION_ALIASES.items()
    for alias in aliases
}

CANONICAL_DISPLAY = {
    "chief_complaint": "Chief Complaint",
    "major_surgical_or_invasive_procedure": "Major Surgical or Invasive Procedure",
    "history_of_present_illness": "History of Present Illness",
    "past_medical_history": "Past Medical History",
    "social_history": "Social History",
    "family_history": "Family History",
    "physical_exam": "Physical Exam",
    "discharge_physical_exam": "Discharge Physical Exam",
    "pertinent_results": "Pertinent Results",
    "hospital_course": "Brief Hospital Course",
    "medications_on_admission": "Medications on Admission",
    "discharge_medications": "Discharge Medications",
    "discharge_disposition": "Discharge Disposition",
    "discharge_diagnosis": "Discharge Diagnosis",
    "discharge_condition": "Discharge Condition",
    "discharge_instructions": "Discharge Instructions",
    "followup_instructions": "Followup Instructions",
    "allergies": "Allergies",
    "service": "Service",
}

ADMISSION_SECTIONS = {
    "chief_complaint",
    "history_of_present_illness",
    "past_medical_history",
    "social_history",
    "family_history",
    "physical_exam",
    "medications_on_admission",
}

OUTCOME_SECTIONS = {
    "hospital_course",
    "discharge_diagnosis",
    "discharge_condition",
    "discharge_medications",
    "discharge_disposition",
    "discharge_instructions",
    "followup_instructions",
}

PERTINENT_RESULTS_SECTION = "pertinent_results"

STROKE_ADMISSION_KEYWORDS = [
    "nihss",
    "nih stroke",
    "hemiparesis",
    "aphasia",
    "weakness",
    "facial droop",
    "onset",
    "presented",
    "found",
    "woke up",
    "last known well",
    "ct",
    "mri",
    "imaging",
]

HINDSIGHT_KEYWORDS = [
    "hemorrhagic transformation",
    "hemorrhagic conversion",
    "haemorrhagic transformation",
    "complication",
    "developed",
    "worsened",
    "improved",
    "discharged",
    "follow-up",
    "follow up",
    "prognosis",
    "craniectomy",
    "tracheostomy",
    "peg",
]

NEURO_NOTE_KEYWORDS = [
    "neuro check",
    "neurological",
    "neuro exam",
    "gcs",
    "glasgow",
    "pupils",
    "pupil",
    "perrl",
    "motor",
    "strength",
    "weakness",
    "hemipar",
    "plegia",
    "speech",
    "aphasia",
    "dysarthria",
    "alert",
    "oriented",
    "confused",
    "lethargic",
    "obtunded",
    "unresponsive",
    "nihss",
    "nih stroke",
    "stroke scale",
    "swallow",
    "dysphagia",
    "aspiration",
]

BRAIN_RAD_FILTER = re.compile(
    r"(head|brain|ct head|mri brain|cta|mra|cranial|stroke|infarct|ischemi|ischaemi|hemorrh|haemorrh)",
    re.IGNORECASE,
)

INITIAL_RAD_KEYWORDS = [
    "acute infarct",
    "early ischemic",
    "no hemorrhage",
    "no haemorrhage",
    "territory",
    "mca",
    "aca",
    "pca",
    "basilar",
    "occlusion",
    "stenosis",
    "thrombus",
    "diffusion restriction",
]

FOLLOWUP_RAD_KEYWORDS = [
    "hemorrhagic transformation",
    "hemorrhagic conversion",
    "haemorrhagic transformation",
    "edema",
    "oedema",
    "midline shift",
    "mass effect",
    "herniation",
    "worsening",
    "new infarct",
]

TABLE_SPECS = {
    "discharge": {"text_col": "discharge_text", "time_cols": ["note_time", "charttime"]},
    "radiology": {"text_col": "radiology_text", "time_cols": ["charttime"]},
    "nursing": {"text_col": "chart_text", "time_cols": ["charttime"]},
    "lab_comment": {"text_col": "lab_comment", "time_cols": ["charttime"]},
}

STROKE_PRIMARY_CODE_RE = re.compile(r"^(I63|433|434|436)", re.IGNORECASE)
STROKE_ANY_CODE_RE = re.compile(r"\b(I63[0-9A-Z.]*|433[0-9A-Z.]*|434[0-9A-Z.]*|436)\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stroke text strategy audit from exported data.")
    p.add_argument("--stroke-cohort", default="")
    p.add_argument("--cohort-v3", default="")
    p.add_argument("--discharge-csv", required=True)
    p.add_argument("--radiology-csv", required=True)
    p.add_argument("--nursing-csv", required=True)
    p.add_argument("--gcs-hourly", required=True)
    p.add_argument("--hourly-grid-dir", required=True)
    p.add_argument("--existing-bq-audit", default="")
    p.add_argument("--lab-comments-csv", default="")
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--sample-size", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def excerpt(text: str, limit: int = 300) -> str:
    return normalize_space(text)[:limit]


def safe_ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0


def pct(n: int, d: int) -> float:
    return round(100.0 * safe_ratio(n, d), 2)


def describe_numeric(values: Sequence[float]) -> dict:
    vals = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "count": len(vals),
        "mean": round(mean(vals), 2),
        "median": round(median(vals), 2),
        "min": round(min(vals), 2),
        "max": round(max(vals), 2),
    }


def normalize_header(line: str) -> str:
    header = normalize_space(line).rstrip(":").strip().lower()
    header = re.sub(r"[^a-z0-9/\-\s]", "", header)
    return re.sub(r"\s+", " ", header).strip()


def detect_header(line: str) -> Optional[str]:
    s = line.strip()
    if not s or len(s) < 3:
        return None
    if s.count(":") != 1 or not s.endswith(":"):
        return None
    candidate = s[:-1].strip()
    if not candidate:
        return None
    if re.fullmatch(r"[_\-\s.]+", candidate):
        return None
    if len(candidate) > 90:
        return None
    norm = normalize_header(s)
    if not norm or re.fullmatch(r"[_\-\s]+", norm):
        return None
    return norm


def canonicalize_header(norm_header: str) -> Tuple[str, str]:
    canonical = ALIAS_TO_CANONICAL.get(norm_header)
    if canonical:
        return canonical, CANONICAL_DISPLAY[canonical]
    display = " ".join(word.capitalize() for word in norm_header.split())
    return norm_header, display


def parse_sections(text: str) -> dict:
    sections: Dict[str, List[str]] = {}
    order: List[str] = []
    current_key: Optional[str] = None
    lines = str(text or "").splitlines()
    for line in lines:
        header = detect_header(line)
        if header:
            key, _display = canonicalize_header(header)
            current_key = key
            if key not in sections:
                sections[key] = []
                order.append(key)
            continue
        if current_key is not None:
            sections[current_key].append(line)
    text_sections = {k: "\n".join(v).strip() for k, v in sections.items()}
    return {"sections": text_sections, "order": order}


def find_distinct_keywords(text: str, keywords: Sequence[str]) -> List[str]:
    text_lower = str(text or "").lower()
    found = []
    for kw in keywords:
        if kw.lower() in text_lower:
            found.append(kw)
    return found


def count_csv_rows(path: Path, usecols: Optional[List[str]] = None, chunksize: int = 100_000) -> int:
    total = 0
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        total += len(chunk)
    return total


def list_parquet_parts(path: Path) -> List[Path]:
    if path.exists() and path.is_dir():
        return sorted(path.glob("part_*.parquet"))
    return []


def is_parquet_parts_dir(path: Path) -> bool:
    return bool(list_parquet_parts(path))


def source_exists(path: Path) -> bool:
    return path.exists() and (path.is_file() or is_parquet_parts_dir(path))


def get_source_columns(path: Path) -> List[str]:
    if path.is_file():
        return pd.read_csv(path, nrows=0).columns.tolist()
    parts = list_parquet_parts(path)
    if not parts:
        return []
    return pq.ParquetFile(parts[0]).schema.names


def read_source_sample(path: Path, nrows: int = 3) -> pd.DataFrame:
    if path.is_file():
        return pd.read_csv(path, nrows=nrows)
    parts = list_parquet_parts(path)
    if not parts:
        return pd.DataFrame()
    sample_rows: List[pd.DataFrame] = []
    remaining = nrows
    for part in parts:
        df = pd.read_parquet(part)
        if df.empty:
            continue
        sample_rows.append(df.head(remaining))
        remaining -= len(sample_rows[-1])
        if remaining <= 0:
            break
    return pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()


def iter_source_chunks(
    path: Path,
    usecols: Optional[List[str]] = None,
    chunksize: int = 100_000,
) -> Iterable[pd.DataFrame]:
    if path.is_file():
        yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize)
        return
    for part in list_parquet_parts(path):
        if usecols:
            available = pq.ParquetFile(part).schema.names
            cols = [c for c in usecols if c in available]
            if not cols:
                continue
            yield pd.read_parquet(part, columns=cols)
        else:
            yield pd.read_parquet(part)


def count_source_rows(path: Path, usecols: Optional[List[str]] = None, chunksize: int = 100_000) -> int:
    if path.is_file():
        return count_csv_rows(path, usecols=usecols, chunksize=chunksize)
    total = 0
    for part in list_parquet_parts(path):
        total += int(pq.ParquetFile(part).metadata.num_rows)
    return total


def normalize_source_path_arg(path_str: str) -> Path:
    return Path(path_str)


def load_stroke_cohort(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["hadm_id", "stay_id"]).copy()
    df["hadm_id"] = df["hadm_id"].astype(int)
    df["stay_id"] = df["stay_id"].astype(int)
    return df


def build_stroke_cohort_from_v3(path: Path) -> pd.DataFrame:
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
        if "has_stroke_final" in chunk.columns:
            stroke_mask = pd.to_numeric(chunk["has_stroke_final"], errors="coerce").fillna(0).astype(int).eq(1)
        else:
            code_series = chunk.get("icd_codes", pd.Series("", index=chunk.index)).fillna("").astype(str)
            dx_series = chunk.get("diagnoses_text", pd.Series("", index=chunk.index)).fillna("").astype(str)
            stroke_mask = code_series.str.contains(STROKE_ANY_CODE_RE) | dx_series.str.contains(r"\bstroke\b|\binfarct\b", case=False, regex=True)
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
        out["primary_dx_flag"] = (
            sub.get("primary_icd_code", pd.Series("", index=sub.index))
            .fillna("")
            .astype(str)
            .str.strip()
            .str.match(STROKE_PRIMARY_CODE_RE)
            .astype(int)
        )
        if "hospital_expire_flag" in sub.columns:
            out["hospital_expire_flag"] = pd.to_numeric(sub["hospital_expire_flag"], errors="coerce")
        elif "label_mortality" in sub.columns:
            out["hospital_expire_flag"] = pd.to_numeric(sub["label_mortality"], errors="coerce")
        else:
            out["hospital_expire_flag"] = pd.Series([math.nan] * len(sub))
        out["gender"] = sub.get("gender", pd.Series([""] * len(sub)))
        out["age"] = pd.to_numeric(sub.get("anchor_age"), errors="coerce") if "anchor_age" in sub.columns else pd.Series([math.nan] * len(sub))
        frames.append(out)
    if not frames:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los", "primary_dx_flag", "hospital_expire_flag", "gender", "age"])
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["subject_id", "hadm_id", "stay_id"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["hadm_id"] = df["hadm_id"].astype(int)
    df["stay_id"] = df["stay_id"].astype(int)
    return df.drop_duplicates(subset=["stay_id"]).reset_index(drop=True)


def build_note_inventory(args: argparse.Namespace, bq_probe: dict) -> dict:
    inventory = {
        "bigquery_note_probe": bq_probe,
        "exported_tables": [],
    }
    specs = [
        ("discharge", Path(args.discharge_csv)),
        ("radiology", Path(args.radiology_csv)),
        ("nursing", Path(args.nursing_csv)),
    ]
    if args.lab_comments_csv:
        specs.append(("lab_comment", Path(args.lab_comments_csv)))
    for name, path in specs:
        if not source_exists(path):
            continue
        sample = read_source_sample(path, nrows=3)
        cols = sample.columns.tolist()
        time_cols = TABLE_SPECS.get(name, {}).get("time_cols", [])
        inventory["exported_tables"].append(
            {
                "name": name,
                "path": str(path),
                "rows": count_source_rows(path),
                "columns": cols,
                "has_timestamp": any(col in cols for col in time_cols),
                "has_category": "category" in cols,
                "text_column": TABLE_SPECS.get(name, {}).get("text_col"),
            }
        )
    return inventory


def load_stroke_discharge_rows(discharge_csv: Path, hadm_ids: set[int]) -> pd.DataFrame:
    keep_rows = []
    usecols = ["stay_id", "subject_id", "hadm_id", "note_id", "note_time", "charttime", "hour_offset", "discharge_text", "text_length"]
    for chunk in iter_source_chunks(discharge_csv, usecols=usecols, chunksize=10_000):
        if "note_time" not in chunk.columns and "charttime" in chunk.columns:
            chunk["note_time"] = chunk["charttime"]
        if "text_length" not in chunk.columns and "discharge_text" in chunk.columns:
            chunk["text_length"] = chunk["discharge_text"].fillna("").astype(str).str.len()
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]
        if not chunk.empty:
            keep_rows.append(chunk)
    if not keep_rows:
        return pd.DataFrame(columns=usecols)
    df = pd.concat(keep_rows, ignore_index=True)
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    return df


def analyze_discharge_sections(discharge_df: pd.DataFrame, sample_size: int, seed: int) -> Tuple[dict, pd.DataFrame]:
    header_counter: Counter[str] = Counter()
    parsed_docs = []
    header_examples: Dict[str, str] = {}
    for row in discharge_df.itertuples(index=False):
        text = getattr(row, "discharge_text", "") or ""
        raw_headers = []
        for line in str(text).splitlines():
            header = detect_header(line)
            if header:
                key, display = canonicalize_header(header)
                raw_headers.append(key)
                header_counter[display] += 1
                header_examples.setdefault(display, line.strip())
        parsed = parse_sections(text)
        parsed_docs.append(
            {
                "hadm_id": int(row.hadm_id),
                "stay_id": int(row.stay_id) if not pd.isna(row.stay_id) else None,
                "note_id": getattr(row, "note_id", None),
                "hour_offset": float(getattr(row, "hour_offset", math.nan)) if not pd.isna(getattr(row, "hour_offset", math.nan)) else None,
                "text_length": int(getattr(row, "text_length", 0)) if not pd.isna(getattr(row, "text_length", math.nan)) else None,
                "sections": parsed["sections"],
                "section_order": parsed["order"],
                "raw_text": text,
            }
        )
    parsed_df = pd.DataFrame(parsed_docs)
    n_docs = len(parsed_df)
    top_headers = [
        {
            "header": header,
            "count": int(count),
            "pct_of_stroke_discharge_summaries": pct(count, n_docs),
            "example": header_examples.get(header, ""),
        }
        for header, count in header_counter.most_common(25)
    ]
    sample_n = min(sample_size, n_docs)
    sample_df = parsed_df.sample(n=sample_n, random_state=seed).reset_index(drop=True) if sample_n else parsed_df.iloc[0:0].copy()

    section_rows = []
    admission_excerpts = []
    outcome_excerpts = []
    failure_examples = []
    pertinent_rows = []

    for row in sample_df.itertuples(index=False):
        sections = row.sections
        n_sections = len(sections)
        has_hpi = "history_of_present_illness" in sections
        has_hospital_course = "hospital_course" in sections
        has_discharge_dx = "discharge_diagnosis" in sections

        admission_text = "\n".join(sections.get(k, "") for k in ADMISSION_SECTIONS if sections.get(k))
        outcome_text = "\n".join(sections.get(k, "") for k in OUTCOME_SECTIONS if sections.get(k))
        admission_keywords = find_distinct_keywords(admission_text, STROKE_ADMISSION_KEYWORDS)
        hindsight_keywords = find_distinct_keywords(outcome_text, HINDSIGHT_KEYWORDS)

        if admission_text and len(admission_excerpts) < 3:
            admission_excerpts.append(
                {
                    "hadm_id": row.hadm_id,
                    "note_id": row.note_id,
                    "text": excerpt(admission_text),
                }
            )
        if outcome_text and len(outcome_excerpts) < 3:
            outcome_excerpts.append(
                {
                    "hadm_id": row.hadm_id,
                    "note_id": row.note_id,
                    "text": excerpt(outcome_text),
                }
            )
        if n_sections < 5 and len(failure_examples) < 3:
            failure_examples.append(
                {
                    "hadm_id": row.hadm_id,
                    "note_id": row.note_id,
                    "section_order": row.section_order,
                    "text_excerpt": excerpt(row.raw_text),
                }
            )

        pertinent = sections.get(PERTINENT_RESULTS_SECTION, "")
        time_marker_count = len(
            re.findall(
                r"(\b\d{1,2}:\d{2}\s?(?:AM|PM)\b|\bPOD\s*\d+\b|\bday\s+\d+\b|\bpost-?cpb\b|\bpre-?cpb\b)",
                pertinent,
                flags=re.IGNORECASE,
            )
        )
        pertinent_rows.append(
            {
                "hadm_id": row.hadm_id,
                "has_pertinent_results": bool(pertinent),
                "pertinent_length": len(pertinent),
                "pertinent_time_marker_count": time_marker_count,
                "pertinent_has_time_markers": bool(time_marker_count),
                "pertinent_has_multiple_time_markers": time_marker_count >= 2,
                "pertinent_hindsight_keywords": find_distinct_keywords(pertinent, HINDSIGHT_KEYWORDS),
                "pertinent_excerpt": excerpt(pertinent),
            }
        )

        section_rows.append(
            {
                "hadm_id": row.hadm_id,
                "note_id": row.note_id,
                "n_sections": n_sections,
                "has_hpi": has_hpi,
                "has_hospital_course": has_hospital_course,
                "has_discharge_diagnosis": has_discharge_dx,
                "admission_char_count": len(admission_text),
                "outcome_char_count": len(outcome_text),
                "admission_ratio": round(len(admission_text) / max(len(admission_text) + len(outcome_text), 1), 4),
                "admission_keyword_count": len(admission_keywords),
                "admission_keywords": admission_keywords,
                "outcome_hindsight_keyword_count": len(hindsight_keywords),
                "outcome_hindsight_keywords": hindsight_keywords,
            }
        )

    section_df = pd.DataFrame(section_rows)
    pertinent_df = pd.DataFrame(pertinent_rows)
    if section_df.empty:
        admission_richness_sufficient = False
    else:
        admission_richness_sufficient = bool(
            median(section_df["admission_char_count"].tolist()) >= 600
            and safe_ratio(int((section_df["admission_keyword_count"] >= 3).sum()), len(section_df)) >= 0.30
        )

    summary = {
        "discharge_summary_coverage": {
            "stroke_unique_hadm_with_discharge": int(discharge_df["hadm_id"].nunique()),
            "stroke_unique_stay_with_discharge": int(discharge_df["stay_id"].dropna().astype(int).nunique()),
        },
        "step_1_1_top_headers": top_headers,
        "step_1_2_section_boundary_reliability": {
            "sample_size": int(len(section_df)),
            "hpi_found_pct": pct(int(section_df["has_hpi"].sum()), len(section_df)),
            "hospital_course_found_pct": pct(int(section_df["has_hospital_course"].sum()), len(section_df)),
            "discharge_diagnosis_found_pct": pct(int(section_df["has_discharge_diagnosis"].sum()), len(section_df)),
            "both_hpi_and_hospital_course_found_pct": pct(
                int((section_df["has_hpi"] & section_df["has_hospital_course"]).sum()),
                len(section_df),
            ),
            "five_or_more_sections_pct": pct(int((section_df["n_sections"] >= 5).sum()), len(section_df)),
        },
        "step_1_3_admission_vs_outcome_content": {
            "admission_char_count": describe_numeric(section_df["admission_char_count"].tolist()),
            "outcome_char_count": describe_numeric(section_df["outcome_char_count"].tolist()),
            "admission_ratio": describe_numeric(section_df["admission_ratio"].tolist()),
            "admission_mentions_nihss_pct": pct(
                int(section_df["admission_keywords"].apply(lambda xs: "nihss" in xs or "nih stroke" in xs).sum()),
                len(section_df),
            ),
            "admission_contains_3_or_more_stroke_keywords_pct": pct(
                int((section_df["admission_keyword_count"] >= 3).sum()),
                len(section_df),
            ),
            "admission_text_richness_sufficient": admission_richness_sufficient,
            "admission_excerpts": admission_excerpts,
            "outcome_excerpts": outcome_excerpts,
        },
        "step_1_4_pertinent_results": {
            "sample_size": int(len(pertinent_df)),
            "documents_with_pertinent_results_pct": pct(
                int(pertinent_df["has_pertinent_results"].sum()), len(pertinent_df)
            ),
            "documents_with_time_markers_pct": pct(
                int(pertinent_df["pertinent_has_time_markers"].sum()), len(pertinent_df)
            ),
            "documents_with_multiple_time_markers_pct": pct(
                int(pertinent_df["pertinent_has_multiple_time_markers"].sum()), len(pertinent_df)
            ),
            "documents_with_hindsight_keywords_pct": pct(
                int(pertinent_df["pertinent_hindsight_keywords"].apply(bool).sum()), len(pertinent_df)
            ),
            "recommendation": (
                "needs_own_temporal_cut"
                if safe_ratio(int(pertinent_df["pertinent_has_time_markers"].sum()), len(pertinent_df)) >= 0.20
                or safe_ratio(int(pertinent_df["pertinent_hindsight_keywords"].apply(bool).sum()), len(pertinent_df)) >= 0.10
                else "possibly_include_with_caution"
            ),
            "examples": [
                {
                    "hadm_id": int(row["hadm_id"]),
                    "excerpt": row["pertinent_excerpt"],
                }
                for row in pertinent_rows
                if row["has_pertinent_results"] and row["pertinent_excerpt"]
            ][:3],
        },
        "failure_examples": failure_examples if summary_is_low(section_df) else [],
    }
    return summary, parsed_df


def summary_is_low(section_df: pd.DataFrame) -> bool:
    if section_df.empty:
        return True
    return safe_ratio(int((section_df["has_hpi"] & section_df["has_hospital_course"]).sum()), len(section_df)) < 0.70


def analyze_timestamped_notes(
    stroke_stays: set[int],
    nursing_csv: Path,
    radiology_csv: Path,
    lab_comments_csv: Optional[Path] = None,
) -> dict:
    available_tables = []
    timestamped_counts = Counter()
    timestamped_first24 = Counter()
    timestamped_24_72 = Counter()
    timestamped_after72 = Counter()
    source_counts = Counter()
    source_hour_bounds: Dict[str, dict] = {}
    nursing_category_counts = Counter()
    neuro_note_count = 0
    neuro_stays = set()
    total_nonrad_timestamped_notes = 0

    def update_time_bins(stay_id: int, hour_offset: float) -> None:
        timestamped_counts[stay_id] += 1
        if pd.isna(hour_offset):
            return
        if hour_offset < 24:
            timestamped_first24[stay_id] += 1
        elif hour_offset < 72:
            timestamped_24_72[stay_id] += 1
        else:
            timestamped_after72[stay_id] += 1

    if source_exists(nursing_csv):
        available_tables.append("nursing")
        source_hour_bounds["nursing"] = {"min_hour": None, "max_hour": None}
        for chunk in iter_source_chunks(
            nursing_csv,
            usecols=["stay_id", "hour_offset", "item_label", "category", "chart_text"],
            chunksize=100_000,
        ):
            chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
            chunk = chunk.dropna(subset=["stay_id"]).copy()
            chunk["stay_id"] = chunk["stay_id"].astype(int)
            chunk = chunk[chunk["stay_id"].isin(stroke_stays)]
            if chunk.empty:
                continue
            vals = pd.to_numeric(chunk["hour_offset"], errors="coerce").dropna()
            if not vals.empty:
                mn = float(vals.min())
                mx = float(vals.max())
                source_hour_bounds["nursing"]["min_hour"] = mn if source_hour_bounds["nursing"]["min_hour"] is None else min(source_hour_bounds["nursing"]["min_hour"], mn)
                source_hour_bounds["nursing"]["max_hour"] = mx if source_hour_bounds["nursing"]["max_hour"] is None else max(source_hour_bounds["nursing"]["max_hour"], mx)
            for row in chunk.itertuples(index=False):
                stay_id = int(row.stay_id)
                source_counts["nursing"] += 1
                total_nonrad_timestamped_notes += 1
                update_time_bins(stay_id, float(row.hour_offset) if not pd.isna(row.hour_offset) else math.nan)
                category = normalize_space(f"{row.category} {row.item_label}")
                if category:
                    nursing_category_counts[category] += 1
                text_blob = normalize_space(f"{row.category} {row.item_label} {row.chart_text}").lower()
                if any(kw in text_blob for kw in NEURO_NOTE_KEYWORDS):
                    neuro_note_count += 1
                    neuro_stays.add(stay_id)

    brain_rad_counts = Counter()
    brain_rad_first24 = Counter()
    brain_rad_total = 0
    first24_brain_reports = 0
    total_brain_reports_after24 = 0
    initial_keyword_hits = Counter()
    followup_keyword_hits = Counter()
    if source_exists(radiology_csv):
        available_tables.append("radiology")
        source_hour_bounds["radiology"] = {"min_hour": None, "max_hour": None}
        for chunk in iter_source_chunks(
            radiology_csv,
            usecols=["stay_id", "hour_offset", "radiology_text"],
            chunksize=50_000,
        ):
            chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
            chunk = chunk.dropna(subset=["stay_id"]).copy()
            chunk["stay_id"] = chunk["stay_id"].astype(int)
            chunk = chunk[chunk["stay_id"].isin(stroke_stays)]
            if chunk.empty:
                continue
            vals = pd.to_numeric(chunk["hour_offset"], errors="coerce").dropna()
            if not vals.empty:
                mn = float(vals.min())
                mx = float(vals.max())
                source_hour_bounds["radiology"]["min_hour"] = mn if source_hour_bounds["radiology"]["min_hour"] is None else min(source_hour_bounds["radiology"]["min_hour"], mn)
                source_hour_bounds["radiology"]["max_hour"] = mx if source_hour_bounds["radiology"]["max_hour"] is None else max(source_hour_bounds["radiology"]["max_hour"], mx)
            for row in chunk.itertuples(index=False):
                stay_id = int(row.stay_id)
                source_counts["radiology"] += 1
                update_time_bins(stay_id, float(row.hour_offset) if not pd.isna(row.hour_offset) else math.nan)
                text = str(row.radiology_text or "")
                if not BRAIN_RAD_FILTER.search(text):
                    continue
                hour_offset = float(row.hour_offset) if not pd.isna(row.hour_offset) else math.nan
                brain_rad_total += 1
                brain_rad_counts[stay_id] += 1
                if not pd.isna(hour_offset) and hour_offset < 24:
                    brain_rad_first24[stay_id] += 1
                    first24_brain_reports += 1
                    for kw in INITIAL_RAD_KEYWORDS:
                        if kw in text.lower():
                            initial_keyword_hits[kw] += 1
                else:
                    total_brain_reports_after24 += 1
                    for kw in FOLLOWUP_RAD_KEYWORDS:
                        if kw in text.lower():
                            followup_keyword_hits[kw] += 1

    if lab_comments_csv and source_exists(lab_comments_csv):
        available_tables.append("lab_comment")
        source_hour_bounds["lab_comment"] = {"min_hour": None, "max_hour": None}
        for chunk in iter_source_chunks(
            lab_comments_csv,
            usecols=["stay_id", "hour_offset"],
            chunksize=100_000,
        ):
            chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
            chunk = chunk.dropna(subset=["stay_id"]).copy()
            chunk["stay_id"] = chunk["stay_id"].astype(int)
            chunk = chunk[chunk["stay_id"].isin(stroke_stays)]
            if chunk.empty:
                continue
            vals = pd.to_numeric(chunk["hour_offset"], errors="coerce").dropna()
            if not vals.empty:
                mn = float(vals.min())
                mx = float(vals.max())
                source_hour_bounds["lab_comment"]["min_hour"] = mn if source_hour_bounds["lab_comment"]["min_hour"] is None else min(source_hour_bounds["lab_comment"]["min_hour"], mn)
                source_hour_bounds["lab_comment"]["max_hour"] = mx if source_hour_bounds["lab_comment"]["max_hour"] is None else max(source_hour_bounds["lab_comment"]["max_hour"], mx)
            for row in chunk.itertuples(index=False):
                stay_id = int(row.stay_id)
                source_counts["lab_comment"] += 1
                total_nonrad_timestamped_notes += 1
                update_time_bins(stay_id, float(row.hour_offset) if not pd.isna(row.hour_offset) else math.nan)

    per_stay_note_counts = [timestamped_counts.get(stay, 0) for stay in stroke_stays]
    source_distribution = [
        {"source": source, "n_notes": int(count)}
        for source, count in source_counts.most_common()
    ]
    top_nursing_categories = [
        {"category": cat, "n_notes": int(count)}
        for cat, count in nursing_category_counts.most_common(20)
    ]

    return {
        "available_timestamped_tables": available_tables,
        "step_2_2_note_type_distribution": {
            "source_distribution": source_distribution,
            "top_nursing_categories": top_nursing_categories,
        },
        "step_2_3_timestamped_note_coverage": {
            "stroke_stays_with_1_or_more_timestamped_notes": int(sum(1 for stay in stroke_stays if timestamped_counts.get(stay, 0) >= 1)),
            "stroke_stays_with_3_or_more_timestamped_notes": int(sum(1 for stay in stroke_stays if timestamped_counts.get(stay, 0) >= 3)),
            "pct_stays_with_1_or_more_timestamped_notes": pct(sum(1 for stay in stroke_stays if timestamped_counts.get(stay, 0) >= 1), len(stroke_stays)),
            "pct_stays_with_3_or_more_timestamped_notes": pct(sum(1 for stay in stroke_stays if timestamped_counts.get(stay, 0) >= 3), len(stroke_stays)),
            "median_timestamped_notes_per_stay": round(median(per_stay_note_counts), 2) if per_stay_note_counts else 0.0,
            "notes_in_first_24h": int(sum(timestamped_first24.values())),
            "notes_in_24_72h": int(sum(timestamped_24_72.values())),
            "notes_after_72h": int(sum(timestamped_after72.values())),
            "source_hour_bounds": source_hour_bounds,
        },
        "step_2_4_neurological_content_in_timestamped_notes": {
            "n_nonradiology_timestamped_notes": int(total_nonrad_timestamped_notes),
            "n_notes_with_neuro_keywords": int(neuro_note_count),
            "pct_nonradiology_timestamped_notes_with_neuro_keywords": pct(neuro_note_count, total_nonrad_timestamped_notes),
            "n_stroke_stays_with_1_or_more_neuro_notes": int(len(neuro_stays)),
            "pct_stroke_stays_with_1_or_more_neuro_notes": pct(len(neuro_stays), len(stroke_stays)),
        },
        "radiology_timing": {
            "brain_radiology_total_reports": int(brain_rad_total),
            "brain_radiology_first24_reports": int(first24_brain_reports),
            "brain_radiology_after24_reports": int(total_brain_reports_after24),
            "median_brain_radiology_reports_per_stay": round(
                median([brain_rad_counts.get(stay, 0) for stay in stroke_stays]), 2
            ) if stroke_stays else 0.0,
            "pct_stays_with_brain_radiology_first24h": pct(sum(1 for stay in stroke_stays if brain_rad_first24.get(stay, 0) >= 1), len(stroke_stays)),
            "pct_stays_with_2_or_more_brain_radiology_reports_total": pct(sum(1 for stay in stroke_stays if brain_rad_counts.get(stay, 0) >= 2), len(stroke_stays)),
            "first24_keyword_prevalence": [
                {"keyword": kw, "count": int(count), "pct_of_first24_brain_reports": pct(count, first24_brain_reports)}
                for kw, count in initial_keyword_hits.most_common()
            ],
            "after24_keyword_prevalence": [
                {"keyword": kw, "count": int(count), "pct_of_after24_brain_reports": pct(count, total_brain_reports_after24)}
                for kw, count in followup_keyword_hits.most_common()
            ],
        },
    }


def analyze_structured_coverage(stroke_stays: set[int], gcs_hourly: Path, hourly_grid_dir: Path) -> dict:
    if not gcs_hourly.exists() or not hourly_grid_dir.exists():
        return {
            "step_4_1_gcs_measurement_frequency": {
                "pct_stays_with_1_or_more_gcs_measurements": None,
                "median_gcs_measurements_per_stay": None,
                "median_time_of_first_gcs_from_icu_admission_hours": None,
                "method_note": "Structured coverage skipped because required parquet inputs were missing.",
            },
            "step_4_2_vital_sign_density_first_72h": {
                "method_note": "Structured coverage skipped because required parquet inputs were missing.",
                "variables": {},
            },
        }
    gcs_df = pd.read_parquet(gcs_hourly)
    gcs_df["stay_id"] = pd.to_numeric(gcs_df["stay_id"], errors="coerce").astype("Int64")
    gcs_df = gcs_df.dropna(subset=["stay_id"]).copy()
    gcs_df["stay_id"] = gcs_df["stay_id"].astype(int)
    gcs_df = gcs_df[gcs_df["stay_id"].isin(stroke_stays)]
    gcs_df = gcs_df[gcs_df["hour"].between(0, 168, inclusive="both")]
    gcs_df["has_any_gcs"] = gcs_df[["gcs_total", "gcs_motor", "gcs_verbal", "gcs_eye"]].notna().any(axis=1)
    gcs_nonmissing = gcs_df[gcs_df["has_any_gcs"]]
    gcs_counts = gcs_nonmissing.groupby("stay_id").size()
    first_gcs = gcs_nonmissing.groupby("stay_id")["hour"].min()

    variables = ["heart_rate", "map_merged", "temperature_c", "spo2"]
    obs_counts = {var: Counter() for var in variables}
    part_paths = sorted(hourly_grid_dir.glob("part_*.parquet"))
    usecols = ["stay_id", "hour"] + variables
    for part in part_paths:
        df = pd.read_parquet(part, columns=usecols)
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["stay_id"]).copy()
        df["stay_id"] = df["stay_id"].astype(int)
        df = df[df["stay_id"].isin(stroke_stays) & df["hour"].between(0, 72, inclusive="both")]
        if df.empty:
            continue
        for var in variables:
            observed = df[df[var].notna()][["stay_id", "hour"]]
            for stay_id, n in observed.groupby("stay_id").size().items():
                obs_counts[var][int(stay_id)] += int(n)

    vitals = {}
    for var in variables:
        values = [obs_counts[var].get(stay, 0) for stay in stroke_stays]
        vitals[var] = {
            "pct_stays_with_1_or_more_observed_hours_0_72h": pct(sum(1 for x in values if x >= 1), len(stroke_stays)),
            "median_observed_hours_0_72h": round(median(values), 2) if values else 0.0,
        }

    return {
        "step_4_1_gcs_measurement_frequency": {
            "pct_stays_with_1_or_more_gcs_measurements": pct(int(gcs_counts.shape[0]), len(stroke_stays)),
            "median_gcs_measurements_per_stay": round(median(gcs_counts.tolist()), 2) if not gcs_counts.empty else 0.0,
            "median_time_of_first_gcs_from_icu_admission_hours": round(median(first_gcs.tolist()), 2) if not first_gcs.empty else None,
            "method_note": "Derived from gcs_hourly_v3 parquet.",
        },
        "step_4_2_vital_sign_density_first_72h": {
            "method_note": "Uses hourly observed bins in hourly_state_grid_168h as a proxy for raw measurement density.",
            "variables": vitals,
        },
    }


def build_decision_matrix(
    stroke_total_stays: int,
    stroke_total_hadm: int,
    discharge_analysis: dict,
    notes_analysis: dict,
) -> dict:
    section_metrics = discharge_analysis["step_1_2_section_boundary_reliability"]
    content_metrics = discharge_analysis["step_1_3_admission_vs_outcome_content"]
    timestamped = notes_analysis["step_2_3_timestamped_note_coverage"]
    radiology = notes_analysis["radiology_timing"]

    temporal_coverage = timestamped["pct_stays_with_1_or_more_timestamped_notes"]
    neuro_coverage = notes_analysis["step_2_4_neurological_content_in_timestamped_notes"]["pct_stroke_stays_with_1_or_more_neuro_notes"]
    rad24_coverage = radiology["pct_stays_with_brain_radiology_first24h"]
    discharge_hadm_with_note = discharge_analysis["discharge_summary_coverage"]["stroke_unique_hadm_with_discharge"]
    discharge_cov = pct(discharge_hadm_with_note, stroke_total_hadm)
    both_found = section_metrics["both_hpi_and_hospital_course_found_pct"]
    admission_nihss = content_metrics["admission_mentions_nihss_pct"]
    admission_rich = content_metrics["admission_text_richness_sufficient"]
    bounds = timestamped.get("source_hour_bounds", {})
    truncated_sources = [
        name for name, meta in bounds.items()
        if meta.get("max_hour") is not None and meta.get("max_hour") <= 24.0
    ]

    approach_a_go = temporal_coverage >= 50.0 and rad24_coverage >= 40.0
    approach_b_go = temporal_coverage >= 50.0 and discharge_cov >= 50.0
    approach_c_go = both_found >= 70.0 and admission_rich

    if approach_b_go and approach_c_go:
        recommendation = "B+C combined"
        reasoning = (
            "Timestamped non-discharge notes/radiology provide a viable temporal layer, and discharge "
            "summaries appear sectionable enough to support a separately controlled sectioning experiment."
        )
    elif approach_b_go:
        recommendation = "B"
        reasoning = (
            "Timestamped notes support a temporal layer and discharge summaries support a retrospective layer, "
            "but discharge sectioning is not strong enough to be the primary temporal strategy."
        )
    elif approach_a_go:
        recommendation = "A"
        reasoning = (
            "Timestamped notes and early radiology coverage are strong enough that a no-discharge temporal strategy is viable."
        )
    elif approach_c_go:
        recommendation = "C"
        reasoning = (
            "Section parsing is reliable enough and admission sections are rich enough to support temporal sectioning, "
            "but timestamped note coverage is weaker than desired."
        )
    else:
        recommendation = "INSUFFICIENT"
        reasoning = (
            "Neither timestamped note coverage nor discharge sectioning currently reaches a comfortable threshold "
            "for a stand-alone temporal stroke text strategy."
        )

    if truncated_sources:
        reasoning += (
            " Current exported timestamped sources are truncated to an early window "
            f"({', '.join(truncated_sources)} max hour <= 24), so Approach A/B temporal coverage should be interpreted as early-window only."
        )

    return {
        "approach_a_feasibility": {
            "description": "No discharge summary, only timestamped notes + radiology",
            "timestamped_note_coverage_pct": temporal_coverage,
            "neuro_note_coverage_pct": neuro_coverage,
            "radiology_24h_coverage_pct": rad24_coverage,
            "median_timestamped_notes_per_stay": timestamped["median_timestamped_notes_per_stay"],
            "scope_note": "Current exported timestamped note sources are limited to the early window if source_hour_bounds max_hour <= 24.",
            "verdict": "GO" if approach_a_go else "INSUFFICIENT",
        },
        "approach_b_feasibility": {
            "description": "Two layers - temporal (timestamped only) + retrospective (full discharge)",
            "temporal_layer_coverage_pct": temporal_coverage,
            "retrospective_layer_coverage_pct": discharge_cov,
            "scope_note": "Temporal layer currently reflects the exported early-window note sources, not an unrestricted multi-day note archive.",
            "verdict": "GO" if approach_b_go else "INSUFFICIENT",
        },
        "approach_c_feasibility": {
            "description": "Temporal sectioning of discharge summary",
            "section_parsing_success_pct": section_metrics["five_or_more_sections_pct"],
            "hpi_found_pct": section_metrics["hpi_found_pct"],
            "hospital_course_found_pct": section_metrics["hospital_course_found_pct"],
            "both_found_pct": both_found,
            "admission_text_richness_sufficient": admission_rich,
            "admission_text_nihss_mention_pct": admission_nihss,
            "verdict": "GO" if approach_c_go else "INSUFFICIENT",
        },
        "recommended_approach": recommendation,
        "reasoning": reasoning,
    }


def render_markdown(audit: dict) -> str:
    dm = audit["decision_matrix"]
    source_bounds = audit["timestamped_note_analysis"]["step_2_3_timestamped_note_coverage"].get("source_hour_bounds", {})
    lines = [
        "# Stroke Text Strategy Audit",
        "",
        "## Cohort",
        f"- Stroke ICU stays: `{audit['stroke_cohort']['n_stays']}`",
        f"- Stroke hospital admissions: `{audit['stroke_cohort']['n_hadm']}`",
        f"- Primary-dx stays: `{audit['stroke_cohort']['n_primary_dx_stays']}`",
        "",
        "## Key Findings",
        f"- Timestamped note coverage (non-discharge): `{dm['approach_a_feasibility']['timestamped_note_coverage_pct']}%`",
        f"- Neuro-note stay coverage: `{dm['approach_a_feasibility']['neuro_note_coverage_pct']}%`",
        f"- Brain radiology in first 24h: `{dm['approach_a_feasibility']['radiology_24h_coverage_pct']}%`",
        f"- Discharge summary coverage by hadm: `{dm['approach_b_feasibility']['retrospective_layer_coverage_pct']}%`",
        f"- HPI + Hospital Course both found in sampled discharge notes: `{dm['approach_c_feasibility']['both_found_pct']}%`",
        f"- Admission-section NIHSS mention rate: `{dm['approach_c_feasibility']['admission_text_nihss_mention_pct']}%`",
        "",
        "## Decision Matrix",
        f"- Approach A: `{dm['approach_a_feasibility']['verdict']}`",
        f"- Approach B: `{dm['approach_b_feasibility']['verdict']}`",
        f"- Approach C: `{dm['approach_c_feasibility']['verdict']}`",
        f"- Recommended approach: `{dm['recommended_approach']}`",
        "",
        "Reasoning:",
        dm["reasoning"],
        "",
        "## Section Parsing",
        f"- Sample size: `{audit['discharge_section_analysis']['step_1_2_section_boundary_reliability']['sample_size']}`",
        f"- HPI found: `{audit['discharge_section_analysis']['step_1_2_section_boundary_reliability']['hpi_found_pct']}%`",
        f"- Hospital Course found: `{audit['discharge_section_analysis']['step_1_2_section_boundary_reliability']['hospital_course_found_pct']}%`",
        f"- Five or more sections parsed: `{audit['discharge_section_analysis']['step_1_2_section_boundary_reliability']['five_or_more_sections_pct']}%`",
        "",
        "## Important Caveats",
        "- Direct BigQuery note-table enumeration was not rerun during this audit; fresh exported v3 note sources and the prior note-module probe were used instead.",
        "- Stroke discharge summaries remain hindsight-rich documents. Any temporal reasoning benchmark must avoid feeding full discharge summaries into forward-looking tasks.",
        "- Structured temporal coverage in Part 4.2 uses hourly observed bins from `hourly_state_grid_168h` rather than raw chart event counts.",
    ]
    if source_bounds:
        bounds_str = ", ".join(
            f"{name}: {meta.get('min_hour')} to {meta.get('max_hour')}h"
            for name, meta in source_bounds.items()
        )
        lines.append(f"- Current exported timestamped note sources are window-limited: {bounds_str}.")
    warnings = audit.get("warnings", [])
    if warnings:
        lines.extend(["", "## Warnings"])
        lines.extend([f"- {w}" for w in warnings])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    stroke_cohort_path = Path(args.stroke_cohort) if args.stroke_cohort else None
    cohort_v3_path = Path(args.cohort_v3) if args.cohort_v3 else None
    if stroke_cohort_path and stroke_cohort_path.exists():
        stroke_cohort = load_stroke_cohort(stroke_cohort_path)
        stroke_cohort_source = str(stroke_cohort_path)
    elif cohort_v3_path and cohort_v3_path.exists():
        stroke_cohort = build_stroke_cohort_from_v3(cohort_v3_path)
        stroke_cohort_source = str(cohort_v3_path) + " (derived fresh stroke cohort)"
    else:
        raise FileNotFoundError("Provide either --stroke-cohort or --cohort-v3 with an existing file.")
    stroke_stays = set(stroke_cohort["stay_id"].astype(int).tolist())
    stroke_hadm = set(stroke_cohort["hadm_id"].astype(int).tolist())

    bq_probe = {
        "status": "not_available",
        "note_module_probe": {},
        "limitation": "Direct note-module BigQuery enumeration not executed in this run.",
    }
    if args.existing_bq_audit and Path(args.existing_bq_audit).exists():
        existing = read_json(Path(args.existing_bq_audit))
        bq_probe = {
            "status": "prior_probe_reused",
            "note_module_probe": existing.get("note_module_probe", {}),
            "source": str(Path(args.existing_bq_audit)),
            "limitation": "This audit reused the prior note-module probe instead of rerunning INFORMATION_SCHEMA queries.",
        }

    discharge_df = load_stroke_discharge_rows(Path(args.discharge_csv), stroke_hadm)
    discharge_analysis, _parsed_df = analyze_discharge_sections(discharge_df, args.sample_size, args.seed)
    notes_analysis = analyze_timestamped_notes(
        stroke_stays=stroke_stays,
        nursing_csv=Path(args.nursing_csv),
        radiology_csv=Path(args.radiology_csv),
        lab_comments_csv=Path(args.lab_comments_csv) if args.lab_comments_csv else None,
    )
    structured_analysis = analyze_structured_coverage(
        stroke_stays=stroke_stays,
        gcs_hourly=Path(args.gcs_hourly),
        hourly_grid_dir=Path(args.hourly_grid_dir),
    )

    decision_matrix = build_decision_matrix(
        stroke_total_stays=len(stroke_stays),
        stroke_total_hadm=len(stroke_hadm),
        discharge_analysis=discharge_analysis,
        notes_analysis=notes_analysis,
    )

    warnings = []
    if decision_matrix["approach_c_feasibility"]["verdict"] != "GO":
        warnings.append("Approach C did not reach the current reliability threshold for clean temporal sectioning.")
    if decision_matrix["approach_a_feasibility"]["verdict"] != "GO":
        warnings.append("Approach A has weaker temporal text coverage than desired.")
    bounds = notes_analysis["step_2_3_timestamped_note_coverage"].get("source_hour_bounds", {})
    truncated_sources = [
        name for name, meta in bounds.items()
        if meta.get("max_hour") is not None and meta.get("max_hour") <= 24.0
    ]
    if truncated_sources:
        warnings.append(
            "Current exported timestamped note sources are early-window only: "
            + ", ".join(f"{name}<=24h" for name in truncated_sources)
            + "."
        )

    audit = {
        "run_metadata": {
            "stroke_cohort_source": stroke_cohort_source,
            "discharge_csv": str(Path(args.discharge_csv)),
            "radiology_csv": str(Path(args.radiology_csv)),
            "nursing_csv": str(Path(args.nursing_csv)),
            "gcs_hourly": str(Path(args.gcs_hourly)),
            "hourly_grid_dir": str(Path(args.hourly_grid_dir)),
            "seed": args.seed,
            "sample_size": args.sample_size,
        },
        "stroke_cohort": {
            "n_stays": int(len(stroke_stays)),
            "n_hadm": int(len(stroke_hadm)),
            "n_primary_dx_stays": int(stroke_cohort["primary_dx_flag"].fillna(0).astype(int).eq(1).sum()),
        },
        "note_table_inventory": build_note_inventory(args, bq_probe),
        "discharge_section_analysis": discharge_analysis,
        "timestamped_note_analysis": notes_analysis,
        "structured_temporal_coverage": structured_analysis,
        "decision_matrix": decision_matrix,
        "warnings": warnings,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    output_md.write_text(render_markdown(audit), encoding="utf-8")


if __name__ == "__main__":
    main()
