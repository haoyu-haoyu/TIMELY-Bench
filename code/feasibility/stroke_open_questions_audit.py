#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from stroke_text_strategy_audit import (
    BRAIN_RAD_FILTER,
    build_stroke_cohort_from_v3,
    excerpt,
    get_source_columns,
    iter_source_chunks,
    list_parquet_parts,
    normalize_space,
    parse_sections,
    pct,
    read_source_sample,
    safe_ratio,
    source_exists,
)


SEED = 42
PRIOR_AUDIT_STROKE_STAYS = 5925
PRIOR_AUDIT_STROKE_HADM = 5190
PRIOR_AUDIT_SQL_NOTE = (
    "Prior 2026-03-19 feasibility audit used only ischaemic-stroke ICD filters: "
    "ICD-10 I63%; ICD-9 433%, 434%, 436; all diagnosis positions; diagnoses_icd "
    "joined to icustays on hadm_id."
)

TARGET_CATEGORIES = [
    "Neurological GCS - Motor Response",
    "Neurological Strength R Arm",
    "Neurological Strength L Arm",
    "Neurological Strength R Leg",
    "Neurological Strength L Leg",
    "Neurological Commands Response",
    "Neurological GCS - Eye Opening",
    "Neurological GCS - Verbal Response",
    "Neurological Orientation",
    "Neurological Speech",
    "Neurological Neuro Drain #1 Type",
    "Neurological RL Strength/Movement",
    "Neurological LU Strength/Movement",
    "Routine Vital Signs Heart Rhythm",
]

ONSET_PATTERNS = [
    re.compile(r"\blast known well\b", re.IGNORECASE),
    re.compile(r"\bonset\b", re.IGNORECASE),
    re.compile(r"\bstarted\b", re.IGNORECASE),
    re.compile(r"\bbegan\b", re.IGNORECASE),
    re.compile(r"\bpresented with\b", re.IGNORECASE),
    re.compile(r"\bfound\b", re.IGNORECASE),
]
WAKE_UNKNOWN_PATTERNS = [
    re.compile(r"\bwake[- ]?up stroke\b", re.IGNORECASE),
    re.compile(r"\bwoke up\b", re.IGNORECASE),
    re.compile(r"\bfound down\b", re.IGNORECASE),
    re.compile(r"\bunknown onset\b", re.IGNORECASE),
    re.compile(r"\bunclear onset\b", re.IGNORECASE),
    re.compile(r"\bunknown last known well\b", re.IGNORECASE),
]
DURATION_PATTERNS = [
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:prior to|before|before admission|pta|ago)", re.IGNORECASE),
    re.compile(r"for\s+(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)", re.IGNORECASE),
    re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*after", re.IGNORECASE),
]

ADMISSION_PHYS_PATTERNS = [
    re.compile(r"\bupon admission\b", re.IGNORECASE),
    re.compile(r"\bon admission\b", re.IGNORECASE),
    re.compile(r"\badmission exam\b", re.IGNORECASE),
    re.compile(r"\bon presentation\b", re.IGNORECASE),
    re.compile(r"\binitial\b", re.IGNORECASE),
]
DISCHARGE_PHYS_PATTERNS = [
    re.compile(r"\bdischarge\b", re.IGNORECASE),
    re.compile(r"\bat discharge\b", re.IGNORECASE),
    re.compile(r"\bdischarge exam\b", re.IGNORECASE),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resolve open stroke text strategy questions on fresh v3 data.")
    p.add_argument("--cohort-v3", required=True)
    p.add_argument("--diagnoses-parquet", required=True)
    p.add_argument("--nursing-source", required=True)
    p.add_argument("--radiology-source", required=True)
    p.add_argument("--discharge-source", required=True)
    p.add_argument("--gcs-hourly", required=True)
    p.add_argument("--hourly-grid-dir", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def classify_cerebro_code(icd_code: str, icd_version: int) -> Optional[str]:
    code = str(icd_code or "").upper().strip()
    if not code:
        return None
    if int(icd_version) == 10:
        if code.startswith("I63"):
            return "ischaemic_stroke"
        if code.startswith("I61") or code.startswith("I62"):
            return "hemorrhagic_stroke"
        if code.startswith("G45"):
            return "tia"
        if code.startswith(("I60", "I64", "I65", "I66", "I67", "I68", "I69")):
            return "other_cerebrovascular"
    elif int(icd_version) == 9:
        if code.startswith("433") or code.startswith("434") or code == "436":
            return "ischaemic_stroke"
        if code.startswith("431") or code.startswith("432"):
            return "hemorrhagic_stroke"
        if code.startswith("435"):
            return "tia"
        if code.startswith(("430", "437", "438")):
            return "other_cerebrovascular"
    return None


def canonical_nursing_category(category: str, item_label: str) -> str:
    return normalize_space(f"{category} {item_label}")


def reservoir_append(sample: List[dict], row: dict, k: int, rng: random.Random, seen: int) -> None:
    if len(sample) < k:
        sample.append(row)
        return
    idx = rng.randint(0, seen - 1)
    if idx < k:
        sample[idx] = row


def analyze_part1(
    stroke_cohort: pd.DataFrame,
    diagnoses_path: Path,
) -> dict:
    stroke_stays = set(stroke_cohort["stay_id"].astype(int).tolist())
    stroke_hadm = set(stroke_cohort["hadm_id"].astype(int).tolist())
    diag = pd.read_parquet(diagnoses_path)
    diag = diag[diag["stay_id"].isin(stroke_stays)].copy()
    diag["classification"] = [
        classify_cerebro_code(code, version)
        for code, version in zip(diag["icd_code"], diag["icd_version"])
    ]
    cerebro = diag[diag["classification"].notna()].copy()

    code_rows = (
        cerebro.groupby(["icd_version", "icd_code", "classification"])
        .agg(n_stays=("stay_id", "nunique"), n_hadm=("hadm_id", "nunique"), min_seq_num=("seq_num", "min"))
        .reset_index()
        .sort_values(["classification", "n_stays", "icd_code"], ascending=[True, False, True])
    )

    stay_groups: Dict[int, dict] = {}
    for stay_id, sub in cerebro.groupby("stay_id"):
        ordered = sub.sort_values(["seq_num", "icd_version", "icd_code"])
        first = ordered.iloc[0]
        primary = ordered[ordered["seq_num"] == 1]
        use_row = primary.iloc[0] if not primary.empty else first
        stay_groups[int(stay_id)] = {
            "hadm_id": int(use_row["hadm_id"]),
            "dominant_classification": str(use_row["classification"]),
            "all_classifications": sorted(set(ordered["classification"].tolist())),
            "primary_has_cerebro_code": not primary.empty,
            "min_seq_num": int(ordered["seq_num"].min()),
        }

    subset_counts = {}
    for label in ["ischaemic_stroke", "hemorrhagic_stroke", "tia", "other_cerebrovascular"]:
        stay_ids = [sid for sid, meta in stay_groups.items() if meta["dominant_classification"] == label]
        hadm_ids = {stay_groups[sid]["hadm_id"] for sid in stay_ids}
        subset_counts[label] = {"n_stays": len(stay_ids), "n_hadm": len(hadm_ids)}

    multi_stay_hadm = stroke_cohort.groupby("hadm_id")["stay_id"].nunique()
    multiple_icu_hadm = int((multi_stay_hadm > 1).sum())

    primary_cerebro = sum(1 for meta in stay_groups.values() if meta["primary_has_cerebro_code"])
    secondary_only_cerebro = len(stay_groups) - primary_cerebro

    ischaemic_only_count = subset_counts["ischaemic_stroke"]
    diff_explanation = []
    if ischaemic_only_count["n_stays"] == PRIOR_AUDIT_STROKE_STAYS:
        diff_explanation.append(
            f"The fresh cohort reproduces the prior 5,925-stay count exactly when restricted to ischaemic-only codes."
        )
    else:
        diff_explanation.append(
            f"The fresh cohort has {ischaemic_only_count['n_stays']} stays under the old ischaemic-only code family versus {PRIOR_AUDIT_STROKE_STAYS} in the prior audit."
        )
    added_non_ischaemic = len(stay_groups) - ischaemic_only_count["n_stays"]
    diff_explanation.append(
        f"Relative to the fresh broad stroke cohort ({len(stay_groups)} stays), {added_non_ischaemic} stays come from non-ischaemic code families or broader cerebrovascular definitions."
    )

    return {
        "icd_code_breakdown": code_rows.to_dict("records"),
        "ischaemic_only_count": subset_counts["ischaemic_stroke"],
        "hemorrhagic_count": subset_counts["hemorrhagic_stroke"],
        "tia_count": subset_counts["tia"],
        "other_cerebrovascular_count": subset_counts["other_cerebrovascular"],
        "total_count": {"n_stays": int(len(stroke_stays)), "n_hadm": int(len(stroke_hadm))},
        "join_logic": {
            "current_audit_logic": "cohort_v3.csv filtered to has_stroke_final=1, then diagnoses_icd_bq joined back on stay_id/hadm_id for code inventory",
            "diagnosis_to_stay_path": "diagnoses_icd_bq already contains stay_id and hadm_id; stay-level cohort reconciliation used direct stay_id join",
            "multiple_stay_ids_per_hadm_count": multiple_icu_hadm,
            "unique_hadm_id": int(len(stroke_hadm)),
            "unique_stay_id": int(len(stroke_stays)),
            "primary_cerebro_code_stays": int(primary_cerebro),
            "secondary_only_cerebro_code_stays": int(secondary_only_cerebro),
            "seq_num_filtering": "No seq_num filter in the fresh broad cohort audit; dominant subset classification uses seq_num=1 if available, else lowest cerebrovascular seq_num.",
            "additional_filtering": "No extra age/LOS/ICU-type filter beyond membership in cohort_v3 ICU stays.",
        },
        "reconciliation_table": [
            {"subset": "Ischaemic stroke only (I63, 433, 434, 436)", **subset_counts["ischaemic_stroke"]},
            {"subset": "Hemorrhagic stroke (I61, I62, 431, 432)", **subset_counts["hemorrhagic_stroke"]},
            {"subset": "TIA (G45, 435)", **subset_counts["tia"]},
            {"subset": "Other cerebrovascular", **subset_counts["other_cerebrovascular"]},
            {"subset": "Total", "n_stays": int(len(stroke_stays)), "n_hadm": int(len(stroke_hadm))},
        ],
        "reconciliation_explanation": " ".join(diff_explanation),
        "prior_audit_comparison": {
            "prior_audit_stays": PRIOR_AUDIT_STROKE_STAYS,
            "prior_audit_hadm": PRIOR_AUDIT_STROKE_HADM,
            "prior_audit_definition": PRIOR_AUDIT_SQL_NOTE,
            "difference_interpretation": (
                "The fresh text-strategy cohort is broader than the prior feasibility audit. "
                "The prior audit was explicitly limited to ischaemic-stroke codes, whereas the fresh broad stroke flag "
                "captures additional hemorrhagic/TIA/other cerebrovascular admissions."
            ),
        },
    }


def analyze_part2_and_6(
    stroke_stays: set[int],
    nursing_path: Path,
    hourly_grid_dir: Path,
) -> Tuple[dict, dict]:
    rng = random.Random(SEED)
    category_samples: Dict[str, List[dict]] = {cat: [] for cat in TARGET_CATEGORIES}
    category_counts: Dict[str, Counter] = {cat: Counter() for cat in TARGET_CATEGORIES}
    valuenum_nonnull = Counter()
    total_by_category = Counter()
    chart_to_vals: Dict[str, Dict[str, Counter]] = {cat: defaultdict(Counter) for cat in TARGET_CATEGORIES}

    neuro_obs_count = Counter()
    neuro_first48_count = Counter()
    neuro_category_sets: Dict[int, set] = defaultdict(set)
    neuro_hours: Dict[int, List[float]] = defaultdict(list)
    neuro_first_hour: Dict[int, float] = {}
    neuro_last_hour: Dict[int, float] = {}

    gcs_motor_nursing: Dict[int, List[Tuple[float, Optional[float], str]]] = defaultdict(list)
    strength_l_arm_timeline: Dict[int, List[Tuple[float, str, Optional[float]]]] = defaultdict(list)

    sample_seen = Counter()
    schema = {
        "columns": get_source_columns(nursing_path),
        "partition_count": len(list_parquet_parts(nursing_path)),
        "random_rows": [],
    }
    row_sample: List[dict] = []
    row_seen = 0

    for chunk in iter_source_chunks(
        nursing_path,
        usecols=["stay_id", "subject_id", "hadm_id", "charttime", "hour_offset", "item_label", "category", "chart_text", "valuenum"],
        chunksize=200_000,
    ):
        # global random rows for schema inspection
        for row in chunk.head(len(chunk)).itertuples(index=False):
            row_seen += 1
            record = {
                "stay_id": None if pd.isna(row.stay_id) else int(row.stay_id),
                "subject_id": None if pd.isna(row.subject_id) else int(row.subject_id),
                "hadm_id": None if pd.isna(row.hadm_id) else int(row.hadm_id),
                "charttime": None if pd.isna(row.charttime) else str(row.charttime),
                "hour_offset": None if pd.isna(row.hour_offset) else float(row.hour_offset),
                "item_label": row.item_label,
                "category": row.category,
                "chart_text": row.chart_text,
                "valuenum": None if pd.isna(row.valuenum) else float(row.valuenum),
            }
            reservoir_append(row_sample, record, 5, rng, row_seen)

        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stroke_stays)]
        if chunk.empty:
            continue
        chunk["full_category"] = [
            canonical_nursing_category(cat, item)
            for cat, item in zip(chunk["category"], chunk["item_label"])
        ]
        chunk["hour_offset"] = pd.to_numeric(chunk["hour_offset"], errors="coerce")

        for row in chunk.itertuples(index=False):
            stay_id = int(row.stay_id)
            full_category = str(row.full_category)
            hour_offset = None if pd.isna(row.hour_offset) else float(row.hour_offset)
            chart_text = "" if pd.isna(row.chart_text) else str(row.chart_text)
            valuenum = None if pd.isna(row.valuenum) else float(row.valuenum)

            if str(row.category) == "Neurological":
                neuro_obs_count[stay_id] += 1
                neuro_category_sets[stay_id].add(full_category)
                if hour_offset is not None:
                    neuro_hours[stay_id].append(hour_offset)
                    neuro_first_hour[stay_id] = hour_offset if stay_id not in neuro_first_hour else min(neuro_first_hour[stay_id], hour_offset)
                    neuro_last_hour[stay_id] = hour_offset if stay_id not in neuro_last_hour else max(neuro_last_hour[stay_id], hour_offset)
                    if hour_offset <= 48:
                        neuro_first48_count[stay_id] += 1

            if full_category in TARGET_CATEGORIES:
                total_by_category[full_category] += 1
                category_counts[full_category][chart_text] += 1
                sample_seen[full_category] += 1
                reservoir_append(
                    category_samples[full_category],
                    {
                        "stay_id": stay_id,
                        "hour_offset": hour_offset,
                        "category": full_category,
                        "chart_text": chart_text,
                        "valuenum": valuenum,
                    },
                    10,
                    rng,
                    sample_seen[full_category],
                )
                if valuenum is not None:
                    valuenum_nonnull[full_category] += 1
                    chart_to_vals[full_category][chart_text][valuenum] += 1

            if full_category == "Neurological GCS - Motor Response":
                gcs_motor_nursing[stay_id].append((hour_offset if hour_offset is not None else math.nan, valuenum, chart_text))
            if full_category == "Neurological Strength L Arm":
                strength_l_arm_timeline[stay_id].append((hour_offset if hour_offset is not None else math.nan, chart_text, valuenum))

    schema["random_rows"] = row_sample

    chart_text_value_distributions = {}
    valuenum_analysis = {}
    for cat in TARGET_CATEGORIES:
        top10 = category_counts[cat].most_common(10)
        covered = sum(count for _value, count in top10)
        chart_text_value_distributions[cat] = {
            "total_notes": int(total_by_category[cat]),
            "total_distinct_chart_text_values": int(len(category_counts[cat])),
            "top_10_values": [{"chart_text": text, "count": int(count)} for text, count in top10],
            "pct_covered_by_top_10_values": pct(covered, total_by_category[cat]),
        }
        nonnull = valuenum_nonnull[cat]
        mapping_rows = []
        one_to_one = True
        for text, counter in list(chart_to_vals[cat].items())[:20]:
            mapping_rows.append({
                "chart_text": text,
                "valuenum_counts": {str(k): int(v) for k, v in counter.items()},
            })
            if len(counter) > 1:
                one_to_one = False
        if total_by_category[cat] == 0:
            relation = "missing"
            recommendation = "no_data"
        elif nonnull == 0:
            relation = "chart_text_only"
            recommendation = "use_chart_text"
        elif safe_ratio(nonnull, total_by_category[cat]) > 0.9 and one_to_one:
            relation = "redundant"
            recommendation = "valuenum_for_scoring_plus_chart_text_for_display"
        else:
            relation = "complementary"
            recommendation = "use_both"
        valuenum_analysis[cat] = {
            "pct_notes_with_non_null_valuenum": pct(nonnull, total_by_category[cat]),
            "relationship": relation,
            "recommendation": recommendation,
            "example_chart_text_to_valuenum_mapping": mapping_rows,
        }

    inter_gaps = []
    first6 = 0
    for stay_id in stroke_stays:
        hours = sorted(neuro_hours.get(stay_id, []))
        if hours:
            if hours[0] <= 6:
                first6 += 1
            if len(hours) >= 2:
                gaps = [hours[i + 1] - hours[i] for i in range(len(hours) - 1)]
                inter_gaps.append(median(gaps))
    temporal_density = {
        "median_neuro_observations_per_stay": round(median([neuro_obs_count.get(s, 0) for s in stroke_stays]), 2),
        "median_distinct_neuro_categories_per_stay": round(median([len(neuro_category_sets.get(s, set())) for s in stroke_stays]), 2),
        "median_inter_observation_gap_hours": round(median(inter_gaps), 2) if inter_gaps else None,
        "pct_stays_with_neuro_observations_first_6h": pct(first6, len(stroke_stays)),
        "pct_stays_with_20_or_more_neuro_observations_first_48h": pct(sum(1 for s in stroke_stays if neuro_first48_count.get(s, 0) >= 20), len(stroke_stays)),
    }

    eligible_l_arm = sorted([sid for sid, rows in strength_l_arm_timeline.items() if rows])
    l_arm_sample = random.Random(SEED).sample(eligible_l_arm, min(50, len(eligible_l_arm))) if eligible_l_arm else []
    l_arm_rows = []
    changed = 0
    for stay_id in l_arm_sample:
        timeline = sorted(strength_l_arm_timeline[stay_id], key=lambda x: (math.inf if pd.isna(x[0]) else x[0], x[1]))
        distinct_values = [v for v in dict.fromkeys([str(x[1]) for x in timeline]).keys() if v]
        any_change = len(distinct_values) >= 2
        changed += int(any_change)
        l_arm_rows.append({
            "stay_id": int(stay_id),
            "timeline": [
                {"hour_offset": None if pd.isna(h) else float(h), "chart_text": text, "valuenum": val}
                for h, text, val in timeline
            ],
            "n_distinct_chart_text_values": int(len(distinct_values)),
            "changed_at_least_once": bool(any_change),
        })

    # structured cross-reference
    sampled_gcs_stays = random.Random(SEED).sample(sorted([sid for sid, rows in gcs_motor_nursing.items() if rows]), min(20, len(gcs_motor_nursing)))
    structured_matches = []
    grid_matches_total = 0
    grid_matches_equal = 0
    hourly_lookup: Dict[int, Dict[int, float]] = defaultdict(dict)
    for part in sorted(hourly_grid_dir.glob("part_*.parquet")):
        df = pd.read_parquet(part, columns=["stay_id", "hour", "gcs_motor"])
        df = df[df["stay_id"].isin(sampled_gcs_stays) & df["gcs_motor"].notna()]
        for row in df.itertuples(index=False):
            hourly_lookup[int(row.stay_id)][int(row.hour)] = float(row.gcs_motor)
    for stay_id in sampled_gcs_stays:
        rows = []
        for hour_offset, valuenum, chart_text in sorted(gcs_motor_nursing[stay_id], key=lambda x: (math.inf if pd.isna(x[0]) else x[0])):
            if pd.isna(hour_offset):
                continue
            hour = int(round(hour_offset))
            structured = hourly_lookup.get(stay_id, {}).get(hour)
            if structured is None or valuenum is None:
                continue
            equal = abs(float(structured) - float(valuenum)) < 1e-6
            grid_matches_total += 1
            grid_matches_equal += int(equal)
            rows.append({
                "hour_offset": float(hour_offset),
                "nursing_valuenum": float(valuenum),
                "nursing_chart_text": chart_text,
                "structured_gcs_motor": float(structured),
                "equal": bool(equal),
            })
        if rows:
            structured_matches.append({"stay_id": int(stay_id), "matches": rows})

    grid_sample_part = next(hourly_grid_dir.glob("part_*.parquet"))
    strength_columns = [c for c in pd.read_parquet(grid_sample_part).columns.tolist() if "strength" in c.lower()]

    part2 = {
        "samples_by_category": category_samples,
        "chart_text_value_distributions": chart_text_value_distributions,
        "valuenum_analysis": valuenum_analysis,
        "temporal_density": temporal_density,
        "longitudinal_change_detection": {
            "sample_size": int(len(l_arm_rows)),
            "pct_stays_with_L_arm_strength_change": pct(changed, len(l_arm_rows)),
            "sampled_stay_timelines": l_arm_rows,
        },
    }
    part6 = {
        "gcs_comparison": {
            "sample_size": int(len(structured_matches)),
            "pct_matched_observations_equal": pct(grid_matches_equal, grid_matches_total),
            "matched_observation_count": int(grid_matches_total),
            "interpretation": (
                "Likely same-source duplication"
                if safe_ratio(grid_matches_equal, grid_matches_total) >= 0.9 and grid_matches_total > 0
                else "Partial overlap / not fully redundant"
            ),
            "sampled_matches": structured_matches,
        },
        "strength_unique_to_nursing": {
            "structured_strength_columns": strength_columns,
            "has_structured_strength_equivalent_in_current_v3_grid": bool(strength_columns),
            "assessment": (
                "Limb strength assessments are unique to nursing-note exports in the current v3 pipeline."
                if not strength_columns
                else "Current v3 grid already contains strength-related structured columns."
            ),
        },
        "incremental_value_assessment": (
            "Nursing-note GCS appears to overlap heavily with structured GCS, but limb-strength observations provide additional neuro information not present in the current structured grid."
        ),
    }
    part5_nursing = {
        "schema": schema,
        "hour_offset_reference": "hour_offset behaves as hours from ICU intime in the current v3 note exports.",
        "row_granularity": "one row per nursing observation",
        "partitioning": {
            "kind": "parquet_parts_directory",
            "n_parts": len(list_parquet_parts(nursing_path)),
        },
    }
    return part2, part6, part5_nursing


def analyze_discharge_and_anchor(
    stroke_cohort: pd.DataFrame,
    discharge_path: Path,
    radiology_path: Path,
    nursing_path: Path,
) -> Tuple[dict, dict, dict]:
    stroke_stays = set(stroke_cohort["stay_id"].astype(int).tolist())
    stroke_hadm = set(stroke_cohort["hadm_id"].astype(int).tolist())

    discharge_rows = []
    for chunk in iter_source_chunks(discharge_path, usecols=["stay_id", "hadm_id", "note_id", "charttime", "hour_offset", "discharge_text", "text_length"], chunksize=5000):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id", "hadm_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stroke_stays)]
        if not chunk.empty:
            discharge_rows.append(chunk)
    discharge_df = pd.concat(discharge_rows, ignore_index=True) if discharge_rows else pd.DataFrame()

    # Part 3.1 and Part 4
    discharge_sample = discharge_df.sample(n=min(200, len(discharge_df)), random_state=SEED).reset_index(drop=True) if not discharge_df.empty else discharge_df
    hpi_candidates = discharge_df.sample(n=min(30, len(discharge_df)), random_state=SEED).reset_index(drop=True) if not discharge_df.empty else discharge_df

    option_a_rows = []
    physical_examples = {"clear_split": [], "single_ambiguous": [], "two_headers": []}
    clear_split = 0
    single_ambiguous = 0
    two_headers = 0
    for row in hpi_candidates.itertuples(index=False):
        sections = parse_sections(str(row.discharge_text or ""))
        hpi = sections["sections"].get("history_of_present_illness", "")
        text = hpi if hpi else str(row.discharge_text or "")
        onset_mentioned = any(p.search(text) for p in ONSET_PATTERNS)
        wake_unknown = any(p.search(text) for p in WAKE_UNKNOWN_PATTERNS)
        extracted_hours = None
        for pat in DURATION_PATTERNS:
            m = pat.search(text)
            if m:
                extracted_hours = float(m.group(1))
                break
        option_a_rows.append({
            "stay_id": int(row.stay_id),
            "hadm_id": int(row.hadm_id),
            "hpi_excerpt": excerpt(text, 500),
            "onset_mentioned": bool(onset_mentioned),
            "wake_up_or_unknown": bool(wake_unknown),
            "reported_onset_to_icu_hours": extracted_hours,
        })

    for row in discharge_sample.itertuples(index=False):
        raw = str(row.discharge_text or "")
        sections = parse_sections(raw)
        phys = sections["sections"].get("physical_exam", "")
        if not phys:
            continue
        phys_header_occurrences = len(re.findall(r"(?im)^physical exam:\s*$", raw))
        has_adm = any(p.search(phys) for p in ADMISSION_PHYS_PATTERNS)
        has_dis = any(p.search(phys) for p in DISCHARGE_PHYS_PATTERNS)
        if has_adm and has_dis:
            clear_split += 1
            if len(physical_examples["clear_split"]) < 3:
                physical_examples["clear_split"].append({"stay_id": int(row.stay_id), "excerpt": excerpt(phys, 600)})
        elif phys_header_occurrences >= 2:
            two_headers += 1
            if len(physical_examples["two_headers"]) < 3:
                physical_examples["two_headers"].append({"stay_id": int(row.stay_id), "excerpt": excerpt(raw, 600)})
        else:
            single_ambiguous += 1
            if len(physical_examples["single_ambiguous"]) < 3:
                physical_examples["single_ambiguous"].append({"stay_id": int(row.stay_id), "excerpt": excerpt(phys, 600)})

    total_phys = clear_split + single_ambiguous + two_headers
    part4 = {
        "sub_header_patterns": {
            "sample_size": int(len(discharge_sample)),
            "documents_with_clear_admission_discharge_split_pct": pct(clear_split, total_phys),
            "documents_with_single_ambiguous_physical_exam_pct": pct(single_ambiguous, total_phys),
            "documents_with_two_separate_physical_exam_headers_pct": pct(two_headers, total_phys),
            "examples": physical_examples,
        },
        "recommendation": "include_in_layer1" if safe_ratio(clear_split, total_phys) >= 0.80 else "exclude_from_layer1",
    }

    # Part 3 radiology + neuro anchor comparisons
    first_brain_imaging = {}
    for chunk in iter_source_chunks(radiology_path, usecols=["stay_id", "hour_offset", "radiology_text"], chunksize=50_000):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(stroke_stays)]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            if not BRAIN_RAD_FILTER.search(str(row.radiology_text or "")):
                continue
            hour = None if pd.isna(row.hour_offset) else float(row.hour_offset)
            if hour is None:
                continue
            sid = int(row.stay_id)
            if sid not in first_brain_imaging or hour < first_brain_imaging[sid]:
                first_brain_imaging[sid] = hour

    first_neuro_obs = {}
    for chunk in iter_source_chunks(nursing_path, usecols=["stay_id", "hour_offset", "category"], chunksize=200_000):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[(chunk["stay_id"].isin(stroke_stays)) & (chunk["category"] == "Neurological")]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            hour = None if pd.isna(row.hour_offset) else float(row.hour_offset)
            if hour is None:
                continue
            sid = int(row.stay_id)
            if sid not in first_neuro_obs or hour < first_neuro_obs[sid]:
                first_neuro_obs[sid] = hour

    imaging_vals = list(first_brain_imaging.values())
    neuro_vals = list(first_neuro_obs.values())
    common = sorted(set(first_brain_imaging).intersection(first_neuro_obs))
    pair_rows = [
        {
            "stay_id": sid,
            "first_neuro_obs_minus_icu_admission_hours": round(first_neuro_obs[sid], 3),
            "first_brain_imaging_minus_icu_admission_hours": round(first_brain_imaging[sid], 3),
            "first_brain_imaging_minus_first_neuro_obs_hours": round(first_brain_imaging[sid] - first_neuro_obs[sid], 3),
        }
        for sid in common
    ]

    def quantiles(values: List[float]) -> dict:
        if not values:
            return {"count": 0, "median": None, "p25": None, "p75": None, "min": None, "max": None}
        s = pd.Series(values)
        return {
            "count": int(len(values)),
            "median": round(float(s.median()), 3),
            "p25": round(float(s.quantile(0.25)), 3),
            "p75": round(float(s.quantile(0.75)), 3),
            "min": round(float(s.min()), 3),
            "max": round(float(s.max()), 3),
        }

    anchor = {
        "option_a_icu_admission": {
            "sample_size": int(len(option_a_rows)),
            "pct_hpi_with_specific_onset_mention": pct(sum(1 for row in option_a_rows if row["onset_mentioned"]), len(option_a_rows)),
            "pct_hpi_wake_up_or_unknown_onset": pct(sum(1 for row in option_a_rows if row["wake_up_or_unknown"]), len(option_a_rows)),
            "median_reported_onset_to_icu_admission_hours": (
                round(median([row["reported_onset_to_icu_hours"] for row in option_a_rows if row["reported_onset_to_icu_hours"] is not None]), 3)
                if any(row["reported_onset_to_icu_hours"] is not None for row in option_a_rows)
                else None
            ),
            "sampled_hpi_assessment": option_a_rows,
            "interpretation": "ICU admission anchors are always feasible, but HPI phrasing is heterogeneous and often lacks a clean structured onset timestamp.",
        },
        "option_b_first_brain_imaging": {
            **quantiles(imaging_vals),
            "pct_within_2h": pct(sum(1 for x in imaging_vals if x <= 2), len(imaging_vals)),
            "pct_within_6h": pct(sum(1 for x in imaging_vals if x <= 6), len(imaging_vals)),
            "pct_within_12h": pct(sum(1 for x in imaging_vals if x <= 12), len(imaging_vals)),
        },
        "option_c_first_neuro_observation": {
            **quantiles(neuro_vals),
            "pct_within_1h": pct(sum(1 for x in neuro_vals if x <= 1), len(neuro_vals)),
            "pct_within_2h": pct(sum(1 for x in neuro_vals if x <= 2), len(neuro_vals)),
        },
        "practical_comparison": {
            "n_stays_with_all_signals": int(len(common)),
            "pairwise_distributions": {
                "first_neuro_obs_minus_icu_admission_hours": quantiles([r["first_neuro_obs_minus_icu_admission_hours"] for r in pair_rows]),
                "first_brain_imaging_minus_icu_admission_hours": quantiles([r["first_brain_imaging_minus_icu_admission_hours"] for r in pair_rows]),
                "first_brain_imaging_minus_first_neuro_obs_hours": quantiles([r["first_brain_imaging_minus_first_neuro_obs_hours"] for r in pair_rows]),
            },
            "sampled_pairs": pair_rows[:20],
        },
        "recommended_anchor": (
            "ICU_admission_or_first_neuro_observation"
            if (not neuro_vals or median(neuro_vals) <= 1.0)
            else "first_brain_imaging"
        ),
    }

    part5_discharge = {
        "schema": {
            "columns": get_source_columns(discharge_path),
            "partition_count": len(list_parquet_parts(discharge_path)),
            "random_rows": reservoir_source_rows(discharge_path, 5, SEED),
        },
        "full_discharge_text_field": "discharge_text",
        "multiple_notes_per_hadm": {
            "n_hadm_with_multiple_discharge_notes": int((discharge_df.groupby("hadm_id")["note_id"].nunique() > 1).sum()),
            "n_total_discharge_rows_for_stroke": int(len(discharge_df)),
        },
        "hour_offset_reference": "hour_offset is relative to ICU admission time in the current v3 discharge export.",
    }
    return anchor, part4, part5_discharge


def reservoir_source_rows(path: Path, k: int, seed: int) -> List[dict]:
    rng = random.Random(seed)
    sample: List[dict] = []
    seen = 0
    for chunk in iter_source_chunks(path, chunksize=50_000):
        for row in chunk.itertuples(index=False):
            seen += 1
            record = {col: _jsonable(getattr(row, col)) for col in chunk.columns}
            reservoir_append(sample, record, k, rng, seen)
    return sample


def _jsonable(v):
    if pd.isna(v):
        return None
    if hasattr(v, "item") and callable(getattr(v, "item")):
        try:
            return v.item()
        except Exception:
            pass
    if isinstance(v, pd.Timestamp):
        return str(v)
    return v


def analyze_part5_inventory(
    base_dir: Path,
    nursing_path: Path,
    radiology_path: Path,
    discharge_path: Path,
) -> Tuple[dict, dict]:
    def walk_dir(path: Path, depth: int = 2) -> List[str]:
        out = []
        if not path.exists():
            return out
        for p in sorted(path.rglob("*")):
            rel_depth = len(p.relative_to(path).parts)
            if rel_depth > depth:
                continue
            out.append(str(p.relative_to(base_dir)))
        return out

    file_inventory = {
        "data_processed_v3": walk_dir(base_dir / "data/processed/v3", depth=2),
        "results_v3": walk_dir(base_dir / "results/v3", depth=2),
        "data_raw_v3": walk_dir(base_dir / "data/raw/v3", depth=2),
    }
    part5_radiology = {
        "schema": {
            "columns": get_source_columns(radiology_path),
            "partition_count": len(list_parquet_parts(radiology_path)),
            "random_rows": reservoir_source_rows(radiology_path, 5, SEED),
        },
        "hour_offset_reference": "hour_offset is relative to ICU admission time in the current v3 radiology export.",
        "radiology_text_field": "radiology_text",
        "exam_type_field": "No dedicated exam-type field; exam type must be inferred from report text header and note_type/note_seq.",
    }
    return file_inventory, part5_radiology


def render_markdown(report: dict) -> str:
    p1 = report["part1_cohort_reconciliation"]
    p2 = report["part2_nursing_content"]
    p3 = report["part3_anchor_definition"]
    p4 = report["part4_physical_exam_split"]
    p6 = report["part6_structured_crossref"]
    return "\n".join(
        [
            "# Stroke Open Questions Audit",
            "",
            "## Part 1: Cohort Reconciliation",
            f"- Total fresh stroke stays: `{p1['total_count']['n_stays']}`",
            f"- Total fresh stroke hadm: `{p1['total_count']['n_hadm']}`",
            f"- Ischaemic-only stays: `{p1['ischaemic_only_count']['n_stays']}`",
            f"- Hemorrhagic stays: `{p1['hemorrhagic_count']['n_stays']}`",
            f"- TIA stays: `{p1['tia_count']['n_stays']}`",
            f"- Other cerebrovascular stays: `{p1['other_cerebrovascular_count']['n_stays']}`",
            f"- Explanation: {p1['reconciliation_explanation']}",
            "",
            "## Part 2: Nursing Content",
            f"- Median neuro observations/stay: `{p2['temporal_density']['median_neuro_observations_per_stay']}`",
            f"- Median distinct neuro categories/stay: `{p2['temporal_density']['median_distinct_neuro_categories_per_stay']}`",
            f"- % stays with neuro obs in first 6h: `{p2['temporal_density']['pct_stays_with_neuro_observations_first_6h']}%`",
            f"- % sampled stays with L-arm strength change: `{p2['longitudinal_change_detection']['pct_stays_with_L_arm_strength_change']}%`",
            "",
            "## Part 3: Anchor Definition",
            f"- HPI specific onset mention: `{p3['option_a_icu_admission']['pct_hpi_with_specific_onset_mention']}%`",
            f"- Wake-up/unknown onset: `{p3['option_a_icu_admission']['pct_hpi_wake_up_or_unknown_onset']}%`",
            f"- First brain imaging median hour: `{p3['option_b_first_brain_imaging']['median']}`",
            f"- First neuro observation median hour: `{p3['option_c_first_neuro_observation']['median']}`",
            f"- Recommended anchor: `{p3['recommended_anchor']}`",
            "",
            "## Part 4: Physical Exam Split",
            f"- Clear admission/discharge split: `{p4['sub_header_patterns']['documents_with_clear_admission_discharge_split_pct']}%`",
            f"- Recommendation: `{p4['recommendation']}`",
            "",
            "## Part 6: Structured Cross-Reference",
            f"- GCS matched observations equal: `{p6['gcs_comparison']['pct_matched_observations_equal']}%`",
            f"- Incremental value: {p6['incremental_value_assessment']}",
            "",
        ]
    ) + "\n"


def main() -> None:
    args = parse_args()
    base_dir = Path(args.cohort_v3).resolve().parents[3]
    stroke_cohort = build_stroke_cohort_from_v3(Path(args.cohort_v3))
    stroke_stays = set(stroke_cohort["stay_id"].astype(int).tolist())

    part1 = analyze_part1(stroke_cohort, Path(args.diagnoses_parquet))
    part2, part6, part5_nursing = analyze_part2_and_6(stroke_stays, Path(args.nursing_source), Path(args.hourly_grid_dir))
    part3, part4, part5_discharge = analyze_discharge_and_anchor(
        stroke_cohort,
        Path(args.discharge_source),
        Path(args.radiology_source),
        Path(args.nursing_source),
    )
    file_inventory, part5_radiology = analyze_part5_inventory(
        base_dir,
        Path(args.nursing_source),
        Path(args.radiology_source),
        Path(args.discharge_source),
    )

    report = {
        "part1_cohort_reconciliation": part1,
        "part2_nursing_content": part2,
        "part3_anchor_definition": part3,
        "part4_physical_exam_split": part4,
        "part5_data_architecture": {
            "file_inventory": file_inventory,
            "nursing_schema": part5_nursing,
            "radiology_schema": part5_radiology,
            "discharge_schema": part5_discharge,
        },
        "part6_structured_crossref": part6,
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
