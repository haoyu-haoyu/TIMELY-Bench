#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "feasibility"))

from stroke_phase12_reaudit import (  # type: ignore
    DEFAULT_OUTPUT_COLUMNS,
    build_assignments,
    classify_stroke_family,
    load_broad_stroke_cohort,
    load_diagnoses,
)
from stroke_text_strategy_audit import BRAIN_RAD_FILTER, iter_source_chunks, normalize_space  # type: ignore


ALLOWED_NEURO_CATEGORIES = [
    "Neurological Strength R Arm",
    "Neurological Strength L Arm",
    "Neurological Strength R Leg",
    "Neurological Strength L Leg",
    "Neurological Commands Response",
    "Neurological GCS - Eye Opening",
    "Neurological GCS - Verbal Response",
    "Neurological GCS - Motor Response",
    "Neurological Orientation",
    "Neurological Speech",
    "Neurological RL Strength/Movement",
    "Neurological LU Strength/Movement",
    "Neurological RU Strength/Movement",
]

GCS_CATEGORIES = {
    "Neurological GCS - Eye Opening",
    "Neurological GCS - Verbal Response",
    "Neurological GCS - Motor Response",
}

NEURO_ITEM_LABEL_TO_CATEGORY = {
    "Strength R Arm": "Neurological Strength R Arm",
    "Neurological Strength R Arm": "Neurological Strength R Arm",
    "RU Strength/Movement": "Neurological RU Strength/Movement",
    "Strength L Arm": "Neurological Strength L Arm",
    "Neurological Strength L Arm": "Neurological Strength L Arm",
    "LU Strength/Movement": "Neurological LU Strength/Movement",
    "Strength R Leg": "Neurological Strength R Leg",
    "Neurological Strength R Leg": "Neurological Strength R Leg",
    "RL Strength/Movement": "Neurological RL Strength/Movement",
    "Strength L Leg": "Neurological Strength L Leg",
    "Neurological Strength L Leg": "Neurological Strength L Leg",
    "LL Strength/Movement": "Neurological LL Strength/Movement",
    "Commands Response": "Neurological Commands Response",
    "Neurological Commands Response": "Neurological Commands Response",
    "GCS - Eye Opening": "Neurological GCS - Eye Opening",
    "Neurological GCS - Eye Opening": "Neurological GCS - Eye Opening",
    "GCS - Verbal Response": "Neurological GCS - Verbal Response",
    "Neurological GCS - Verbal Response": "Neurological GCS - Verbal Response",
    "GCS - Motor Response": "Neurological GCS - Motor Response",
    "Neurological GCS - Motor Response": "Neurological GCS - Motor Response",
    "Orientation": "Neurological Orientation",
    "Neurological Orientation": "Neurological Orientation",
    "Speech": "Neurological Speech",
    "Neurological Speech": "Neurological Speech",
}

EXPECTED_TIER_COUNTS = {"A": 3570, "B": 1406, "C": 342, "D": 187}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stroke subtype columns, nursing/radiology timelines, and per-stay metadata for v3.")
    p.add_argument("--cohort-v3", required=True)
    p.add_argument("--diagnoses-parquet", required=True)
    p.add_argument("--nursing-source", required=True)
    p.add_argument("--radiology-source", required=True)
    p.add_argument("--discharge-source", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--summary-json", required=True)
    return p.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def normalize_item_label(text: object) -> str:
    return normalize_space("" if text is None else str(text))


def canonicalize_neuro_category(item_label: object) -> str | None:
    norm = normalize_item_label(item_label)
    mapped = NEURO_ITEM_LABEL_TO_CATEGORY.get(norm)
    if mapped is not None:
        return mapped
    lower = norm.lower()
    if lower.startswith("orientation"):
        return "Neurological Orientation"
    return None


def build_assignments_and_update_cohort(
    cohort_v3: Path,
    diagnoses_parquet: Path,
    out_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broad_stroke = load_broad_stroke_cohort(cohort_v3)
    dx_df = load_diagnoses(diagnoses_parquet, set(broad_stroke["stay_id"].astype(int).tolist()))
    assignments = build_assignments(broad_stroke, dx_df).copy()
    assignments["stroke_subtype_priority"] = assignments["priority_subtype"]
    assignments["stroke_subtype_mixed"] = assignments["mixed_subtype"]

    assign_csv = out_dir / "stroke_subtype_assignments.csv"
    assign_parquet = out_dir / "stroke_subtype_assignments.parquet"
    ensure_parent(assign_csv)
    assignments[DEFAULT_OUTPUT_COLUMNS + ["primary_dx_flag", "stroke_subtype_priority", "stroke_subtype_mixed"]].to_csv(assign_csv, index=False)
    assignments.to_parquet(assign_parquet, index=False)

    full = pd.read_csv(cohort_v3)
    for col in ("stroke_subtype_priority", "stroke_subtype_mixed"):
        if col in full.columns:
            full = full.drop(columns=[col])
    merged = full.merge(
        assignments[["stay_id", "stroke_subtype_priority", "stroke_subtype_mixed"]],
        on="stay_id",
        how="left",
    )
    for col in ("stroke_subtype_priority", "stroke_subtype_mixed"):
        merged[col] = merged[col].fillna("")
    atomic_write_csv(merged, cohort_v3)
    return broad_stroke, assignments, merged


def build_nursing_timeline(nursing_source: Path, ischaemic_stays: set[int], out_path: Path) -> tuple[pd.DataFrame, dict]:
    frames: List[pd.DataFrame] = []
    category_counter = Counter()
    for chunk in iter_source_chunks(
        nursing_source,
        usecols=["stay_id", "hour_offset", "item_label", "category", "chart_text", "valuenum"],
        chunksize=100_000,
    ):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(ischaemic_stays)]
        if chunk.empty:
            continue
        chunk["source_category"] = chunk["category"].map(normalize_item_label)
        chunk = chunk[chunk["source_category"] == "Neurological"].copy()
        if chunk.empty:
            continue
        chunk["item_label_norm"] = chunk["item_label"].map(normalize_item_label)
        chunk["category"] = chunk["item_label_norm"].apply(canonicalize_neuro_category)
        chunk = chunk.dropna(subset=["category"]).copy()
        if chunk.empty:
            continue
        chunk["hour_offset"] = pd.to_numeric(chunk["hour_offset"], errors="coerce")
        chunk = chunk.dropna(subset=["hour_offset"]).copy()
        if chunk.empty:
            continue
        chunk["is_incremental"] = ~chunk["category"].isin(GCS_CATEGORIES)
        out = chunk[["stay_id", "hour_offset", "category", "chart_text", "valuenum", "is_incremental"]].copy()
        out["valuenum"] = pd.to_numeric(out["valuenum"], errors="coerce")
        frames.append(out)
        category_counter.update(out["category"].tolist())

    if frames:
        timeline = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["stay_id", "hour_offset", "category"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        timeline = pd.DataFrame(columns=["stay_id", "hour_offset", "category", "chart_text", "valuenum", "is_incremental"])
    ensure_parent(out_path)
    timeline.to_parquet(out_path, index=False)
    summary = {
        "total_rows": int(len(timeline)),
        "unique_stays": int(timeline["stay_id"].nunique()) if not timeline.empty else 0,
        "incremental_rows": int(timeline["is_incremental"].sum()) if not timeline.empty else 0,
        "non_incremental_rows": int((~timeline["is_incremental"]).sum()) if not timeline.empty else 0,
        "category_distribution": [
            {"category": category, "n_rows": int(count)}
            for category, count in sorted(category_counter.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
    }
    return timeline, summary


def build_radiology_timeline(radiology_source: Path, ischaemic_stays: set[int], out_path: Path) -> tuple[pd.DataFrame, dict]:
    frames: List[pd.DataFrame] = []
    temporal_counter = Counter()
    for chunk in iter_source_chunks(
        radiology_source,
        usecols=["stay_id", "hour_offset", "radiology_text"],
        chunksize=50_000,
    ):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk = chunk[chunk["stay_id"].isin(ischaemic_stays)]
        if chunk.empty:
            continue
        chunk["hour_offset"] = pd.to_numeric(chunk["hour_offset"], errors="coerce")
        chunk = chunk.dropna(subset=["hour_offset"]).copy()
        if chunk.empty:
            continue
        chunk["report_text"] = chunk["radiology_text"].fillna("").astype(str)
        chunk = chunk[chunk["report_text"].map(lambda text: bool(BRAIN_RAD_FILTER.search(text)))]
        if chunk.empty:
            continue
        chunk["temporal_category"] = chunk["hour_offset"].apply(lambda x: "early_diagnostic" if float(x) <= 24.0 else "follow_up")
        out = chunk[["stay_id", "hour_offset", "report_text", "temporal_category"]].copy()
        frames.append(out)
        temporal_counter.update(out["temporal_category"].tolist())

    if frames:
        timeline = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["stay_id", "hour_offset"], kind="mergesort")
            .reset_index(drop=True)
        )
    else:
        timeline = pd.DataFrame(columns=["stay_id", "hour_offset", "report_text", "temporal_category"])
    ensure_parent(out_path)
    timeline.to_parquet(out_path, index=False)
    summary = {
        "total_rows": int(len(timeline)),
        "unique_stays": int(timeline["stay_id"].nunique()) if not timeline.empty else 0,
        "temporal_category_distribution": [
            {"temporal_category": key, "n_rows": int(value)}
            for key, value in sorted(temporal_counter.items())
        ],
    }
    return timeline, summary


def collect_discharge_hadm_ids(discharge_source: Path, hadm_ids: set[int]) -> set[int]:
    found: set[int] = set()
    for chunk in iter_source_chunks(discharge_source, usecols=["hadm_id"], chunksize=10_000):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        if chunk.empty:
            continue
        found.update(set(chunk.loc[chunk["hadm_id"].isin(hadm_ids), "hadm_id"].tolist()))
    return found


def build_stay_metadata(
    cohort_enriched: pd.DataFrame,
    assignments: pd.DataFrame,
    nursing_timeline: pd.DataFrame,
    radiology_timeline: pd.DataFrame,
    discharge_hadm_ids: set[int],
    out_path: Path,
) -> tuple[pd.DataFrame, dict]:
    main_assign = assignments[assignments["stroke_subtype_priority"] == "ischaemic"].copy()
    main_cohort = cohort_enriched[cohort_enriched["stay_id"].isin(main_assign["stay_id"])].copy()
    main = main_cohort.merge(
        main_assign[[
            "stay_id",
            "primary_icd_family",
        ]],
        on="stay_id",
        how="left",
    )
    main["primary_dx_flag"] = main["primary_icd_family"].eq("ischaemic")
    main["has_discharge_summary"] = main["hadm_id"].isin(discharge_hadm_ids)

    if not nursing_timeline.empty and {"stay_id", "is_incremental"}.issubset(nursing_timeline.columns):
        incremental = nursing_timeline[nursing_timeline["is_incremental"]].copy()
        n_incremental = incremental.groupby("stay_id").size().rename("n_incremental_neuro_obs")
    else:
        n_incremental = pd.Series(dtype="int64", name="n_incremental_neuro_obs")
    if not radiology_timeline.empty and {"stay_id", "hour_offset"}.issubset(radiology_timeline.columns):
        rad_counts = radiology_timeline.groupby("stay_id").size().rename("n_brain_radiology_reports")
        first_rad = radiology_timeline.groupby("stay_id")["hour_offset"].min().rename("first_brain_imaging_hour")
        rad_first24 = (
            radiology_timeline[radiology_timeline["hour_offset"] <= 24]
            .groupby("stay_id")
            .size()
            .rename("brain_first24_count")
        )
    else:
        rad_counts = pd.Series(dtype="int64", name="n_brain_radiology_reports")
        first_rad = pd.Series(dtype="float64", name="first_brain_imaging_hour")
        rad_first24 = pd.Series(dtype="int64", name="brain_first24_count")

    main = main.merge(n_incremental, on="stay_id", how="left")
    main = main.merge(rad_counts, on="stay_id", how="left")
    main = main.merge(first_rad, on="stay_id", how="left")
    main = main.merge(rad_first24, on="stay_id", how="left")
    main["n_incremental_neuro_obs"] = pd.to_numeric(main["n_incremental_neuro_obs"], errors="coerce").fillna(0).astype(int)
    main["n_brain_radiology_reports"] = pd.to_numeric(main["n_brain_radiology_reports"], errors="coerce").fillna(0).astype(int)
    main["first_brain_imaging_hour"] = pd.to_numeric(main["first_brain_imaging_hour"], errors="coerce")
    main["has_brain_imaging_first24h"] = pd.to_numeric(main["brain_first24_count"], errors="coerce").fillna(0).astype(int) > 0

    def assign_tier(row: pd.Series) -> str:
        ge10 = int(row["n_incremental_neuro_obs"]) >= 10
        has_discharge = bool(row["has_discharge_summary"])
        if has_discharge and ge10:
            return "A"
        if (not has_discharge) and ge10:
            return "B"
        if has_discharge and (not ge10):
            return "C"
        return "D"

    main["tier"] = main.apply(assign_tier, axis=1)
    main["layer1_eligible"] = main["tier"].isin(["A", "B"])
    main["layer2_eligible"] = main["tier"].isin(["A", "C"])
    metadata = main[[
        "stay_id",
        "stroke_subtype_priority",
        "stroke_subtype_mixed",
        "tier",
        "layer1_eligible",
        "layer2_eligible",
        "has_discharge_summary",
        "n_incremental_neuro_obs",
        "n_brain_radiology_reports",
        "first_brain_imaging_hour",
        "has_brain_imaging_first24h",
        "primary_dx_flag",
    ]].copy()
    metadata = metadata.sort_values("stay_id", kind="mergesort").reset_index(drop=True)
    ensure_parent(out_path)
    metadata.to_parquet(out_path, index=False)

    tier_counts = metadata["tier"].value_counts().to_dict()
    tier_distribution = [
        {"tier": tier, "n_stays": int(tier_counts.get(tier, 0))}
        for tier in ["A", "B", "C", "D"]
    ]
    summary = {
        "total_rows": int(len(metadata)),
        "unique_stays": int(metadata["stay_id"].nunique()),
        "tier_distribution": tier_distribution,
        "layer1_eligible_stays": int(metadata["layer1_eligible"].sum()),
        "layer2_eligible_stays": int(metadata["layer2_eligible"].sum()),
        "has_discharge_summary_stays": int(metadata["has_discharge_summary"].sum()),
        "expected_tier_distribution": EXPECTED_TIER_COUNTS,
    }
    return metadata, summary


def build_summary(
    broad_stroke: pd.DataFrame,
    assignments: pd.DataFrame,
    nursing_summary: dict,
    radiology_summary: dict,
    metadata_summary: dict,
) -> dict:
    broad_n = int(broad_stroke["stay_id"].nunique())
    assignment_missing = int(
        broad_stroke.loc[~broad_stroke["stay_id"].isin(assignments["stay_id"]), "stay_id"].nunique()
    )
    tier_actual = {item["tier"]: item["n_stays"] for item in metadata_summary["tier_distribution"]}
    tier_mismatches = {
        tier: {"expected": expected, "actual": int(tier_actual.get(tier, 0))}
        for tier, expected in EXPECTED_TIER_COUNTS.items()
        if int(tier_actual.get(tier, 0)) != expected
    }
    flags: List[str] = []
    if assignment_missing != 0:
        flags.append(f"join_loss: {assignment_missing} broad stroke stays are missing subtype assignments")
    if nursing_summary["total_rows"] == 0:
        flags.append("unexpected_zero: nursing timeline has zero rows")
    if radiology_summary["total_rows"] == 0:
        flags.append("unexpected_zero: radiology timeline has zero rows")
    if metadata_summary["total_rows"] == 0:
        flags.append("unexpected_zero: stroke metadata has zero rows")
    if tier_mismatches:
        flags.append("tier_mismatch: metadata tier distribution does not match confirmed Phase 1-2 counts")
    if metadata_summary["layer1_eligible_stays"] == 0 or metadata_summary["layer2_eligible_stays"] == 0:
        flags.append("unexpected_zero: one or more layer eligibility counts are zero")

    return {
        "cohort_update": {
            "broad_stroke_stays": broad_n,
            "assignment_missing_stays": assignment_missing,
            "assignment_rows": int(assignments["stay_id"].nunique()),
            "assignment_source_breakdown": [
                {"source": str(source), "n_stays": int(count)}
                for source, count in assignments["assignment_source"].value_counts().items()
            ],
            "cohort_csv_columns_written": ["stroke_subtype_priority", "stroke_subtype_mixed"],
        },
        "nursing_timeline": nursing_summary,
        "radiology_timeline": radiology_summary,
        "metadata": metadata_summary,
        "tier_mismatches": tier_mismatches,
        "flags": flags,
    }


def main() -> None:
    args = parse_args()
    cohort_v3 = Path(args.cohort_v3)
    diagnoses_parquet = Path(args.diagnoses_parquet)
    nursing_source = Path(args.nursing_source)
    radiology_source = Path(args.radiology_source)
    discharge_source = Path(args.discharge_source)
    out_dir = Path(args.out_dir)
    summary_json = Path(args.summary_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(summary_json)

    broad_stroke, assignments, cohort_enriched = build_assignments_and_update_cohort(
        cohort_v3=cohort_v3,
        diagnoses_parquet=diagnoses_parquet,
        out_dir=out_dir,
    )

    ischaemic_stays = set(assignments.loc[assignments["stroke_subtype_priority"] == "ischaemic", "stay_id"].astype(int).tolist())
    ischaemic_hadm = set(assignments.loc[assignments["stroke_subtype_priority"] == "ischaemic", "hadm_id"].astype(int).tolist())

    nursing_timeline, nursing_summary = build_nursing_timeline(
        nursing_source=nursing_source,
        ischaemic_stays=ischaemic_stays,
        out_path=out_dir / "stroke_nursing_neuro_timeline.parquet",
    )
    radiology_timeline, radiology_summary = build_radiology_timeline(
        radiology_source=radiology_source,
        ischaemic_stays=ischaemic_stays,
        out_path=out_dir / "stroke_brain_radiology_timeline.parquet",
    )
    discharge_hadm_ids = collect_discharge_hadm_ids(discharge_source, ischaemic_hadm)
    metadata, metadata_summary = build_stay_metadata(
        cohort_enriched=cohort_enriched,
        assignments=assignments,
        nursing_timeline=nursing_timeline,
        radiology_timeline=radiology_timeline,
        discharge_hadm_ids=discharge_hadm_ids,
        out_path=out_dir / "stroke_stay_metadata.parquet",
    )

    summary = build_summary(
        broad_stroke=broad_stroke,
        assignments=assignments,
        nursing_summary=nursing_summary,
        radiology_summary=radiology_summary,
        metadata_summary=metadata_summary,
    )
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(
        {
            "summary_json": str(summary_json),
            "tier_distribution": metadata_summary["tier_distribution"],
            "flags": summary["flags"],
        },
        indent=2,
        ensure_ascii=False,
    ))


if __name__ == "__main__":
    main()
