#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "feasibility"))

from stroke_phase12_reaudit import load_best_discharge_rows  # type: ignore
from stroke_text_strategy_audit import parse_sections  # type: ignore
from v3.constants import (  # type: ignore
    DEFAULT_CONTEXT_JSONL,
    DEFAULT_DIAGNOSIS_COMORBIDITIES,
    DEFAULT_DIAGNOSIS_PATHWAY_EVENTS,
    DEFAULT_MEDICATION_EVENTS,
    ROOT_DIR,
    V3_PROCESSED_DIR,
    V3_RESULTS_DIR,
    V3_SOURCE_COHORT_FILE,
)
from v3.io_utils import read_table, relativize_value, write_table  # type: ignore


ANTICOAG_KEYWORDS = [
    "apixaban",
    "eliquis",
    "rivaroxaban",
    "xarelto",
    "warfarin",
    "coumadin",
    "dabigatran",
    "pradaxa",
    "edoxaban",
    "heparin",
    "enoxaparin",
    "lovenox",
]
ANTIPLATELET_KEYWORDS = [
    "aspirin",
    "clopidogrel",
    "plavix",
    "ticagrelor",
    "brilinta",
    "prasugrel",
    "dipyridamole",
    "aggrenox",
]
STATIN_KEYWORDS = [
    "atorvastatin",
    "rosuvastatin",
    "pravastatin",
    "simvastatin",
    "statin",
]
MECHANISM_PATTERNS = {
    "cardioembolic": [r"cardioembolic", r"atrial fibrillation", r"a[\-\s]?fib", r"embol(ic|ism)"],
    "large_artery": [r"large artery", r"athero", r"carotid", r"stenosis", r"large vessel"],
    "small_vessel": [r"lacunar", r"small vessel"],
    "cryptogenic": [r"cryptogenic", r"esus", r"undetermined source"],
}
COMPLICATION_PATTERNS = {
    "hemorrhagic_transformation": [r"hemorrhagic transformation", r"hemorrhagic conversion", r"haemorrhagic transformation"],
    "cerebral_edema": [r"cerebral edema", r"cerebral oedema", r"edema", r"oedema", r"midline shift", r"mass effect", r"herniation"],
    "aspiration_pneumonia": [r"aspiration pneumonia", r"aspiration pneumon", r"aspirat"],
}
NIHSS_VALUE_RE = re.compile(r"nihss[^0-9]{0,12}(\d{1,2})", re.IGNORECASE)
LEFT_PATTERN = re.compile(r"\bleft\b", re.IGNORECASE)
RIGHT_PATTERN = re.compile(r"\bright\b", re.IGNORECASE)
BILATERAL_PATTERN = re.compile(r"\bbilateral\b", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 stroke task instances.")
    p.add_argument("--cohort-v3", default=str(V3_SOURCE_COHORT_FILE))
    p.add_argument(
        "--stroke-metadata",
        default=str(V3_PROCESSED_DIR / "stroke" / "stroke_stay_metadata.parquet"),
    )
    p.add_argument(
        "--stroke-nursing-timeline",
        default=str(V3_PROCESSED_DIR / "stroke" / "stroke_nursing_neuro_timeline.parquet"),
    )
    p.add_argument(
        "--stroke-radiology-timeline",
        default=str(V3_PROCESSED_DIR / "stroke" / "stroke_brain_radiology_timeline.parquet"),
    )
    p.add_argument(
        "--diagnosis-comorbidities",
        default=str(DEFAULT_DIAGNOSIS_COMORBIDITIES),
    )
    p.add_argument(
        "--diagnosis-pathway-events",
        default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS),
    )
    p.add_argument(
        "--medication-events",
        default=str(DEFAULT_MEDICATION_EVENTS),
    )
    p.add_argument(
        "--discharge-source",
        default=str((ROOT_DIR / "data" / "raw" / "v3" / "discharge_notes_v3.parquet")),
    )
    p.add_argument(
        "--contexts-jsonl",
        default=str(DEFAULT_CONTEXT_JSONL),
    )
    p.add_argument(
        "--out-dir",
        default=str(V3_PROCESSED_DIR / "stroke" / "tasks"),
    )
    p.add_argument(
        "--summary-json",
        default=str(V3_RESULTS_DIR / "stroke" / "stroke_task_build_summary.json"),
    )
    return p.parse_args()


def normalize_space(text: object) -> str:
    return re.sub(r"\s+", " ", "" if text is None else str(text)).strip()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def strength_site(category: object) -> Optional[Tuple[str, str]]:
    text = normalize_space(category).lower()
    if "r arm" in text or "ru strength" in text:
        return ("right", "arm")
    if "l arm" in text or "lu strength" in text:
        return ("left", "arm")
    if "r leg" in text or "rl strength" in text:
        return ("right", "leg")
    if "l leg" in text or "ll strength" in text:
        return ("left", "leg")
    return None


def compute_first_strength_worsening_hour(strength_df: pd.DataFrame) -> float | None:
    first_hour = None
    for _, group in strength_df.sort_values(["category", "hour_offset"], kind="mergesort").groupby("category", sort=False):
        prev = None
        for row in group.itertuples(index=False):
            val = getattr(row, "valuenum", math.nan)
            if pd.isna(val):
                continue
            hour = float(getattr(row, "hour_offset"))
            if prev is not None and float(val) < prev:
                first_hour = hour if first_hour is None else min(first_hour, hour)
                break
            prev = float(val)
    return first_hour


def compute_latest_side_means(strength_df: pd.DataFrame, anchor_hour: float) -> Dict[str, float]:
    pre = strength_df[strength_df["hour_offset"] <= anchor_hour].copy()
    if pre.empty:
        return {}
    pre["site"] = pre["category"].apply(strength_site)
    pre = pre.dropna(subset=["site"]).copy()
    if pre.empty:
        return {}
    pre[["side", "limb"]] = pd.DataFrame(pre["site"].tolist(), index=pre.index)
    latest = pre.sort_values(["side", "limb", "hour_offset"], kind="mergesort").groupby(["side", "limb"], sort=False).tail(1)
    out = latest.groupby("side")["valuenum"].mean().to_dict()
    return {str(k): float(v) for k, v in out.items() if not pd.isna(v)}


def derive_deficit_side(side_means: Dict[str, float]) -> str:
    left = side_means.get("left")
    right = side_means.get("right")
    if left is None and right is None:
        return "unknown"
    if left is None or right is None:
        return "unknown"
    if left + 0.25 < right:
        return "left"
    if right + 0.25 < left:
        return "right"
    if left < 5.0 or right < 5.0:
        return "bilateral_or_none"
    return "none"


def derive_future_trend(strength_df: pd.DataFrame, anchor_hour: float, affected_side: str, horizon: float = 12.0) -> str:
    if affected_side not in {"left", "right"}:
        return "uncertain"
    pre_means = compute_latest_side_means(strength_df, anchor_hour)
    anchor_mean = pre_means.get(affected_side)
    if anchor_mean is None:
        return "uncertain"
    future = strength_df[(strength_df["hour_offset"] > anchor_hour) & (strength_df["hour_offset"] <= anchor_hour + horizon)].copy()
    if future.empty:
        return "unobserved"
    future["site"] = future["category"].apply(strength_site)
    future = future.dropna(subset=["site"]).copy()
    if future.empty:
        return "unobserved"
    future[["side", "limb"]] = pd.DataFrame(future["site"].tolist(), index=future.index)
    side_future = future[future["side"] == affected_side]
    if side_future.empty:
        return "unobserved"
    future_mean = pd.to_numeric(side_future["valuenum"], errors="coerce").dropna().mean()
    if pd.isna(future_mean):
        return "unobserved"
    if future_mean < anchor_mean - 0.25:
        return "worsening"
    if future_mean > anchor_mean + 0.25:
        return "improving"
    return "stable"

def detect_imaging_side(report_text: object) -> str:
    text = normalize_space(report_text).lower()
    if not text:
        return "unknown"
    bilateral = bool(BILATERAL_PATTERN.search(text))
    left = bool(LEFT_PATTERN.search(text))
    right = bool(RIGHT_PATTERN.search(text))
    if bilateral or (left and right):
        return "bilateral"
    if left:
        return "left"
    if right:
        return "right"
    return "unknown"


def consistency_from_sides(imaging_side: str, deficit_side: str) -> str:
    if imaging_side == "unknown" or deficit_side == "unknown":
        return "uncertain"
    if imaging_side == "bilateral":
        return "consistent" if deficit_side == "bilateral_or_none" else "uncertain"
    if imaging_side == "left" and deficit_side == "right":
        return "consistent"
    if imaging_side == "right" and deficit_side == "left":
        return "consistent"
    if imaging_side == deficit_side:
        return "inconsistent"
    return "uncertain"


def extract_mechanism_label(text: object, afib_flag: int = 0) -> Tuple[str, str]:
    lower = normalize_space(text).lower()
    if not lower:
        return ("unknown", "none")
    for label, patterns in MECHANISM_PATTERNS.items():
        if any(re.search(p, lower) for p in patterns):
            return (label, "text")
    if afib_flag:
        return ("cardioembolic", "comorbidity_proxy")
    return ("unknown", "none")


def extract_treatment_strategy(text: object, med_flags: Dict[str, int]) -> Dict[str, object]:
    lower = normalize_space(text).lower()
    anticoag = any(k in lower for k in ANTICOAG_KEYWORDS) or bool(med_flags.get("has_anticoag_event_168h", 0))
    antiplatelet = any(k in lower for k in ANTIPLATELET_KEYWORDS) or bool(med_flags.get("has_antiplatelet_event_168h", 0))
    statin = any(k in lower for k in STATIN_KEYWORDS) or bool(med_flags.get("has_statin_event_168h", 0))
    if anticoag and antiplatelet:
        strategy = "both"
    elif anticoag:
        strategy = "anticoagulation"
    elif antiplatelet:
        strategy = "antiplatelet"
    else:
        strategy = "neither"
    return {
        "strategy": strategy,
        "mentions_anticoag": int(anticoag),
        "mentions_antiplatelet": int(antiplatelet),
        "mentions_statin": int(statin),
    }


def appropriateness_proxy(strategy: str, mechanism_label: str, afib_flag: int) -> str:
    cardio_context = mechanism_label == "cardioembolic" or bool(afib_flag)
    if cardio_context and strategy in {"anticoagulation", "both"}:
        return "appropriate"
    if cardio_context and strategy == "antiplatelet":
        return "potentially_inappropriate"
    if (not cardio_context) and strategy == "antiplatelet":
        return "appropriate"
    return "uncertain"


def extract_nihss_details(text: object) -> Dict[str, object]:
    raw = normalize_space(text)
    if not raw:
        return {
            "n_nihss_mentions": 0,
            "nihss_values_json": "[]",
            "nihss_admission": None,
            "nihss_peak": None,
            "nihss_discharge": None,
        }
    sentence_like = re.split(r"(?<=[\.;])\s+", raw)
    values: List[Dict[str, object]] = []
    admission = None
    peak = None
    discharge = None
    for sent in sentence_like:
        lower = sent.lower()
        for match in NIHSS_VALUE_RE.finditer(sent):
            value = int(match.group(1))
            bucket = "other"
            if any(k in lower for k in ["presented", "admission", "arrival", "initial", "on presentation"]):
                bucket = "admission"
                admission = value if admission is None else admission
            elif any(k in lower for k in ["peak", "worse", "worsened", "worst", "maximum"]):
                bucket = "peak"
                peak = value if peak is None else max(int(peak), value)
            elif "discharge" in lower:
                bucket = "discharge"
                discharge = value
            values.append({"value": value, "context": bucket, "sentence": sent[:200]})
    return {
        "n_nihss_mentions": len(values),
        "nihss_values_json": json.dumps(values, ensure_ascii=False),
        "nihss_admission": admission,
        "nihss_peak": peak,
        "nihss_discharge": discharge,
    }


def extract_complications(text: object) -> Dict[str, int]:
    lower = normalize_space(text).lower()
    out = {}
    for name, patterns in COMPLICATION_PATTERNS.items():
        out[f"label_{name}"] = int(any(re.search(p, lower) for p in patterns))
    out["label_any_complication"] = int(any(out.values()))
    return out


def load_support_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = read_table(args.stroke_metadata)
    nursing = read_table(args.stroke_nursing_timeline)
    radiology = read_table(args.stroke_radiology_timeline)
    cohort = pd.read_csv(
        args.cohort_v3,
        usecols=["stay_id", "hadm_id", "stroke_subtype_priority", "stroke_subtype_mixed", "primary_icd_code"],
    )
    comorb = read_table(args.diagnosis_comorbidities)
    pathway = read_table(args.diagnosis_pathway_events)
    return metadata, nursing, radiology, cohort, comorb, pathway


def load_medication_flags(path: str | Path, stay_ids: set[int]) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame(columns=["stay_id", "has_anticoag_event_168h", "has_antiplatelet_event_168h", "has_statin_event_168h"])
    meds = read_table(path, columns=["stay_id", "event_name"])
    meds["stay_id"] = pd.to_numeric(meds["stay_id"], errors="coerce").astype("Int64")
    meds = meds.dropna(subset=["stay_id"]).copy()
    meds["stay_id"] = meds["stay_id"].astype(int)
    meds = meds[meds["stay_id"].isin(stay_ids)].copy()
    meds["event_name"] = meds["event_name"].fillna("").astype(str).str.lower()
    rows = []
    for stay_id, g in meds.groupby("stay_id", sort=False):
        names = " ".join(g["event_name"].tolist())
        rows.append(
            {
                "stay_id": int(stay_id),
                "has_anticoag_event_168h": int(any(k in names for k in ANTICOAG_KEYWORDS)),
                "has_antiplatelet_event_168h": int(any(k in names for k in ANTIPLATELET_KEYWORDS)),
                "has_statin_event_168h": int(any(k in names for k in STATIN_KEYWORDS)),
            }
        )
    return pd.DataFrame(rows)


def load_discharge_rows_from_contexts(contexts_jsonl: str | Path, stay_to_hadm: Dict[int, int], stay_ids: set[int]) -> pd.DataFrame:
    path = Path(contexts_jsonl)
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "stay_id",
                "subject_id",
                "hadm_id",
                "note_id",
                "note_seq",
                "charttime",
                "storetime",
                "hour_offset",
                "discharge_text",
                "text_length",
            ]
        )
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            stay_id = pd.to_numeric(rec.get("stay_id"), errors="coerce")
            if pd.isna(stay_id):
                continue
            stay_id = int(stay_id)
            if stay_id not in stay_ids:
                continue
            best = None
            best_len = -1
            for note in rec.get("notes", []):
                if str(note.get("note_type", "")).lower() != "discharge":
                    continue
                text = normalize_space(note.get("text"))
                if not text:
                    continue
                tlen = len(text)
                if tlen > best_len:
                    best = note
                    best_len = tlen
            if best is None:
                continue
            rows.append(
                {
                    "stay_id": stay_id,
                    "subject_id": pd.NA,
                    "hadm_id": int(stay_to_hadm[stay_id]),
                    "note_id": best.get("note_id"),
                    "note_seq": pd.NA,
                    "charttime": best.get("charttime"),
                    "storetime": pd.NA,
                    "hour_offset": best.get("hour"),
                    "discharge_text": normalize_space(best.get("text")),
                    "text_length": best_len,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "stay_id",
                "subject_id",
                "hadm_id",
                "note_id",
                "note_seq",
                "charttime",
                "storetime",
                "hour_offset",
                "discharge_text",
                "text_length",
            ]
        )
    df = pd.DataFrame(rows)
    df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce")
    return df


def build_temporal_tasks(base: pd.DataFrame, nursing: pd.DataFrame, radiology: pd.DataFrame, pathway: pd.DataFrame, out_dir: Path) -> Dict[str, dict]:
    summaries: Dict[str, dict] = {}
    layer1 = base[base["layer1_eligible"]].copy()
    layer1_ids = set(layer1["stay_id"].astype(int).tolist())
    nursing = nursing[nursing["stay_id"].isin(layer1_ids)].copy()
    radiology = radiology[radiology["stay_id"].isin(layer1_ids)].copy()
    pathway = pathway[pathway["stay_id"].isin(layer1_ids)].copy()

    nursing["hour_offset"] = pd.to_numeric(nursing["hour_offset"], errors="coerce")
    nursing["valuenum"] = pd.to_numeric(nursing["valuenum"], errors="coerce")
    radiology["hour_offset"] = pd.to_numeric(radiology["hour_offset"], errors="coerce")
    pathway["event_time_hour"] = pd.to_numeric(pathway["event_time_hour"], errors="coerce")

    strength = nursing[nursing["category"].astype(str).str.contains("Strength", case=False, na=False)].copy()
    by_strength = {int(stay_id): g.sort_values("hour_offset", kind="mergesort").copy() for stay_id, g in strength.groupby("stay_id", sort=False)}
    by_nursing = {int(stay_id): g.sort_values("hour_offset", kind="mergesort").copy() for stay_id, g in nursing.groupby("stay_id", sort=False)}
    by_rad = {int(stay_id): g.sort_values("hour_offset", kind="mergesort").copy() for stay_id, g in radiology.groupby("stay_id", sort=False)}
    by_pathway = {int(stay_id): g.sort_values("event_time_hour", kind="mergesort").copy() for stay_id, g in pathway.groupby("stay_id", sort=False)}

    # S-T1
    st1_rows: List[dict] = []
    for row in layer1.itertuples(index=False):
        stay_id = int(row.stay_id)
        strength_df = by_strength.get(stay_id)
        if strength_df is None or strength_df.empty:
            continue
        for anchor in range(6, 49, 6):
            pre = strength_df[strength_df["hour_offset"] <= anchor]
            if pre.empty:
                continue
            latest = pre.sort_values(["category", "hour_offset"], kind="mergesort").groupby("category", sort=False).tail(1)
            future = strength_df[(strength_df["hour_offset"] > anchor) & (strength_df["hour_offset"] <= anchor + 12)].copy()
            worsening = 0
            first_worsening_hour = None
            worsening_category = ""
            if not future.empty:
                for last_row in latest.itertuples(index=False):
                    cat = str(last_row.category)
                    val = float(last_row.valuenum) if not pd.isna(last_row.valuenum) else math.nan
                    if pd.isna(val):
                        continue
                    future_cat = future[future["category"] == cat]
                    if future_cat.empty:
                        continue
                    worse = future_cat[future_cat["valuenum"] < val]
                    if worse.empty:
                        continue
                    worsening = 1
                    hour = float(worse["hour_offset"].min())
                    if first_worsening_hour is None or hour < first_worsening_hour:
                        first_worsening_hour = hour
                        worsening_category = cat
            st1_rows.append(
                {
                    "task_id": "S-T1",
                    "layer": "Layer1",
                    "stay_id": stay_id,
                    "hadm_id": int(row.hadm_id),
                    "tier": str(row.tier),
                    "anchor_hour": float(anchor),
                    "horizon_hours": 12,
                    "label_strength_worsening": int(worsening),
                    "label_first_worsening_hour": first_worsening_hour,
                    "label_worsening_category": worsening_category,
                }
            )
    st1 = pd.DataFrame(st1_rows)
    write_table(st1, out_dir / "stroke_S-T1_instances.parquet", index=False)
    summaries["S-T1"] = {
        "rows": int(len(st1)),
        "unique_stays": int(st1["stay_id"].nunique()) if not st1.empty else 0,
        "positive_rate": round(float(st1["label_strength_worsening"].mean()), 4) if not st1.empty else None,
    }

    # S-T2
    st2_rows: List[dict] = []
    for row in layer1.itertuples(index=False):
        stay_id = int(row.stay_id)
        strength_df = by_strength.get(stay_id)
        if strength_df is None or strength_df.empty:
            continue
        tmp = strength_df.copy()
        tmp["site"] = tmp["category"].apply(strength_site)
        tmp = tmp.dropna(subset=["site"]).copy()
        if tmp.empty:
            continue
        tmp[["side", "limb"]] = pd.DataFrame(tmp["site"].tolist(), index=tmp.index)
        bilateral_hour = None
        for hour, hour_df in tmp.groupby("hour_offset", sort=True):
            sides = set(hour_df["side"].tolist())
            if {"left", "right"}.issubset(sides):
                bilateral_hour = float(hour)
                break
        if bilateral_hour is None:
            continue
        side_means = compute_latest_side_means(tmp, bilateral_hour)
        affected_side = derive_deficit_side(side_means)
        trend = derive_future_trend(tmp, bilateral_hour, "left" if affected_side == "left" else "right" if affected_side == "right" else affected_side)
        st2_rows.append(
            {
                "task_id": "S-T2",
                "layer": "Layer1",
                "stay_id": stay_id,
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                "anchor_hour": bilateral_hour,
                "label_affected_side": affected_side,
                "label_trend_12h": trend,
                "label_left_mean_at_anchor": side_means.get("left"),
                "label_right_mean_at_anchor": side_means.get("right"),
            }
        )
    st2 = pd.DataFrame(st2_rows)
    write_table(st2, out_dir / "stroke_S-T2_instances.parquet", index=False)
    summaries["S-T2"] = {
        "rows": int(len(st2)),
        "unique_stays": int(st2["stay_id"].nunique()) if not st2.empty else 0,
        "affected_side_distribution": st2["label_affected_side"].value_counts().to_dict() if not st2.empty else {},
    }

    # S-T3
    st3_rows: List[dict] = []
    st3_base = layer1[layer1["has_brain_imaging_first24h"]].copy()
    for row in st3_base.itertuples(index=False):
        stay_id = int(row.stay_id)
        rad_df = by_rad.get(stay_id)
        if rad_df is None or rad_df.empty:
            continue
        first_rad = rad_df.iloc[0]
        anchor = float(first_rad["hour_offset"])
        strength_df = by_strength.get(stay_id)
        side_means = compute_latest_side_means(strength_df, anchor) if strength_df is not None else {}
        deficit_side = derive_deficit_side(side_means)
        imaging_side = detect_imaging_side(first_rad["report_text"])
        consistency = consistency_from_sides(imaging_side, deficit_side)
        st3_rows.append(
            {
                "task_id": "S-T3",
                "layer": "Layer1",
                "stay_id": stay_id,
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                "anchor_hour": anchor,
                "label_imaging_side": imaging_side,
                "label_deficit_side": deficit_side,
                "label_consistency": consistency,
                "report_excerpt": normalize_space(first_rad["report_text"])[:300],
            }
        )
    st3 = pd.DataFrame(st3_rows)
    write_table(st3, out_dir / "stroke_S-T3_instances.parquet", index=False)
    summaries["S-T3"] = {
        "rows": int(len(st3)),
        "unique_stays": int(st3["stay_id"].nunique()) if not st3.empty else 0,
        "consistency_distribution": st3["label_consistency"].value_counts().to_dict() if not st3.empty else {},
    }

    # S-T4
    st4_rows: List[dict] = []
    for row in layer1.itertuples(index=False):
        stay_id = int(row.stay_id)
        nursing_df = by_nursing.get(stay_id)
        if nursing_df is None or nursing_df.empty:
            continue
        strength_df = by_strength.get(stay_id)
        rad_df = by_rad.get(stay_id)
        path_df = by_pathway.get(stay_id)
        first_neuro = float(nursing_df["hour_offset"].min())
        first_worsen = compute_first_strength_worsening_hour(strength_df) if strength_df is not None else None
        first_rad = float(rad_df["hour_offset"].min()) if rad_df is not None and not rad_df.empty else None
        first_path = float(path_df["event_time_hour"].min()) if path_df is not None and not path_df.empty else None
        order = []
        for name, hour in [
            ("neuro", first_neuro),
            ("pathway", first_path),
            ("imaging", first_rad),
            ("worsening", first_worsen),
        ]:
            if hour is not None and not pd.isna(hour):
                order.append((float(hour), name))
        order.sort()
        st4_rows.append(
            {
                "task_id": "S-T4",
                "layer": "Layer1",
                "stay_id": stay_id,
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                "anchor_hour": 48.0,
                "label_first_neuro_hour": first_neuro,
                "label_first_pathway_hour": first_path,
                "label_first_brain_imaging_hour": first_rad,
                "label_first_strength_worsening_hour": first_worsen,
                "label_sequence_signature": " > ".join(name for _, name in order),
            }
        )
    st4 = pd.DataFrame(st4_rows)
    write_table(st4, out_dir / "stroke_S-T4_instances.parquet", index=False)
    summaries["S-T4"] = {
        "rows": int(len(st4)),
        "unique_stays": int(st4["stay_id"].nunique()) if not st4.empty else 0,
    }
    return summaries

def build_retrospective_tasks(
    base: pd.DataFrame,
    discharge_df: pd.DataFrame,
    comorb: pd.DataFrame,
    med_flags: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, dict]:
    summaries: Dict[str, dict] = {}
    layer2 = base[base["layer2_eligible"]].copy()
    layer2_ids = set(layer2["stay_id"].astype(int).tolist())
    dis = discharge_df[discharge_df["stay_id"].isin(layer2_ids)].copy()
    if dis.empty:
        empty = pd.DataFrame(columns=["task_id", "layer", "stay_id"])
        for task_id in ["S-R1", "S-R2", "S-R3", "S-R4"]:
            write_table(empty, out_dir / f"stroke_{task_id}_instances.parquet", index=False)
            summaries[task_id] = {"rows": 0, "unique_stays": 0}
        return summaries

    dis["stay_id"] = pd.to_numeric(dis["stay_id"], errors="coerce").astype(int)
    dis["discharge_text"] = dis["discharge_text"].fillna("").astype(str)
    comorb_small = comorb[["stay_id", "atrial_fibrillation"]].copy() if "atrial_fibrillation" in comorb.columns else pd.DataFrame(columns=["stay_id", "atrial_fibrillation"])
    comorb_small["stay_id"] = pd.to_numeric(comorb_small["stay_id"], errors="coerce").astype("Int64")
    comorb_small = comorb_small.dropna(subset=["stay_id"]).copy()
    comorb_small["stay_id"] = comorb_small["stay_id"].astype(int)
    comorb_small["atrial_fibrillation"] = pd.to_numeric(comorb_small["atrial_fibrillation"], errors="coerce").fillna(0).astype(int)
    med_flags = med_flags.copy()
    if not med_flags.empty:
        med_flags["stay_id"] = pd.to_numeric(med_flags["stay_id"], errors="coerce").astype("Int64")
        med_flags = med_flags.dropna(subset=["stay_id"]).copy()
        med_flags["stay_id"] = med_flags["stay_id"].astype(int)

    base2 = layer2.merge(dis[["stay_id", "discharge_text"]], on="stay_id", how="left")
    base2 = base2.merge(comorb_small, on="stay_id", how="left")
    base2 = base2.merge(med_flags, on="stay_id", how="left")
    base2["atrial_fibrillation"] = pd.to_numeric(base2["atrial_fibrillation"], errors="coerce").fillna(0).astype(int)
    for col in ["has_anticoag_event_168h", "has_antiplatelet_event_168h", "has_statin_event_168h"]:
        if col not in base2.columns:
            base2[col] = 0
        base2[col] = pd.to_numeric(base2[col], errors="coerce").fillna(0).astype(int)
    base2["discharge_text"] = base2["discharge_text"].fillna("").astype(str)

    # S-R1
    sr1_rows = []
    for row in base2.itertuples(index=False):
        mech, mech_source = extract_mechanism_label(row.discharge_text, int(row.atrial_fibrillation))
        sr1_rows.append(
            {
                "task_id": "S-R1",
                "layer": "Layer2",
                "task_mode": "retrospective",
                "stay_id": int(row.stay_id),
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                "label_mechanism": mech,
                "label_source": mech_source,
            }
        )
    sr1 = pd.DataFrame(sr1_rows)
    write_table(sr1, out_dir / "stroke_S-R1_instances.parquet", index=False)
    summaries["S-R1"] = {
        "rows": int(len(sr1)),
        "unique_stays": int(sr1["stay_id"].nunique()) if not sr1.empty else 0,
        "mechanism_distribution": sr1["label_mechanism"].value_counts().to_dict() if not sr1.empty else {},
    }

    # S-R2
    sr2_rows = []
    for row in base2.itertuples(index=False):
        mech, _ = extract_mechanism_label(row.discharge_text, int(row.atrial_fibrillation))
        meds = extract_treatment_strategy(
            row.discharge_text,
            {
                "has_anticoag_event_168h": int(row.has_anticoag_event_168h),
                "has_antiplatelet_event_168h": int(row.has_antiplatelet_event_168h),
                "has_statin_event_168h": int(row.has_statin_event_168h),
            },
        )
        sr2_rows.append(
            {
                "task_id": "S-R2",
                "layer": "Layer2",
                "task_mode": "retrospective",
                "stay_id": int(row.stay_id),
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                "label_strategy": meds["strategy"],
                "label_appropriateness_proxy": appropriateness_proxy(meds["strategy"], mech, int(row.atrial_fibrillation)),
                "label_mechanism_context": mech,
                "atrial_fibrillation": int(row.atrial_fibrillation),
                "mentions_anticoag": int(meds["mentions_anticoag"]),
                "mentions_antiplatelet": int(meds["mentions_antiplatelet"]),
                "mentions_statin": int(meds["mentions_statin"]),
            }
        )
    sr2 = pd.DataFrame(sr2_rows)
    write_table(sr2, out_dir / "stroke_S-R2_instances.parquet", index=False)
    summaries["S-R2"] = {
        "rows": int(len(sr2)),
        "unique_stays": int(sr2["stay_id"].nunique()) if not sr2.empty else 0,
        "appropriateness_distribution": sr2["label_appropriateness_proxy"].value_counts().to_dict() if not sr2.empty else {},
    }

    # S-R3
    sr3_rows = []
    for row in base2.itertuples(index=False):
        details = extract_nihss_details(row.discharge_text)
        if int(details["n_nihss_mentions"]) <= 0:
            continue
        sr3_rows.append(
            {
                "task_id": "S-R3",
                "layer": "Layer2",
                "task_mode": "retrospective",
                "stay_id": int(row.stay_id),
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                "n_nihss_mentions": int(details["n_nihss_mentions"]),
                "nihss_values_json": str(details["nihss_values_json"]),
                "label_nihss_admission": details["nihss_admission"],
                "label_nihss_peak": details["nihss_peak"],
                "label_nihss_discharge": details["nihss_discharge"],
            }
        )
    sr3 = pd.DataFrame(sr3_rows)
    write_table(sr3, out_dir / "stroke_S-R3_instances.parquet", index=False)
    summaries["S-R3"] = {
        "rows": int(len(sr3)),
        "unique_stays": int(sr3["stay_id"].nunique()) if not sr3.empty else 0,
    }

    # S-R4
    sr4_rows = []
    for row in base2.itertuples(index=False):
        comp = extract_complications(row.discharge_text)
        sr4_rows.append(
            {
                "task_id": "S-R4",
                "layer": "Layer2",
                "task_mode": "retrospective",
                "stay_id": int(row.stay_id),
                "hadm_id": int(row.hadm_id),
                "tier": str(row.tier),
                **comp,
            }
        )
    sr4 = pd.DataFrame(sr4_rows)
    write_table(sr4, out_dir / "stroke_S-R4_instances.parquet", index=False)
    summaries["S-R4"] = {
        "rows": int(len(sr4)),
        "unique_stays": int(sr4["stay_id"].nunique()) if not sr4.empty else 0,
        "any_complication_rate": round(float(sr4["label_any_complication"].mean()), 4) if not sr4.empty else None,
    }
    return summaries


def build_summary_json(
    base: pd.DataFrame,
    task_summaries: Dict[str, dict],
    summary_path: Path,
    out_dir: Path,
) -> None:
    flags: List[str] = []
    for task_id, meta in task_summaries.items():
        if int(meta.get("rows", 0)) == 0:
            flags.append(f"unexpected_zero: {task_id} has zero rows")
    summary = {
        "base_counts": {
            "main_stays": int(len(base)),
            "layer1_eligible": int(base["layer1_eligible"].sum()),
            "layer2_eligible": int(base["layer2_eligible"].sum()),
            "tier_distribution": base["tier"].value_counts().sort_index().to_dict(),
        },
        "tasks": task_summaries,
        "flags": flags,
        "outputs_dir": str(out_dir),
    }
    ensure_parent(summary_path)
    summary_path.write_text(
        json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = Path(args.summary_json)

    metadata, nursing, radiology, cohort, comorb, pathway = load_support_tables(args)
    base = metadata.merge(
        cohort[["stay_id", "hadm_id", "stroke_subtype_priority", "stroke_subtype_mixed"]],
        on="stay_id",
        how="left",
        suffixes=("", "_cohort"),
    )
    if "stroke_subtype_priority_cohort" in base.columns:
        if "stroke_subtype_priority" in base.columns:
            base["stroke_subtype_priority"] = base["stroke_subtype_priority"].fillna(base["stroke_subtype_priority_cohort"])
        else:
            base["stroke_subtype_priority"] = base["stroke_subtype_priority_cohort"]
    if "stroke_subtype_mixed_cohort" in base.columns:
        if "stroke_subtype_mixed" in base.columns:
            base["stroke_subtype_mixed"] = base["stroke_subtype_mixed"].fillna(base["stroke_subtype_mixed_cohort"])
        else:
            base["stroke_subtype_mixed"] = base["stroke_subtype_mixed_cohort"]
    base = base[base["stroke_subtype_priority"] == "ischaemic"].copy()
    keep_cols = [
        "stay_id",
        "hadm_id",
        "tier",
        "layer1_eligible",
        "layer2_eligible",
        "has_discharge_summary",
        "has_brain_imaging_first24h",
        "stroke_subtype_priority",
        "stroke_subtype_mixed",
        "primary_dx_flag",
    ]
    base = base[keep_cols].drop_duplicates(subset=["stay_id"]).sort_values("stay_id", kind="mergesort").reset_index(drop=True)

    tier_expected = {"A": 3570, "B": 1406, "C": 342, "D": 187}
    tier_actual = base["tier"].value_counts().sort_index().to_dict()
    if tier_actual != tier_expected:
        raise ValueError(f"Stroke metadata tier distribution mismatch: expected {tier_expected}, got {tier_actual}")

    # discharge rows for layer2 stays only
    discharge_hadm = set(base.loc[base["layer2_eligible"], "hadm_id"].astype(int).tolist())
    discharge_source = Path(args.discharge_source)
    if discharge_source.exists():
        discharge_df = load_best_discharge_rows(discharge_source, discharge_hadm)
    else:
        stay_to_hadm = dict(zip(base["stay_id"].astype(int), base["hadm_id"].astype(int)))
        discharge_df = load_discharge_rows_from_contexts(
            args.contexts_jsonl,
            stay_to_hadm=stay_to_hadm,
            stay_ids=set(base.loc[base["layer2_eligible"], "stay_id"].astype(int).tolist()),
        )

    med_flags = load_medication_flags(args.medication_events, set(base["stay_id"].astype(int).tolist()))

    temporal_summaries = build_temporal_tasks(base, nursing, radiology, pathway, out_dir)
    retrospective_summaries = build_retrospective_tasks(base, discharge_df, comorb, med_flags, out_dir)
    task_summaries = {**temporal_summaries, **retrospective_summaries}
    build_summary_json(base, task_summaries, summary_path, out_dir)
    print(json.dumps({"summary_json": str(summary_path), "tasks": task_summaries}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
