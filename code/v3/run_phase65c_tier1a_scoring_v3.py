#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "cres_v3" / "phase65c_tier1a_scores"
DEFAULT_RESPONSES_DIR = ROOT_DIR / "results" / "cres_v3" / "phase65c_tier1a_full"
DEFAULT_SAMPLE_PATH = ROOT_DIR / "data" / "processed" / "v3" / "cres" / "cres_eval_sample_12k.parquet"
DEFAULT_PROMPTS_PATH = ROOT_DIR / "data" / "processed" / "v3" / "cres" / "cres_eval_prompts_12k.jsonl"

TASK_SPECS = {
    "AKI-T1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "aki" / "tasks" / "aki_stage2plus_instances.parquet", "anchor_col": "anchor_hour"},
    "AKI-S1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "aki" / "tasks" / "aki_rrt_proxy_instances.parquet", "anchor_col": "anchor_hour"},
    "DEL-T1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "delirium" / "tasks" / "delirium_persistence_instances.parquet", "anchor_col": "prediction_hour"},
    "DEL-S1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "delirium" / "tasks" / "delirium_resolution_instances.parquet", "anchor_col": "prediction_hour"},
    "SEP-T1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "sepsis" / "tasks" / "sepsis_shock_instances.parquet", "anchor_col": "prediction_hour"},
    "SEP-S1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "sepsis" / "tasks" / "sepsis_lactate_clearance_instances.parquet", "anchor_col": "prediction_hour"},
    "S-T1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-T1_instances.parquet", "anchor_col": "anchor_hour"},
    "S-T2": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-T2_instances.parquet", "anchor_col": "anchor_hour"},
    "S-T3": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-T3_instances.parquet", "anchor_col": "anchor_hour"},
    "S-T4": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-T4_instances.parquet", "anchor_col": "anchor_hour"},
    "S-R1": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-R1_instances.parquet", "anchor_col": None},
    "S-R2": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-R2_instances.parquet", "anchor_col": None},
    "S-R3": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-R3_instances.parquet", "anchor_col": None},
    "S-R4": {"path": ROOT_DIR / "data" / "processed" / "v3" / "stroke" / "tasks" / "stroke_S-R4_instances.parquet", "anchor_col": None},
}

AUTO_SCORING_RULES = {
    ("AKI-T1", "D1"): {"kind": "event_time", "truth_hour_col": "stage2plus_onset_hour", "anchor_col": "anchor_hour"},
    ("AKI-T1", "D2"): {"kind": "event_time", "truth_hour_col": "stage2plus_onset_hour", "anchor_col": "anchor_hour"},
    ("AKI-T1", "D5"): {"kind": "categorical", "truth_builder": "aki_typicality", "parser": "typicality"},
    ("AKI-S1", "D1"): {"kind": "event_time", "truth_hour_col": "stage1_onset_hour", "anchor_col": "anchor_hour"},
    ("AKI-S1", "D2"): {"kind": "event_time", "truth_hour_col": "rrt_proxy_hour", "anchor_col": "anchor_hour"},
    ("DEL-T1", "D1"): {"kind": "event_time", "truth_hour_col": "delirium_onset_hour", "anchor_col": "prediction_hour"},
    ("DEL-T1", "D2"): {"kind": "event_time", "truth_hour_col": "delirium_onset_hour", "anchor_col": "prediction_hour"},
    ("DEL-S1", "D1"): {"kind": "event_time", "truth_hour_col": "delirium_onset_hour", "anchor_col": "prediction_hour"},
    ("SEP-T1", "D1"): {"kind": "event_time", "truth_hour_col": "sepsis_onset_hour", "anchor_col": "prediction_hour"},
    ("SEP-T1", "D2"): {"kind": "event_time", "truth_hour_col": "shock_onset_hour", "anchor_col": "prediction_hour"},
    ("SEP-T1", "D5"): {"kind": "categorical", "truth_builder": "sepsis_onset_confidence", "parser": "confidence_high_low"},
    ("SEP-S1", "D1"): {"kind": "event_time", "truth_hour_col": "sepsis_onset_hour", "anchor_col": "prediction_hour"},
    ("S-T1", "D1"): {"kind": "event_time", "truth_hour_col": "label_first_worsening_hour", "anchor_col": "anchor_hour"},
    ("S-T1", "D3"): {"kind": "binary_from_trend", "truth_col": "label_strength_worsening", "parser": "trend_binary_worsening"},
    ("S-T2", "D4"): {"kind": "categorical", "truth_col": "label_affected_side", "parser": "affected_side"},
    ("S-T3", "D4"): {"kind": "categorical", "truth_col": "label_consistency", "parser": "consistency"},
    ("S-R1", "D4"): {"kind": "categorical", "truth_col": "label_mechanism", "parser": "mechanism"},
    ("S-R2", "D4"): {"kind": "categorical", "truth_col": "label_strategy", "parser": "strategy"},
    ("S-R3", "D2"): {"kind": "numeric_exact", "truth_col": "label_nihss_peak", "parser": "numeric"},
    ("S-R4", "D4"): {"kind": "binary_from_yesno", "truth_col": "label_any_complication", "parser": "yes_no"},
}

ALL_TASK_DIMENSIONS = {
    "AKI-T1": ["D1", "D2", "D3", "D4", "D5", "D6"],
    "AKI-S1": ["D1", "D2", "D3", "D4", "D6"],
    "DEL-T1": ["D1", "D2", "D3", "D4", "D5", "D6"],
    "DEL-S1": ["D1", "D3", "D4", "D6"],
    "SEP-T1": ["D1", "D2", "D3", "D4", "D5", "D6"],
    "SEP-S1": ["D1", "D2", "D3", "D6"],
    "S-T1": ["D1", "D3", "D4", "D6"],
    "S-T2": ["D4"],
    "S-T3": ["D4"],
    "S-T4": ["D3", "D6"],
    "S-R1": ["D4", "D6"],
    "S-R2": ["D4", "D6"],
    "S-R3": ["D2", "D6"],
    "S-R4": ["D4", "D6"],
}

CONF_MAP = {"low": 0.55, "medium": 0.70, "high": 0.90}
HOUR_RE = re.compile(r"\bhour\s*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)
NUM_RE = re.compile(r"[-+]?[0-9]+(?:\.[0-9]+)?")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Formal Phase 6.5C Tier 1A scoring package")
    p.add_argument("--root", default=str(ROOT_DIR))
    p.add_argument("--responses-dir", default=str(DEFAULT_RESPONSES_DIR))
    p.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
    p.add_argument("--prompts-path", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--variant-id", default="full_multimodal")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonl_iter(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))


def format_anchor(anchor: object) -> str:
    if anchor is None or pd.isna(anchor):
        return "na"
    value = float(anchor)
    return str(int(value)) if value.is_integer() else str(value)


def build_instance_id(task_id: str, stay_id: int, anchor_hour: object | None) -> str:
    if anchor_hour is None or pd.isna(anchor_hour):
        return f"{task_id}::{stay_id}"
    return f"{task_id}::{stay_id}::h{format_anchor(anchor_hour)}"


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def first_hour(text: str) -> Optional[float]:
    match = HOUR_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def first_number(text: str) -> Optional[float]:
    match = NUM_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def confidence_to_prob(confidence: object) -> float:
    value = normalize_text(confidence)
    return CONF_MAP.get(value, 0.67)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j + 2) / 2.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def safe_auroc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=int)
    ys = np.asarray(list(y_score), dtype=float)
    pos = int(yt.sum())
    neg = int((1 - yt).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    ranks = rankdata_average(ys)
    sum_ranks_pos = ranks[yt == 1].sum()
    return float((sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg))


def safe_auprc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=int)
    ys = np.asarray(list(y_score), dtype=float)
    pos = int(yt.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-ys, kind="mergesort")
    yt = yt[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.concatenate(([1.0], precision))
    recall = np.concatenate(([0.0], recall))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def safe_brier(y_true: Iterable[int], y_prob: Iterable[float]) -> float:
    yt = np.asarray(list(y_true), dtype=float)
    yp = np.asarray(list(y_prob), dtype=float)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean((yt - yp) ** 2))


def safe_accuracy(y_true: Iterable[object], y_pred: Iterable[object]) -> float:
    yt = list(y_true)
    yp = list(y_pred)
    if not yt:
        return float("nan")
    return float(sum(int(a == b) for a, b in zip(yt, yp)) / len(yt))


def safe_macro_f1(y_true: Iterable[object], y_pred: Iterable[object]) -> float:
    yt = list(y_true)
    yp = list(y_pred)
    labels = sorted(set(yt) | set(yp))
    if not yt or not labels:
        return float("nan")
    scores = []
    for label in labels:
        tp = sum(1 for a, b in zip(yt, yp) if a == label and b == label)
        fp = sum(1 for a, b in zip(yt, yp) if a != label and b == label)
        fn = sum(1 for a, b in zip(yt, yp) if a == label and b != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(f1)
    return float(sum(scores) / len(scores))


def safe_balanced_accuracy(y_true: Iterable[object], y_pred: Iterable[object]) -> float:
    yt = list(y_true)
    yp = list(y_pred)
    labels = sorted(set(yt))
    if not yt or not labels:
        return float("nan")
    recalls = []
    for label in labels:
        tp = sum(1 for a, b in zip(yt, yp) if a == label and b == label)
        fn = sum(1 for a, b in zip(yt, yp) if a == label and b != label)
        recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
    return float(sum(recalls) / len(recalls))


def parse_event(answer: object) -> Tuple[Optional[int], Optional[float]]:
    text = normalize_text(answer)
    negative_markers = [
        "no documented",
        "no clear",
        "does not",
        "did not",
        "not cross",
        "not met",
        "not documented",
        "not clearly",
        "no conclusion",
        "cannot be determined",
        "insufficient",
    ]
    if any(marker in text for marker in negative_markers) or text.startswith("no"):
        return 0, None
    hour = first_hour(text)
    if text.startswith("yes") or hour is not None:
        return 1, hour
    return None, hour


def parse_trend_binary_worsening(answer: object) -> Optional[int]:
    text = normalize_text(answer)
    if any(x in text for x in ["worsen", "progression", "progressive", "declin"]):
        return 1
    if any(x in text for x in ["stable", "improv", "resolution", "resolving", "better"]):
        return 0
    return None


def parse_yes_no(answer: object) -> Optional[int]:
    text = normalize_text(answer)
    if text.startswith("yes") or text.startswith("yes-") or text.startswith("yes—"):
        return 1
    if text.startswith("no") or text.startswith("no-") or text.startswith("no—"):
        return 0
    return None


def parse_numeric(answer: object) -> Optional[float]:
    text = normalize_text(answer)
    return first_number(text)


def parse_typicality(answer: object) -> Optional[str]:
    text = normalize_text(answer)
    if "atypical" in text:
        return "atypical"
    if "typical" in text:
        return "typical"
    return None


def parse_confidence_high_low(answer: object) -> Optional[str]:
    text = normalize_text(answer)
    if "high-confidence" in text or "high confidence" in text:
        return "high"
    if "low-confidence" in text or "low confidence" in text or "lower-confidence" in text:
        return "low"
    if "atypical" in text:
        return "low"
    return None


def parse_affected_side(answer: object) -> Optional[str]:
    text = normalize_text(answer)
    if "cannot be determined" in text or "unknown" in text or "insufficient" in text or "no conclusion" in text:
        return "unknown"
    if "bilateral" in text or ("left" in text and "right" in text):
        return "bilateral_or_none"
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    if "none" in text or "no focal" in text or "intact motor" in text or "largely intact" in text:
        return "none"
    return None


def parse_consistency(answer: object) -> Optional[str]:
    text = normalize_text(answer)
    if "inconsistent" in text:
        return "inconsistent"
    if "consistent" in text:
        return "consistent"
    if "insufficient" in text or "uncertain" in text or "cannot be determined" in text or "no conclusion" in text:
        return "uncertain"
    return None


def parse_mechanism(answer: object) -> Optional[str]:
    text = normalize_text(answer)
    if "cardioembolic" in text or "atrial fibrillation" in text:
        return "cardioembolic"
    if "large artery" in text or "large-artery" in text or "carotid" in text or "athero" in text:
        return "large_artery"
    if "small vessel" in text or "lacunar" in text:
        return "small_vessel"
    if "cryptogenic" in text:
        return "cryptogenic"
    if "unknown" in text or "cannot be determined" in text or "insufficient" in text:
        return "unknown"
    return None


def parse_strategy(answer: object) -> Optional[str]:
    text = normalize_text(answer)
    has_anticoag = "anticoag" in text
    has_antiplatelet = "antiplatelet" in text
    if "both" in text or (has_anticoag and has_antiplatelet):
        return "both"
    if has_anticoag:
        return "anticoagulation"
    if has_antiplatelet:
        return "antiplatelet"
    if "neither" in text or "no antithrombot" in text:
        return "neither"
    return None


PARSERS = {
    "yes_no": parse_yes_no,
    "numeric": parse_numeric,
    "typicality": parse_typicality,
    "confidence_high_low": parse_confidence_high_low,
    "affected_side": parse_affected_side,
    "consistency": parse_consistency,
    "mechanism": parse_mechanism,
    "strategy": parse_strategy,
    "trend_binary_worsening": parse_trend_binary_worsening,
}


def load_prompts(prompts_path: Path, variant_id: str) -> pd.DataFrame:
    keep = ["prompt_id", "instance_id", "task_id", "dimension_id", "variant_id", "condition", "task_mode"]
    rows = []
    for row in jsonl_iter(prompts_path):
        if row.get("variant_id") != variant_id:
            continue
        rows.append({k: row.get(k) for k in keep})
    return pd.DataFrame(rows)


def load_sample(sample_path: Path) -> pd.DataFrame:
    cols = [
        "instance_id",
        "condition",
        "task_id",
        "anchor_hour",
        "trajectory_tier",
        "left_censored",
        "onset_confidence",
        "stroke_layer",
        "stroke_tier",
        "representation_profile",
    ]
    df = pd.read_parquet(sample_path, columns=cols)
    df["anchor_time_bin"] = pd.to_numeric(df["anchor_hour"], errors="coerce").map(anchor_time_bin)
    return df


def anchor_time_bin(hour: object) -> str:
    if hour is None or pd.isna(hour):
        return "retrospective_or_na"
    value = float(hour)
    if value <= 12:
        return "T=6"
    if value <= 36:
        return "T=24"
    return "T=48"


def load_provider_responses(path: Path) -> pd.DataFrame:
    keep = [
        "prompt_id",
        "instance_id",
        "task_id",
        "dimension_id",
        "variant_id",
        "provider",
        "model_name",
        "parse_success",
        "parsed_response",
    ]
    rows = []
    for row in jsonl_iter(path):
        rows.append({k: row.get(k) for k in keep})
    return pd.DataFrame(rows)


def load_truth_tables() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for task_id, spec in TASK_SPECS.items():
        df = pd.read_parquet(spec["path"]).copy()
        anchor_col = spec["anchor_col"]
        df["instance_id"] = [
            build_instance_id(task_id, int(stay_id), row[anchor_col] if anchor_col else None)
            for stay_id, (_, row) in zip(df["stay_id"].astype(int), df.iterrows())
        ]
        df["task_id"] = task_id
        frames.append(df)
    return pd.concat(frames, ignore_index=True, sort=False)


def sepsis_onset_confidence(row: pd.Series) -> Optional[str]:
    source = normalize_text(row.get("onset_confidence") or row.get("onset_source"))
    if source in {"high", "diagnosis_pathway"}:
        return "high"
    if source in {"low", "fallback_zero"}:
        return "low"
    return None


def aki_typicality(row: pd.Series) -> Optional[str]:
    tier = normalize_text(row.get("trajectory_tier"))
    if not tier:
        return None
    if tier.startswith("typical"):
        return "typical"
    if tier.startswith("atypical"):
        return "atypical"
    return None


TRUTH_BUILDERS = {
    "sepsis_onset_confidence": sepsis_onset_confidence,
    "aki_typicality": aki_typicality,
}


def enrich_truth_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "left_censored" in df.columns:
        df["left_censored"] = df["left_censored"].astype("string")
    if "trajectory_tier" in df.columns:
        df["trajectory_tier"] = df["trajectory_tier"].astype("string")
    if "onset_confidence" in df.columns:
        df["onset_confidence"] = df["onset_confidence"].astype("string")
    if "stroke_tier" in df.columns:
        df["stroke_tier"] = df["stroke_tier"].astype("string")
    return df


def build_joined_frame(args: argparse.Namespace) -> pd.DataFrame:
    prompts = load_prompts(Path(args.prompts_path), args.variant_id)
    sample = load_sample(Path(args.sample_path))
    truth = enrich_truth_columns(load_truth_tables())

    meta = prompts.merge(sample, on="instance_id", how="left", suffixes=("", "_sample"))
    joined = meta.merge(truth, on=["instance_id", "task_id"], how="left", suffixes=("", "_truth"))
    return joined


def derive_truth_and_prediction(row: pd.Series) -> Optional[dict]:
    key = (row["task_id"], row["dimension_id"])
    rule = AUTO_SCORING_RULES.get(key)
    if rule is None:
        return None

    parsed = row.get("parsed_response") or {}
    answer = parsed.get("answer", "")
    confidence = parsed.get("confidence")
    out = {
        "provider": row["provider"],
        "model_name": row["model_name"],
        "prompt_id": row["prompt_id"],
        "instance_id": row["instance_id"],
        "condition": row["condition"],
        "task_id": row["task_id"],
        "dimension_id": row["dimension_id"],
        "score_kind": rule["kind"],
        "anchor_hour": row.get("anchor_hour"),
        "anchor_time_bin": row.get("anchor_time_bin"),
        "trajectory_tier": row.get("trajectory_tier"),
        "left_censored": row.get("left_censored"),
        "onset_confidence": row.get("onset_confidence"),
        "stroke_layer": row.get("stroke_layer"),
        "stroke_tier": row.get("stroke_tier"),
        "representation_profile": row.get("representation_profile"),
        "parse_success": bool(row.get("parse_success")),
        "confidence_value": normalize_text(confidence),
    }

    kind = rule["kind"]
    if kind == "event_time":
        truth_hour = pd.to_numeric(row.get(rule["truth_hour_col"]), errors="coerce")
        anchor_hour = pd.to_numeric(row.get(rule["anchor_col"]), errors="coerce")
        truth_event = int(pd.notna(truth_hour) and pd.notna(anchor_hour) and float(truth_hour) <= float(anchor_hour))
        pred_event, pred_hour = parse_event(answer)
        if pred_event is None:
            return None
        conf_prob = confidence_to_prob(confidence)
        pred_prob = conf_prob if pred_event == 1 else 1.0 - conf_prob
        out.update(
            {
                "truth_binary": truth_event,
                "truth_hour": float(truth_hour) if truth_event else np.nan,
                "pred_binary": int(pred_event),
                "pred_hour": float(pred_hour) if pred_hour is not None else np.nan,
                "pred_prob": pred_prob,
            }
        )
        return out

    if kind == "categorical":
        if "truth_builder" in rule:
            truth_label = TRUTH_BUILDERS[rule["truth_builder"]](row)
        else:
            truth_label = row.get(rule["truth_col"])
        truth_label = normalize_text(truth_label)
        if not truth_label or truth_label in {"<na>", "nan"}:
            return None
        pred_label = PARSERS[rule["parser"]](answer)
        if pred_label is None:
            return None
        out.update({"truth_label": truth_label, "pred_label": normalize_text(pred_label)})
        return out

    if kind == "binary_from_yesno":
        truth_value = pd.to_numeric(row.get(rule["truth_col"]), errors="coerce")
        if pd.isna(truth_value):
            return None
        pred_binary = PARSERS[rule["parser"]](answer)
        if pred_binary is None:
            return None
        conf_prob = confidence_to_prob(confidence)
        pred_prob = conf_prob if pred_binary == 1 else 1.0 - conf_prob
        out.update(
            {
                "truth_binary": int(truth_value),
                "pred_binary": int(pred_binary),
                "pred_prob": pred_prob,
            }
        )
        return out

    if kind == "binary_from_trend":
        truth_value = pd.to_numeric(row.get(rule["truth_col"]), errors="coerce")
        if pd.isna(truth_value):
            return None
        pred_binary = PARSERS[rule["parser"]](answer)
        if pred_binary is None:
            return None
        conf_prob = confidence_to_prob(confidence)
        pred_prob = conf_prob if pred_binary == 1 else 1.0 - conf_prob
        out.update(
            {
                "truth_binary": int(truth_value),
                "pred_binary": int(pred_binary),
                "pred_prob": pred_prob,
            }
        )
        return out

    if kind == "numeric_exact":
        truth_num = pd.to_numeric(row.get(rule["truth_col"]), errors="coerce")
        if pd.isna(truth_num):
            return None
        pred_num = PARSERS[rule["parser"]](answer)
        if pred_num is None:
            return None
        out.update({"truth_numeric": float(truth_num), "pred_numeric": float(pred_num)})
        return out

    return None


def aggregate_event_time(df: pd.DataFrame) -> dict:
    yt = df["truth_binary"].astype(int).tolist()
    yp = df["pred_binary"].astype(int).tolist()
    ys = df["pred_prob"].astype(float).tolist()
    positives = df[df["truth_binary"] == 1].copy()
    if not positives.empty:
        positives["abs_hour_error"] = (positives["truth_hour"] - positives["pred_hour"]).abs()
        exact = float((positives["abs_hour_error"] == 0).mean())
        tol1 = float((positives["abs_hour_error"] <= 1).mean())
        median_abs = float(positives["abs_hour_error"].median()) if positives["abs_hour_error"].notna().any() else float("nan")
    else:
        exact = float("nan")
        tol1 = float("nan")
        median_abs = float("nan")
    return {
        "n_rows": int(len(df)),
        "n_positive_truth": int(sum(yt)),
        "n_positive_pred": int(sum(yp)),
        "binary_accuracy": safe_accuracy(yt, yp),
        "event_presence_auroc": safe_auroc(yt, ys),
        "event_presence_auprc": safe_auprc(yt, ys),
        "brier": safe_brier(yt, ys),
        "positive_exact_hour_rate": exact,
        "positive_tolerance_1h_rate": tol1,
        "median_abs_hour_error": median_abs,
    }


def aggregate_binary(df: pd.DataFrame) -> dict:
    yt = df["truth_binary"].astype(int).tolist()
    yp = df["pred_binary"].astype(int).tolist()
    ys = df["pred_prob"].astype(float).tolist()
    return {
        "n_rows": int(len(df)),
        "binary_accuracy": safe_accuracy(yt, yp),
        "event_presence_auroc": safe_auroc(yt, ys),
        "event_presence_auprc": safe_auprc(yt, ys),
        "brier": safe_brier(yt, ys),
    }


def aggregate_categorical(df: pd.DataFrame) -> dict:
    yt = df["truth_label"].astype(str).tolist()
    yp = df["pred_label"].astype(str).tolist()
    return {
        "n_rows": int(len(df)),
        "accuracy": safe_accuracy(yt, yp),
        "macro_f1": safe_macro_f1(yt, yp),
        "balanced_accuracy": safe_balanced_accuracy(yt, yp),
        "truth_distribution": dict(sorted(Counter(yt).items())),
        "pred_distribution": dict(sorted(Counter(yp).items())),
    }


def aggregate_numeric(df: pd.DataFrame) -> dict:
    truth = df["truth_numeric"].astype(float)
    pred = df["pred_numeric"].astype(float)
    exact = float((truth == pred).mean()) if len(df) else float("nan")
    tol1 = float(((truth - pred).abs() <= 1).mean()) if len(df) else float("nan")
    medae = float((truth - pred).abs().median()) if len(df) else float("nan")
    return {"n_rows": int(len(df)), "exact_match": exact, "tolerance_1": tol1, "median_abs_error": medae}


def aggregate_group(df: pd.DataFrame) -> dict:
    kind = df["score_kind"].iloc[0]
    if kind == "event_time":
        return aggregate_event_time(df)
    if kind in {"binary_from_yesno", "binary_from_trend"}:
        return aggregate_binary(df)
    if kind == "categorical":
        return aggregate_categorical(df)
    if kind == "numeric_exact":
        return aggregate_numeric(df)
    raise ValueError(f"unsupported kind: {kind}")


def flatten_metric_row(base: dict, metrics: dict) -> dict:
    out = dict(base)
    for key, value in metrics.items():
        if isinstance(value, (dict, list)):
            out[key] = json.dumps(value, ensure_ascii=False)
        else:
            out[key] = value
    return out


def build_metric_tables(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[dict] = []
    for keys, group in scored.groupby(["provider", "model_name", "task_id", "dimension_id", "score_kind"], dropna=False):
        provider, model_name, task_id, dimension_id, score_kind = keys
        metrics = aggregate_group(group)
        metric_rows.append(
            flatten_metric_row(
                {
                    "provider": provider,
                    "model_name": model_name,
                    "task_id": task_id,
                    "dimension_id": dimension_id,
                    "score_kind": score_kind,
                },
                metrics,
            )
        )
    per_task_dim = pd.DataFrame(metric_rows).sort_values(["provider", "task_id", "dimension_id"]).reset_index(drop=True)

    strat_rows: List[dict] = []
    strat_keys = [
        "anchor_time_bin",
        "trajectory_tier",
        "left_censored",
        "onset_confidence",
        "stroke_layer",
        "stroke_tier",
        "representation_profile",
    ]
    for strat_key in strat_keys:
        subset = scored[scored[strat_key].notna() & (scored[strat_key].astype(str) != "<NA>")].copy()
        if subset.empty:
            continue
        for keys, group in subset.groupby(["provider", "model_name", "task_id", "dimension_id", "score_kind", strat_key], dropna=False):
            provider, model_name, task_id, dimension_id, score_kind, strat_value = keys
            metrics = aggregate_group(group)
            strat_rows.append(
                flatten_metric_row(
                    {
                        "provider": provider,
                        "model_name": model_name,
                        "task_id": task_id,
                        "dimension_id": dimension_id,
                        "score_kind": score_kind,
                        "strat_key": strat_key,
                        "strat_value": strat_value,
                    },
                    metrics,
                )
            )
    stratified = pd.DataFrame(strat_rows).sort_values(["provider", "task_id", "dimension_id", "strat_key", "strat_value"]).reset_index(drop=True)
    return per_task_dim, stratified


def build_audit(joined: pd.DataFrame, scored: pd.DataFrame, output_dir: Path) -> dict:
    all_pairs = [(task, dim) for task, dims in ALL_TASK_DIMENSIONS.items() for dim in dims]
    supported_pairs = sorted({(task, dim) for task, dim in AUTO_SCORING_RULES})
    deferred_pairs = sorted(set(all_pairs) - set(supported_pairs))

    manual_tail4 = output_dir.parent / "phase65c_tier1a_full" / "gemini31pro_manual_tail4_outputs.json"
    manual_parsefix = output_dir.parent / "phase65c_tier1a_full" / "gemini31pro_manual_parsefix_row.jsonl"

    manual_tail4_count = 0
    manual_parsefix_count = 0
    if manual_tail4.exists():
        manual_tail4_count = len(json.loads(manual_tail4.read_text()))
    if manual_parsefix.exists():
        manual_parsefix_count = sum(1 for _ in jsonl_iter(manual_parsefix))

    pair_counts = joined.groupby(["task_id", "dimension_id"]).size().to_dict()
    scored_counts = scored.groupby(["task_id", "dimension_id"]).size().to_dict()

    return {
        "total_prompt_pairs": len(all_pairs),
        "auto_scored_prompt_pairs": len(supported_pairs),
        "deferred_prompt_pairs": len(deferred_pairs),
        "supported_pairs": [{"task_id": t, "dimension_id": d, "n_rows": int(pair_counts.get((t, d), 0)), "n_scored_rows": int(scored_counts.get((t, d), 0))} for t, d in supported_pairs],
        "deferred_pairs": [{"task_id": t, "dimension_id": d, "reason": "judge_only_or_no_direct_ground_truth"} for t, d in deferred_pairs],
        "gemini_manual_recovery": {
            "manual_tail4_count": manual_tail4_count,
            "manual_parsefix_count": manual_parsefix_count,
        },
        "notes": [
            "Representation-branch comparisons are not computed in Tier 1A because this run uses only the full_multimodal prompt variant.",
            "D6 evidence attribution remains deferred to later judge/evidence analyses.",
            "Some task-dimension combinations are scored with conservative coarse mappings where only partial automatic ground truth is available (e.g. S-T1 D3 worsening vs non-worsening).",
        ],
    }


def load_provider_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    joined = build_joined_frame(args)

    provider_files = {
        "gpt54": Path(args.responses_dir) / "gpt54_full_responses.jsonl",
        "gemini31pro": Path(args.responses_dir) / "gemini31pro_full_responses.jsonl",
    }

    scored_rows: List[dict] = []
    provider_run_summary = {}

    for provider, path in provider_files.items():
        responses = load_provider_responses(path)
        provider_joined = joined.merge(
            responses,
            on=["prompt_id", "instance_id", "task_id", "dimension_id", "variant_id"],
            how="inner",
            suffixes=("", "_resp"),
        )
        for row in provider_joined.to_dict(orient="records"):
            derived = derive_truth_and_prediction(pd.Series(row))
            if derived is not None:
                scored_rows.append(derived)

        summary_path = Path(args.responses_dir) / f"{provider}_full_summary.json"
        provider_run_summary[provider] = load_provider_summary(summary_path)

    scored = pd.DataFrame(scored_rows)
    per_task_dim, stratified = build_metric_tables(scored)
    audit = build_audit(joined, scored, output_dir)

    scored_prompts_path = output_dir / "phase65c_tier1a_scored_prompts.parquet"
    per_task_dim_path = output_dir / "phase65c_tier1a_per_task_dimension_metrics.csv"
    stratified_path = output_dir / "phase65c_tier1a_stratified_metrics.csv"
    audit_path = output_dir / "phase65c_tier1a_audit.json"
    summary_path = output_dir / "phase65c_tier1a_scoring_summary.json"
    md_path = output_dir / "phase65c_tier1a_formal_summary.md"

    scored.to_parquet(scored_prompts_path, index=False)
    per_task_dim.to_csv(per_task_dim_path, index=False)
    stratified.to_csv(stratified_path, index=False)
    write_json(audit_path, audit)

    summary = {
        "variant_id": args.variant_id,
        "providers": provider_run_summary,
        "auto_scoring": {
            "scored_prompt_rows": int(len(scored)),
            "supported_task_dimensions": int(len(AUTO_SCORING_RULES)),
            "per_task_dimension_rows": int(len(per_task_dim)),
            "stratified_rows": int(len(stratified)),
        },
        "audit_path": str(audit_path),
        "outputs": {
            "scored_prompts": str(scored_prompts_path),
            "per_task_dimension_metrics": str(per_task_dim_path),
            "stratified_metrics": str(stratified_path),
            "formal_summary_markdown": str(md_path),
        },
    }
    write_json(summary_path, summary)

    md_lines = [
        "# Phase 6.5C Tier 1A Formal Summary",
        "",
        "## Run Completeness",
        "",
    ]
    for provider, payload in provider_run_summary.items():
        md_lines.extend(
            [
                f"- `{provider}`: rows={payload['rows']}, ok={payload['ok_rows']}, parse_success={payload['parse_success_rows']}, parse_success_rate={payload['parse_success_rate']:.6f}",
                f"  confidence={json.dumps(payload.get('confidence_distribution', {}), ensure_ascii=False)}",
            ]
        )
    md_lines.extend(
        [
            "",
            "## Auto-Scoring Scope",
            "",
            f"- scored_prompt_rows: `{len(scored)}`",
            f"- supported_task_dimensions: `{len(AUTO_SCORING_RULES)}`",
            f"- deferred_task_dimensions: `{audit['deferred_prompt_pairs']}`",
            "",
            "## Notes",
            "",
        ]
    )
    for note in audit["notes"]:
        md_lines.append(f"- {note}")
    md_lines.append("")
    md_path.write_text("\n".join(md_lines))


if __name__ == "__main__":
    main()
