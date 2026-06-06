#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from run_phase65c_tier1a_scoring_v3 import (
    ALL_TASK_DIMENSIONS,
    AUTO_SCORING_RULES,
    aggregate_group,
    build_joined_frame,
    derive_truth_and_prediction,
    ensure_dir,
    flatten_metric_row,
    jsonl_iter,
    normalize_text,
    parse_confidence_high_low,
    parse_consistency,
    parse_event,
    parse_mechanism,
    parse_numeric,
    parse_strategy,
    parse_yes_no,
    write_json,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_ROOT = ROOT_DIR / "results" / "cres_v3"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "phase65f_frozen_eval"
DEFAULT_SAMPLE_PATH = ROOT_DIR / "data" / "processed" / "v3" / "cres" / "cres_eval_sample_12k.parquet"
DEFAULT_PROMPTS_PATH = ROOT_DIR / "data" / "processed" / "v3" / "cres" / "cres_eval_prompts_12k.jsonl"
DEFAULT_VARIANT_ID = "full_multimodal"
SCORE_BATCH_ROWS = 1000
FROZEN_TIER1A_PROVIDERS = {"gpt54", "gemini31pro"}
MANUAL_JUDGE_CONTESTANT_PROVIDERS = [
    "gpt54",
    "deepseek_chat",
    "aloe70b",
    "medgemma15_4b_it",
]
JUDGE_CONTESTANT_ROSTER_RULE = "manual_fixed_vendor_diversity_and_parameter_range_coverage"
JUDGE_CONTESTANT_ROSTER_RATIONALE = "vendor diversity + parameter range coverage"
JUDGE_CONTESTANT_JUDGE_OVERLAP_NOTE = (
    "GPT-5.4 is included both as a contestant and as one cross-check judge. "
    "Bias risk is documented as controlled because Claude Opus 4.6 remains the primary judge "
    "and GPT-5.4 contributes only one of two cross-check opinions."
)
JUDGE_MODEL_ROSTER = {
    "primary_judge": "Claude Opus 4.6",
    "cross_check_judges": ["GPT-5.4", "Gemini 3.1 Pro"],
}

DIRECT_PROVIDER_SPECS = {
    "gpt54": {
        "tier": "tier1a",
        "canonical_direct": "phase65c_tier1a_full/gpt54_full_responses.jsonl",
        "summary_path": "phase65c_tier1a_full/gpt54_full_summary.json",
    },
    "gemini31pro": {
        "tier": "tier1a",
        "canonical_direct": "phase65c_tier1a_full/gemini31pro_full_responses.jsonl",
        "summary_path": "phase65c_tier1a_full/gemini31pro_full_summary.json",
    },
    "deepseek_chat": {
        "tier": "tier1b",
        "canonical_direct": "phase65d_tier1b_full/deepseek_chat_full_responses.jsonl",
        "summary_path": "phase65d_tier1b_full/deepseek_chat_full_summary.json",
    },
    "qwen35": {
        "tier": "tier1b",
        "canonical_direct": "phase65d_tier1b_full/qwen35_full_responses.jsonl",
        "summary_path": "phase65d_tier1b_full/qwen35_full_summary.json",
    },
    "gemma4_26b": {
        "tier": "tier1b",
        "canonical_direct": "phase65d_tier1b_full/gemma4_26b_full_responses.jsonl",
        "summary_path": "phase65d_tier1b_full/gemma4_26b_full_summary.json",
    },
}

OVERLAY_PROVIDER_SPECS = {
    "aloe70b": {
        "tier": "tier2",
        "source_mode": "overlay_repair_chain",
        "summary_path": "phase65e_tier2_full_aloe70b/aloe70b_full_summary.json",
        "filename_allow_patterns": ["aloe70b"],
        "supporting_summary_paths": [
            "phase65e_tier2_formal_summary.md",
        ],
        "source_dirs": [
            "phase65e_tier2_full_aloe70b",
        ],
    },
    "aloe7b": {
        "tier": "tier2",
        "source_mode": "overlay_repair_chain",
        "summary_path": "phase65e_tier2_aloe7b_repair_final_summary.json",
        "filename_allow_patterns": ["aloe7b"],
        "source_dirs": [
            "phase65e_tier2_full_aloe7b",
            "phase65e_tier2_aloe7b_repair_jsonstrict_v1",
            "phase65e_tier2_aloe7b_repair_compact_v2",
            "phase65e_tier2_aloe7b_repair_ultracompact_v3",
            "phase65e_tier2_aloe7b_repair_reformat_v4",
            "phase65e_tier2_aloe7b_repair_micro_v5",
        ],
    },
    "meditron3_8b": {
        "tier": "tier2",
        "source_mode": "overlay_repair_chain",
        "summary_path": "phase65e_tier2_meditron3_postrepair_summary_v10.json",
        "filename_allow_patterns": ["meditron3_8b"],
        "source_dirs": [
            "phase65e_tier2_full_meditron3_8b",
            "phase65e_tier2_meditron3_repairpilot16_schema",
            "phase65e_tier2_meditron3_repairpilot24_promptcompact",
            "phase65e_tier2_meditron3_repairpilot24_compact2",
            "phase65e_tier2_meditron3_tailrepair_v2",
            "phase65e_tier2_meditron3_tailrepair_v3",
            "phase65e_tier2_meditron3_tailrepair_v4",
            "phase65e_tier2_meditron3_tailrepair_v5",
            "phase65e_tier2_meditron3_tailrepair_v6",
            "phase65e_tier2_meditron3_tailrepair_v7",
            "phase65e_tier2_meditron3_tailrepair_v8",
            "phase65e_tier2_meditron3_tailrepair_v9",
            "phase65e_tier2_meditron3_tailrepair_v10",
            "phase65e_tier2_meditron3_repairpilot40",
            "phase65e_tier2_meditron3_promptcompact_fullrepair",
        ],
    },
    "medgemma15_4b_it": {
        "tier": "tier2",
        "source_mode": "overlay_repair_chain",
        "summary_path": "phase65e_tier2_medgemma_final151_twostage_20260417/summary.json",
        "filename_allow_patterns": ["medgemma15_4b", "medgemma15_4b_two_stage"],
        "supporting_summary_paths": [
            "phase65e_tier2_formal_summary.md",
            "phase65e_tier2_medgemma_dualrepair_v1/phase65e_medgemma_dualrepair_v1_summary.json",
            "phase65e_tier2_medgemma_vllm_fullrepair1009_max32k/phase65e_medgemma_fullrepair1009_summary.json",
            "phase65e_tier2_medgemma_vllm_fullrepair1009_max40k_pass2/phase65e_medgemma_fullrepair1009_unresolved_summary.json",
            "phase65e_tier2_medgemma_vllm_fullrepair1009_max50k_pass3/phase65e_medgemma_fullrepair1009_unresolved_pass3_summary.json",
        ],
        "source_dirs": [
            "phase65e_tier2_full",
            "phase65e_tier2_full_b200a",
            "phase65e_tier2_full_b200b",
            "phase65e_tier2_full_h200pair_a",
            "phase65e_tier2_full_h200pair_b",
            "phase65e_tier2_medgemma_global6",
            "phase65e_tier2_medgemma_salvage_v1",
            "phase65e_tier2_medgemma_scaleout_v2",
            "phase65e_tier2_medgemma_dualrepair_v1",
            "phase65e_tier2_medgemma_repair33210778_compact1800",
            "phase65e_tier2_medgemma_repair33257488_b200_bs32",
            "phase65e_tier2_medgemma_repair33257488_b200_bs32_max8000",
            "phase65e_tier2_medgemma_repair33257488_b200_bs8_max5000",
            "phase65e_tier2_medgemma_repair_b200_h200_round3",
            "phase65e_tier2_medgemma_repair_h200_round4",
            "phase65e_tier2_medgemma_hardtail_v3",
            "phase65e_tier2_medgemma_repair_split_20260416",
            "phase65e_tier2_medgemma_vllm_fullrepair1009_max32k",
            "phase65e_tier2_medgemma_vllm_fullrepair1009_max40k_pass2",
            "phase65e_tier2_medgemma_vllm_fullrepair1009_max50k_pass3_sharded",
            "phase65e_tier2_medgemma_final151_twostage_20260417",
        ],
    },
}

PROVIDER_SPECS = {
    **{
        provider: {
            "tier": spec["tier"],
            "source_mode": "direct_merged",
            **spec,
        }
        for provider, spec in DIRECT_PROVIDER_SPECS.items()
    },
    **OVERLAY_PROVIDER_SPECS,
}

DEFERRED_STRAT_KEYS = [
    "trajectory_tier",
    "left_censored",
    "onset_confidence",
    "stroke_layer",
    "stroke_tier",
]

PRIMARY_SCORE_BY_KIND = {
    "event_time": "binary_accuracy",
    "binary_from_yesno": "binary_accuracy",
    "binary_from_trend": "binary_accuracy",
    "categorical": "macro_f1",
    "numeric_exact": "exact_match",
}

JUDGE_RUBRIC_MD = """# Phase 6.5F Judge Rubric

Each judged row corresponds to one contestant response to one frozen `full_multimodal`
CRES prompt. Judges must score the response using the following schema:

- `overall_quality_1to5`
- `clinical_correctness_1to5`
- `temporal_grounding_1to5_or_na`
- `evidence_grounding_1to5`
- `confidence_calibration_1to5`
- `brief_rationale`

Scoring guidance:

- `overall_quality_1to5`
  - holistic judgment of answer usefulness and correctness
- `clinical_correctness_1to5`
  - whether the answer matches the clinical truth implied by the benchmark context
- `temporal_grounding_1to5_or_na`
  - whether timestamps and ordering are used correctly for temporal tasks
  - use `na` when temporal grounding is not relevant
- `evidence_grounding_1to5`
  - whether cited measurements / notes support the answer
- `confidence_calibration_1to5`
  - whether stated confidence matches the actual uncertainty
- `brief_rationale`
  - short explanation grounded in the prompt and response
"""

MEDGEMMA_ANSWER_RE = re.compile(r'"answer"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"', re.DOTALL)
MEDGEMMA_CONFIDENCE_RE = re.compile(r'"confidence"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"', re.DOTALL)
MEDGEMMA_EVIDENCE_RE = re.compile(
    r'\{\s*"timestamp"\s*:\s*"(?P<timestamp>(?:\\.|[^"\\])*)"\s*,\s*"measurement"\s*:\s*"(?P<measurement>(?:\\.|[^"\\])*)"\s*,\s*"value"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"',
    re.DOTALL,
)
MEDGEMMA_INLINE_EVIDENCE_RE = re.compile(
    r"(?P<measurement>[A-Za-z][A-Za-z0-9_ /()%.\-]{1,48})\s*[=:]\s*(?P<value>[<>]?\d+(?:\.\d+)?(?:\s*[A-Za-z/%]+)?)\s+at Hour\s+(?P<hour>\d+)",
    re.IGNORECASE,
)
MEDGEMMA_HOUR_FIRST_EVIDENCE_RE = re.compile(
    r"Hour\s+(?P<hour>\d+)\s*:\s*(?P<measurement>[A-Za-z][A-Za-z0-9_ /()%.\-]{1,48})\s*[=:]\s*(?P<value>[<>]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
MEDGEMMA_NIHSS_RE = re.compile(r"nihss[^0-9><]{0,24}(?P<value>[<>]?\d+(?:\.\d+)?)", re.IGNORECASE)
MEDGEMMA_HOUR_RE = re.compile(r"Hour\s+(?P<hour>\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5F frozen package canonicalization, scoring, and judge packet build")
    p.add_argument("--mode", choices=["canonicalize", "score", "finalize_score", "build_judge"], required=True)
    p.add_argument("--project-root", default=str(ROOT_DIR))
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--sample-path", default=str(DEFAULT_SAMPLE_PATH))
    p.add_argument("--prompts-path", default=str(DEFAULT_PROMPTS_PATH))
    p.add_argument("--variant-id", default=DEFAULT_VARIANT_ID)
    p.add_argument("--judge-seed", type=int, default=65)
    p.add_argument("--judge-size", type=int, default=500)
    p.add_argument("--judge-per-condition", type=int, default=125)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_quality(row: dict | None) -> tuple[int, int]:
    if row is None:
        return (-1, -1)
    status_ok = int(row.get("status") == "ok")
    parse_ok = int(row.get("parse_success") is True)
    return (status_ok + parse_ok, parse_ok)


def normalize_confidence_value(row: dict) -> Optional[str]:
    value = row.get("confidence_value")
    if value:
        value = normalize_text(value)
        if value in {"high", "medium", "low"}:
            return value
    parsed = row.get("parsed_response") or {}
    if isinstance(parsed, dict):
        parsed_value = normalize_text(parsed.get("confidence"))
        if parsed_value in {"high", "medium", "low"}:
            return parsed_value
    return None


def read_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _json_unescape(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except Exception:  # noqa: BLE001
        return text


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].strip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def extract_partial_reasoning(raw: str) -> str:
    marker = '"reasoning"'
    idx = raw.find(marker)
    if idx < 0:
        return ""
    key_end = raw.find(":", idx + len(marker))
    if key_end < 0:
        return ""
    value_start = raw.find('"', key_end)
    if value_start < 0:
        return ""
    content = raw[value_start + 1 :]
    end_candidates = []
    for token in ['",\n  "answer"', '", "answer"', '"\n  "answer"', '",\n  "confidence"', '", "confidence"']:
        token_idx = content.find(token)
        if token_idx >= 0:
            end_candidates.append(token_idx)
    if end_candidates:
        content = content[: min(end_candidates)]
    return _json_unescape(content).strip()


def compact_reasoning(text: str, max_len: int = 500) -> str:
    compact = strip_code_fence(text)
    compact = re.sub(r"<unused\d+>thought", " ", compact, flags=re.IGNORECASE)
    compact = re.sub(r"```json", " ", compact, flags=re.IGNORECASE)
    compact = compact.replace("```", " ")
    compact = re.sub(r"\s+", " ", compact).strip()
    if not compact:
        return ""
    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", compact)
    if sentence_match:
        compact = sentence_match.group(1).strip()
    return compact[:max_len]


def extract_generic_evidence(raw: str, limit: int = 3) -> List[dict]:
    evidence_items: List[dict] = []
    seen: set[tuple[str, str, str]] = set()

    def maybe_add(hour: str, measurement: str, value: str) -> None:
        timestamp = f"Hour {hour}".strip()
        measurement = measurement.strip().strip("\"'")[:64]
        value = value.strip().strip("\"'")[:128]
        if not (timestamp and measurement and value):
            return
        key = (timestamp, measurement, value)
        if key in seen:
            return
        evidence_items.append({"timestamp": timestamp, "measurement": measurement, "value": value})
        seen.add(key)

    for match in MEDGEMMA_INLINE_EVIDENCE_RE.finditer(raw):
        maybe_add(match.group("hour"), match.group("measurement"), match.group("value"))
        if len(evidence_items) >= limit:
            return evidence_items
    for match in MEDGEMMA_HOUR_FIRST_EVIDENCE_RE.finditer(raw):
        maybe_add(match.group("hour"), match.group("measurement"), match.group("value"))
        if len(evidence_items) >= limit:
            return evidence_items
    return evidence_items


def mechanism_answer_from_raw(raw: str) -> str:
    text = normalize_text(raw)
    label = parse_mechanism(text)
    if label is None:
        if any(token in text for token in ["septic embol", "endocarditis", "embolic", "emboli", "atrial fibrillation", "afib"]):
            label = "cardioembolic"
        elif any(token in text for token in ["moyamoya", "carotid", "ica", "mca", "vertebral", "basilar", "stenosis", "athero", "atheromatous"]):
            label = "large_artery"
        else:
            label = "unknown"
    mapping = {
        "cardioembolic": "Cardioembolic stroke",
        "large_artery": "Large artery atherosclerosis",
        "small_vessel": "Small vessel / lacunar stroke",
        "cryptogenic": "Cryptogenic stroke",
        "unknown": "Unknown mechanism",
    }
    return mapping[label]


def strategy_answer_from_raw(raw: str) -> str:
    text = normalize_text(raw)
    label = parse_strategy(text)
    if label is None:
        anticoag_tokens = [
            "anticoag",
            "heparin",
            "apixaban",
            "warfarin",
            "argatroban",
            "enoxaparin",
            "rivaroxaban",
            "dabigatran",
        ]
        antiplatelet_tokens = [
            "antiplatelet",
            "aspirin",
            "clopidogrel",
            "plavix",
            "ticagrelor",
            "dipyridamole",
        ]
        has_anticoag = any(token in text for token in anticoag_tokens)
        has_antiplatelet = any(token in text for token in antiplatelet_tokens)
        if has_anticoag and has_antiplatelet:
            label = "both"
        elif has_anticoag:
            label = "anticoagulation"
        elif has_antiplatelet:
            label = "antiplatelet"
        else:
            label = "neither"
    mapping = {
        "both": "Both antiplatelet and anticoagulation",
        "anticoagulation": "Anticoagulation",
        "antiplatelet": "Antiplatelet",
        "neither": "Neither",
    }
    return mapping[label]


def yes_no_answer_from_raw(raw: str) -> str:
    parsed = parse_yes_no(raw)
    if parsed is not None:
        return "Yes" if parsed == 1 else "No"
    text = normalize_text(raw)
    negative_tokens = [
        "no complication",
        "no evidence of complication",
        "no hemorrhagic conversion",
        "no hemorrhage",
        "no seizure",
        "no edema",
    ]
    if any(token in text for token in negative_tokens):
        return "No"
    positive_tokens = [
        "complication",
        "hemorrhagic",
        "hemorrhage",
        "edema",
        "aspiration",
        "pneumonia",
        "seizure",
        "deterioration",
        "worsening",
        "reintubat",
    ]
    if any(token in text for token in positive_tokens):
        return "Yes"
    return "No"


def numeric_answer_from_raw(raw: str) -> str:
    nihss_values = MEDGEMMA_NIHSS_RE.findall(raw)
    if nihss_values:
        def numeric_key(token: str) -> float:
            clean = token.replace(">", "").strip()
            try:
                return float(clean)
            except Exception:  # noqa: BLE001
                return float("-inf")

        return max(nihss_values, key=numeric_key).strip()
    numeric_value = parse_numeric(raw)
    if numeric_value is None:
        return ""
    if float(numeric_value).is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def confidence_answer_from_raw(raw: str) -> str:
    label = parse_confidence_high_low(raw)
    if label is None:
        text = normalize_text(raw)
        if any(token in text for token in ["uncertain", "possible", "suggest", "atypical", "equivocal"]):
            label = "low"
        else:
            label = "high"
    return f"{label} confidence"


def event_answer_from_raw(raw: str) -> str:
    parsed_event, parsed_hour = parse_event(raw)
    if parsed_event is not None:
        if parsed_event == 1 and parsed_hour is not None:
            return f"Yes, at Hour {int(parsed_hour)}"
        return "Yes" if parsed_event == 1 else "No"
    text = normalize_text(raw)
    positive_tokens = ["rrt", "crrt", "cvvh", "dialysis", "hemodialysis", "renal replacement"]
    if any(token in text for token in positive_tokens):
        hour_match = MEDGEMMA_HOUR_RE.search(raw)
        if hour_match:
            return f"Yes, at Hour {hour_match.group('hour')}"
        return "Yes"
    return "No"


def infer_medgemma_answer(row: dict, raw: str, reasoning: str) -> str:
    answer_match = MEDGEMMA_ANSWER_RE.search(raw)
    if answer_match:
        return _json_unescape(answer_match.group("value")).strip()

    key = (row.get("task_id"), row.get("dimension_id"))
    if key == ("S-R1", "D4"):
        return mechanism_answer_from_raw(raw)
    if key == ("S-R2", "D4"):
        return strategy_answer_from_raw(raw)
    if key == ("S-R3", "D2"):
        return numeric_answer_from_raw(raw)
    if key == ("S-R4", "D4"):
        return yes_no_answer_from_raw(raw)
    if key == ("S-T3", "D4"):
        parsed = parse_consistency(raw)
        if parsed == "consistent":
            return "Consistent"
        if parsed == "inconsistent":
            return "Inconsistent"
        return "Cannot be determined"
    if key == ("SEP-T1", "D5"):
        return confidence_answer_from_raw(raw)
    if key == ("AKI-S1", "D2"):
        return event_answer_from_raw(raw)
    return reasoning or compact_reasoning(raw, max_len=240)


def infer_medgemma_confidence(row: dict, raw: str, answer: str) -> str:
    confidence_match = MEDGEMMA_CONFIDENCE_RE.search(raw)
    if confidence_match:
        confidence = _json_unescape(confidence_match.group("value")).strip().lower()
        if confidence in {"high", "medium", "low"}:
            return confidence
    key = (row.get("task_id"), row.get("dimension_id"))
    if key == ("SEP-T1", "D5"):
        label = parse_confidence_high_low(answer)
        if label in {"high", "low"}:
            return label
    return "medium"


def maybe_salvage_medgemma_row(row: dict) -> dict:
    model_name = str(row.get("model_name") or "").lower()
    provider = str(row.get("provider") or "").lower()
    if "medgemma" not in model_name and "medgemma" not in provider:
        return row
    if row.get("parse_success") is True:
        return row
    raw_content = row.get("raw_content")
    if not isinstance(raw_content, str) or not raw_content.strip():
        return row

    raw = strip_code_fence(raw_content)
    evidence_items: List[dict] = []
    seen_evidence: set[tuple[str, str, str]] = set()
    for match in MEDGEMMA_EVIDENCE_RE.finditer(raw):
        timestamp = _json_unescape(match.group("timestamp")).strip()
        measurement = _json_unescape(match.group("measurement")).strip()
        value = _json_unescape(match.group("value")).strip()
        if not (timestamp and measurement and value):
            continue
        key = (timestamp, measurement, value)
        if key in seen_evidence:
            continue
        evidence_items.append(
            {
                "timestamp": timestamp[:32],
                "measurement": measurement[:64],
                "value": value[:128],
            }
        )
        seen_evidence.add(key)
        if len(evidence_items) >= 3:
            break

    reasoning = extract_partial_reasoning(raw)
    if not reasoning:
        reasoning = compact_reasoning(raw)
    else:
        reasoning = compact_reasoning(reasoning)

    if not evidence_items:
        evidence_items = extract_generic_evidence(raw)

    answer = infer_medgemma_answer(row, raw, reasoning)
    if not answer:
        return row
    confidence = infer_medgemma_confidence(row, raw, answer)

    parsed_response = {"answer": answer[:240]}
    if reasoning:
        parsed_response["reasoning"] = reasoning
    if evidence_items:
        parsed_response["evidence"] = evidence_items
    if confidence:
        parsed_response["confidence"] = confidence

    repaired = dict(row)
    repaired["parse_success"] = True
    repaired["parsed_response"] = parsed_response
    repaired["parse_error"] = "phase65f_medgemma_salvaged_partial_json" if evidence_items else "phase65f_medgemma_salvaged_reasoning_only"
    if confidence:
        repaired["confidence_value"] = confidence
    return repaired


def candidate_file(path: Path, provider: str) -> bool:
    if not path.is_file() or path.suffix != ".jsonl":
        return False
    name = path.name
    if "manifest" in name:
        return False
    if name.endswith("_raw_rows.jsonl") or name.endswith("_salvaged_rows.jsonl"):
        return False
    if "responses" in name or name.endswith("full_responses.jsonl") or name.endswith("two_stage.jsonl"):
        return True
    return False


def gather_overlay_files(results_root: Path, provider: str, spec: dict) -> List[Path]:
    files: List[Path] = []
    allow_patterns = [token.lower() for token in spec.get("filename_allow_patterns", [])]
    for rel_dir in spec["source_dirs"]:
        base = results_root / rel_dir
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.jsonl"), key=lambda p: (p.stat().st_mtime, str(p))):
            name_lower = path.name.lower()
            if allow_patterns and not any(token in name_lower for token in allow_patterns):
                continue
            if candidate_file(path, provider):
                files.append(path)
    seen: set[Path] = set()
    deduped: List[Path] = []
    for path in files:
        if path in seen:
            continue
        deduped.append(path)
        seen.add(path)
    return deduped


def canonicalize_direct(provider: str, spec: dict, results_root: Path) -> tuple[List[dict], List[dict]]:
    path = results_root / spec["canonical_direct"]
    rows = list(iter_jsonl(path))
    return rows, [{"path": str(path), "n_rows": len(rows)}]


def canonicalize_overlay(provider: str, spec: dict, results_root: Path) -> tuple[List[dict], List[dict]]:
    files = gather_overlay_files(results_root, provider, spec)
    best: Dict[str, tuple[int, dict]] = {}
    source_stats: List[dict] = []
    for source_idx, path in enumerate(files):
        n_rows = 0
        for row in iter_jsonl(path):
            if provider == "medgemma15_4b_it":
                row = maybe_salvage_medgemma_row(row)
            prompt_id = row.get("prompt_id")
            if not prompt_id:
                continue
            n_rows += 1
            current = best.get(prompt_id)
            if current is None:
                best[prompt_id] = (source_idx, row)
                continue
            current_idx, current_row = current
            if row_quality(row) > row_quality(current_row):
                best[prompt_id] = (source_idx, row)
            elif row_quality(row) == row_quality(current_row) and source_idx >= current_idx:
                best[prompt_id] = (source_idx, row)
        source_stats.append({"path": str(path), "n_rows": n_rows})
    rows = [payload for _, payload in sorted(best.values(), key=lambda item: item[1].get("prompt_id", ""))]
    return rows, source_stats


def summarize_canonical_rows(provider: str, tier: str, rows: List[dict], spec: dict, source_stats: List[dict]) -> dict:
    status_counts = Counter(str(row.get("status", "missing")) for row in rows)
    parse_success_rows = int(sum(row.get("parse_success") is True for row in rows))
    ok_rows = int(sum(row.get("status") == "ok" for row in rows))
    unique_prompt_ids = int(len({row.get("prompt_id") for row in rows if row.get("prompt_id")}))
    latencies = [float(row["latency_seconds"]) for row in rows if row.get("latency_seconds") is not None]
    usage_total = [
        float(row["usage_total_tokens"])
        for row in rows
        if row.get("usage_total_tokens") is not None and not pd.isna(row.get("usage_total_tokens"))
    ]
    model_name_counts = Counter(str(row.get("model_name")) for row in rows if row.get("model_name"))
    parse_error_counts = Counter(str(row.get("parse_error")) for row in rows if row.get("parse_success") is not True and row.get("parse_error"))
    summary = {
        "provider": provider,
        "tier": tier,
        "rows": len(rows),
        "unique_prompt_ids": unique_prompt_ids,
        "ok_rows": ok_rows,
        "parse_success_rows": parse_success_rows,
        "rows_expected": 53070,
        "parse_success_expected": 53070,
        "matches_rows_expected": unique_prompt_ids == 53070,
        "matches_parse_success_expected": parse_success_rows == 53070,
        "status_counts": dict(sorted(status_counts.items())),
        "avg_latency_seconds": float(np.mean(latencies)) if latencies else None,
        "usage_total_tokens": int(round(sum(usage_total))) if usage_total else None,
        "model_name_counts": dict(sorted(model_name_counts.items())),
        "primary_model_name": model_name_counts.most_common(1)[0][0] if model_name_counts else None,
        "parse_error_top": parse_error_counts.most_common(10),
        "source_mode": spec["source_mode"],
        "source_file_count": len(source_stats),
        "source_files": source_stats,
    }
    return summary


def run_canonicalize(args: argparse.Namespace) -> None:
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    canonical_dir = output_dir / "canonical"
    ensure_dir(canonical_dir)

    registry: List[dict] = []
    provider_summaries: dict[str, dict] = {}

    for provider, spec in PROVIDER_SPECS.items():
        tier = spec["tier"]
        if spec["source_mode"] == "direct_merged":
            rows, source_stats = canonicalize_direct(provider, spec, results_root)
        else:
            rows, source_stats = canonicalize_overlay(provider, spec, results_root)

        canonical_path = canonical_dir / f"{provider}_canonical_responses.jsonl"
        write_jsonl(canonical_path, rows)
        canonical_summary = summarize_canonical_rows(provider, tier, rows, spec, source_stats)
        canonical_summary_path = output_dir / f"{provider}_canonical_summary.json"
        write_json(canonical_summary_path, canonical_summary)

        entry = {
            "provider": provider,
            "tier": tier,
            "source_mode": spec["source_mode"],
            "canonical_response_path": str(canonical_path),
            "summary_path": str((results_root / spec["summary_path"]).resolve()),
            "canonical_summary_path": str(canonical_summary_path),
            "rows_expected": 53070,
            "parse_success_expected": 53070,
            "rows_actual": canonical_summary["unique_prompt_ids"],
            "parse_success_actual": canonical_summary["parse_success_rows"],
            "ok_rows_actual": canonical_summary["ok_rows"],
            "avg_latency_seconds": canonical_summary["avg_latency_seconds"],
            "usage_total_tokens": canonical_summary["usage_total_tokens"],
            "primary_model_name": canonical_summary["primary_model_name"],
            "model_name_counts": canonical_summary["model_name_counts"],
        }
        if spec.get("supporting_summary_paths"):
            entry["supporting_summary_paths"] = [
                str((results_root / rel).resolve()) for rel in spec["supporting_summary_paths"]
            ]
        registry.append(entry)
        provider_summaries[provider] = canonical_summary

    registry_path = output_dir / "phase65f_frozen_provider_registry.json"
    canonicalization_summary_path = output_dir / "phase65f_canonicalization_summary.json"
    write_json(registry_path, registry)
    write_json(
        canonicalization_summary_path,
        {
            "results_root": str(results_root),
            "output_dir": str(output_dir),
            "providers": provider_summaries,
            "all_rows_match_expected": all(item["rows_actual"] == 53070 for item in registry),
            "all_parse_success_match_expected": all(item["parse_success_actual"] == 53070 for item in registry),
        },
    )


def load_registry(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def resolve_tier1a_scored_prompts_path(project_root: Path, results_root: Path) -> Path:
    candidates = [
        results_root / "phase65c_tier1a_scores" / "phase65c_tier1a_scored_prompts.parquet",
        project_root / "results" / "cres_v3" / "phase65c_tier1a_scores" / "phase65c_tier1a_scored_prompts.parquet",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise SystemExit(
            "Missing frozen Tier 1A scored prompts parquet; expected one of: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return path


def load_frozen_tier1a_provider_scored(
    *,
    provider: str,
    tier: str,
    primary_model_name: Optional[str],
    project_root: Path,
    results_root: Path,
) -> pd.DataFrame:
    path = resolve_tier1a_scored_prompts_path(project_root, results_root)
    df = pd.read_parquet(path)
    df = df[df["provider"] == provider].copy()
    if df.empty:
        raise SystemExit(f"Frozen Tier 1A scored parquet does not contain provider={provider}")
    df["tier"] = tier
    df["primary_model_name"] = primary_model_name
    df["row_correct"] = df.apply(row_correctness, axis=1)
    df["confidence_value"] = df["confidence_value"].map(normalize_text)
    keep_cols = [
        "provider",
        "prompt_id",
        "condition",
        "task_id",
        "dimension_id",
        "score_kind",
        "anchor_hour",
        "anchor_time_bin",
        "trajectory_tier",
        "left_censored",
        "onset_confidence",
        "stroke_layer",
        "stroke_tier",
        "representation_profile",
        "confidence_value",
        "truth_binary",
        "truth_hour",
        "pred_binary",
        "pred_hour",
        "pred_prob",
        "tier",
        "primary_model_name",
        "row_correct",
        "truth_label",
        "pred_label",
        "truth_numeric",
        "pred_numeric",
    ]
    for column in keep_cols:
        if column not in df.columns:
            df[column] = pd.NA
    return df[keep_cols].reset_index(drop=True)


def load_canonical_responses(
    path: Path,
    provider: str,
    include_raw_content: bool = True,
    prompt_id_allowlist: Optional[set[str]] = None,
    minimal_parsed_response: bool = False,
) -> pd.DataFrame:
    keep_rows: List[dict] = []
    for row in iter_jsonl(path):
        prompt_id = row.get("prompt_id")
        if prompt_id_allowlist is not None and prompt_id not in prompt_id_allowlist:
            continue
        parsed_response = row.get("parsed_response")
        if minimal_parsed_response:
            parsed = parsed_response if isinstance(parsed_response, dict) else {}
            parsed_response = {"answer": parsed.get("answer", "")}
            confidence = normalize_confidence_value(row)
            if confidence:
                parsed_response["confidence"] = confidence
        payload = {
            "prompt_id": prompt_id,
            "instance_id": row.get("instance_id"),
            "task_id": row.get("task_id"),
            "dimension_id": row.get("dimension_id"),
            "variant_id": row.get("variant_id"),
            "provider": provider,
            "model_name": row.get("model_name"),
            "parse_success": row.get("parse_success"),
            "parsed_response": parsed_response,
            "confidence_value": normalize_confidence_value(row),
            "status": row.get("status"),
            "latency_seconds": row.get("latency_seconds"),
            "usage_total_tokens": row.get("usage_total_tokens"),
        }
        if include_raw_content:
            payload["raw_content"] = row.get("raw_content")
        keep_rows.append(payload)
    return pd.DataFrame(keep_rows)


def row_correctness(row: pd.Series) -> Optional[bool]:
    kind = row["score_kind"]
    if kind in {"event_time", "binary_from_yesno", "binary_from_trend"}:
        return bool(int(row["truth_binary"]) == int(row["pred_binary"]))
    if kind == "categorical":
        return bool(str(row["truth_label"]) == str(row["pred_label"]))
    if kind == "numeric_exact":
        return bool(float(row["truth_numeric"]) == float(row["pred_numeric"]))
    return None


def normalize_phase65f_answer(task_id: object, dimension_id: object, parsed_response: dict) -> dict:
    if not isinstance(parsed_response, dict):
        return {"answer": ""}
    normalized = dict(parsed_response)
    if (task_id, dimension_id) == ("S-T3", "D4"):
        answer = normalize_text(normalized.get("answer"))
        if answer in {"yes", "yes."}:
            normalized["answer"] = "Consistent"
        elif answer in {"no", "no."}:
            normalized["answer"] = "Inconsistent"
    return normalized


def select_primary_score(score_kind: str, metrics: dict) -> tuple[str, Optional[float]]:
    key = PRIMARY_SCORE_BY_KIND[score_kind]
    value = metrics.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return key, None
    return key, float(value)


def build_metric_tables_v65f(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: List[dict] = []
    group_cols = ["provider", "tier", "primary_model_name", "condition", "task_id", "dimension_id", "score_kind"]
    for keys, group in scored.groupby(group_cols, dropna=False):
        provider, tier, primary_model_name, condition, task_id, dimension_id, score_kind = keys
        metrics = aggregate_group(group)
        primary_score_name, primary_score = select_primary_score(score_kind, metrics)
        metric_rows.append(
            flatten_metric_row(
                {
                    "provider": provider,
                    "tier": tier,
                    "primary_model_name": primary_model_name,
                    "condition": condition,
                    "task_id": task_id,
                    "dimension_id": dimension_id,
                    "score_kind": score_kind,
                    "primary_score_name": primary_score_name,
                    "primary_score": primary_score,
                },
                metrics,
            )
        )
    per_task_dim = pd.DataFrame(metric_rows).sort_values(["provider", "task_id", "dimension_id"]).reset_index(drop=True)

    strat_rows: List[dict] = []
    strat_keys = DEFERRED_STRAT_KEYS
    for strat_key in strat_keys:
        subset = scored[scored[strat_key].notna() & (scored[strat_key].astype(str) != "<NA>")].copy()
        if subset.empty:
            continue
        for keys, group in subset.groupby(group_cols + [strat_key], dropna=False):
            provider, tier, primary_model_name, condition, task_id, dimension_id, score_kind, strat_value = keys
            metrics = aggregate_group(group)
            primary_score_name, primary_score = select_primary_score(score_kind, metrics)
            strat_rows.append(
                flatten_metric_row(
                    {
                        "provider": provider,
                        "tier": tier,
                        "primary_model_name": primary_model_name,
                        "condition": condition,
                        "task_id": task_id,
                        "dimension_id": dimension_id,
                        "score_kind": score_kind,
                        "strat_key": strat_key,
                        "strat_value": strat_value,
                        "primary_score_name": primary_score_name,
                        "primary_score": primary_score,
                    },
                    metrics,
                )
            )
    stratified = pd.DataFrame(strat_rows).sort_values(
        ["provider", "task_id", "dimension_id", "strat_key", "strat_value"]
    ).reset_index(drop=True)
    return per_task_dim, stratified


def assign_temporal_bucket(hour: object) -> Optional[str]:
    if hour is None or pd.isna(hour):
        return None
    value = float(hour)
    if 4.0 <= value <= 8.0:
        return "T6"
    if 18.0 <= value <= 30.0:
        return "T24"
    if 46.0 <= value <= 50.0:
        return "T48"
    return None


def build_temporal_degradation(scored: pd.DataFrame) -> pd.DataFrame:
    temporal_task_ids = {
        "AKI-T1",
        "AKI-S1",
        "DEL-T1",
        "DEL-S1",
        "SEP-T1",
        "SEP-S1",
        "S-T1",
    }
    df = scored[scored["task_id"].isin(temporal_task_ids)].copy()
    df["temporal_bucket"] = df["anchor_hour"].map(assign_temporal_bucket)
    df = df[df["temporal_bucket"].notna()].copy()
    rows: List[dict] = []
    group_cols = ["provider", "tier", "primary_model_name", "condition", "task_id", "dimension_id", "score_kind", "temporal_bucket"]
    for keys, group in df.groupby(group_cols, dropna=False):
        provider, tier, primary_model_name, condition, task_id, dimension_id, score_kind, temporal_bucket = keys
        metrics = aggregate_group(group)
        primary_score_name, primary_score = select_primary_score(score_kind, metrics)
        support_status = "ok" if int(metrics["n_rows"]) >= 25 else "insufficient_support"
        rows.append(
            flatten_metric_row(
                {
                    "provider": provider,
                    "tier": tier,
                    "primary_model_name": primary_model_name,
                    "condition": condition,
                    "task_id": task_id,
                    "dimension_id": dimension_id,
                    "score_kind": score_kind,
                    "temporal_bucket": temporal_bucket,
                    "primary_score_name": primary_score_name,
                    "primary_score": primary_score if support_status == "ok" else None,
                    "support_status": support_status,
                },
                metrics,
            )
        )
    return pd.DataFrame(rows).sort_values(
        ["provider", "task_id", "dimension_id", "temporal_bucket"]
    ).reset_index(drop=True)


def build_condition_heatmap(per_task_dim: pd.DataFrame) -> pd.DataFrame:
    df = per_task_dim[per_task_dim["primary_score"].notna()].copy()
    out = (
        df.groupby(["provider", "tier", "condition"], dropna=False)
        .agg(
            macro_primary_score=("primary_score", "mean"),
            n_pairs=("primary_score", "size"),
        )
        .reset_index()
        .sort_values(["provider", "condition"])
        .reset_index(drop=True)
    )
    return out


def build_tier_comparison(condition_heatmap: pd.DataFrame) -> pd.DataFrame:
    return (
        condition_heatmap.groupby(["tier", "condition"], dropna=False)
        .agg(
            macro_primary_score=("macro_primary_score", "mean"),
            n_providers=("provider", "nunique"),
        )
        .reset_index()
        .sort_values(["tier", "condition"])
        .reset_index(drop=True)
    )


def build_provider_metrics(per_task_dim: pd.DataFrame, registry: list[dict]) -> pd.DataFrame:
    registry_df = pd.DataFrame(registry)[
        [
            "provider",
            "tier",
            "primary_model_name",
            "avg_latency_seconds",
            "usage_total_tokens",
            "rows_actual",
            "ok_rows_actual",
            "parse_success_actual",
        ]
    ].copy()
    metrics = (
        per_task_dim[per_task_dim["primary_score"].notna()]
        .groupby(["provider", "tier"], dropna=False)
        .agg(
            overall_macro_primary_score=("primary_score", "mean"),
            supported_pairs_scored=("primary_score", "size"),
        )
        .reset_index()
    )
    return metrics.merge(registry_df, on=["provider", "tier"], how="left").sort_values("provider").reset_index(drop=True)


def compute_pair_wins(per_task_dim: pd.DataFrame) -> pd.DataFrame:
    df = per_task_dim[per_task_dim["primary_score"].notna()].copy()
    df["pair_max"] = df.groupby(["tier", "task_id", "dimension_id"])["primary_score"].transform("max")
    wins = (
        df[df["primary_score"] == df["pair_max"]]
        .groupby(["provider", "tier"], dropna=False)
        .size()
        .reset_index(name="pair_win_count")
    )
    return wins


def build_deferred_pairs(joined: pd.DataFrame) -> dict:
    all_pairs = [(task, dim) for task, dims in ALL_TASK_DIMENSIONS.items() for dim in dims]
    supported_pairs = sorted({(task, dim) for task, dim in AUTO_SCORING_RULES})
    deferred_pairs = sorted(set(all_pairs) - set(supported_pairs))
    pair_counts = joined.groupby(["task_id", "dimension_id"]).size().to_dict()
    return {
        "supported_pairs": [
            {"task_id": task, "dimension_id": dim, "n_rows": int(pair_counts.get((task, dim), 0))}
            for task, dim in supported_pairs
        ],
        "deferred_pairs": [
            {
                "task_id": task,
                "dimension_id": dim,
                "n_rows": int(pair_counts.get((task, dim), 0)),
                "reason": "judge_only_or_no_direct_ground_truth",
            }
            for task, dim in deferred_pairs
        ],
    }


def tier1a_parity_check(per_task_dim: pd.DataFrame, project_root: Path, results_root: Path) -> dict:
    reference_candidates = [
        results_root / "phase65c_tier1a_scores" / "phase65c_tier1a_provider_metrics.csv",
        project_root / "results" / "cres_v3" / "phase65c_tier1a_scores" / "phase65c_tier1a_provider_metrics.csv",
    ]
    reference_path = next((path for path in reference_candidates if path.exists()), reference_candidates[0])
    reference = pd.read_csv(reference_path)
    current = per_task_dim[per_task_dim["provider"].isin(["gpt54", "gemini31pro"])].copy()
    keep_cols = [
        "provider",
        "task_id",
        "dimension_id",
        "score_kind",
        "n_rows",
        "binary_accuracy",
        "event_presence_auroc",
        "event_presence_auprc",
        "positive_tolerance_1h_rate",
        "median_abs_hour_error",
        "accuracy",
        "macro_f1",
        "exact_match",
        "tolerance_1",
    ]
    merged = current[keep_cols].merge(reference[keep_cols], on=["provider", "task_id", "dimension_id", "score_kind"], suffixes=("_current", "_reference"))
    metric_cols = [col for col in keep_cols if col not in {"provider", "task_id", "dimension_id", "score_kind"}]
    max_abs_diff = 0.0
    mismatches = []
    for _, row in merged.iterrows():
        row_max_diff = 0.0
        for col in metric_cols:
            current_value = row[f"{col}_current"]
            reference_value = row[f"{col}_reference"]
            if pd.isna(current_value) and pd.isna(reference_value):
                continue
            current_float = float(current_value) if not pd.isna(current_value) else None
            reference_float = float(reference_value) if not pd.isna(reference_value) else None
            if current_float is None or reference_float is None:
                mismatches.append(
                    {
                        "provider": row["provider"],
                        "task_id": row["task_id"],
                        "dimension_id": row["dimension_id"],
                        "metric": col,
                        "current": current_value,
                        "reference": reference_value,
                    }
                )
                continue
            diff = abs(current_float - reference_float)
            row_max_diff = max(row_max_diff, diff)
            max_abs_diff = max(max_abs_diff, diff)
            if diff > 1e-9:
                mismatches.append(
                    {
                        "provider": row["provider"],
                        "task_id": row["task_id"],
                        "dimension_id": row["dimension_id"],
                        "metric": col,
                        "current": current_float,
                        "reference": reference_float,
                        "abs_diff": diff,
                    }
                )
    return {
        "rows_compared": int(len(merged)),
        "max_abs_diff": max_abs_diff,
        "match_within_rounding": max_abs_diff <= 1e-9 and not mismatches,
        "mismatch_examples": mismatches[:25],
    }


def write_combined_scored_parquet(temp_paths: List[Path], output_path: Path) -> None:
    if not temp_paths:
        pd.DataFrame().to_parquet(output_path, index=False)
        return
    output_path.unlink(missing_ok=True)
    writer = None
    canonical_schema: Optional[pa.Schema] = None
    try:
        for path in temp_paths:
            parquet_file = pq.ParquetFile(path)
            if canonical_schema is None:
                canonical_schema = parquet_file.schema_arrow
            assert canonical_schema is not None
            for batch in parquet_file.iter_batches(batch_size=4096):
                table = pa.Table.from_batches([batch])
                arrays = []
                for field in canonical_schema:
                    if field.name in table.column_names:
                        column = table[field.name]
                        if not column.type.equals(field.type):
                            column = column.cast(field.type, safe=False)
                    else:
                        column = pa.nulls(table.num_rows, type=field.type)
                    arrays.append(column)
                normalized = pa.Table.from_arrays(arrays, schema=canonical_schema)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, canonical_schema)
                writer.write_table(normalized)
    finally:
        if writer is not None:
            writer.close()


def finalize_score_outputs(
    *,
    project_root: Path,
    results_root: Path,
    output_dir: Path,
    registry: list[dict],
    joined: pd.DataFrame,
    scored_temp_paths: List[Path],
    total_scored_rows: int,
    variant_id: str,
) -> None:
    ensure_dir(output_dir)
    if not scored_temp_paths:
        raise SystemExit("No scored temp parquet files found for Phase 6.5F finalization")

    provider_run_summary: dict[str, dict] = {}
    per_task_dim_frames: List[pd.DataFrame] = []
    stratified_frames: List[pd.DataFrame] = []
    temporal_frames: List[pd.DataFrame] = []

    for entry in registry:
        provider = entry["provider"]
        temp_path = output_dir / "_score_tmp" / f"{provider}_scored.parquet"
        if not temp_path.exists():
            raise SystemExit(f"Missing scored parquet for provider={provider}: {temp_path}")
        provider_scored = pd.read_parquet(temp_path)
        per_task_dim_provider, stratified_provider = build_metric_tables_v65f(provider_scored)
        temporal_provider = build_temporal_degradation(provider_scored)
        per_task_dim_frames.append(per_task_dim_provider)
        stratified_frames.append(stratified_provider)
        temporal_frames.append(temporal_provider)
        provider_run_summary[provider] = {
            "provider": provider,
            "tier": entry["tier"],
            "rows": entry["rows_actual"],
            "ok_rows": entry["ok_rows_actual"],
            "parse_success_rows": entry["parse_success_actual"],
            "avg_latency_seconds": entry.get("avg_latency_seconds"),
            "usage_total_tokens": entry.get("usage_total_tokens"),
            "primary_model_name": entry.get("primary_model_name"),
            "summary_path": entry.get("summary_path"),
        }

    per_task_dim = pd.concat(per_task_dim_frames, ignore_index=True) if per_task_dim_frames else pd.DataFrame()
    stratified = pd.concat(stratified_frames, ignore_index=True) if stratified_frames else pd.DataFrame()
    condition_heatmap = build_condition_heatmap(per_task_dim)
    tier_comparison = build_tier_comparison(condition_heatmap)
    temporal_degradation = pd.concat(temporal_frames, ignore_index=True) if temporal_frames else pd.DataFrame()
    provider_metrics = build_provider_metrics(per_task_dim, registry)
    pair_wins = compute_pair_wins(per_task_dim)
    provider_metrics = provider_metrics.merge(pair_wins, on=["provider", "tier"], how="left")
    provider_metrics["pair_win_count"] = provider_metrics["pair_win_count"].fillna(0).astype(int)
    deferred_pairs = build_deferred_pairs(joined)
    parity = tier1a_parity_check(per_task_dim, project_root, results_root)

    scored_prompts_path = output_dir / "phase65f_scored_prompts.parquet"
    per_task_dim_path = output_dir / "phase65f_per_task_dimension_metrics.csv"
    provider_metrics_path = output_dir / "phase65f_provider_metrics.csv"
    heatmap_path = output_dir / "phase65f_condition_heatmap_data.csv"
    stratified_path = output_dir / "phase65f_stratified_metrics.csv"
    temporal_path = output_dir / "phase65f_temporal_degradation.csv"
    tier_comparison_path = output_dir / "phase65f_tier_comparison.csv"
    deferred_pairs_path = output_dir / "phase65f_deferred_pairs.json"
    scoring_summary_path = output_dir / "phase65f_scoring_summary.json"

    write_combined_scored_parquet(scored_temp_paths, scored_prompts_path)
    per_task_dim.to_csv(per_task_dim_path, index=False)
    provider_metrics.to_csv(provider_metrics_path, index=False)
    condition_heatmap.to_csv(heatmap_path, index=False)
    stratified.to_csv(stratified_path, index=False)
    temporal_degradation.to_csv(temporal_path, index=False)
    tier_comparison.to_csv(tier_comparison_path, index=False)
    write_json(deferred_pairs_path, deferred_pairs)

    scoring_summary = {
        "variant_id": variant_id,
        "providers": provider_run_summary,
        "auto_scoring": {
            "scored_prompt_rows": int(total_scored_rows),
            "supported_task_dimensions": int(len(AUTO_SCORING_RULES)),
            "per_task_dimension_rows": int(len(per_task_dim)),
            "provider_rows": int(len(provider_metrics)),
            "condition_heatmap_rows": int(len(condition_heatmap)),
            "stratified_rows": int(len(stratified)),
            "temporal_rows": int(len(temporal_degradation)),
        },
        "parity_check_tier1a": parity,
        "outputs": {
            "scored_prompts": str(scored_prompts_path),
            "per_task_dimension_metrics": str(per_task_dim_path),
            "provider_metrics": str(provider_metrics_path),
            "condition_heatmap_data": str(heatmap_path),
            "stratified_metrics": str(stratified_path),
            "temporal_degradation": str(temporal_path),
            "tier_comparison": str(tier_comparison_path),
            "deferred_pairs": str(deferred_pairs_path),
        },
    }
    write_json(scoring_summary_path, scoring_summary)


def run_score(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    registry_path = output_dir / "phase65f_frozen_provider_registry.json"
    registry = load_registry(registry_path)
    ensure_dir(output_dir)

    joined = build_joined_frame(
        argparse.Namespace(
            sample_path=str(Path(args.sample_path).resolve()),
            prompts_path=str(Path(args.prompts_path).resolve()),
            variant_id=args.variant_id,
        )
    )
    joined_prompt_ids = set(joined["prompt_id"].dropna().astype(str).tolist())
    joined_key_map: Dict[tuple, dict] = {}
    joined_columns = list(joined.columns)
    for values in joined.itertuples(index=False, name=None):
        row = dict(zip(joined_columns, values))
        key = (
            row.get("prompt_id"),
            row.get("instance_id"),
            row.get("task_id"),
            row.get("dimension_id"),
            row.get("variant_id"),
        )
        joined_key_map[key] = row

    score_tmp_dir = output_dir / "_score_tmp"
    ensure_dir(score_tmp_dir)
    scored_temp_paths: List[Path] = []
    total_scored_rows = 0
    for entry in registry:
        provider = entry["provider"]
        tier = entry["tier"]
        response_path = Path(entry["canonical_response_path"])
        provider_scored_rows: List[dict] = []
        provider_batch_paths: List[Path] = []

        if provider in FROZEN_TIER1A_PROVIDERS:
            provider_scored = load_frozen_tier1a_provider_scored(
                provider=provider,
                tier=tier,
                primary_model_name=entry.get("primary_model_name"),
                project_root=project_root,
                results_root=results_root,
            )
            temp_path = score_tmp_dir / f"{provider}_scored.parquet"
            provider_scored.to_parquet(temp_path, index=False)
            scored_temp_paths.append(temp_path)
            total_scored_rows += int(len(provider_scored))
            continue

        def flush_provider_batch() -> None:
            nonlocal total_scored_rows
            if not provider_scored_rows:
                return
            batch_df = pd.DataFrame(provider_scored_rows)
            drop_cols = [col for col in ["model_name", "instance_id", "parse_success"] if col in batch_df.columns]
            if drop_cols:
                batch_df = batch_df.drop(columns=drop_cols)
            total_scored_rows += int(len(batch_df))
            batch_path = score_tmp_dir / f"{provider}_batch{len(provider_batch_paths):03d}.parquet"
            batch_df.to_parquet(batch_path, index=False)
            provider_batch_paths.append(batch_path)
            provider_scored_rows.clear()

        for response_row in iter_jsonl(response_path):
            prompt_id = response_row.get("prompt_id")
            if prompt_id not in joined_prompt_ids:
                continue
            key = (
                prompt_id,
                response_row.get("instance_id"),
                response_row.get("task_id"),
                response_row.get("dimension_id"),
                response_row.get("variant_id"),
            )
            joined_row = joined_key_map.get(key)
            if joined_row is None:
                continue
            parsed_response = response_row.get("parsed_response")
            if isinstance(parsed_response, dict):
                minimal_parsed = {"answer": parsed_response.get("answer", "")}
            else:
                minimal_parsed = {"answer": ""}
            confidence = normalize_confidence_value(response_row)
            if confidence:
                minimal_parsed["confidence"] = confidence
            minimal_parsed = normalize_phase65f_answer(
                response_row.get("task_id"),
                response_row.get("dimension_id"),
                minimal_parsed,
            )
            merged_row = dict(joined_row)
            merged_row.update(
                {
                    "provider": provider,
                    "model_name": response_row.get("model_name"),
                    "parse_success": response_row.get("parse_success"),
                    "parsed_response": minimal_parsed,
                    "confidence_value": confidence,
                    "status": response_row.get("status"),
                }
            )
            derived = derive_truth_and_prediction(merged_row)
            if derived is None:
                continue
            derived["tier"] = tier
            derived["primary_model_name"] = entry.get("primary_model_name")
            derived["row_correct"] = row_correctness(derived)
            derived["confidence_value"] = normalize_text(derived.get("confidence_value"))
            provider_scored_rows.append(derived)
            if len(provider_scored_rows) >= SCORE_BATCH_ROWS:
                flush_provider_batch()
        flush_provider_batch()
        if provider_batch_paths:
            provider_scored = pd.concat([pd.read_parquet(path) for path in provider_batch_paths], ignore_index=True)
            temp_path = score_tmp_dir / f"{provider}_scored.parquet"
            provider_scored.to_parquet(temp_path, index=False)
            scored_temp_paths.append(temp_path)
            for path in provider_batch_paths:
                path.unlink(missing_ok=True)
            del provider_scored
        del provider_scored_rows

    finalize_score_outputs(
        project_root=project_root,
        results_root=results_root,
        output_dir=output_dir,
        registry=registry,
        joined=joined,
        scored_temp_paths=scored_temp_paths,
        total_scored_rows=total_scored_rows,
        variant_id=args.variant_id,
    )


def run_finalize_score(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    registry = load_registry(output_dir / "phase65f_frozen_provider_registry.json")
    joined = build_joined_frame(
        argparse.Namespace(
            sample_path=str(Path(args.sample_path).resolve()),
            prompts_path=str(Path(args.prompts_path).resolve()),
            variant_id=args.variant_id,
        )
    )
    scored_temp_paths = [
        output_dir / "_score_tmp" / f"{entry['provider']}_scored.parquet"
        for entry in registry
    ]
    missing = [str(path) for path in scored_temp_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing scored temp parquet files: {missing}")
    total_scored_rows = int(sum(pq.ParquetFile(path).metadata.num_rows for path in scored_temp_paths))
    finalize_score_outputs(
        project_root=project_root,
        results_root=results_root,
        output_dir=output_dir,
        registry=registry,
        joined=joined,
        scored_temp_paths=scored_temp_paths,
        total_scored_rows=total_scored_rows,
        variant_id=args.variant_id,
    )


def choose_tier_representatives(provider_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: List[dict] = []
    for tier, subset in provider_metrics.groupby("tier", dropna=False):
        ordered = subset.sort_values(
            ["overall_macro_primary_score", "pair_win_count", "avg_latency_seconds", "provider"],
            ascending=[False, False, True, True],
            na_position="last",
        ).reset_index(drop=True)
        rep = ordered.iloc[0].to_dict()
        rep["selection_rule"] = "best_overall_macro_primary_score_then_pair_wins_then_lower_latency"
        rows.append(rep)
    return pd.DataFrame(rows).sort_values("tier").reset_index(drop=True)


def build_manual_judge_contestant_roster(provider_metrics: pd.DataFrame) -> pd.DataFrame:
    roster = provider_metrics[provider_metrics["provider"].isin(MANUAL_JUDGE_CONTESTANT_PROVIDERS)].copy()
    expected = set(MANUAL_JUDGE_CONTESTANT_PROVIDERS)
    actual = set(roster["provider"].tolist())
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise SystemExit(f"Manual judge contestant roster mismatch: missing={missing}, extra={extra}")
    order_map = {provider: idx for idx, provider in enumerate(MANUAL_JUDGE_CONTESTANT_PROVIDERS)}
    roster["judge_roster_order"] = roster["provider"].map(order_map)
    roster["selection_rule"] = JUDGE_CONTESTANT_ROSTER_RULE
    roster["selection_rationale"] = JUDGE_CONTESTANT_ROSTER_RATIONALE
    roster["judge_overlap_note"] = roster["provider"].map(
        lambda provider: JUDGE_CONTESTANT_JUDGE_OVERLAP_NOTE if provider == "gpt54" else ""
    )
    return roster.sort_values("judge_roster_order").reset_index(drop=True)


def allocate_by_task(df: pd.DataFrame, n_target: int, rng: random.Random, min_per_task: int = 15) -> pd.DataFrame:
    if n_target <= 0 or df.empty:
        return df.iloc[0:0].copy()
    task_counts = df["task_id"].value_counts().to_dict()
    tasks = sorted(task_counts)
    if len(df) <= n_target:
        return df.sample(frac=1.0, random_state=rng.randint(0, 10_000_000)).reset_index(drop=True)

    if len(tasks) * min_per_task > n_target:
        min_per_task = 0

    allocation = {task: 0 for task in tasks}
    remaining = n_target

    for task in tasks:
        base = min(min_per_task, task_counts[task])
        allocation[task] = base
        remaining -= base

    residual = {task: max(task_counts[task] - allocation[task], 0) for task in tasks}
    total_residual = sum(residual.values())
    if remaining > 0 and total_residual > 0:
        fractional: List[tuple[float, str]] = []
        for task in tasks:
            share = remaining * residual[task] / total_residual if total_residual else 0.0
            whole = int(math.floor(share))
            whole = min(whole, residual[task])
            allocation[task] += whole
            fractional.append((share - whole, task))
        allocated = sum(allocation.values())
        leftover = n_target - allocated
        fractional.sort(reverse=True)
        idx = 0
        while leftover > 0 and idx < len(fractional):
            _, task = fractional[idx]
            if allocation[task] < task_counts[task]:
                allocation[task] += 1
                leftover -= 1
            idx += 1

    pieces: List[pd.DataFrame] = []
    for task in tasks:
        need = allocation[task]
        if need <= 0:
            continue
        task_df = df[df["task_id"] == task].copy()
        task_df = task_df.sample(frac=1.0, random_state=rng.randint(0, 10_000_000))
        pieces.append(task_df.head(need))
    out = pd.concat(pieces, ignore_index=True) if pieces else df.iloc[0:0].copy()
    if len(out) > n_target:
        out = out.sample(n=n_target, random_state=rng.randint(0, 10_000_000)).reset_index(drop=True)
    return out


def run_build_judge(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).resolve()
    registry = load_registry(output_dir / "phase65f_frozen_provider_registry.json")
    prompts = pd.DataFrame(
        [
            row
            for row in iter_jsonl(Path(args.prompts_path).resolve())
            if row.get("variant_id") == args.variant_id
        ]
    )
    scored = pd.read_parquet(output_dir / "phase65f_scored_prompts.parquet")
    provider_metrics = pd.read_csv(output_dir / "phase65f_provider_metrics.csv")

    roster = build_manual_judge_contestant_roster(provider_metrics)
    roster_path = output_dir / "phase65f_judge_contestant_roster.json"
    legacy_reps_path = output_dir / "phase65f_tier_representatives.json"
    write_json(roster_path, roster.to_dict(orient="records"))
    write_json(legacy_reps_path, roster.to_dict(orient="records"))

    rep_providers = roster["provider"].tolist()
    rep_response_meta_frames: List[pd.DataFrame] = []
    for entry in registry:
        if entry["provider"] not in rep_providers:
            continue
        df = load_canonical_responses(
            Path(entry["canonical_response_path"]),
            entry["provider"],
            include_raw_content=False,
            minimal_parsed_response=True,
        )
        df["tier"] = entry["tier"]
        rep_response_meta_frames.append(df)
    rep_response_meta = pd.concat(rep_response_meta_frames, ignore_index=True)

    prompts = prompts.copy()
    prompts["is_supported_pair"] = prompts.apply(lambda row: (row["task_id"], row["dimension_id"]) in AUTO_SCORING_RULES, axis=1)

    deferred_pool = prompts[~prompts["is_supported_pair"]].copy()

    scored_rep = scored[scored["provider"].isin(rep_providers)].copy()
    response_meta = rep_response_meta[["prompt_id", "provider", "confidence_value", "model_name"]].copy()
    correctness = scored_rep[["prompt_id", "provider", "row_correct", "score_kind", "condition", "task_id", "dimension_id"]].copy()

    prompt_eval = (
        prompts[prompts["is_supported_pair"]][["prompt_id", "condition", "task_id", "dimension_id", "prompt_text", "instance_id", "variant_id"]]
        .assign(_key=1)
        .merge(pd.DataFrame({"provider": rep_providers, "_key": 1}), on="_key", how="inner")
        .drop(columns="_key")
        .merge(response_meta, on=["prompt_id", "provider"], how="left")
        .merge(correctness[["prompt_id", "provider", "row_correct", "score_kind"]], on=["prompt_id", "provider"], how="left")
    )
    prompt_eval["row_correct"] = prompt_eval["row_correct"].fillna(False)
    prompt_eval["confidence_value"] = prompt_eval["confidence_value"].fillna("missing")

    hard_rows: List[dict] = []
    for prompt_id, group in prompt_eval.groupby("prompt_id", dropna=False):
        correctness_values = {bool(value) for value in group["row_correct"].tolist()}
        confidences = {normalize_text(value) for value in group["confidence_value"].tolist()}
        if len(correctness_values) > 1:
            bucket = "prediction_disagreement"
        elif "high" in confidences and "low" in confidences:
            bucket = "confidence_disagreement_secondary"
        else:
            continue
        base_row = group.iloc[0]
        hard_rows.append(
            {
                "prompt_id": prompt_id,
                "condition": base_row["condition"],
                "task_id": base_row["task_id"],
                "dimension_id": base_row["dimension_id"],
                "instance_id": base_row["instance_id"],
                "variant_id": base_row["variant_id"],
                "prompt_text": base_row["prompt_text"],
                "selection_bucket": bucket,
            }
        )
    hard_pool = pd.DataFrame(hard_rows)

    rng = random.Random(args.judge_seed)
    condition_targets = {condition: args.judge_per_condition for condition in sorted(prompts["condition"].unique())}
    prompt_selection_rows: List[pd.DataFrame] = []
    composition_rows: List[dict] = []

    for condition, total_target in condition_targets.items():
        deferred_target = int(round(total_target * 0.60))
        hard_target = total_target - deferred_target

        cond_deferred = deferred_pool[deferred_pool["condition"] == condition].copy()
        cond_hard = hard_pool[hard_pool["condition"] == condition].copy()

        selected_deferred = allocate_by_task(cond_deferred, min(deferred_target, len(cond_deferred)), rng)
        selected_hard = allocate_by_task(
            cond_hard[cond_hard["selection_bucket"] == "prediction_disagreement"],
            min(hard_target, len(cond_hard[cond_hard["selection_bucket"] == "prediction_disagreement"])),
            rng,
        )
        shortfall = hard_target - len(selected_hard)
        if shortfall > 0:
            fallback_pool = cond_hard[
                ~cond_hard["prompt_id"].isin(selected_hard["prompt_id"])
                & cond_hard["selection_bucket"].eq("confidence_disagreement_secondary")
            ].copy()
            fallback = allocate_by_task(fallback_pool, min(shortfall, len(fallback_pool)), rng, min_per_task=0)
            selected_hard = pd.concat([selected_hard, fallback], ignore_index=True)

        combined = pd.concat([selected_deferred, selected_hard], ignore_index=True)
        if len(combined) < total_target:
            fallback_pool = pd.concat(
                [
                    cond_deferred[~cond_deferred["prompt_id"].isin(combined["prompt_id"])],
                    cond_hard[~cond_hard["prompt_id"].isin(combined["prompt_id"])],
                ],
                ignore_index=True,
            )
            fallback = allocate_by_task(fallback_pool.drop_duplicates("prompt_id"), min(total_target - len(combined), len(fallback_pool)), rng, min_per_task=0)
            combined = pd.concat([combined, fallback], ignore_index=True)

        combined = combined.drop_duplicates("prompt_id").head(total_target).copy()
        prompt_selection_rows.append(combined)
        composition_rows.append(
            {
                "condition": condition,
                "target_total": total_target,
                "selected_total": int(len(combined)),
                "selected_deferred": int(sum(combined.get("selection_bucket", pd.Series(dtype="object")).fillna("deferred_pool").eq("deferred_pool"))),
                "selected_prediction_disagreement": int(sum(combined.get("selection_bucket", pd.Series(dtype="object")).eq("prediction_disagreement"))),
                "selected_confidence_disagreement_secondary": int(sum(combined.get("selection_bucket", pd.Series(dtype="object")).eq("confidence_disagreement_secondary"))),
            }
        )

    selected_prompts = pd.concat(prompt_selection_rows, ignore_index=True).drop_duplicates("prompt_id").reset_index(drop=True)
    if len(selected_prompts) > args.judge_size:
        selected_prompts = selected_prompts.sample(n=args.judge_size, random_state=args.judge_seed).reset_index(drop=True)
    elif len(selected_prompts) < args.judge_size:
        raise SystemExit(f"Judge prompt selection underfilled: expected {args.judge_size}, got {len(selected_prompts)}")

    selected_prompt_ids = set(selected_prompts["prompt_id"].astype(str).tolist())
    rep_full_frames: List[pd.DataFrame] = []
    for entry in registry:
        if entry["provider"] not in rep_providers:
            continue
        df = load_canonical_responses(
            Path(entry["canonical_response_path"]),
            entry["provider"],
            include_raw_content=True,
            prompt_id_allowlist=selected_prompt_ids,
            minimal_parsed_response=False,
        )
        rep_full_frames.append(df)
    rep_responses = pd.concat(rep_full_frames, ignore_index=True)

    judge_rows = (
        selected_prompts.assign(_key=1)
        .merge(
            roster[["provider", "tier", "primary_model_name", "selection_rationale", "judge_overlap_note"]].assign(_key=1),
            on="_key",
            how="inner",
        )
        .drop(columns="_key")
        .merge(rep_responses[["prompt_id", "provider", "model_name", "raw_content", "parsed_response", "confidence_value"]], on=["prompt_id", "provider"], how="left")
        .merge(scored_rep[["prompt_id", "provider", "row_correct", "score_kind"]], on=["prompt_id", "provider"], how="left")
    )
    judge_rows["judge_row_id"] = judge_rows.apply(lambda row: f"{row['prompt_id']}::{row['provider']}", axis=1)
    judge_rows["row_correct"] = judge_rows["row_correct"].fillna(False)

    judge_prompt_manifest_path = output_dir / "phase65f_judge500_prompt_instances.jsonl"
    judge_response_manifest_path = output_dir / "phase65f_judge500_manifest.jsonl"
    judge_summary_path = output_dir / "phase65f_judge500_summary.json"
    judge_rubric_path = output_dir / "phase65f_judge_rubric.md"
    formal_summary_md_path = output_dir / "phase65f_formal_summary.md"
    formal_summary_json_path = output_dir / "phase65f_formal_summary.json"

    write_jsonl(judge_prompt_manifest_path, selected_prompts.to_dict(orient="records"))
    write_jsonl(judge_response_manifest_path, judge_rows.to_dict(orient="records"))
    judge_rubric_path.write_text(JUDGE_RUBRIC_MD)

    judge_summary = {
        "judge_seed": args.judge_seed,
        "prompt_instances": int(len(selected_prompts)),
        "judged_response_rows": int(len(judge_rows)),
        "contestant_roster_mode": JUDGE_CONTESTANT_ROSTER_RULE,
        "contestant_roster_rationale": JUDGE_CONTESTANT_ROSTER_RATIONALE,
        "contestant_roster": roster.to_dict(orient="records"),
        "judge_model_roster": JUDGE_MODEL_ROSTER,
        "contestant_judge_overlap_note": JUDGE_CONTESTANT_JUDGE_OVERLAP_NOTE,
        "composition_by_condition": composition_rows,
        "outputs": {
            "prompt_instances_manifest": str(judge_prompt_manifest_path),
            "judge_response_manifest": str(judge_response_manifest_path),
            "judge_rubric": str(judge_rubric_path),
            "judge_contestant_roster": str(roster_path),
            "legacy_tier_representatives_alias": str(legacy_reps_path),
        },
        "execution_status": "manifest_ready_judge_calls_not_executed",
    }
    write_json(judge_summary_path, judge_summary)

    scoring_summary = read_summary(output_dir / "phase65f_scoring_summary.json")
    canonicalization_summary = read_summary(output_dir / "phase65f_canonicalization_summary.json")
    formal_summary = {
        "phase": "6.5F",
        "variant_id": args.variant_id,
        "frozen_contestant_set": list(PROVIDER_SPECS.keys()),
        "excluded_supplementary_provider": "openbiollm70b",
        "human_llm_agreement": "deferred",
        "new_contestant_inference": False,
        "canonicalization": canonicalization_summary,
        "scoring": scoring_summary,
        "judge": judge_summary,
    }
    write_json(formal_summary_json_path, formal_summary)

    md_lines = [
        "# Phase 6.5F Formal Summary",
        "",
        "## Frozen Set",
        "",
        "- comparative providers: `gpt54`, `gemini31pro`, `deepseek_chat`, `qwen35`, `gemma4_26b`, `aloe70b`, `aloe7b`, `meditron3_8b`, `medgemma15_4b_it`",
        "- excluded supplementary provider: `openbiollm70b`",
        "- variant: `full_multimodal`",
        "",
        "## Canonicalization",
        "",
    ]
    for provider, payload in canonicalization_summary.get("providers", {}).items():
        md_lines.append(
            f"- `{provider}`: rows={payload['unique_prompt_ids']}, ok={payload['ok_rows']}, parse_success={payload['parse_success_rows']}"
        )
    md_lines.extend(
        [
            "",
            "## Auto-Scoring",
            "",
            f"- scored_prompt_rows: `{scoring_summary.get('auto_scoring', {}).get('scored_prompt_rows')}`",
            f"- supported_task_dimensions: `{scoring_summary.get('auto_scoring', {}).get('supported_task_dimensions')}`",
            f"- parity_with_tier1a: `{scoring_summary.get('parity_check_tier1a', {}).get('match_within_rounding')}`",
            "",
            "## Judge Packet",
            "",
            f"- prompt_instances: `{judge_summary['prompt_instances']}`",
            f"- judged_response_rows: `{judge_summary['judged_response_rows']}`",
            f"- contestant_roster_mode: `{judge_summary['contestant_roster_mode']}`",
            f"- execution_status: `{judge_summary['execution_status']}`",
            "",
            "## Judge Contestants",
            "",
        ]
    )
    for row in roster.to_dict(orient="records"):
        md_lines.append(
            f"- `{row['provider']}` [{row['tier']}] (overall_macro_primary_score={row['overall_macro_primary_score']:.6f}, pair_win_count={int(row['pair_win_count'])})"
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `openbiollm70b` remains excluded from formal comparative tables.",
            "- `human-LLM agreement` is deferred and not executed in this phase.",
            "- Manual judge contestant roster is fixed to `gpt54`, `deepseek_chat`, `aloe70b`, `medgemma15_4b_it` for vendor diversity and parameter range coverage.",
            "- `GPT-5.4` appears both as a contestant and as a cross-check judge; this overlap is documented, while `Claude Opus 4.6` remains the primary judge.",
            "- Judge manifests are built, but judge API calls are not executed by this script.",
            "",
        ]
    )
    formal_summary_md_path.write_text("\n".join(md_lines))


def main() -> None:
    args = parse_args()
    if args.mode == "canonicalize":
        run_canonicalize(args)
        return
    if args.mode == "score":
        run_score(args)
        return
    if args.mode == "finalize_score":
        run_finalize_score(args)
        return
    if args.mode == "build_judge":
        run_build_judge(args)
        return
    raise SystemExit(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
