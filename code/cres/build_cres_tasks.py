"""
Build CRES tasks (opt-in).

Tasks:
1) trend_threshold
2) temporal_grounding (+ evidence index)
3) diagnostic_consistency
4) contrastive_inference
"""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR


EPISODES_DIR = ROOT_DIR / "episodes" / "episodes_enhanced"
OUT_DIR = ROOT_DIR / "results" / "cres"

VITALS_FEATURES = ["heart_rate", "sbp", "mbp", "resp_rate", "temperature", "spo2"]
ALLOWED_GROUNDING_NOTE_TYPES = ("nursing", "radiology", "lab_comment")
GROUNDING_QUALITY_ORDER = ("high", "medium", "low", "unknown")
GROUNDING_LABEL_ORDER = ("high", "medium", "low")
DIAG_LABEL_ORDER = ("consistent", "inconsistent", "ambiguous")
CONTRAST_LABEL_ORDER = ("A_more_plausible", "B_more_plausible", "tie")
LABEL_DERIVATION_VERSION = "grounding_v2_20260212"

THRESHOLDS = {
    "heart_rate": 100,
    "sbp": 90,
    "mbp": 70,
    "resp_rate": 20,
    "temperature": 38.0,
    "spo2": 94,
}

TARGET_CONDITIONS = ("sepsis", "aki", "delirium", "stroke")
CONDITION_ALIASES = {
    "sepsis": "sepsis",
    "septic": "sepsis",
    "aki": "aki",
    "acute kidney injury": "aki",
    "delirium": "delirium",
    "encephalopathy": "delirium",
    "stroke": "stroke",
    "cva": "stroke",
}
CONTEXT_ALIASES = {
    **CONDITION_ALIASES,
    "ards": "ards",
    "acute respiratory distress syndrome": "ards",
    "ckd": "ckd",
    "chronic kidney disease": "ckd",
}
SEVERITY_WEIGHT = {"mild": 1.0, "moderate": 1.5, "severe": 2.0, "critical": 2.3}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episode_vitals(ep_path: Path):
    with ep_path.open() as f:
        ep = json.load(f)
    vitals = ep.get("timeseries", {}).get("vitals", [])
    if not vitals:
        return None
    df = pd.DataFrame(vitals)
    if "hour" not in df.columns and "hour_offset" in df.columns:
        df["hour"] = df["hour_offset"]
    return ep, df


def _normalize_quality(raw_quality: str) -> str:
    q = str(raw_quality or "").strip().lower()
    if q in {"high", "medium", "low"}:
        return q
    return "unknown"


def _normalize_condition(text: str):
    s = str(text or "").strip().lower()
    for key, value in CONDITION_ALIASES.items():
        if key in s:
            return value
    return None


def _normalize_context_condition(text: str):
    s = str(text or "").strip().lower()
    for key, value in CONTEXT_ALIASES.items():
        if key in s:
            return value
    return None


def _pattern_keyword_hit(pattern_name: str, note_text: str) -> bool:
    text = str(note_text or "").strip().lower()
    if not text:
        return False
    tokens = [t for t in str(pattern_name).strip().lower().split("_") if len(t) >= 4]
    return any(t in text for t in tokens)


def _derive_grounding_label(row: dict) -> str:
    dt = abs(float(row.get("time_delta_hours", 99)))
    quality = row.get("alignment_quality", "unknown")
    note_text = row.get("note_text_relevant", "")
    pattern_name = row.get("pattern_name", "")

    if dt <= 1:
        temporal_score = 3
    elif dt <= 3:
        temporal_score = 2
    elif dt <= 6:
        temporal_score = 1
    else:
        temporal_score = 0

    quality_score = {"high": 2, "medium": 1, "low": 0, "unknown": 0}.get(quality, 0)
    semantic_score = 1 if _pattern_keyword_hit(pattern_name, note_text) else 0
    conflict_penalty = -1 if dt > 12 else 0
    if quality == "unknown":
        conflict_penalty -= 1

    total = temporal_score + quality_score + semantic_score + conflict_penalty
    if total >= 5:
        return "high"
    if total >= 3:
        return "medium"
    return "low"


def _parse_grounding_row(row: dict):
    note_type = str(row.get("note_type", "")).strip().lower()
    if note_type not in ALLOWED_GROUNDING_NOTE_TYPES:
        return None

    try:
        stay_id = int(float(row.get("stay_id")))
        note_hour = float(row.get("note_hour", "nan"))
        pattern_hour = float(row.get("pattern_hour", "nan"))
        time_delta = float(row.get("time_delta_hours", "nan"))
    except (TypeError, ValueError):
        return None

    if not (0 <= note_hour < 24):
        return None
    if time_delta > 0:
        return None

    pattern_name = str(row.get("pattern_name", "")).strip()
    note_id = str(row.get("note_id", "")).strip()
    if not pattern_name or not note_id:
        return None

    parsed = {
        "stay_id": stay_id,
        "pattern_hour": pattern_hour,
        "pattern_name": pattern_name,
        "note_id": note_id,
        "note_hour": note_hour,
        "note_type": note_type,
        "note_text_relevant": row.get("note_text_relevant", ""),
        "time_delta_hours": time_delta,
        "alignment_quality": _normalize_quality(row.get("alignment_quality", "unknown")),
    }
    parsed["derived_label"] = _derive_grounding_label(parsed)
    return parsed


def _allocate_note_type_targets(n_samples: int, note_type_counts: dict) -> dict:
    available = [
        (nt, int(note_type_counts.get(nt, 0)))
        for nt in ALLOWED_GROUNDING_NOTE_TYPES
        if note_type_counts.get(nt, 0) > 0
    ]
    if not available:
        return {}

    base = n_samples // len(available)
    remainder = n_samples % len(available)
    ordered = sorted(available, key=lambda x: (-x[1], x[0]))
    targets = {}

    for idx, (nt, cnt) in enumerate(ordered):
        take = base + (1 if idx < remainder else 0)
        targets[nt] = min(take, cnt)

    assigned = sum(targets.values())
    while assigned < n_samples:
        progressed = False
        for nt, cnt in ordered:
            if targets[nt] < cnt:
                targets[nt] += 1
                assigned += 1
                progressed = True
                if assigned >= n_samples:
                    break
        if not progressed:
            break

    return targets


def _allocate_derived_label_targets(note_type_target: int, label_counts: Counter) -> dict:
    label_counts = {k: int(label_counts.get(k, 0)) for k in GROUNDING_LABEL_ORDER}
    targets = {k: 0 for k in GROUNDING_LABEL_ORDER}
    target_ratio = {"high": 0.30, "medium": 0.40, "low": 0.30}

    for label in GROUNDING_LABEL_ORDER:
        ideal = int(round(note_type_target * target_ratio[label]))
        targets[label] = min(ideal, label_counts[label])

    assigned = sum(targets.values())
    while assigned < note_type_target:
        progressed = False
        for label in sorted(GROUNDING_LABEL_ORDER, key=lambda k: label_counts[k] - targets[k], reverse=True):
            if targets[label] < label_counts[label]:
                targets[label] += 1
                assigned += 1
                progressed = True
                if assigned >= note_type_target:
                    break
        if not progressed:
            break

    return targets


def _row_uid(row: dict):
    return (
        row["stay_id"],
        row["note_id"],
        row["pattern_name"],
        round(float(row["note_hour"]), 4),
        round(float(row["pattern_hour"]), 4),
    )


def _pick_rows_with_pattern_cap(pool: list, required: int, max_per_pattern: int, used: set):
    if required <= 0:
        return []
    if not pool:
        return []

    random.shuffle(pool)
    chosen = []
    pattern_counter = Counter()

    for row in pool:
        if len(chosen) >= required:
            break
        uid = _row_uid(row)
        if uid in used:
            continue
        pattern = row["pattern_name"]
        if pattern_counter[pattern] >= max_per_pattern:
            continue
        chosen.append(row)
        used.add(uid)
        pattern_counter[pattern] += 1

    if len(chosen) < required:
        for row in pool:
            if len(chosen) >= required:
                break
            uid = _row_uid(row)
            if uid in used:
                continue
            chosen.append(row)
            used.add(uid)

    return chosen


def _balanced_sample_by_label(rows: list, n_samples: int, label_key: str, order: tuple, ratio: dict):
    if not rows or n_samples <= 0:
        return []

    buckets = defaultdict(list)
    for row in rows:
        buckets[row.get(label_key, "unknown")].append(row)
    for key in buckets:
        random.shuffle(buckets[key])

    selected = []
    targets = {}
    for key in order:
        targets[key] = int(round(n_samples * ratio.get(key, 0.0)))
    assigned = sum(targets.values())
    while assigned < n_samples:
        for key in order:
            targets[key] += 1
            assigned += 1
            if assigned >= n_samples:
                break

    for key in order:
        take = min(targets.get(key, 0), len(buckets.get(key, [])))
        selected.extend(buckets.get(key, [])[:take])
        buckets[key] = buckets.get(key, [])[take:]

    if len(selected) < n_samples:
        remainder = []
        for key in order:
            remainder.extend(buckets.get(key, []))
        random.shuffle(remainder)
        selected.extend(remainder[: max(0, n_samples - len(selected))])

    random.shuffle(selected)
    return selected[:n_samples]


def build_trend_threshold(n_samples):
    examples = []
    files = list(EPISODES_DIR.glob("TIMELY_v2_*.json"))
    random.shuffle(files)
    for ep_path in files:
        if len(examples) >= n_samples:
            break
        loaded = load_episode_vitals(ep_path)
        if loaded is None:
            continue
        ep, df = loaded
        stay_id = ep.get("stay_id")
        if stay_id is None:
            continue

        for feat in VITALS_FEATURES:
            if feat not in df.columns:
                continue
            values = pd.to_numeric(df[feat], errors="coerce").dropna()
            if values.empty:
                continue

            thr = THRESHOLDS.get(feat)
            if thr is None:
                continue
            mean_val = float(values.mean())
            answer = "yes" if mean_val >= thr else "no"
            examples.append({
                "id": f"{stay_id}_{feat}_threshold",
                "task": "trend_threshold",
                "subtask": "threshold",
                "stay_id": int(stay_id),
                "feature": feat,
                "question": f"Is mean {feat} >= {thr} in 0-24h?",
                "value": mean_val,
                "threshold": thr,
                "answer": answer,
                "evidence": {"feature": feat, "mean": mean_val, "threshold": thr},
            })
            if len(examples) >= n_samples:
                break

            if "hour" in df.columns:
                early = df[df["hour"] < 6]
                late = df[df["hour"] >= 18]
                if not early.empty and not late.empty:
                    early_mean = float(pd.to_numeric(early[feat], errors="coerce").dropna().mean())
                    late_mean = float(pd.to_numeric(late[feat], errors="coerce").dropna().mean())
                    if not (pd.isna(early_mean) or pd.isna(late_mean)):
                        trend = "up" if late_mean > early_mean else "down"
                        examples.append({
                            "id": f"{stay_id}_{feat}_trend",
                            "task": "trend_threshold",
                            "subtask": "trend",
                            "stay_id": int(stay_id),
                            "feature": feat,
                            "question": f"Is {feat} increasing from 0-6h to 18-24h?",
                            "value": {"early_mean": early_mean, "late_mean": late_mean},
                            "answer": "yes" if trend == "up" else "no",
                            "evidence": {"early_mean": early_mean, "late_mean": late_mean},
                        })
            if len(examples) >= n_samples:
                break

    return examples


def build_temporal_grounding(n_samples, max_rows=0):
    alignment_path = TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"
    if not alignment_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {alignment_path}")

    note_type_counts = Counter()
    derived_counts = defaultdict(Counter)
    total_eligible = 0

    with alignment_path.open() as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            row = _parse_grounding_row(raw)
            if row is None:
                continue
            total_eligible += 1
            nt = row["note_type"]
            lb = row["derived_label"]
            note_type_counts[nt] += 1
            derived_counts[nt][lb] += 1

    note_type_targets = _allocate_note_type_targets(n_samples, note_type_counts)
    if not note_type_targets:
        return [], {
            "n_eligible_rows": 0,
            "note_type_counts": {},
            "note_type_targets": {},
            "derived_label_targets": {},
            "realized_note_type_counts": {},
            "realized_label_counts": {},
        }

    label_targets = {
        nt: _allocate_derived_label_targets(note_type_targets[nt], derived_counts[nt])
        for nt in note_type_targets
    }

    key_targets = {}
    for nt, targets in label_targets.items():
        for lb, tgt in targets.items():
            if tgt > 0:
                key_targets[(nt, lb)] = int(tgt)

    reservoirs = {k: [] for k in key_targets}
    seen = Counter()
    with alignment_path.open() as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            row = _parse_grounding_row(raw)
            if row is None:
                continue
            key = (row["note_type"], row["derived_label"])
            if key not in key_targets:
                continue
            seen[key] += 1
            cap = max(300, key_targets[key] * 12)
            bucket = reservoirs[key]
            if len(bucket) < cap:
                bucket.append(row)
                continue
            j = random.randint(1, seen[key])
            if j <= cap:
                bucket[j - 1] = row

    selected = []
    used = set()
    realized_note_type_counts = Counter()
    realized_label_counts = Counter()

    for nt, nt_target in note_type_targets.items():
        nt_selected = []
        nt_pattern_cap = max(1, int(math.ceil(nt_target * 0.35)))
        for lb in GROUNDING_LABEL_ORDER:
            need = label_targets.get(nt, {}).get(lb, 0)
            if need <= 0:
                continue
            pool = list(reservoirs.get((nt, lb), []))
            picked = _pick_rows_with_pattern_cap(pool, need, nt_pattern_cap, used)
            nt_selected.extend(picked)

        if len(nt_selected) < nt_target:
            fallback = []
            for lb in GROUNDING_LABEL_ORDER:
                fallback.extend(reservoirs.get((nt, lb), []))
            nt_selected.extend(
                _pick_rows_with_pattern_cap(
                    fallback, nt_target - len(nt_selected), nt_pattern_cap, used
                )
            )
        selected.extend(nt_selected)
        realized_note_type_counts[nt] += len(nt_selected)
        for row in nt_selected:
            realized_label_counts[row["derived_label"]] += 1

    if len(selected) < n_samples:
        global_pool = []
        for rows in reservoirs.values():
            global_pool.extend(rows)
        global_cap = max(1, int(math.ceil(n_samples * 0.45)))
        backfill = _pick_rows_with_pattern_cap(
            global_pool, n_samples - len(selected), global_cap, used
        )
        selected.extend(backfill)
        for row in backfill:
            realized_note_type_counts[row["note_type"]] += 1
            realized_label_counts[row["derived_label"]] += 1

    random.shuffle(selected)
    selected = selected[:n_samples]

    examples = []
    for row in selected:
        examples.append({
            "id": f"{row['stay_id']}_{row['note_id']}_{row['pattern_name']}",
            "task": "temporal_grounding",
            "stay_id": int(row["stay_id"]),
            "pattern_hour": float(row["pattern_hour"]),
            "pattern_name": row["pattern_name"],
            "note_id": str(row["note_id"]),
            "note_hour": float(row["note_hour"]),
            "note_type": row["note_type"],
            "note_text_relevant": row.get("note_text_relevant", ""),
            "time_delta_hours": float(row["time_delta_hours"]),
            "label": row["derived_label"],
            "source_alignment_quality": row["alignment_quality"],
            "label_derivation_version": LABEL_DERIVATION_VERSION,
        })

    summary = {
        "n_eligible_rows": int(total_eligible),
        "note_type_counts": {k: int(v) for k, v in note_type_counts.items()},
        "note_type_targets": {k: int(v) for k, v in note_type_targets.items()},
        "derived_label_targets": {
            nt: {k: int(v) for k, v in targets.items()} for nt, targets in label_targets.items()
        },
        "realized_note_type_counts": {k: int(v) for k, v in realized_note_type_counts.items()},
        "realized_label_counts": {k: int(v) for k, v in realized_label_counts.items()},
    }
    return examples, summary


def _extract_detected_patterns(ep: dict):
    patterns = []
    detected = ep.get("reasoning", {}).get("detected_patterns", [])
    if not isinstance(detected, list):
        return patterns
    for item in detected:
        if not isinstance(item, dict):
            continue
        disease = _normalize_condition(item.get("disease", ""))
        if disease is None:
            continue
        if disease not in TARGET_CONDITIONS:
            continue
        patterns.append({
            "pattern_name": str(item.get("pattern_name", "")),
            "severity": str(item.get("severity", "mild")).strip().lower(),
            "disease": disease,
            "hour": float(item.get("detection_hour", 0) or 0),
        })
    return patterns


def _extract_condition_context(ep: dict):
    context = set()

    # Legacy fallback (older episodes may have top-level "conditions")
    raw_conditions = ep.get("conditions", [])
    if isinstance(raw_conditions, list):
        for value in raw_conditions:
            norm = _normalize_context_condition(value)
            if norm:
                context.add(norm)

    labels = ep.get("labels", {})
    if isinstance(labels, dict):
        bool_map = {
            "has_sepsis": "sepsis",
            "has_aki": "aki",
            "has_ards": "ards",
            "has_delirium": "delirium",
            "has_stroke": "stroke",
            "has_ckd": "ckd",
        }
        for key, value in bool_map.items():
            if bool(labels.get(key, False)):
                context.add(value)

        diagnoses_text = labels.get("diagnoses_text", [])
        if isinstance(diagnoses_text, str):
            diagnoses_text = [diagnoses_text]
        if isinstance(diagnoses_text, list):
            for item in diagnoses_text:
                norm = _normalize_context_condition(item)
                if norm:
                    context.add(norm)

        icd_codes = labels.get("icd_codes", [])
        if isinstance(icd_codes, str):
            icd_codes = [icd_codes]
        if isinstance(icd_codes, list):
            for code in icd_codes:
                c = str(code).strip().upper()
                # CKD ICD-10 N18*; ICD-9 585*
                if c.startswith("N18") or c.startswith("585"):
                    context.add("ckd")

    return sorted(context)


def _diagnostic_label_from_delta(delta: float) -> str:
    if delta >= 2.0:
        return "consistent"
    if delta <= -2.0:
        return "inconsistent"
    return "ambiguous"


def _enforce_multimorbidity_ratio(rows: list, pool: list, min_ratio: float) -> list:
    if not rows:
        return rows
    if min_ratio <= 0:
        return rows

    need_with_context = int(math.ceil(len(rows) * min_ratio))
    have_with_context = sum(1 for r in rows if r.get("multimorbidity_context"))
    if have_with_context >= need_with_context:
        return rows

    selected_ids = {r["id"] for r in rows}
    candidates = [
        r for r in pool
        if r.get("multimorbidity_context") and r["id"] not in selected_ids
    ]
    random.shuffle(candidates)
    if not candidates:
        return rows

    rows = list(rows)
    need = need_with_context - have_with_context
    for cand in candidates:
        if need <= 0:
            break
        # Replace a same-label row without context first, then any row without context.
        replace_idx = None
        for idx, cur in enumerate(rows):
            if cur.get("multimorbidity_context"):
                continue
            if cur.get("label") == cand.get("label"):
                replace_idx = idx
                break
        if replace_idx is None:
            for idx, cur in enumerate(rows):
                if not cur.get("multimorbidity_context"):
                    replace_idx = idx
                    break
        if replace_idx is None:
            break
        rows[replace_idx] = cand
        selected_ids.add(cand["id"])
        need -= 1

    return rows


def build_diagnostic_consistency(n_samples, min_multimorbidity_ratio=0.3):
    files = list(EPISODES_DIR.glob("TIMELY_v2_*.json"))
    random.shuffle(files)
    pool = []

    for ep_path in files:
        try:
            ep = json.loads(ep_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        stay_id = ep.get("stay_id")
        if stay_id is None:
            continue

        patterns = _extract_detected_patterns(ep)
        if not patterns:
            continue

        score = Counter()
        for p in patterns:
            sev = p["severity"]
            score[p["disease"]] += SEVERITY_WEIGHT.get(sev, 1.0)

        ranking = sorted(score.items(), key=lambda x: x[1], reverse=True)
        primary = ranking[0][0]
        alternative = ranking[1][0] if len(ranking) > 1 else random.choice(
            [c for c in TARGET_CONDITIONS if c != primary]
        )

        primary_score = float(score.get(primary, 0.0))
        alt_score = float(score.get(alternative, 0.0))
        delta = primary_score - alt_score
        label = _diagnostic_label_from_delta(delta)
        support = [p["pattern_name"] for p in patterns if p["disease"] == primary][:8]
        conflict = [p["pattern_name"] for p in patterns if p["disease"] == alternative][:8]
        context = [c for c in _extract_condition_context(ep) if c not in {primary, alternative}]
        pool.append({
            "id": f"{stay_id}_{primary}_vs_{alternative}_primary",
            "task": "diagnostic_consistency",
            "stay_id": int(stay_id),
            "window": "0-24h",
            "primary_hypothesis": primary,
            "alternative_hypothesis": alternative,
            "supporting_patterns": support,
            "conflicting_patterns": conflict,
            "multimorbidity_context": context,
            "score_primary": primary_score,
            "score_alternative": alt_score,
            "score_delta": delta,
            "label": label,
            "rationale": f"delta={delta:.2f} from pattern severity aggregation",
        })

        if abs(delta) >= 1.5:
            swap_delta = -delta
            pool.append({
                "id": f"{stay_id}_{alternative}_vs_{primary}_swap",
                "task": "diagnostic_consistency",
                "stay_id": int(stay_id),
                "window": "0-24h",
                "primary_hypothesis": alternative,
                "alternative_hypothesis": primary,
                "supporting_patterns": conflict,
                "conflicting_patterns": support,
                "multimorbidity_context": context,
                "score_primary": alt_score,
                "score_alternative": primary_score,
                "score_delta": swap_delta,
                "label": _diagnostic_label_from_delta(swap_delta),
                "rationale": f"swapped-hypothesis delta={swap_delta:.2f}",
            })

        if len(pool) >= n_samples * 4:
            break

    sampled = _balanced_sample_by_label(
        pool,
        n_samples,
        "label",
        DIAG_LABEL_ORDER,
        {"consistent": 0.40, "inconsistent": 0.30, "ambiguous": 0.30},
    )
    sampled = _enforce_multimorbidity_ratio(
        rows=sampled,
        pool=pool,
        min_ratio=min_multimorbidity_ratio,
    )
    return sampled


def _plausibility_from_grounding(row: dict) -> int:
    label_score = {"high": 2, "medium": 1, "low": 0}.get(row.get("derived_label", "low"), 0)
    dt = abs(float(row.get("time_delta_hours", 99)))
    if dt <= 1:
        temporal = 2
    elif dt <= 3:
        temporal = 1
    else:
        temporal = 0
    return label_score + temporal


def build_contrastive_inference(n_samples, max_rows=0):
    alignment_path = TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"
    if not alignment_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {alignment_path}")

    grouped = defaultdict(list)
    with alignment_path.open() as f:
        reader = csv.DictReader(f)
        for idx, raw in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            row = _parse_grounding_row(raw)
            if row is None:
                continue
            key = (row["stay_id"], row["pattern_name"])
            grouped[key].append(row)

    pool = []
    for (stay_id, pattern_name), rows in grouped.items():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=_plausibility_from_grounding, reverse=True)
        best = rows[0]
        worst = rows[-1]
        score_best = _plausibility_from_grounding(best)
        score_worst = _plausibility_from_grounding(worst)
        diff = score_best - score_worst

        if len(rows) > 2:
            near = rows[1]
            near_diff = abs(score_best - _plausibility_from_grounding(near))
            if near_diff <= 1:
                pool.append({
                    "id": f"{stay_id}_{pattern_name}_tie",
                    "task": "contrastive_inference",
                    "stay_id": int(stay_id),
                    "pattern_name": pattern_name,
                    "question": f"Which option better supports {pattern_name} around event time?",
                    "option_a": {
                        "note_id": best["note_id"],
                        "note_type": best["note_type"],
                        "note_hour": best["note_hour"],
                        "time_delta_hours": best["time_delta_hours"],
                        "source_alignment_quality": best["alignment_quality"],
                    },
                    "option_b": {
                        "note_id": near["note_id"],
                        "note_type": near["note_type"],
                        "note_hour": near["note_hour"],
                        "time_delta_hours": near["time_delta_hours"],
                        "source_alignment_quality": near["alignment_quality"],
                    },
                    "label": "tie",
                    "score_delta": float(score_best - _plausibility_from_grounding(near)),
                })

        if diff <= 0:
            continue

        forward_id = f"{stay_id}_{pattern_name}_ab"
        reverse_id = f"{stay_id}_{pattern_name}_ba"
        pool.append({
            "id": forward_id,
            "task": "contrastive_inference",
            "stay_id": int(stay_id),
            "pattern_name": pattern_name,
            "question": f"Which option better supports {pattern_name} around event time?",
            "option_a": {
                "note_id": best["note_id"],
                "note_type": best["note_type"],
                "note_hour": best["note_hour"],
                "time_delta_hours": best["time_delta_hours"],
                "source_alignment_quality": best["alignment_quality"],
            },
            "option_b": {
                "note_id": worst["note_id"],
                "note_type": worst["note_type"],
                "note_hour": worst["note_hour"],
                "time_delta_hours": worst["time_delta_hours"],
                "source_alignment_quality": worst["alignment_quality"],
            },
            "label": "A_more_plausible",
            "score_delta": float(diff),
        })
        pool.append({
            "id": reverse_id,
            "task": "contrastive_inference",
            "stay_id": int(stay_id),
            "pattern_name": pattern_name,
            "question": f"Which option better supports {pattern_name} around event time?",
            "option_a": {
                "note_id": worst["note_id"],
                "note_type": worst["note_type"],
                "note_hour": worst["note_hour"],
                "time_delta_hours": worst["time_delta_hours"],
                "source_alignment_quality": worst["alignment_quality"],
            },
            "option_b": {
                "note_id": best["note_id"],
                "note_type": best["note_type"],
                "note_hour": best["note_hour"],
                "time_delta_hours": best["time_delta_hours"],
                "source_alignment_quality": best["alignment_quality"],
            },
            "label": "B_more_plausible",
            "score_delta": float(-diff),
        })

        if len(pool) >= n_samples * 4:
            break

    return _balanced_sample_by_label(
        pool,
        n_samples,
        "label",
        CONTRAST_LABEL_ORDER,
        {"A_more_plausible": 0.35, "B_more_plausible": 0.35, "tie": 0.30},
    )


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trend", type=int, default=2000)
    parser.add_argument("--n-grounding", type=int, default=2000)
    parser.add_argument("--n-diagnostic", type=int, default=1200)
    parser.add_argument("--n-contrastive", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Max rows scanned from alignment CSV (0 = full file).",
    )
    parser.add_argument(
        "--min-multimorbidity-ratio",
        type=float,
        default=0.3,
        help="Minimum ratio of diagnostic_consistency samples with non-empty multimorbidity_context.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    trend_rows = build_trend_threshold(args.n_trend)
    grounding_rows, grounding_sampling = build_temporal_grounding(args.n_grounding, args.max_rows)
    diagnostic_rows = build_diagnostic_consistency(
        args.n_diagnostic,
        min_multimorbidity_ratio=args.min_multimorbidity_ratio,
    )
    contrastive_rows = build_contrastive_inference(args.n_contrastive, args.max_rows)

    trend_path = OUT_DIR / "trend_threshold.jsonl"
    grounding_path = OUT_DIR / "temporal_grounding.jsonl"
    grounding_index_path = OUT_DIR / "temporal_grounding_index.jsonl"
    diagnostic_path = OUT_DIR / "diagnostic_consistency.jsonl"
    contrastive_path = OUT_DIR / "contrastive_inference.jsonl"

    write_jsonl(trend_path, trend_rows)
    write_jsonl(grounding_path, grounding_rows)
    write_jsonl(diagnostic_path, diagnostic_rows)
    write_jsonl(contrastive_path, contrastive_rows)

    with grounding_index_path.open("w") as f:
        for row in grounding_rows:
            key = {
                "stay_id": row["stay_id"],
                "pattern_hour": row["pattern_hour"],
                "pattern_name": row["pattern_name"],
                "note_id": row["note_id"],
                "note_hour": row["note_hour"],
            }
            f.write(json.dumps(key, ensure_ascii=True) + "\n")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trend_threshold": {"n": len(trend_rows), "path": str(trend_path)},
        "temporal_grounding": {"n": len(grounding_rows), "path": str(grounding_path)},
        "temporal_grounding_index": {"n": len(grounding_rows), "path": str(grounding_index_path)},
        "diagnostic_consistency": {"n": len(diagnostic_rows), "path": str(diagnostic_path)},
        "contrastive_inference": {"n": len(contrastive_rows), "path": str(contrastive_path)},
        "alignment_source": str(TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"),
        "grounding_sampling": grounding_sampling,
        "diagnostic_multimorbidity_nonempty_ratio": (
            float(
                sum(1 for r in diagnostic_rows if r.get("multimorbidity_context")) / len(diagnostic_rows)
            )
            if diagnostic_rows
            else 0.0
        ),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "min_multimorbidity_ratio": args.min_multimorbidity_ratio,
    }

    meta_path = OUT_DIR / "cres_build_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True))

    alignment_path = Path(meta["alignment_source"])
    inputs = []
    if alignment_path.exists():
        inputs.append({"path": str(alignment_path), "sha256": sha256_file(alignment_path)})

    episodes_tar = ROOT_DIR / "episodes_enhanced.tar.gz"
    if episodes_tar.exists():
        inputs.append({"path": str(episodes_tar), "sha256": sha256_file(episodes_tar)})

    outputs = [
        {"path": str(trend_path), "sha256": sha256_file(trend_path)},
        {"path": str(grounding_path), "sha256": sha256_file(grounding_path)},
        {"path": str(grounding_index_path), "sha256": sha256_file(grounding_index_path)},
        {"path": str(diagnostic_path), "sha256": sha256_file(diagnostic_path)},
        {"path": str(contrastive_path), "sha256": sha256_file(contrastive_path)},
        {"path": str(meta_path), "sha256": sha256_file(meta_path)},
    ]

    manifest = {
        "timestamp": meta["timestamp"],
        "n_trend": len(trend_rows),
        "n_grounding": len(grounding_rows),
        "n_diagnostic": len(diagnostic_rows),
        "n_contrastive": len(contrastive_rows),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "inputs": inputs,
        "outputs": outputs,
    }
    manifest_path = OUT_DIR / "cres_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))

    print(f"Wrote {trend_path}")
    print(f"Wrote {grounding_path}")
    print(f"Wrote {diagnostic_path}")
    print(f"Wrote {contrastive_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
