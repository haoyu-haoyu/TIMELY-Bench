"""
Utilities for building windowed / note-type MedCAT "has_*" features.

We intentionally mirror the semantics of:
  code/data_processing/build_medcat_has_features.py

But add:
- time-window filtering (6h / 12h / 24h and D0 daily)
- note_type filtering (nursing / radiology / lab_comment / discharge)

This module streams `medcat_note_concepts_24h.csv` to avoid loading it into memory.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


PATTERN_KEYS: List[str] = [
    "has_sepsis",
    "has_pneumonia",
    "has_infection",
    "has_antibiotic",
    "has_aki",
    "has_ards",
]

_PATTERNS = {
    "has_sepsis": re.compile(r"\bsepsis\b|\bseptic\b|\bsepticemia\b|\bsepticaemia\b", re.I),
    "has_pneumonia": re.compile(r"\bpneumonia\b|\bpneumonitis\b", re.I),
    "has_infection": re.compile(
        r"\binfection\b|\binfectious\b|\bbacteremia\b|\bbacteraemia\b|\bcellulitis\b"
        r"|\babscess\b|\buti\b|\burinary tract infection\b|\bmeningitis\b"
        r"|\bendocarditis\b|\bosteomyelitis\b",
        re.I,
    ),
    "has_antibiotic": re.compile(
        r"\bantibiotic\b|\bantibacterial\b|\bantimicrobial\b|\banti-?infective\b",
        re.I,
    ),
    "has_aki": re.compile(
        r"\bacute kidney injury\b|\bacute renal failure\b|\bacute renal injury\b"
        r"|\bacute kidney failure\b|\bacute renal insufficiency\b|\baki\b",
        re.I,
    ),
    "has_ards": re.compile(
        r"\bacute respiratory distress syndrome\b|\badult respiratory distress syndrome\b"
        r"|\bacute lung injury\b|\bards\b",
        re.I,
    ),
}

NEGATION_MARKERS = ("neg", "no", "denied", "absent", "without")
HISTORICAL_MARKERS = ("histor", "past", "prior")


def _safe_float(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_negated(value) -> bool:
    if value is None:
        return False
    val = str(value).strip().lower()
    if not val or val == "nan":
        return False
    return any(marker in val for marker in NEGATION_MARKERS)


def _is_historical(value) -> bool:
    if value is None:
        return False
    val = str(value).strip().lower()
    if not val or val == "nan":
        return False
    return any(marker in val for marker in HISTORICAL_MARKERS)


def match_bits(concept_name: str) -> int:
    """
    Convert a concept name into a bitmask over PATTERN_KEYS.
    Bit i corresponds to PATTERN_KEYS[i].
    """
    name = (concept_name or "").strip().lower()
    if not name:
        return 0
    bits = 0
    for i, key in enumerate(PATTERN_KEYS):
        if _PATTERNS[key].search(name):
            bits |= 1 << i
    return bits


@dataclass(frozen=True)
class WindowConfig:
    name: str
    hours: Optional[float] = None  # if None, caller must provide per-stay cutoff via d0_cutoff_hours
    is_d0: bool = False


def build_window_masks(
    note_concepts_csv: Path,
    stay_id_to_index: Mapping[int, int],
    windows: Sequence[WindowConfig],
    allowed_note_types: Optional[Iterable[str]],
    d0_cutoff_hours: Optional[np.ndarray],
    observation_horizon_hours: float = 24.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Stream note-level MedCAT concepts and build (pos_mask, neg_mask) per window.

    Returns:
      {window_name: (pos_mask_uint16, neg_mask_uint16)}
    """
    n = len(stay_id_to_index)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for w in windows:
        out[w.name] = (np.zeros(n, dtype=np.uint16), np.zeros(n, dtype=np.uint16))

    allowed = None
    if allowed_note_types is not None:
        allowed = {str(x).strip().lower() for x in allowed_note_types}

    if any(w.is_d0 for w in windows) and d0_cutoff_hours is None:
        raise ValueError("d0_cutoff_hours is required when windows include is_d0=True")

    with open(note_concepts_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay_id = _safe_int(row.get("stay_id"))
            if stay_id is None:
                continue
            idx = stay_id_to_index.get(stay_id)
            if idx is None:
                continue

            note_type = (row.get("note_type") or "").strip().lower()
            if allowed is not None and note_type not in allowed:
                continue

            chart_hour = _safe_float(row.get("chart_hour"))
            if chart_hour is None or chart_hour < 0:
                continue
            if chart_hour >= observation_horizon_hours:
                continue

            bits = match_bits(row.get("name") or "")
            if bits == 0:
                continue

            is_neg = _is_negated(row.get("negation")) or _is_historical(row.get("temporality"))

            for w in windows:
                if w.is_d0:
                    cutoff = float(d0_cutoff_hours[idx])
                else:
                    cutoff = float(w.hours) if w.hours is not None else observation_horizon_hours
                if chart_hour < cutoff:
                    pos_mask, neg_mask = out[w.name]
                    if is_neg:
                        neg_mask[idx] |= bits
                    else:
                        pos_mask[idx] |= bits

    return out


def build_note_type_masks(
    note_concepts_csv: Path,
    stay_id_to_index: Mapping[int, int],
    note_types: Sequence[str],
    window_hours: float = 24.0,
    observation_horizon_hours: float = 24.0,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Build (pos_mask, neg_mask) per note_type within a fixed chart_hour window.
    """
    n = len(stay_id_to_index)
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    wanted = [str(x).strip().lower() for x in note_types]
    for nt in wanted:
        out[nt] = (np.zeros(n, dtype=np.uint16), np.zeros(n, dtype=np.uint16))

    wanted_set = set(wanted)

    with open(note_concepts_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stay_id = _safe_int(row.get("stay_id"))
            if stay_id is None:
                continue
            idx = stay_id_to_index.get(stay_id)
            if idx is None:
                continue

            note_type = (row.get("note_type") or "").strip().lower()
            if note_type not in wanted_set:
                continue

            chart_hour = _safe_float(row.get("chart_hour"))
            if chart_hour is None or chart_hour < 0:
                continue
            if chart_hour >= observation_horizon_hours:
                continue
            if chart_hour >= window_hours:
                continue

            bits = match_bits(row.get("name") or "")
            if bits == 0:
                continue

            is_neg = _is_negated(row.get("negation")) or _is_historical(row.get("temporality"))
            pos_mask, neg_mask = out[note_type]
            if is_neg:
                neg_mask[idx] |= bits
            else:
                pos_mask[idx] |= bits

    return out


def masks_to_frame(
    stay_ids: Sequence[int],
    pos_mask: np.ndarray,
    neg_mask: np.ndarray,
) -> "np.ndarray":
    """
    Convert masks into a dense feature matrix with columns:
      PATTERN_KEYS + [f"{k}_neg" for k in PATTERN_KEYS]

    Returns:
      np.ndarray shape (N, 2*len(PATTERN_KEYS)) with int values 0/1.
    """
    n = len(stay_ids)
    d = len(PATTERN_KEYS)
    X = np.zeros((n, 2 * d), dtype=np.int8)
    for i in range(d):
        X[:, i] = ((pos_mask >> i) & 1).astype(np.int8)
        X[:, d + i] = ((neg_mask >> i) & 1).astype(np.int8)
    return X

