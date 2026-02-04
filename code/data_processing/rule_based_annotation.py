"""
Rule-based annotation for pattern-note relations (opt-in).
"""

import json
import re
from typing import Dict, Any


RULE_CONFIG = {
    "version": "v1",
    "negation_terms": [
        "no ", "denies ", "without ", "negative for ", "rule out ", "ruled out ", "not ",
        "absent ", "free of ", "no evidence of ", "no signs of ",
    ],
    "max_negation_window": 40,
    "max_evidence_len": 240,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _find_first(term: str, text: str):
    idx = text.find(term)
    return idx if idx >= 0 else None


def _is_negated(term: str, text: str, window: int) -> bool:
    idx = _find_first(term, text)
    if idx is None:
        return False
    start = max(0, idx - window)
    prefix = text[start:idx]
    for neg in RULE_CONFIG["negation_terms"]:
        if neg in prefix:
            return True
    return False


def _extract_evidence(text: str, term: str, max_len: int) -> str:
    if not text:
        return ""
    idx = _find_first(term, text)
    if idx is None:
        return text[:max_len]
    start = max(0, idx - max_len // 2)
    end = min(len(text), idx + max_len // 2)
    return text[start:end]


def annotate_record(rec: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
    cfg = config or RULE_CONFIG
    text_raw = str(rec.get("note_text_relevant", "") or "")
    text = _normalize(text_raw)
    pattern = str(rec.get("pattern_name", "") or "").lower()

    label = "UNRELATED"
    evidence_span = ""
    evidence_note = ""

    if not text:
        label = "UNRELATED"
        evidence_note = "empty_note_text"
        return {
            "label": label,
            "evidence_span": evidence_span,
            "evidence_note": evidence_note,
        }

    if pattern and pattern in text:
        if _is_negated(pattern, text, cfg.get("max_negation_window", 40)):
            label = "CONTRADICTORY"
        else:
            label = "SUPPORTIVE"
        evidence_span = _extract_evidence(text_raw, pattern, cfg.get("max_evidence_len", 240))
    else:
        label = "UNRELATED"
        evidence_span = text_raw[: cfg.get("max_evidence_len", 240)]

    return {
        "label": label,
        "evidence_span": evidence_span,
        "evidence_note": evidence_note,
    }


def rule_config_hash() -> str:
    payload = json.dumps(RULE_CONFIG, sort_keys=True, ensure_ascii=True).encode("utf-8")
    import hashlib
    return hashlib.sha256(payload).hexdigest()

