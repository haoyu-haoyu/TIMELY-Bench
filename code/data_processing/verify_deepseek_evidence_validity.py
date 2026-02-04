#!/usr/bin/env python3
"""
verify_deepseek_evidence_validity.py

Validates that evidence_span values in DeepSeek annotation JSONL files
are genuine substrings of the corresponding note_text_relevant from the
annotation set CSV. Produces a JSON report with validity statistics and
a sample of failure cases.

Deployed path: code/data_processing/verify_deepseek_evidence_validity.py
"""

import json
import csv
import glob
import shutil
import sys
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path bootstrap – allow import of project config regardless of cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LLM_DIR = ROOT_DIR / "results" / "llm_annotations"
ANNOTATION_GLOB = str(LLM_DIR / "annotations_deepseek_*.jsonl")
METADATA_PATH = LLM_DIR / "ANNOTATION_METADATA_deepseek.json"
CSV_PATH = LLM_DIR / "llm_annotation_set.csv"
OUTPUT_PATH = LLM_DIR / "evidence_validity_deepseek.json"
FINAL_RELEASE_DIR = ROOT_DIR / "final_release" / "llm_annotations"
FINAL_RELEASE_PATH = FINAL_RELEASE_DIR / "evidence_validity_deepseek.json"

MAX_FAILURE_SAMPLES = 20
EVIDENCE_TRUNC = 80
NOTE_SNIPPET_LEN = 120


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trunc(text: str | None, limit: int) -> str:
    """Return text truncated to *limit* chars with ellipsis if needed."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def _make_join_key(stay_id, pattern_hour, pattern_name, note_id) -> tuple:
    """Normalise the four-part join key to (str, str, str, str)."""
    return (str(stay_id), str(pattern_hour), str(pattern_name), str(note_id))


def build_note_text_lookup(csv_path: Path) -> dict[tuple, str]:
    """
    Read llm_annotation_set.csv and return a dict mapping
    (stay_id, pattern_hour, pattern_name, note_id) -> note_text_relevant.
    NaN / empty values are stored as empty string.
    """
    lookup: dict[tuple, str] = {}
    with open(csv_path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = _make_join_key(
                row["stay_id"],
                row["pattern_hour"],
                row["pattern_name"],
                row["note_id"],
            )
            text = row.get("note_text_relevant", "") or ""
            # Pandas may write NaN as literal "nan"
            if text.strip().lower() == "nan":
                text = ""
            lookup[key] = text
    return lookup


def load_jsonl_files(pattern: str) -> list[dict]:
    """Glob for JSONL files and return all parsed records.
    Excludes audited files to avoid double-counting."""
    records: list[dict] = []
    paths = sorted(glob.glob(pattern))
    paths = [p for p in paths if "_audited" not in p]
    if not paths:
        print(f"[WARN] No files matched pattern: {pattern}")
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    print(f"[WARN] JSON decode error in {p} line {lineno}: {exc}")
    return records


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------

def validate(records: list[dict], lookup: dict[tuple, str]) -> dict:
    """
    Walk every annotation record and classify its evidence validity.
    Returns a dict ready for JSON serialisation.
    """
    total_records = len(records)
    empty_evidence_count = 0
    empty_evidence_unrelated_count = 0
    join_miss_count = 0
    substring_valid_count = 0
    substring_mismatch_count = 0
    index_present_count = 0
    index_valid_count = 0
    index_oob_count = 0

    failure_samples: list[dict] = []

    for rec in records:
        evidence_span = rec.get("evidence_span")
        label = rec.get("label", "")

        # --- (a) empty evidence ------------------------------------------------
        if evidence_span is None or evidence_span == "":
            empty_evidence_count += 1
            if str(label).upper() == "UNRELATED":
                empty_evidence_unrelated_count += 1
            continue

        # --- build join key ----------------------------------------------------
        key = _make_join_key(
            rec.get("stay_id", ""),
            rec.get("pattern_hour", ""),
            rec.get("pattern_name", ""),
            rec.get("note_id", ""),
        )

        # --- (b) join miss -----------------------------------------------------
        if key not in lookup:
            join_miss_count += 1
            continue

        note_text = lookup[key]

        # --- (c) / (d) substring check ----------------------------------------
        if evidence_span in note_text:
            substring_valid_count += 1
        else:
            substring_mismatch_count += 1
            if len(failure_samples) < MAX_FAILURE_SAMPLES:
                failure_samples.append({
                    "stay_id": rec.get("stay_id"),
                    "pattern_name": rec.get("pattern_name"),
                    "note_type": rec.get("note_type"),
                    "note_hour": rec.get("note_hour"),
                    "label": label,
                    "evidence_span": _trunc(evidence_span, EVIDENCE_TRUNC),
                    "note_text_relevant_snippet": _trunc(note_text, NOTE_SNIPPET_LEN),
                    "mismatch_reason": "evidence_span not found as substring of note_text_relevant",
                })

        # --- (e) index-based check (evidence_start / evidence_end) -------------
        ev_start = rec.get("evidence_start")
        ev_end = rec.get("evidence_end")
        if ev_start is not None and ev_end is not None:
            index_present_count += 1
            try:
                start_idx = int(ev_start)
                end_idx = int(ev_end)
            except (ValueError, TypeError):
                # Non-integer indices – treat as OOB / invalid
                index_oob_count += 1
                if len(failure_samples) < MAX_FAILURE_SAMPLES:
                    failure_samples.append({
                        "stay_id": rec.get("stay_id"),
                        "pattern_name": rec.get("pattern_name"),
                        "note_type": rec.get("note_type"),
                        "note_hour": rec.get("note_hour"),
                        "label": label,
                        "evidence_span": _trunc(evidence_span, EVIDENCE_TRUNC),
                        "note_text_relevant_snippet": _trunc(note_text, NOTE_SNIPPET_LEN),
                        "mismatch_reason": "evidence_start/end not valid integers",
                    })
                continue

            if start_idx < 0 or end_idx > len(note_text) or start_idx > end_idx:
                index_oob_count += 1
                if len(failure_samples) < MAX_FAILURE_SAMPLES:
                    failure_samples.append({
                        "stay_id": rec.get("stay_id"),
                        "pattern_name": rec.get("pattern_name"),
                        "note_type": rec.get("note_type"),
                        "note_hour": rec.get("note_hour"),
                        "label": label,
                        "evidence_span": _trunc(evidence_span, EVIDENCE_TRUNC),
                        "note_text_relevant_snippet": _trunc(note_text, NOTE_SNIPPET_LEN),
                        "mismatch_reason": (
                            f"index out of bounds: start={start_idx}, end={end_idx}, "
                            f"text_len={len(note_text)}"
                        ),
                    })
            elif note_text[start_idx:end_idx] == evidence_span:
                index_valid_count += 1
            else:
                # Indices in range but slice doesn't match evidence_span
                if len(failure_samples) < MAX_FAILURE_SAMPLES:
                    failure_samples.append({
                        "stay_id": rec.get("stay_id"),
                        "pattern_name": rec.get("pattern_name"),
                        "note_type": rec.get("note_type"),
                        "note_hour": rec.get("note_hour"),
                        "label": label,
                        "evidence_span": _trunc(evidence_span, EVIDENCE_TRUNC),
                        "note_text_relevant_snippet": _trunc(note_text, NOTE_SNIPPET_LEN),
                        "mismatch_reason": (
                            "index slice does not match evidence_span: "
                            f"note_text[{start_idx}:{end_idx}] != evidence_span"
                        ),
                    })

    # --- derived statistics ----------------------------------------------------
    records_with_text = total_records - empty_evidence_count - join_miss_count
    substring_valid_rate = (
        substring_valid_count / records_with_text if records_with_text > 0 else 0.0
    )
    empty_evidence_rate = (
        empty_evidence_count / total_records if total_records > 0 else 0.0
    )
    index_valid_rate = (
        index_valid_count / index_present_count if index_present_count > 0 else 0.0
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_glob": ANNOTATION_GLOB,
        "statistics": {
            "total_records": total_records,
            "empty_evidence_count": empty_evidence_count,
            "empty_evidence_rate": round(empty_evidence_rate, 6),
            "empty_evidence_unrelated_count": empty_evidence_unrelated_count,
            "join_miss_count": join_miss_count,
            "records_with_text": records_with_text,
            "substring_valid_count": substring_valid_count,
            "substring_valid_rate": round(substring_valid_rate, 6),
            "substring_mismatch_count": substring_mismatch_count,
            "index_present_count": index_present_count,
            "index_valid_count": index_valid_count,
            "index_valid_rate": round(index_valid_rate, 6),
            "index_oob_count": index_oob_count,
        },
        "failure_samples": failure_samples,
    }
    return report


# ---------------------------------------------------------------------------
# I/O and entry point
# ---------------------------------------------------------------------------

def print_summary(report: dict) -> None:
    """Pretty-print key statistics to stdout."""
    s = report["statistics"]
    print("=" * 64)
    print("  DeepSeek Evidence Validity Report")
    print("=" * 64)
    print(f"  Generated at:                {report['generated_at']}")
    print(f"  Total records:               {s['total_records']}")
    print(f"  Empty evidence:              {s['empty_evidence_count']}  "
          f"({s['empty_evidence_rate']:.2%})")
    print(f"    of which UNRELATED:        {s['empty_evidence_unrelated_count']}")
    print(f"  Join misses:                 {s['join_miss_count']}")
    print(f"  Records with text:           {s['records_with_text']}")
    print(f"  Substring valid:             {s['substring_valid_count']}  "
          f"({s['substring_valid_rate']:.2%})")
    print(f"  Substring mismatch:          {s['substring_mismatch_count']}")
    print("-" * 64)
    print(f"  Index fields present:        {s['index_present_count']}")
    print(f"  Index valid:                 {s['index_valid_count']}  "
          f"({s['index_valid_rate']:.2%})")
    print(f"  Index out-of-bounds:         {s['index_oob_count']}")
    print("=" * 64)
    if report["failure_samples"]:
        print(f"\n  First {len(report['failure_samples'])} failure sample(s):")
        for i, fs in enumerate(report["failure_samples"], 1):
            print(f"    [{i}] stay_id={fs['stay_id']}  pattern={fs['pattern_name']}  "
                  f"label={fs['label']}")
            print(f"        reason: {fs['mismatch_reason']}")
            print(f"        evidence: {fs['evidence_span']}")
    print()


def main() -> None:
    # 1. Load metadata (informational; not strictly needed for validation)
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)
        print(f"[INFO] Loaded metadata: {len(metadata.get('outputs', []))} output entries")
    else:
        print(f"[WARN] Metadata file not found: {METADATA_PATH}")

    # 2. Build note_text_relevant lookup from CSV
    if not CSV_PATH.exists():
        print(f"[ERROR] CSV not found: {CSV_PATH}")
        sys.exit(1)
    lookup = build_note_text_lookup(CSV_PATH)
    print(f"[INFO] CSV lookup built: {len(lookup)} unique keys")

    # 3. Load all DeepSeek JSONL annotation records
    records = load_jsonl_files(ANNOTATION_GLOB)
    print(f"[INFO] Loaded {len(records)} annotation records from JSONL files")

    if not records:
        print("[WARN] No annotation records found – writing empty report.")

    # 4. Validate
    report = validate(records, lookup)

    # 5. Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"[INFO] Report written to {OUTPUT_PATH}")

    # 6. Copy to final_release
    FINAL_RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUTPUT_PATH, FINAL_RELEASE_PATH)
    print(f"[INFO] Report copied to {FINAL_RELEASE_PATH}")

    # 7. Print summary
    print_summary(report)


if __name__ == "__main__":
    main()
