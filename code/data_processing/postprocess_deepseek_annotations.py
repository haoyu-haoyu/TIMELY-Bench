#!/usr/bin/env python3
"""
postprocess_deepseek_annotations.py

Post-process DeepSeek LLM annotation JSONL files:
  1. Extract raw_response_excerpt (truncated content_preview, max 500 chars)
  2. Derive parse_error boolean from raw_response.parse_status
  3. Remove full raw_response field from output
  4. Write audited JSONL, compute SHA256, produce ANNOTATION_AUDIT_PATCH.json
  5. Copy outputs to final_release/llm_annotations/

Deployed to: code/data_processing/postprocess_deepseek_annotations.py
"""

import json
import re
import sys
import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup – this file lives in code/data_processing/ on the HPC server,
# so we add the parent (code/) to sys.path to import config.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR

# ---------------------------------------------------------------------------
# Directory constants
# ---------------------------------------------------------------------------
LLM_ANNOTATIONS_DIR = ROOT_DIR / "results" / "llm_annotations"
FINAL_RELEASE_DIR = ROOT_DIR / "final_release" / "llm_annotations"

# Pattern to match source files and extract the embedded timestamp
SOURCE_PATTERN = re.compile(r"^annotations_deepseek_(\d{8}_\d{6})(?:_part\d+)?\.jsonl$")


def compute_sha256(filepath: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def process_record(record: dict) -> dict:
    """
    Transform a single JSONL record:
      - Add raw_response_excerpt  (str | None)
      - Add parse_error           (bool | None)
      - Remove raw_response
    Returns the modified record (mutated in-place).
    """
    raw = record.get("raw_response")

    if raw is not None and isinstance(raw, dict):
        # --- raw_response_excerpt ---
        content_preview = raw.get("content_preview")
        if content_preview is not None:
            record["raw_response_excerpt"] = str(content_preview)[:500]
        else:
            record["raw_response_excerpt"] = None

        # --- parse_error ---
        parse_status = raw.get("parse_status")
        if parse_status == "json":
            record["parse_error"] = False
        elif parse_status is not None:
            record["parse_error"] = True
        else:
            record["parse_error"] = None
    else:
        # raw_response missing or not a dict
        record["raw_response_excerpt"] = None
        record["parse_error"] = None

    # Remove the full raw_response regardless
    record.pop("raw_response", None)

    return record


def process_file(source_path: Path, timestamp_str: str) -> dict | None:
    """
    Process a single source JSONL file.

    Returns a dict with metadata about the audited output, or None on failure.
    """
    # Build audited filename: preserve original stem (including _partNNNN if present)
    stem = source_path.stem  # e.g. annotations_deepseek_20260127_151413_part0001
    audited_filename = f"{stem}_audited.jsonl"
    audited_path = LLM_ANNOTATIONS_DIR / audited_filename

    records_out: list[dict] = []
    stats = {
        "has_raw_response": 0,
        "parse_error_false": 0,
        "parse_error_true": 0,
        "parse_error_null": 0,
        "excerpt_present": 0,
        "excerpt_null": 0,
    }

    with open(source_path, "r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"  [WARN] Skipping malformed JSON at {source_path.name}:{line_no} – {exc}"
                )
                continue

            # Track whether the original had a raw_response
            if "raw_response" in record and isinstance(
                record.get("raw_response"), dict
            ):
                stats["has_raw_response"] += 1

            record = process_record(record)

            # Collect stats on derived fields
            pe = record.get("parse_error")
            if pe is None:
                stats["parse_error_null"] += 1
            elif pe is False:
                stats["parse_error_false"] += 1
            else:
                stats["parse_error_true"] += 1

            if record.get("raw_response_excerpt") is not None:
                stats["excerpt_present"] += 1
            else:
                stats["excerpt_null"] += 1

            records_out.append(record)

    # Write audited JSONL
    with open(audited_path, "w", encoding="utf-8") as fout:
        for rec in records_out:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sha = compute_sha256(audited_path)
    row_count = len(records_out)

    print(f"  Wrote {row_count} records -> {audited_path.name}  (SHA256: {sha[:16]}...)")

    return {
        "source_path": str(source_path),
        "audited_path": str(audited_path),
        "sha256": sha,
        "rows": row_count,
        "stats": stats,
    }


def main() -> None:
    print("=" * 60)
    print("postprocess_deepseek_annotations.py")
    print("=" * 60)

    # Ensure directories exist
    LLM_ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    # Discover source files
    source_files: list[tuple[Path, str]] = []
    for p in sorted(LLM_ANNOTATIONS_DIR.glob("annotations_deepseek_*.jsonl")):
        m = SOURCE_PATTERN.match(p.name)
        if m:
            source_files.append((p, m.group(1)))

    if not source_files:
        print(
            f"[ERROR] No matching annotations_deepseek_*.jsonl files found in "
            f"{LLM_ANNOTATIONS_DIR}"
        )
        sys.exit(1)

    print(f"Found {len(source_files)} source file(s):\n")

    # Aggregate containers
    all_source_paths: list[str] = []
    all_audited_info: list[dict] = []
    total_stats = {
        "total_records": 0,
        "has_raw_response": 0,
        "parse_error_false": 0,
        "parse_error_true": 0,
        "parse_error_null": 0,
        "excerpt_present": 0,
        "excerpt_null": 0,
    }

    for source_path, ts in source_files:
        print(f"Processing: {source_path.name}")
        result = process_file(source_path, ts)
        if result is None:
            continue

        all_source_paths.append(result["source_path"])
        all_audited_info.append(
            {
                "path": result["audited_path"],
                "sha256": result["sha256"],
                "rows": result["rows"],
            }
        )

        total_stats["total_records"] += result["rows"]
        for key in (
            "has_raw_response",
            "parse_error_false",
            "parse_error_true",
            "parse_error_null",
            "excerpt_present",
            "excerpt_null",
        ):
            total_stats[key] += result["stats"][key]

    # Build ANNOTATION_AUDIT_PATCH.json
    audit_patch = {
        "audit_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_files": all_source_paths,
        "audited_files": all_audited_info,
        "fields_added": ["raw_response_excerpt", "parse_error"],
        "fields_removed": ["raw_response"],
        "notes": [
            "raw_response_excerpt is truncated to 500 chars from content_preview",
            "parse_error derived from raw_response.parse_status: json->false, other->true",
            "Full raw_response removed from audited output to reduce file size",
        ],
        "stats": total_stats,
    }

    audit_patch_path = LLM_ANNOTATIONS_DIR / "ANNOTATION_AUDIT_PATCH.json"
    with open(audit_patch_path, "w", encoding="utf-8") as f:
        json.dump(audit_patch, f, indent=2, ensure_ascii=False)
    print(f"\nWrote audit patch -> {audit_patch_path}")

    # Copy outputs to final_release/llm_annotations/
    print(f"\nCopying outputs to {FINAL_RELEASE_DIR} ...")
    for info in all_audited_info:
        src = Path(info["path"])
        dst = FINAL_RELEASE_DIR / src.name
        shutil.copy2(src, dst)
        print(f"  Copied {src.name}")

    shutil.copy2(audit_patch_path, FINAL_RELEASE_DIR / audit_patch_path.name)
    print(f"  Copied {audit_patch_path.name}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total records processed : {total_stats['total_records']}")
    print(f"  Had raw_response        : {total_stats['has_raw_response']}")
    print(f"  parse_error = false     : {total_stats['parse_error_false']}")
    print(f"  parse_error = true      : {total_stats['parse_error_true']}")
    print(f"  parse_error = null      : {total_stats['parse_error_null']}")
    print(f"  excerpt present         : {total_stats['excerpt_present']}")
    print(f"  excerpt null            : {total_stats['excerpt_null']}")
    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
