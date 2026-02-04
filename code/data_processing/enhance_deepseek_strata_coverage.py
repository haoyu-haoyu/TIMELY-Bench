#!/usr/bin/env python3
"""
enhance_deepseek_strata_coverage.py

Deployed to: code/data_processing/enhance_deepseek_strata_coverage.py

Part 1: Update summary_strata_deepseek.json with time_delta audit stats.
Part 2: Generate deepseek_coverage_summary.json with full coverage statistics.
"""

import sys
import json
import glob
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup – this script lives in code/data_processing/ on the HPC server.
# Insert the parent (code/) so we can import config.
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(ROOT_DIR)

CSV_PATH = ROOT / "results" / "llm_annotations" / "llm_annotation_set.csv"
SUMMARY_STRATA_PATH = ROOT / "results" / "llm_annotations" / "summary_strata_deepseek.json"
JSONL_GLOB = str(ROOT / "results" / "llm_annotations" / "annotations_deepseek_*.jsonl")

COVERAGE_OUT = ROOT / "results" / "llm_annotations" / "deepseek_coverage_summary.json"
FINAL_RELEASE_DIR = ROOT / "final_release" / "llm_annotations"


def _ensure_dir(path: Path) -> None:
    """Create parent directories if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ===================================================================
# Part 1 – Update summary_strata_deepseek.json with time_delta stats
# ===================================================================
def part1_update_summary_strata(df: pd.DataFrame) -> None:
    """Read the existing summary JSON, add time-delta audit fields, overwrite."""

    # Load existing summary
    with open(SUMMARY_STRATA_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    td = df["time_delta_hours"].dropna().astype(float)

    # Percentiles
    pct_keys = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_values = np.percentile(td.values, pct_keys).tolist()
    percentiles = {f"p{str(k).zfill(2)}": round(v, 4) for k, v in zip(pct_keys, pct_values)}

    # Bin edges and labels
    bin_edges = [-1e9, -12, -6, 0]
    bin_labels = ["<=-12h", "-12h~-6h", "-6h~0h"]

    # Write new fields
    summary["time_delta_min"] = round(float(td.min()), 4)
    summary["time_delta_max"] = round(float(td.max()), 4)
    summary["time_delta_percentiles"] = percentiles
    summary["time_delta_bins"] = bin_edges
    summary["time_delta_bins_labels"] = bin_labels
    summary["time_delta_empty_bucket_explanation"] = (
        "All sampled note-pattern pairs have time_delta_hours in [-12, 0]; "
        "the annotation window is capped at 12h before pattern onset, "
        "so the <=-12h bucket is structurally empty."
    )

    # Overwrite in place
    _ensure_dir(SUMMARY_STRATA_PATH)
    with open(SUMMARY_STRATA_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Part 1] Updated {SUMMARY_STRATA_PATH}")

    # Copy to final_release
    dest = FINAL_RELEASE_DIR / "summary_strata_deepseek.json"
    _ensure_dir(dest)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[Part 1] Copied to {dest}")


# ===================================================================
# Part 2 – Generate deepseek_coverage_summary.json
# ===================================================================
def _read_all_jsonl(pattern: str) -> list[dict]:
    """Read all JSONL files matching *pattern* and return a list of records.
    Excludes audited files to avoid double-counting."""
    records = []
    for fpath in sorted(glob.glob(pattern)):
        if "_audited" in fpath:
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def part2_generate_coverage_summary(df: pd.DataFrame) -> None:
    """Compute coverage statistics from the CSV + JSONL and write JSON."""

    # ------------------------------------------------------------------
    # Basic counts from CSV
    # ------------------------------------------------------------------
    unique_stay_ids = int(df["stay_id"].nunique())
    unique_patterns = int(df["pattern_name"].nunique())
    unique_note_ids = int(df["note_id"].nunique())
    total_samples = len(df)

    # Note type distribution
    note_type_dist = (
        df["note_type"]
        .value_counts()
        .to_dict()
    )
    note_type_dist = {k: int(v) for k, v in note_type_dist.items()}

    # Severity distribution
    severity_dist = (
        df["pattern_severity"]
        .value_counts()
        .to_dict()
    )
    severity_dist = {k: int(v) for k, v in severity_dist.items()}

    # ------------------------------------------------------------------
    # Pattern sample counts
    # ------------------------------------------------------------------
    pattern_counts = df["pattern_name"].value_counts()
    top_10 = [
        {"pattern": str(pat), "count": int(cnt)}
        for pat, cnt in pattern_counts.head(10).items()
    ]
    min_per = int(pattern_counts.min())
    median_per = int(np.median(pattern_counts.values))
    max_per = int(pattern_counts.max())
    total_unique = int(len(pattern_counts))

    # Long-tail note: patterns with fewer than 5 samples (reasonable threshold)
    threshold = 5
    long_tail_count = int((pattern_counts < threshold).sum())
    long_tail_note = f"{long_tail_count} patterns have fewer than {threshold} samples"

    pattern_sample_counts = {
        "top_10": top_10,
        "min_per_pattern": min_per,
        "median_per_pattern": median_per,
        "max_per_pattern": max_per,
        "total_unique_patterns": total_unique,
        "long_tail_note": long_tail_note,
    }

    # ------------------------------------------------------------------
    # Label distribution from JSONL
    # ------------------------------------------------------------------
    jsonl_records = _read_all_jsonl(JSONL_GLOB)
    label_counts: dict[str, int] = {}
    evidence_lengths: list[int] = []

    for rec in jsonl_records:
        lbl = rec.get("label", "UNKNOWN")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Evidence span length (if present)
        evidence = rec.get("evidence_span") or rec.get("evidence", "")
        if isinstance(evidence, str) and evidence:
            evidence_lengths.append(len(evidence))

    # ------------------------------------------------------------------
    # Evidence span length stats
    # ------------------------------------------------------------------
    if evidence_lengths:
        ev_arr = np.array(evidence_lengths)
        evidence_span_length_stats = {
            "min": int(ev_arr.min()),
            "max": int(ev_arr.max()),
            "mean": round(float(ev_arr.mean()), 2),
            "median": int(np.median(ev_arr)),
        }
    else:
        evidence_span_length_stats = {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0,
        }

    # ------------------------------------------------------------------
    # Note text length stats (from CSV column note_text_relevant)
    # ------------------------------------------------------------------
    text_lengths = df["note_text_relevant"].dropna().astype(str).str.len()
    if len(text_lengths) > 0:
        tl_arr = text_lengths.values.astype(int)
        note_text_length_stats = {
            "min": int(tl_arr.min()),
            "max": int(tl_arr.max()),
            "mean": round(float(tl_arr.mean()), 2),
            "median": int(np.median(tl_arr)),
        }
    else:
        note_text_length_stats = {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0,
        }

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    coverage = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "unique_stay_ids": unique_stay_ids,
        "unique_patterns": unique_patterns,
        "unique_note_ids": unique_note_ids,
        "total_samples": total_samples,
        "note_type_distribution": note_type_dist,
        "severity_distribution": severity_dist,
        "pattern_sample_counts": pattern_sample_counts,
        "label_distribution": label_counts,
        "evidence_span_length_stats": evidence_span_length_stats,
        "note_text_length_stats": note_text_length_stats,
    }

    # Write to results/
    _ensure_dir(COVERAGE_OUT)
    with open(COVERAGE_OUT, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"[Part 2] Wrote {COVERAGE_OUT}")

    # Copy to final_release/
    dest = FINAL_RELEASE_DIR / "deepseek_coverage_summary.json"
    _ensure_dir(dest)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(coverage, f, indent=2, ensure_ascii=False)
    print(f"[Part 2] Copied to {dest}")


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    print("=" * 60)
    print("enhance_deepseek_strata_coverage.py")
    print("=" * 60)

    # Read the main CSV once and share across both parts
    print(f"Reading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    print(f"  Loaded {len(df)} rows, {df.shape[1]} columns.")

    # Part 1
    print("-" * 60)
    part1_update_summary_strata(df)

    # Part 2
    print("-" * 60)
    part2_generate_coverage_summary(df)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
