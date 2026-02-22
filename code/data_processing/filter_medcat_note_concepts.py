"""
Filter note-level MedCAT concepts by note_type.

Default canonical policy removes discharge notes at the data layer.
This script streams CSV rows to avoid loading the full file into memory.
"""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = BASE_DIR / "data" / "processed" / "medcat_full" / "medcat_note_concepts_24h.csv"
DEFAULT_OUTPUT = DEFAULT_INPUT
DEFAULT_REPORT = BASE_DIR / "results" / "qc" / "medcat_note_concepts_filter_report.json"


def _normalize_types(raw: str):
    values = [x.strip().lower() for x in raw.split(",")]
    return sorted(set([x for x in values if x]))


def filter_csv(input_path: Path, output_path: Path, exclude_types):
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    same_file = input_path == output_path
    if same_file:
        fd, tmp_path_str = tempfile.mkstemp(
            prefix="medcat_note_concepts_",
            suffix=".csv",
            dir=str(output_path.parent),
            text=True,
        )
        os.close(fd)
        write_path = Path(tmp_path_str)
    else:
        write_path = output_path

    total_rows = 0
    kept_rows = 0
    removed_rows = 0
    note_type_counts_before = Counter()
    note_type_counts_after = Counter()
    removed_by_type = Counter()

    with open(input_path, "r", newline="", encoding="utf-8") as fin, open(
        write_path, "w", newline="", encoding="utf-8"
    ) as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise ValueError("Input CSV missing header")
        if "note_type" not in reader.fieldnames:
            raise ValueError("Input CSV missing required column: note_type")

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            note_type = (row.get("note_type") or "").strip().lower()
            note_type_counts_before[note_type] += 1
            if note_type in exclude_types:
                removed_rows += 1
                removed_by_type[note_type] += 1
                continue
            writer.writerow(row)
            kept_rows += 1
            note_type_counts_after[note_type] += 1

    if same_file:
        write_path.replace(output_path)

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "total_rows": total_rows,
        "kept_rows": kept_rows,
        "removed_rows": removed_rows,
        "excluded_note_types": sorted(exclude_types),
        "removed_by_type": dict(sorted(removed_by_type.items())),
        "note_type_counts_before": dict(sorted(note_type_counts_before.items())),
        "note_type_counts_after": dict(sorted(note_type_counts_after.items())),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter medcat_note_concepts_24h.csv by note_type (canonical excludes discharge)."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument(
        "--exclude-note-types",
        type=str,
        default="discharge",
        help="Comma-separated note types to remove (default: discharge).",
    )
    parser.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    exclude_types = _normalize_types(args.exclude_note_types)
    if not exclude_types:
        raise ValueError("exclude-note-types is empty")

    summary = filter_csv(Path(args.input), Path(args.output), set(exclude_types))
    summary["timestamp"] = datetime.now().isoformat(timespec="seconds")

    report_path = Path(args.report).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print("MedCAT note concepts filtering complete")
    print(f"  Input : {summary['input_path']}")
    print(f"  Output: {summary['output_path']}")
    print(f"  Rows  : {summary['total_rows']:,} -> {summary['kept_rows']:,}")
    print(f"  Removed rows: {summary['removed_rows']:,}")
    print(f"  Removed by type: {summary['removed_by_type']}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
