"""
Verify LLM annotation artefacts integrity and alignment constraints.

Supports both rule-based and DeepSeek LLM branches via --metadata-path.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR


OUT_DIR = ROOT_DIR / "results" / "llm_annotations"
FINAL_LLM_DIR = ROOT_DIR / "final_release" / "llm_annotations"
ALLOWED_LABELS = {"SUPPORTIVE", "CONTRADICTORY", "AMBIGUOUS", "UNRELATED"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_alignment_hash_from_base() -> Tuple[str, str]:
    base_meta = OUT_DIR / "ANNOTATION_METADATA.json"
    if base_meta.exists():
        meta = json.loads(base_meta.read_text())
        return meta.get("alignment_source", ""), meta.get("alignment_sha256", "")
    return "", ""


def iter_annotation_files(meta: Dict[str, object]) -> List[Path]:
    # DeepSeek branch uses outputs list; rule-based uses annotations_path
    outputs = meta.get("outputs")
    paths: List[Path] = []
    if isinstance(outputs, list) and outputs:
        for item in outputs:
            if isinstance(item, dict) and item.get("path"):
                paths.append(Path(str(item["path"])))
    else:
        ann_path = meta.get("annotations_path")
        if ann_path:
            paths.append(Path(str(ann_path)))
    resolved: List[Path] = []
    for p in paths:
        if not p.is_absolute():
            p = (ROOT_DIR / p).resolve()
        if p.exists():
            resolved.append(p)
    return resolved


def check_alignment_hash(meta: Dict[str, object], compute_hash: bool) -> None:
    align_info = meta.get("alignment") if isinstance(meta.get("alignment"), dict) else {}
    meta_align_path = align_info.get("path") or meta.get("alignment_source")
    meta_align_hash = align_info.get("sha256") or meta.get("alignment_sha256")

    base_align_path, base_align_hash = load_alignment_hash_from_base()
    # First, compare against base metadata to avoid re-hashing huge files
    if base_align_hash and meta_align_hash and base_align_hash != meta_align_hash:
        raise ValueError("alignment sha256 mismatch vs base metadata")

    if compute_hash:
        align_path = Path(meta_align_path or (TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"))
        if not align_path.is_absolute():
            align_path = (ROOT_DIR / align_path).resolve()
        if not align_path.exists():
            raise FileNotFoundError(f"Alignment file missing: {align_path}")
        align_hash = sha256_file(align_path)
        if meta_align_hash and meta_align_hash != align_hash:
            raise ValueError("alignment sha256 mismatch vs computed hash")


def verify_annotation_rows(
    ann_files: List[Path],
) -> Tuple[int, int, int, int, Dict[str, int]]:
    required_fields = {
        "stay_id",
        "pattern_name",
        "pattern_hour",
        "note_id",
        "note_hour",
        "note_type",
        "label",
        "evidence_span",
    }

    seen: Set[Tuple[object, object, object, object]] = set()
    total_rows = 0
    discharge_rows = 0
    out_of_range_rows = 0
    duplicate_rows = 0
    label_counts: Dict[str, int] = {}

    for ann_path in ann_files:
        with ann_path.open() as f:
            for i, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                total_rows += 1

                missing = required_fields - set(rec.keys())
                if missing:
                    raise ValueError(f"Annotation record missing fields at {ann_path}:{i}: {sorted(missing)}")

                label = str(rec.get("label", ""))
                if label not in ALLOWED_LABELS:
                    raise ValueError(f"Invalid label at {ann_path}:{i}: {label}")
                label_counts[label] = label_counts.get(label, 0) + 1

                note_type = str(rec.get("note_type", "")).lower()
                if note_type == "discharge":
                    discharge_rows += 1

                try:
                    note_hour = float(rec.get("note_hour"))
                    if note_hour < 0 or note_hour >= 24:
                        out_of_range_rows += 1
                except Exception:
                    out_of_range_rows += 1

                evidence_span = str(rec.get("evidence_span", ""))
                evidence_note = str(rec.get("evidence_note", ""))
                if evidence_span == "" and evidence_note == "":
                    raise ValueError(f"Empty evidence_span without reason at {ann_path}:{i}")

                key = (
                    rec.get("stay_id"),
                    rec.get("pattern_hour"),
                    rec.get("pattern_name"),
                    rec.get("note_id"),
                )
                if key in seen:
                    duplicate_rows += 1
                else:
                    seen.add(key)

    return total_rows, discharge_rows, out_of_range_rows, duplicate_rows, label_counts


def summarize_strata(sample_df: pd.DataFrame, label_counts: Dict[str, int]) -> Dict[str, object]:
    df = sample_df.copy()
    df["time_delta_bucket"] = pd.cut(
        df["time_delta_hours"],
        bins=[-1e9, -12, -6, 0],
        labels=["<=-12h", "-12h~-6h", "-6h~0h"],
    )
    summary = {
        "n_samples": int(len(df)),
        "note_type_counts": df["note_type"].value_counts().to_dict(),
        "severity_counts": df["pattern_severity"].value_counts().to_dict(),
        "time_delta_bucket_counts": df["time_delta_bucket"].value_counts().to_dict(),
        "label_counts": label_counts,
        "dedup_key": "stay_id+pattern_hour+pattern_name+note_id",
    }
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata-path",
        type=str,
        default=str(OUT_DIR / "ANNOTATION_METADATA.json"),
        help="Path to metadata JSON (rule-based or DeepSeek).",
    )
    parser.add_argument(
        "--summary-suffix",
        type=str,
        default="",
        help="Optional suffix for summary output (e.g., deepseek).",
    )
    parser.add_argument(
        "--compute-alignment-hash",
        action="store_true",
        help="Compute alignment sha256 (expensive on huge files).",
    )
    args = parser.parse_args()

    meta_path = Path(args.metadata_path)
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata not found: {meta_path}")

    sample_path = OUT_DIR / "llm_annotation_set.csv"
    if not sample_path.exists():
        raise FileNotFoundError("llm_annotation_set.csv not found; run build_llm_annotation_set.py first")

    meta = json.loads(meta_path.read_text())

    # Alignment hash consistency check (without rehashing by default)
    check_alignment_hash(meta, compute_hash=args.compute_alignment_hash)

    df = pd.read_csv(sample_path)

    note_type_ok = (df["note_type"].astype(str).str.lower() != "discharge").all()
    if not note_type_ok:
        raise ValueError("Found discharge notes in annotation set")

    hour_ok = ((df["note_hour"] >= 0) & (df["note_hour"] < 24)).all()
    if not hour_ok:
        raise ValueError("Found note_hour outside 0<=hour<24")

    key_cols = ["stay_id", "pattern_hour", "pattern_name", "note_id"]
    dup_mask = df.duplicated(subset=key_cols)
    if dup_mask.any():
        raise ValueError("Duplicate keys found in annotation set")

    ann_files = iter_annotation_files(meta)
    if not ann_files:
        raise FileNotFoundError("No annotation output files found from metadata")

    total_rows, discharge_rows, out_of_range_rows, duplicate_rows, label_counts = verify_annotation_rows(ann_files)

    if discharge_rows:
        raise ValueError(f"Found discharge notes in annotations: {discharge_rows}")
    if out_of_range_rows:
        raise ValueError(f"Found note_hour outside window in annotations: {out_of_range_rows}")
    if duplicate_rows:
        raise ValueError(f"Found duplicate keys in annotations: {duplicate_rows}")

    summary = summarize_strata(df, label_counts)
    summary.update(
        {
            "metadata_path": str(meta_path),
            "annotation_files": [str(p) for p in ann_files],
            "annotation_total_rows": total_rows,
        }
    )

    suffix = f"_{args.summary_suffix}" if args.summary_suffix else ""
    summary_path = OUT_DIR / f"summary_strata{suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))

    FINAL_LLM_DIR.mkdir(parents=True, exist_ok=True)
    (FINAL_LLM_DIR / summary_path.name).write_bytes(summary_path.read_bytes())

    print("PASS: LLM annotation set verified")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
