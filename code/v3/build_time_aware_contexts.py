#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable

import pandas as pd
try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_processing"))

from v3.constants import (  # type: ignore
    DEFAULT_CONTEXT_CLEAN_JSONL,
    DEFAULT_CONTEXT_JSONL,
    DEFAULT_CONTEXT_SUMMARY_JSON,
    DEFAULT_DIAGNOSIS_PATHWAY_EVENTS,
    DEFAULT_HOURLY_STATE_GRID,
    DEFAULT_MEDICATION_EVENTS,
    DEFAULT_PROCEDURE_EVENTS,
    RAW_NOTE_FILES,
    ROOT_DIR,
    V3_MAX_HOURS,
    ensure_v3_directories,
)
from v3.io_utils import chunk_dir_path, iter_table_chunks, portable_path, relativize_value, table_exists  # type: ignore

try:
    from doctime_rel_classifier import classify_note
except Exception:  # pragma: no cover
    classify_note = None


NOTE_SCHEMA = {
    "discharge": {"text_col": "discharge_text", "time_col": "hour_offset", "id_col": "note_id"},
    "nursing": {"text_col": "chart_text", "time_col": "hour_offset", "id_col": None},
    "lab_comment": {"text_col": "lab_comment", "time_col": "hour_offset", "id_col": None},
    "radiology": {"text_col": "radiology_text", "time_col": "hour_offset", "id_col": "note_id"},
}

STATIC_CONTEXT_FIELDS = [
    "subject_id",
    "hadm_id",
    "anchor_age",
    "gender",
    "label_mortality",
    "los_hours",
    "los_days",
    "prolonged_los_3d",
    "prolonged_los_5d",
    "prolonged_los_7d",
    "ckd",
    "readmission_30d",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 time-aware patient contexts.")
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--pathway-events", default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS))
    p.add_argument("--medication-events", default=str(DEFAULT_MEDICATION_EVENTS))
    p.add_argument("--procedure-events", default=str(DEFAULT_PROCEDURE_EVENTS))
    p.add_argument("--out-jsonl", default=str(DEFAULT_CONTEXT_JSONL))
    p.add_argument("--clean-jsonl", default=str(DEFAULT_CONTEXT_CLEAN_JSONL))
    p.add_argument("--summary-json", default=str(DEFAULT_CONTEXT_SUMMARY_JSON))
    p.add_argument("--hours", type=int, default=V3_MAX_HOURS)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--stay-batch-size", type=int, default=250)
    p.add_argument("--include-empty-hours", action="store_true")
    p.add_argument("--note-char-limit", type=int, default=1600)
    p.add_argument("--note-read-chunksize", type=int, default=20_000)
    p.add_argument("--chunk-start", type=int, default=None)
    p.add_argument("--chunk-end", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-merged-jsonl", action="store_true")
    p.add_argument("--skip-summary", action="store_true")
    p.add_argument("--disable-clean-classifier", action="store_true")
    return p.parse_args()


def _load_event_table(path: Path, allowed_stays: set[int], sort_cols: list[str]) -> pd.DataFrame:
    if not table_exists(path):
        return pd.DataFrame(columns=["stay_id"])
    parts = []
    for chunk in iter_table_chunks(path):
        if "stay_id" not in chunk.columns:
            continue
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype("int64")
        chunk = chunk[chunk["stay_id"].isin(allowed_stays)].copy()
        if not chunk.empty:
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["stay_id"])
    df = pd.concat(parts, ignore_index=True)
    available = [col for col in sort_cols if col in df.columns]
    if available:
        df = df.sort_values(available, kind="mergesort").reset_index(drop=True)
    return df


def _iter_stay_batches(stay_ids: list[int], batch_size: int) -> Iterable[list[int]]:
    size = max(1, int(batch_size))
    for start in range(0, len(stay_ids), size):
        yield stay_ids[start : start + size]


def _clean_text(text: object) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    return " ".join(str(text).split())


def _note_variants(text: str, note_hour: int, note_type: str, note_char_limit: int) -> dict:
    original = _clean_text(text)
    original = original[:note_char_limit]
    result = {
        "text_original": original,
        "text_clean": original,
        "n_sentences": 0,
        "n_after_sentences": 0,
        "after_sentence_ratio": 0.0,
        "cleaning_policy": "no_sentence_classifier_available",
    }
    if not original or classify_note is None:
        return result

    classified = classify_note(original, note_hour=float(note_hour), note_type=note_type)
    classes = classified.get("classifications") or []
    if not classes:
        result["cleaning_policy"] = "heuristic_sentence_after_removal"
        return result

    kept = [str(item.get("sentence", "")).strip() for item in classes if str(item.get("docTimeRel")) != "AFTER"]
    n_sentences = len(classes)
    n_after = sum(1 for item in classes if str(item.get("docTimeRel")) == "AFTER")
    clean_text = " ".join(s for s in kept if s).strip()[:note_char_limit]
    result.update(
        {
            "text_clean": clean_text,
            "n_sentences": n_sentences,
            "n_after_sentences": n_after,
            "after_sentence_ratio": float(n_after / n_sentences) if n_sentences else 0.0,
            "cleaning_policy": "heuristic_sentence_after_removal",
        }
    )
    return result


def _load_notes_for_stays(
    allowed_stays: set[int],
    hours: int,
    note_char_limit: int,
    note_read_chunksize: int,
) -> dict[int, list[dict]]:
    notes_by_stay: dict[int, list[dict]] = defaultdict(list)
    allowed_stays_list = sorted(int(v) for v in allowed_stays)
    for note_type, file_path in RAW_NOTE_FILES.items():
        if not table_exists(file_path):
            continue
        schema = NOTE_SCHEMA[note_type]
        wanted_cols = {"stay_id", schema["text_col"], schema["time_col"], "charttime"}
        if schema["id_col"]:
            wanted_cols.add(schema["id_col"])
        path = Path(file_path).expanduser().resolve(strict=False)
        parts_dir = chunk_dir_path(path)
        if parts_dir.exists():
            part_files = sorted(
                p for p in parts_dir.iterdir() if p.is_file() and p.suffix in {".parquet", ".csv"}
            )
        elif path.exists():
            part_files = [path]
        elif path.with_suffix(".csv").exists():
            part_files = [path.with_suffix(".csv")]
        else:
            part_files = []

        for part in part_files:
            if part.suffix == ".csv":
                chunk_iter = pd.read_csv(
                    part,
                    usecols=lambda c: c in wanted_cols,
                    low_memory=False,
                    chunksize=max(int(note_read_chunksize), 1000),
                )
            elif pq is not None:
                parquet_file = pq.ParquetFile(part)
                chunk_iter = (
                    batch.to_pandas()
                    for batch in parquet_file.iter_batches(
                        batch_size=max(int(note_read_chunksize), 1000),
                        columns=sorted(wanted_cols),
                        use_threads=False,
                    )
                )
            else:
                try:
                    chunk_iter = [
                        pd.read_parquet(
                            part,
                            columns=sorted(wanted_cols),
                            filters=[("stay_id", "in", allowed_stays_list)],
                        )
                    ]
                except Exception:
                    chunk_iter = [pd.read_parquet(part, columns=sorted(wanted_cols))]

            for chunk in chunk_iter:
                if chunk.empty:
                    continue
                if "stay_id" not in chunk.columns:
                    continue
                available_cols = [col for col in chunk.columns if col in wanted_cols]
                chunk = chunk[available_cols].copy()
                chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
                chunk = chunk.dropna(subset=["stay_id"]).copy()
                chunk["stay_id"] = chunk["stay_id"].astype("int64")
                chunk = chunk[chunk["stay_id"].isin(allowed_stays)].copy()
                if chunk.empty:
                    continue
                time_col = schema["time_col"]
                text_col = schema["text_col"]
                chunk[time_col] = pd.to_numeric(chunk[time_col], errors="coerce")
                chunk = chunk[chunk[time_col].between(0, hours - 1, inclusive="both")].copy()
                if chunk.empty:
                    continue
                chunk = chunk.reset_index(drop=True)
                id_col = schema["id_col"]
                note_id_values = chunk[id_col].astype(str).tolist() if id_col and id_col in chunk.columns else None
                charttime_values = chunk["charttime"].astype(str).tolist() if "charttime" in chunk.columns else None
                stay_values = chunk["stay_id"].astype("int64").tolist()
                hour_values = pd.to_numeric(chunk[time_col], errors="coerce").fillna(-1).astype(int).tolist()
                text_values = chunk[text_col].tolist()

                for idx, (stay_id, note_hour, text) in enumerate(zip(stay_values, hour_values, text_values)):
                    variants = _note_variants(
                        text=text,
                        note_hour=int(note_hour),
                        note_type=note_type,
                        note_char_limit=note_char_limit,
                    )
                    note_id = note_id_values[idx] if note_id_values is not None else f"{note_type}_{idx}"
                    charttime = charttime_values[idx] if charttime_values is not None else ""
                    notes_by_stay[int(stay_id)].append(
                        {
                            "stay_id": int(stay_id),
                            "hour": int(note_hour),
                            "note_type": note_type,
                            "note_id": note_id,
                            "charttime": charttime,
                            "text_original": variants["text_original"],
                            "text_clean": variants["text_clean"],
                            "n_sentences": int(variants["n_sentences"]),
                            "n_after_sentences": int(variants["n_after_sentences"]),
                            "after_sentence_ratio": float(variants["after_sentence_ratio"]),
                            "cleaning_policy": str(variants["cleaning_policy"]),
                        }
                    )
                del chunk
                gc.collect()

    for stay_id, items in notes_by_stay.items():
        items.sort(key=lambda row: (int(row["hour"]), str(row["note_type"]), str(row["note_id"])))
    return dict(notes_by_stay)


def _structured_hour_rows(df: pd.DataFrame, include_empty_hours: bool) -> list[dict]:
    reserved = {
        "stay_id",
        "subject_id",
        "hadm_id",
        "hour",
        "intime",
        "outtime",
        "deathtime",
        "anchor_age",
        "gender",
        "label_mortality",
        "los_hours",
        "los_days",
        "prolonged_los_3d",
        "prolonged_los_5d",
        "prolonged_los_7d",
        "ckd",
        "readmission_30d",
    }
    rows = []
    value_cols = [col for col in df.columns if col not in reserved]
    for _, row in df.iterrows():
        values = {}
        for col in value_cols:
            value = row[col]
            if pd.notna(value):
                values[col] = _json_safe_value(value)
        if include_empty_hours or values:
            rows.append({"hour": int(row["hour"]), "values": values})
    return rows


def _json_safe_value(value):
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if hasattr(value, "item"):
        return _json_safe_value(value.item())
    return value


def _records_from_df(df: pd.DataFrame) -> list[dict]:
    records = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for key, value in row.items():
            if pd.isna(value):
                continue
            clean[key] = _json_safe_value(value)
        records.append(clean)
    return records


def _note_records(items: list[dict], clean_variant: bool) -> list[dict]:
    records = []
    for row in items:
        text_key = "text_clean" if clean_variant else "text_original"
        text = str(row.get(text_key, "") or "").strip()
        if clean_variant and not text:
            continue
        records.append(
            {
                "stay_id": int(row["stay_id"]),
                "hour": int(row["hour"]),
                "note_type": row["note_type"],
                "note_id": row["note_id"],
                "charttime": row.get("charttime", ""),
                "text": text,
                "text_variant": "clean" if clean_variant else "original",
                "n_sentences": int(row.get("n_sentences", 0) or 0),
                "n_after_sentences": int(row.get("n_after_sentences", 0) or 0),
                "after_sentence_ratio": float(row.get("after_sentence_ratio", 0.0) or 0.0),
                "cleaning_policy": row.get("cleaning_policy", ""),
            }
        )
    return records


def _group_by_stay(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    if df.empty or "stay_id" not in df.columns:
        return {}
    grouped: dict[int, pd.DataFrame] = {}
    for stay_id, subdf in df.groupby("stay_id", sort=False):
        grouped[int(stay_id)] = subdf.copy()
    return grouped


def _concatenate_jsonl_parts(parts_dir: Path, out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for part in sorted(parts_dir.glob("part_*.jsonl")):
            with part.open("r", encoding="utf-8") as in_f:
                for line in in_f:
                    out_f.write(line)
                    total += 1
    tmp_path.replace(out_path)
    return total


def _count_jsonl_parts(parts_dir: Path) -> tuple[int, int]:
    n_lines = 0
    unique_stays = set()
    for part in sorted(parts_dir.glob("part_*.jsonl")):
        with part.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                n_lines += 1
                unique_stays.add(int(obj["stay_id"]))
    return n_lines, len(unique_stays)


def _iter_stay_batches(stay_ids: list[int], batch_size: int) -> Iterable[list[int]]:
    batch_size = max(int(batch_size), 1)
    for i in range(0, len(stay_ids), batch_size):
        yield stay_ids[i : i + batch_size]


def main() -> None:
    args = parse_args()
    global classify_note
    if args.disable_clean_classifier or os.environ.get("TIMELY_DISABLE_CLEAN_CLASSIFIER") == "1":
        classify_note = None
    clean_policy_name = (
        "fallback_original_text_no_sentence_classifier"
        if classify_note is None
        else "heuristic_sentence_after_removal"
    )
    ensure_v3_directories()

    original_out = Path(args.out_jsonl)
    clean_out = Path(args.clean_jsonl)
    summary_path = Path(args.summary_json)
    for path in [original_out, clean_out, summary_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    original_parts_dir = chunk_dir_path(original_out)
    clean_parts_dir = chunk_dir_path(clean_out)
    if original_parts_dir.exists() and not args.resume:
        shutil.rmtree(original_parts_dir)
    if clean_parts_dir.exists() and not args.resume:
        shutil.rmtree(clean_parts_dir)
    original_parts_dir.mkdir(parents=True, exist_ok=True)
    clean_parts_dir.mkdir(parents=True, exist_ok=True)

    if original_out.exists() and not args.resume:
        original_out.unlink()
    if clean_out.exists() and not args.resume:
        clean_out.unlink()

    processed_stays = 0
    for chunk_idx, grid in enumerate(iter_table_chunks(args.hourly_grid), start=1):
        if args.chunk_start is not None and chunk_idx < int(args.chunk_start):
            continue
        if args.chunk_end is not None and chunk_idx > int(args.chunk_end):
            continue

        original_part = original_parts_dir / f"part_{chunk_idx:05d}.jsonl"
        clean_part = clean_parts_dir / f"part_{chunk_idx:05d}.jsonl"
        if args.resume and original_part.exists() and clean_part.exists():
            if original_part.stat().st_size > 0 and clean_part.stat().st_size > 0:
                continue
            original_part.unlink(missing_ok=True)
            clean_part.unlink(missing_ok=True)

        stay_ids = [int(v) for v in grid["stay_id"].drop_duplicates().tolist()]
        if args.stay_limit is not None:
            remaining = int(args.stay_limit) - processed_stays
            if remaining <= 0:
                break
            stay_ids = stay_ids[:remaining]
            grid = grid[grid["stay_id"].isin(stay_ids)].copy()
        if not stay_ids:
            continue

        with original_part.open("w", encoding="utf-8") as original_f, clean_part.open("w", encoding="utf-8") as clean_f:
            stay_batch_size = max(int(args.stay_batch_size), 1)
            n_batches = max(1, math.ceil(len(stay_ids) / stay_batch_size))
            for batch_idx, stay_batch in enumerate(_iter_stay_batches(stay_ids, stay_batch_size), start=1):
                batch_allowed_stays = set(stay_batch)
                batch_grid = grid[grid["stay_id"].isin(stay_batch)].copy()
                pathway_events = _load_event_table(
                    Path(args.pathway_events),
                    batch_allowed_stays,
                    ["stay_id", "event_time_hour", "event_name"],
                )
                medication_events = _load_event_table(
                    Path(args.medication_events),
                    batch_allowed_stays,
                    ["stay_id", "event_start_hour", "event_name"],
                )
                procedure_events = _load_event_table(
                    Path(args.procedure_events),
                    batch_allowed_stays,
                    ["stay_id", "event_start_hour", "event_name"],
                )
                notes_by_stay = _load_notes_for_stays(
                    batch_allowed_stays,
                    int(args.hours),
                    int(args.note_char_limit),
                    int(args.note_read_chunksize),
                )

                pathway_by_stay = _group_by_stay(pathway_events)
                meds_by_stay = _group_by_stay(medication_events)
                procs_by_stay = _group_by_stay(procedure_events)

                for sid, stay_grid in batch_grid.groupby("stay_id", sort=False):
                    sid = int(sid)
                    stay_grid = stay_grid.sort_values("hour", kind="mergesort")
                    if stay_grid.empty:
                        continue
                    stay_pathway = pathway_by_stay.get(sid, pd.DataFrame())
                    stay_meds = meds_by_stay.get(sid, pd.DataFrame())
                    stay_procs = procs_by_stay.get(sid, pd.DataFrame())
                    stay_notes = notes_by_stay.get(sid, [])
                    first = stay_grid.iloc[0]
                    static_context = {}
                    for field in STATIC_CONTEXT_FIELDS:
                        if field in stay_grid.columns and pd.notna(first[field]):
                            value = first[field]
                            static_context[field] = _json_safe_value(value)

                    common_record = {
                        "stay_id": sid,
                        "static_context": static_context,
                        "structured_timeline": _structured_hour_rows(stay_grid, include_empty_hours=bool(args.include_empty_hours)),
                        "medication_timeline": _records_from_df(stay_meds),
                        "procedure_timeline": _records_from_df(stay_procs),
                        "diagnosis_pathway_events": _records_from_df(stay_pathway),
                    }
                    original_record = dict(common_record)
                    original_record["notes"] = _note_records(stay_notes, clean_variant=False)
                    original_record["context_variant"] = "original"

                    clean_record = dict(common_record)
                    clean_record["notes"] = _note_records(stay_notes, clean_variant=True)
                    clean_record["context_variant"] = "clean"
                    clean_record["clean_policy"] = clean_policy_name

                    original_f.write(json.dumps(original_record, ensure_ascii=False) + "\n")
                    clean_f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

                print(
                    f"Chunk {chunk_idx}: processed stay batch {batch_idx}/{n_batches} "
                    f"(n_stays={len(stay_batch)})",
                    flush=True,
                )
                del batch_grid, pathway_events, medication_events, procedure_events, notes_by_stay
                del pathway_by_stay, meds_by_stay, procs_by_stay
                gc.collect()

        processed_stays += len(stay_ids)
        print(f"Wrote {original_part}", flush=True)
        print(f"Wrote {clean_part}", flush=True)
        del grid
        gc.collect()

    if args.skip_summary:
        return

    summary = {
        "hours": int(args.hours),
        "original_parts_dir": portable_path(original_parts_dir, root=ROOT_DIR),
        "clean_parts_dir": portable_path(clean_parts_dir, root=ROOT_DIR),
        "n_parts_original": len(list(original_parts_dir.glob('part_*.jsonl'))),
        "n_parts_clean": len(list(clean_parts_dir.glob('part_*.jsonl'))),
        "clean_policy": clean_policy_name,
        "top_level_keys": [
            "stay_id",
            "static_context",
            "structured_timeline",
            "medication_timeline",
            "procedure_timeline",
            "diagnosis_pathway_events",
            "notes",
            "context_variant",
        ],
    }

    if not args.skip_merged_jsonl:
        summary["n_records_original"] = _concatenate_jsonl_parts(original_parts_dir, original_out)
        summary["n_records_clean"] = _concatenate_jsonl_parts(clean_parts_dir, clean_out)
        summary["original_jsonl"] = portable_path(original_out, root=ROOT_DIR)
        summary["clean_jsonl"] = portable_path(clean_out, root=ROOT_DIR)
    else:
        n_records_original, _ = _count_jsonl_parts(original_parts_dir)
        n_records_clean, _ = _count_jsonl_parts(clean_parts_dir)
        summary["n_records_original"] = n_records_original
        summary["n_records_clean"] = n_records_clean

    _, unique_original = _count_jsonl_parts(original_parts_dir)
    _, unique_clean = _count_jsonl_parts(clean_parts_dir)
    summary["unique_stays_original"] = unique_original
    summary["unique_stays_clean"] = unique_clean
    summary_path.write_text(json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
