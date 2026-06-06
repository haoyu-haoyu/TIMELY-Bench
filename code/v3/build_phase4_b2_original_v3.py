#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import DEFAULT_CONTEXT_JSONL, ROOT_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR  # type: ignore
from v3.io_utils import read_table, relativize_value, table_exists  # type: ignore


MAX_CONTEXT_HOUR = 167


CONDITION_CONFIGS = {
    "aki": {
        "mode": "temporal",
        "task_files": [
            V3_PROCESSED_DIR / "aki" / "tasks" / "aki_stage2plus_instances.parquet",
            V3_PROCESSED_DIR / "aki" / "tasks" / "aki_rrt_proxy_instances.parquet",
        ],
        "anchor_col": "anchor_hour",
        "representations_dir": V3_PROCESSED_DIR / "aki" / "representations",
        "summary_json": V3_RESULTS_DIR / "aki" / "aki_B2_original_build_summary.json",
        "exclude_discharge": False,
    },
    "delirium": {
        "mode": "temporal",
        "task_files": [
            V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_persistence_instances.parquet",
            V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_resolution_instances.parquet",
        ],
        "anchor_col": "prediction_hour",
        "representations_dir": V3_PROCESSED_DIR / "delirium" / "representations",
        "summary_json": V3_RESULTS_DIR / "delirium" / "delirium_B2_original_build_summary.json",
        "exclude_discharge": False,
    },
    "sepsis": {
        "mode": "temporal",
        "task_files": [
            V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_shock_instances.parquet",
            V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_lactate_clearance_instances.parquet",
        ],
        "anchor_col": "prediction_hour",
        "representations_dir": V3_PROCESSED_DIR / "sepsis" / "representations",
        "summary_json": V3_RESULTS_DIR / "sepsis" / "sepsis_B2_original_build_summary.json",
        "exclude_discharge": False,
    },
    "stroke": {
        "mode": "stroke",
        "temporal_task_files": [
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T1_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T2_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T3_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-T4_instances.parquet",
        ],
        "retrospective_task_files": [
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R1_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R2_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R3_instances.parquet",
            V3_PROCESSED_DIR / "stroke" / "tasks" / "stroke_S-R4_instances.parquet",
        ],
        "anchor_col": "anchor_hour",
        "representations_dir": V3_PROCESSED_DIR / "stroke" / "representations",
        "summary_json": V3_RESULTS_DIR / "stroke" / "stroke_B2_original_build_summary.json",
        "exclude_discharge": True,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 Phase 4C B2 original representations.")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["aki", "delirium", "sepsis", "stroke"],
        choices=sorted(CONDITION_CONFIGS.keys()),
    )
    p.add_argument("--contexts-jsonl", default=str(DEFAULT_CONTEXT_JSONL))
    p.add_argument(
        "--combined-summary-json",
        default=str(V3_RESULTS_DIR / "representations" / "phase4c_B2_original_build_summary.json"),
    )
    p.add_argument("--chunk-size", type=int, default=50000)
    return p.parse_args()


def _parts_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}.parts")


class ParquetPartWriter:
    def __init__(self, output_path: Path, chunk_size: int = 50000) -> None:
        self.output_path = output_path
        self.parts_dir = _parts_dir(output_path)
        if self.parts_dir.exists():
            shutil.rmtree(self.parts_dir)
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = int(chunk_size)
        self.buffer: list[dict[str, Any]] = []
        self.part_idx = 0
        self.total_rows = 0

    def write_row(self, row: dict[str, Any]) -> None:
        self.buffer.append(row)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        self.part_idx += 1
        part_path = self.parts_dir / f"part_{self.part_idx:05d}.parquet"
        pd.DataFrame(self.buffer).to_parquet(part_path, index=False)
        self.total_rows += len(self.buffer)
        self.buffer = []

    def close(self) -> dict[str, Any]:
        self.flush()
        return {
            "parts_dir": str(self.parts_dir),
            "n_parts": int(self.part_idx),
            "rows": int(self.total_rows),
        }


def _normalize_unique_str(values: pd.Series) -> str | None:
    vals = [str(v) for v in pd.unique(values.dropna()) if str(v) != ""]
    if not vals:
        return None
    return "|".join(sorted(vals))


def _aggregate_unique_join(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    output_col: str,
) -> pd.DataFrame:
    if value_col not in df.columns:
        out = df[group_cols].drop_duplicates().copy()
        out[output_col] = None
        return out
    base = df[group_cols + [value_col]].dropna().copy()
    if base.empty:
        out = df[group_cols].drop_duplicates().copy()
        out[output_col] = None
        return out
    base[value_col] = base[value_col].astype(str)
    base = base.loc[base[value_col] != ""].drop_duplicates()
    if base.empty:
        out = df[group_cols].drop_duplicates().copy()
        out[output_col] = None
        return out
    out = (
        base.sort_values(group_cols + [value_col], kind="mergesort")
        .groupby(group_cols, sort=False)[value_col]
        .agg("|".join)
        .reset_index()
        .rename(columns={value_col: output_col})
    )
    return out


def _load_temporal_anchor_index(task_files: list[Path], anchor_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    task_row_counts: dict[str, int] = {}
    for path in task_files:
        if not table_exists(path):
            raise FileNotFoundError(path)
        df = read_table(path)
        if "eligible" in df.columns:
            df = df.loc[pd.to_numeric(df["eligible"], errors="coerce").fillna(0).astype(int) == 1].copy()
        if anchor_col not in df.columns:
            raise KeyError(f"{path} missing anchor column {anchor_col}")
        keep = ["stay_id", anchor_col]
        for col in ["task_id", "hadm_id", "tier", "layer", "task_mode"]:
            if col in df.columns:
                keep.append(col)
        frame = df[keep].copy()
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
        frame["anchor_hour_requested"] = pd.to_numeric(frame[anchor_col], errors="coerce")
        frame = frame.dropna(subset=["stay_id", "anchor_hour_requested"]).copy()
        frame["stay_id"] = frame["stay_id"].astype("int64")
        frame["anchor_hour_requested"] = frame["anchor_hour_requested"].astype("int64")
        frame["anchor_hour_clipped"] = frame["anchor_hour_requested"].clip(upper=MAX_CONTEXT_HOUR)
        frame["anchor_was_clipped"] = (frame["anchor_hour_requested"] > MAX_CONTEXT_HOUR).astype(int)
        if "task_id" not in frame.columns:
            frame["task_id"] = path.stem
        task_row_counts[path.stem] = int(len(frame))
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    group_cols = ["stay_id", "anchor_hour_requested", "anchor_hour_clipped"]
    base = merged[group_cols + ["anchor_was_clipped"]].copy()
    agg = (
        base.groupby(group_cols, sort=False)["anchor_was_clipped"]
        .max()
        .reset_index()
    )
    task_base = merged[group_cols + ["task_id"]].drop_duplicates()
    task_ids = (
        task_base.sort_values(group_cols + ["task_id"], kind="mergesort")
        .groupby(group_cols, sort=False)["task_id"]
        .agg("|".join)
        .reset_index()
        .rename(columns={"task_id": "source_task_ids"})
    )
    task_counts = (
        task_base.groupby(group_cols, sort=False)
        .size()
        .reset_index(name="n_source_tasks")
    )
    agg = agg.merge(task_ids, on=group_cols, how="left").merge(task_counts, on=group_cols, how="left")
    for value_col, output_col in [
        ("hadm_id", "hadm_ids"),
        ("tier", "tier_values"),
        ("layer", "layer_values"),
        ("task_mode", "task_mode_values"),
    ]:
        agg = agg.merge(_aggregate_unique_join(merged, group_cols, value_col, output_col), on=group_cols, how="left")
    return agg.sort_values(group_cols, kind="mergesort").reset_index(drop=True), task_row_counts


def _load_retrospective_index(task_files: list[Path]) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    task_row_counts: dict[str, int] = {}
    for path in task_files:
        if not table_exists(path):
            raise FileNotFoundError(path)
        df = read_table(path)
        keep = ["stay_id"]
        for col in ["task_id", "hadm_id", "tier", "layer", "task_mode"]:
            if col in df.columns:
                keep.append(col)
        frame = df[keep].copy()
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["stay_id"]).copy()
        frame["stay_id"] = frame["stay_id"].astype("int64")
        if "task_id" not in frame.columns:
            frame["task_id"] = path.stem
        task_row_counts[path.stem] = int(len(frame))
        frames.append(frame)
    merged = pd.concat(frames, ignore_index=True)
    group_cols = ["stay_id"]
    task_base = merged[group_cols + ["task_id"]].drop_duplicates()
    agg = (
        task_base.sort_values(group_cols + ["task_id"], kind="mergesort")
        .groupby(group_cols, sort=False)["task_id"]
        .agg("|".join)
        .reset_index()
        .rename(columns={"task_id": "source_task_ids"})
    )
    task_counts = task_base.groupby(group_cols, sort=False).size().reset_index(name="n_source_tasks")
    agg = agg.merge(task_counts, on=group_cols, how="left")
    for value_col, output_col in [
        ("hadm_id", "hadm_ids"),
        ("tier", "tier_values"),
        ("layer", "layer_values"),
        ("task_mode", "task_mode_values"),
    ]:
        agg = agg.merge(_aggregate_unique_join(merged, group_cols, value_col, output_col), on=group_cols, how="left")
    agg["anchor_hour_requested"] = pd.NA
    agg["anchor_hour_clipped"] = MAX_CONTEXT_HOUR
    agg["anchor_was_clipped"] = 0
    return agg, task_row_counts


def _extract_event_hours(items: list[dict[str, Any]], key: str) -> list[int]:
    hours: list[int] = []
    for item in items:
        val = item.get(key)
        try:
            if val is None:
                continue
            hour = int(float(val))
            hours.append(hour)
        except Exception:
            continue
    hours.sort()
    return hours


def _extract_note_hours_by_type(notes: list[dict[str, Any]]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for note in notes:
        note_type = str(note.get("note_type", "unknown")).lower()
        try:
            hour = int(float(note.get("hour")))
        except Exception:
            continue
        out.setdefault(note_type, []).append(hour)
    for vals in out.values():
        vals.sort()
    return out


def _count_leq(sorted_hours: list[int], anchor: int) -> int:
    return int(bisect_right(sorted_hours, int(anchor)))


def _iter_contexts(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            yield line_number, json.loads(line)


def _build_temporal_condition(
    condition: str,
    contexts_path: Path,
    anchor_df: pd.DataFrame,
    output_path: Path,
    chunk_size: int,
    exclude_discharge: bool,
) -> tuple[dict[str, Any], list[str]]:
    writer = ParquetPartWriter(output_path, chunk_size=chunk_size)
    target_by_stay = {int(k): g.copy() for k, g in anchor_df.groupby("stay_id", sort=False)}
    found_stays: set[int] = set()

    for line_number, rec in _iter_contexts(contexts_path):
        stay_id = int(rec["stay_id"])
        group = target_by_stay.get(stay_id)
        if group is None:
            continue
        found_stays.add(stay_id)

        structured_len = int(len(rec.get("structured_timeline", [])))
        med_hours = _extract_event_hours(rec.get("medication_timeline", []), "event_start_hour")
        proc_hours = _extract_event_hours(rec.get("procedure_timeline", []), "event_start_hour")
        pathway_hours = _extract_event_hours(rec.get("diagnosis_pathway_events", []), "event_time_hour")
        notes = rec.get("notes", [])
        note_hours_by_type = _extract_note_hours_by_type(notes)

        for row in group.itertuples(index=False):
            anchor_requested = int(row.anchor_hour_requested)
            anchor_clipped = int(row.anchor_hour_clipped)
            total_notes = 0
            notes_by_type: dict[str, int] = {}
            for note_type, hours in note_hours_by_type.items():
                if exclude_discharge and note_type == "discharge":
                    continue
                count = _count_leq(hours, anchor_clipped)
                notes_by_type[note_type] = count
                total_notes += count
            writer.write_row(
                {
                    "representation_id": f"{condition}:{stay_id}:{anchor_requested}",
                    "condition": condition,
                    "context_mode": "temporal_original",
                    "stay_id": stay_id,
                    "context_line_number": int(line_number),
                    "anchor_hour_requested": anchor_requested,
                    "anchor_hour_clipped": anchor_clipped,
                    "anchor_was_clipped": int(row.anchor_was_clipped),
                    "source_task_ids": row.source_task_ids,
                    "n_source_tasks": int(row.n_source_tasks),
                    "hadm_ids": getattr(row, "hadm_ids", None),
                    "tier_values": getattr(row, "tier_values", None),
                    "layer_values": getattr(row, "layer_values", None),
                    "task_mode_values": getattr(row, "task_mode_values", None),
                    "notes_policy": "exclude_discharge" if exclude_discharge else "include_all_note_types",
                    "n_structured_hours": min(anchor_clipped + 1, structured_len),
                    "n_medication_events": _count_leq(med_hours, anchor_clipped),
                    "n_procedure_events": _count_leq(proc_hours, anchor_clipped),
                    "n_pathway_events": _count_leq(pathway_hours, anchor_clipped),
                    "n_notes_total": total_notes,
                    "n_notes_nursing": notes_by_type.get("nursing", 0),
                    "n_notes_radiology": notes_by_type.get("radiology", 0),
                    "n_notes_lab_comment": notes_by_type.get("lab_comment", 0),
                    "n_notes_discharge": notes_by_type.get("discharge", 0),
                }
            )

    output_info = writer.close()
    missing_stays = sorted(set(target_by_stay) - found_stays)
    flags: list[str] = []
    if missing_stays:
        flags.append(f"context_missing_stays={len(missing_stays)}")
    return output_info, flags


def _build_retrospective_stroke(
    contexts_path: Path,
    retro_df: pd.DataFrame,
    output_path: Path,
    chunk_size: int,
) -> tuple[dict[str, Any], list[str]]:
    writer = ParquetPartWriter(output_path, chunk_size=chunk_size)
    target_by_stay = {int(k): g.copy() for k, g in retro_df.groupby("stay_id", sort=False)}
    found_stays: set[int] = set()

    for line_number, rec in _iter_contexts(contexts_path):
        stay_id = int(rec["stay_id"])
        group = target_by_stay.get(stay_id)
        if group is None:
            continue
        found_stays.add(stay_id)
        structured_len = int(len(rec.get("structured_timeline", [])))
        med_count = int(len(rec.get("medication_timeline", [])))
        proc_count = int(len(rec.get("procedure_timeline", [])))
        pathway_count = int(len(rec.get("diagnosis_pathway_events", [])))
        notes = rec.get("notes", [])
        note_hours_by_type = _extract_note_hours_by_type(notes)
        notes_by_type = {note_type: len(hours) for note_type, hours in note_hours_by_type.items()}

        row = group.iloc[0]
        writer.write_row(
            {
                "representation_id": f"stroke:retrospective:{stay_id}",
                "condition": "stroke",
                "context_mode": "retrospective_original",
                "stay_id": stay_id,
                "context_line_number": int(line_number),
                "anchor_hour_requested": pd.NA,
                "anchor_hour_clipped": MAX_CONTEXT_HOUR,
                "anchor_was_clipped": 0,
                "source_task_ids": row["source_task_ids"],
                "n_source_tasks": int(row["n_source_tasks"]),
                "hadm_ids": row.get("hadm_ids"),
                "tier_values": row.get("tier_values"),
                "layer_values": row.get("layer_values"),
                "task_mode_values": row.get("task_mode_values"),
                "notes_policy": "full_context_including_discharge",
                "n_structured_hours": structured_len,
                "n_medication_events": med_count,
                "n_procedure_events": proc_count,
                "n_pathway_events": pathway_count,
                "n_notes_total": int(len(notes)),
                "n_notes_nursing": notes_by_type.get("nursing", 0),
                "n_notes_radiology": notes_by_type.get("radiology", 0),
                "n_notes_lab_comment": notes_by_type.get("lab_comment", 0),
                "n_notes_discharge": notes_by_type.get("discharge", 0),
            }
        )

    output_info = writer.close()
    missing_stays = sorted(set(target_by_stay) - found_stays)
    flags: list[str] = []
    if missing_stays:
        flags.append(f"context_missing_stays={len(missing_stays)}")
    return output_info, flags


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(relativize_value(payload, root=ROOT_DIR), indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_temporal(condition: str, anchor_df: pd.DataFrame, output_info: dict[str, Any], flags: list[str], summary_path: Path) -> dict[str, Any]:
    summary = {
        "condition": condition,
        "representation_family": "B2",
        "representation_layout": "global_original_context_bank_plus_task_aware_anchor_index",
        "context_mode": "temporal_original",
        "anchor_index": {
            "rows": int(len(anchor_df)),
            "unique_stays": int(anchor_df["stay_id"].nunique()),
            "min_anchor_hour_requested": int(anchor_df["anchor_hour_requested"].min()) if len(anchor_df) else None,
            "max_anchor_hour_requested": int(anchor_df["anchor_hour_requested"].max()) if len(anchor_df) else None,
            "clipped_anchor_rows": int(anchor_df["anchor_was_clipped"].sum()) if "anchor_was_clipped" in anchor_df.columns else 0,
        },
        "outputs": {
            "context_index_parts_dir": relativize_value(output_info["parts_dir"], root=ROOT_DIR),
            "summary_json": relativize_value(str(summary_path), root=ROOT_DIR),
        },
        "build_stats": {
            "rows_written": int(output_info["rows"]),
            "n_parts": int(output_info["n_parts"]),
        },
        "settings": {
            "anchor_clip_policy": f"clip_to_{MAX_CONTEXT_HOUR}",
            "source_context_variant": "original",
        },
        "flags": flags,
    }
    return summary


def _summarize_stroke(
    temporal_df: pd.DataFrame,
    temporal_output: dict[str, Any],
    temporal_flags: list[str],
    retro_df: pd.DataFrame,
    retro_output: dict[str, Any],
    retro_flags: list[str],
    summary_path: Path,
) -> dict[str, Any]:
    summary = {
        "condition": "stroke",
        "representation_family": "B2",
        "representation_layout": "global_original_context_bank_plus_task_aware_index",
        "temporal": {
            "context_mode": "temporal_original",
            "rows": int(len(temporal_df)),
            "unique_stays": int(temporal_df["stay_id"].nunique()),
            "clipped_anchor_rows": int(temporal_df["anchor_was_clipped"].sum()),
            "outputs": relativize_value(temporal_output["parts_dir"], root=ROOT_DIR),
            "flags": temporal_flags,
            "notes_policy": "exclude_discharge",
        },
        "retrospective": {
            "context_mode": "retrospective_original",
            "rows": int(len(retro_df)),
            "unique_stays": int(retro_df["stay_id"].nunique()),
            "outputs": relativize_value(retro_output["parts_dir"], root=ROOT_DIR),
            "flags": retro_flags,
            "notes_policy": "full_context_including_discharge",
        },
        "outputs": {
            "summary_json": relativize_value(str(summary_path), root=ROOT_DIR),
        },
        "settings": {
            "anchor_clip_policy": f"clip_to_{MAX_CONTEXT_HOUR} for temporal only",
            "source_context_variant": "original",
        },
    }
    return summary


def main() -> None:
    args = parse_args()
    contexts_path = Path(args.contexts_jsonl)
    if not contexts_path.exists():
        raise FileNotFoundError(contexts_path)

    combined: dict[str, Any] = {
        "phase": "4C",
        "representation_family": "B2",
        "source_contexts": relativize_value(str(contexts_path), root=ROOT_DIR),
        "conditions": {},
    }

    for condition in args.conditions:
        cfg = CONDITION_CONFIGS[condition]
        representations_dir = Path(cfg["representations_dir"])
        representations_dir.mkdir(parents=True, exist_ok=True)
        summary_path = Path(cfg["summary_json"])

        if cfg["mode"] == "temporal":
            anchor_df, _ = _load_temporal_anchor_index([Path(p) for p in cfg["task_files"]], str(cfg["anchor_col"]))
            output_path = representations_dir / f"{condition}_B2_original_context_index.parquet"
            output_info, flags = _build_temporal_condition(
                condition=condition,
                contexts_path=contexts_path,
                anchor_df=anchor_df,
                output_path=output_path,
                chunk_size=int(args.chunk_size),
                exclude_discharge=bool(cfg["exclude_discharge"]),
            )
            summary = _summarize_temporal(condition, anchor_df, output_info, flags, summary_path)
            _write_summary(summary_path, summary)
            combined["conditions"][condition] = summary
            continue

        temporal_df, _ = _load_temporal_anchor_index(
            [Path(p) for p in cfg["temporal_task_files"]],
            str(cfg["anchor_col"]),
        )
        retro_df, _ = _load_retrospective_index([Path(p) for p in cfg["retrospective_task_files"]])
        temporal_out = representations_dir / "stroke_B2_original_temporal_index.parquet"
        retro_out = representations_dir / "stroke_B2_original_retrospective_index.parquet"
        temporal_output, temporal_flags = _build_temporal_condition(
            condition="stroke",
            contexts_path=contexts_path,
            anchor_df=temporal_df,
            output_path=temporal_out,
            chunk_size=int(args.chunk_size),
            exclude_discharge=True,
        )
        retro_output, retro_flags = _build_retrospective_stroke(
            contexts_path=contexts_path,
            retro_df=retro_df,
            output_path=retro_out,
            chunk_size=int(args.chunk_size),
        )
        summary = _summarize_stroke(temporal_df, temporal_output, temporal_flags, retro_df, retro_output, retro_flags, summary_path)
        _write_summary(summary_path, summary)
        combined["conditions"][condition] = summary

    combined_summary_path = Path(args.combined_summary_json)
    _write_summary(combined_summary_path, combined)
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
