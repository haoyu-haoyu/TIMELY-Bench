#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import DEFAULT_STATE_VECTORS, ROOT_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR  # type: ignore
from v3.io_utils import iter_table_chunks, read_table, relativize_value, table_exists, write_table  # type: ignore


CONDITION_CONFIGS = {
    "aki": {
        "task_files": [
            V3_PROCESSED_DIR / "aki" / "tasks" / "aki_stage2plus_instances.parquet",
            V3_PROCESSED_DIR / "aki" / "tasks" / "aki_rrt_proxy_instances.parquet",
        ],
        "anchor_col": "anchor_hour",
        "representations_dir": V3_PROCESSED_DIR / "aki" / "representations",
        "summary_json": V3_RESULTS_DIR / "aki" / "aki_B3_build_summary.json",
    },
    "delirium": {
        "task_files": [
            V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_persistence_instances.parquet",
            V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_resolution_instances.parquet",
        ],
        "anchor_col": "prediction_hour",
        "representations_dir": V3_PROCESSED_DIR / "delirium" / "representations",
        "summary_json": V3_RESULTS_DIR / "delirium" / "delirium_B3_build_summary.json",
    },
    "sepsis": {
        "task_files": [
            V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_shock_instances.parquet",
            V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_lactate_clearance_instances.parquet",
        ],
        "anchor_col": "prediction_hour",
        "representations_dir": V3_PROCESSED_DIR / "sepsis" / "representations",
        "summary_json": V3_RESULTS_DIR / "sepsis" / "sepsis_B3_build_summary.json",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 Phase 4B B3 representations.")
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["aki", "delirium", "sepsis"],
        choices=sorted(CONDITION_CONFIGS.keys()),
    )
    p.add_argument("--state-vectors", default=str(DEFAULT_STATE_VECTORS))
    p.add_argument(
        "--combined-summary-json",
        default=str(V3_RESULTS_DIR / "representations" / "phase4b_B3_build_summary.json"),
    )
    return p.parse_args()


def _parts_dir(path: Path) -> Path:
    return path.with_name(f"{path.name}.parts")


def _load_anchor_index(condition: str, task_files: list[Path], anchor_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
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
        keep_cols = ["stay_id", anchor_col]
        task_id_col = "task_id" if "task_id" in df.columns else None
        if task_id_col:
            keep_cols.append(task_id_col)
        frame = df[keep_cols].copy()
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
        frame["anchor_hour"] = pd.to_numeric(frame[anchor_col], errors="coerce")
        frame = frame.dropna(subset=["stay_id", "anchor_hour"]).copy()
        frame["stay_id"] = frame["stay_id"].astype("int64")
        frame["anchor_hour"] = frame["anchor_hour"].astype("int64")
        if task_id_col:
            frame["task_id"] = frame[task_id_col].astype(str)
        else:
            frame["task_id"] = path.stem
        task_row_counts[Path(path).stem] = int(len(frame))
        frames.append(frame[["stay_id", "anchor_hour", "task_id"]])
    if not frames:
        raise RuntimeError(f"No task frames found for {condition}")
    anchors_long = pd.concat(frames, ignore_index=True)
    grouped = (
        anchors_long.groupby(["stay_id", "anchor_hour"], sort=False)["task_id"]
        .agg(lambda s: "|".join(sorted(pd.unique(s.astype(str)))))
        .reset_index()
        .rename(columns={"task_id": "source_task_ids"})
    )
    grouped["n_source_tasks"] = grouped["source_task_ids"].str.split("|").str.len().astype(int)
    grouped["condition"] = condition
    grouped["representation_id"] = (
        condition + ":"
        + grouped["stay_id"].astype(str)
        + ":"
        + grouped["anchor_hour"].astype(str)
    )
    grouped["history_start_hour"] = 0
    grouped["history_end_hour"] = grouped["anchor_hour"]
    grouped["history_length_hours"] = grouped["anchor_hour"] + 1
    grouped = grouped[
        [
            "representation_id",
            "condition",
            "stay_id",
            "anchor_hour",
            "history_start_hour",
            "history_end_hour",
            "history_length_hours",
            "n_source_tasks",
            "source_task_ids",
        ]
    ].sort_values(["stay_id", "anchor_hour"], kind="mergesort").reset_index(drop=True)
    return grouped, task_row_counts


def _write_state_bank(condition: str, state_vectors_path: Path, stay_ids: set[int], output_path: Path) -> dict[str, object]:
    parts_dir = _parts_dir(output_path)
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    part_idx = 0
    chunk_idx = 0
    unique_stays: set[int] = set()
    min_state_hour: int | None = None
    max_state_hour: int | None = None
    columns_written: list[str] | None = None

    for chunk in iter_table_chunks(state_vectors_path):
        chunk_idx += 1
        if "stay_id" not in chunk.columns or "hour" not in chunk.columns:
            raise KeyError("state_vectors chunk missing stay_id/hour")
        df = chunk.loc[chunk["stay_id"].isin(stay_ids)].copy()
        if df.empty:
            print(f"[{condition}] state_vectors chunk {chunk_idx} kept_rows=0")
            continue
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
        df = df.dropna(subset=["stay_id", "hour"]).copy()
        df["stay_id"] = df["stay_id"].astype("int64")
        df["hour"] = df["hour"].astype("int64")
        df = df.rename(columns={"hour": "state_hour"})
        df.insert(0, "condition", condition)
        part_idx += 1
        part_path = parts_dir / f"part_{part_idx:05d}.parquet"
        df.to_parquet(part_path, index=False)
        total_rows += int(len(df))
        unique_stays.update(df["stay_id"].astype(int).unique().tolist())
        cur_min = int(df["state_hour"].min())
        cur_max = int(df["state_hour"].max())
        min_state_hour = cur_min if min_state_hour is None else min(min_state_hour, cur_min)
        max_state_hour = cur_max if max_state_hour is None else max(max_state_hour, cur_max)
        if columns_written is None:
            columns_written = df.columns.tolist()
        print(f"[{condition}] state_vectors chunk {chunk_idx} kept_rows={len(df)} wrote_part={part_idx}")

    if part_idx == 0:
        raise RuntimeError(f"No state vector rows found for {condition}")

    return {
        "rows": int(total_rows),
        "unique_stays": int(len(unique_stays)),
        "n_parts": int(part_idx),
        "min_state_hour": min_state_hour,
        "max_state_hour": max_state_hour,
        "columns": columns_written or [],
        "parts_dir": str(parts_dir),
    }


def _coverage_summary(anchor_df: pd.DataFrame, state_bank_parts_dir: Path) -> dict[str, object]:
    max_hour_df_frames: list[pd.DataFrame] = []
    part_files = sorted(state_bank_parts_dir.glob("*.parquet"))
    for part in part_files:
        df = pd.read_parquet(part, columns=["stay_id", "state_hour"])
        max_hour_df_frames.append(
            df.groupby("stay_id", sort=False)["state_hour"].max().reset_index().rename(columns={"state_hour": "max_state_hour"})
        )
    max_hour_df = pd.concat(max_hour_df_frames, ignore_index=True)
    max_hour_df = (
        max_hour_df.groupby("stay_id", sort=False)["max_state_hour"]
        .max()
        .reset_index()
    )
    cov = anchor_df.merge(max_hour_df, on="stay_id", how="left")
    missing_stays = int(cov["max_state_hour"].isna().sum())
    anchor_hour_exceeds = int((cov["max_state_hour"].notna() & (cov["anchor_hour"] > cov["max_state_hour"])).sum())
    return {
        "anchor_rows": int(len(anchor_df)),
        "anchor_unique_stays": int(anchor_df["stay_id"].nunique()),
        "anchor_stays_missing_in_state_bank": missing_stays,
        "anchor_rows_exceeding_max_state_hour": anchor_hour_exceeds,
    }


def _condition_summary(
    condition: str,
    anchor_df: pd.DataFrame,
    task_row_counts: dict[str, int],
    state_bank_info: dict[str, object],
    anchor_index_path: Path,
    summary_json_path: Path,
) -> dict[str, object]:
    state_bank_parts_dir = Path(str(state_bank_info["parts_dir"]))
    coverage = _coverage_summary(anchor_df, state_bank_parts_dir)
    summary = {
        "condition": condition,
        "representation_family": "B3",
        "representation_layout": "state_bank_plus_anchor_index",
        "inputs": relativize_value(
            {
                "task_row_counts": task_row_counts,
            },
            root=ROOT_DIR,
        ),
        "anchor_index": {
            "rows": int(len(anchor_df)),
            "unique_stays": int(anchor_df["stay_id"].nunique()),
            "min_anchor_hour": int(anchor_df["anchor_hour"].min()) if len(anchor_df) else None,
            "max_anchor_hour": int(anchor_df["anchor_hour"].max()) if len(anchor_df) else None,
            "outputs": relativize_value(str(anchor_index_path), root=ROOT_DIR),
        },
        "state_bank": {
            "rows": int(state_bank_info["rows"]),
            "unique_stays": int(state_bank_info["unique_stays"]),
            "n_parts": int(state_bank_info["n_parts"]),
            "min_state_hour": state_bank_info["min_state_hour"],
            "max_state_hour": state_bank_info["max_state_hour"],
            "n_columns": int(len(state_bank_info["columns"])),
            "outputs": relativize_value(str(state_bank_parts_dir), root=ROOT_DIR),
        },
        "coverage": coverage,
        "settings": {
            "history_rule": "Use state bank rows where state_hour <= anchor_hour and state_hour >= 0",
            "anchor_deduplication": "Unique per (stay_id, anchor_hour) across all eligible Phase 3 tasks within condition",
            "forward_fill_policy": "No additional forward fill in B3 builder; inherits hourly_state_grid/state_vectors values and explicit missingness masks",
        },
        "outputs": relativize_value(
            {
                "anchor_index": str(anchor_index_path),
                "state_bank_parts_dir": str(state_bank_parts_dir),
                "summary_json": str(summary_json_path),
            },
            root=ROOT_DIR,
        ),
        "flags": [],
    }
    if coverage["anchor_stays_missing_in_state_bank"] != 0:
        summary["flags"].append(f"anchor_stays_missing_in_state_bank={coverage['anchor_stays_missing_in_state_bank']}")
    if coverage["anchor_rows_exceeding_max_state_hour"] != 0:
        summary["flags"].append(f"anchor_rows_exceeding_max_state_hour={coverage['anchor_rows_exceeding_max_state_hour']}")
    return summary


def _build_condition(condition: str, cfg: dict[str, object], state_vectors_path: Path) -> dict[str, object]:
    task_files = [Path(p) for p in cfg["task_files"]]
    anchor_col = str(cfg["anchor_col"])
    representations_dir = Path(cfg["representations_dir"])
    summary_json_path = Path(cfg["summary_json"])

    anchor_index_df, task_row_counts = _load_anchor_index(condition, task_files, anchor_col)
    representations_dir.mkdir(parents=True, exist_ok=True)
    anchor_index_path = representations_dir / f"{condition}_B3_anchor_index.parquet"
    state_bank_path = representations_dir / f"{condition}_B3_state_bank.parquet"
    write_table(anchor_index_df, anchor_index_path, index=False)

    state_bank_info = _write_state_bank(
        condition=condition,
        state_vectors_path=state_vectors_path,
        stay_ids=set(anchor_index_df["stay_id"].astype(int).tolist()),
        output_path=state_bank_path,
    )
    summary = _condition_summary(
        condition=condition,
        anchor_df=anchor_index_df,
        task_row_counts=task_row_counts,
        state_bank_info=state_bank_info,
        anchor_index_path=anchor_index_path,
        summary_json_path=summary_json_path,
    )
    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    summary_json_path.write_text(
        json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    args = parse_args()
    state_vectors_path = Path(args.state_vectors)
    if not table_exists(state_vectors_path):
        raise FileNotFoundError(state_vectors_path)

    combined: dict[str, object] = {
        "phase": "4B",
        "representation_family": "B3",
        "state_vectors_source": relativize_value(str(state_vectors_path), root=ROOT_DIR),
        "conditions": {},
    }

    for condition in args.conditions:
        summary = _build_condition(condition, CONDITION_CONFIGS[condition], state_vectors_path)
        combined["conditions"][condition] = summary

    combined_summary_path = Path(args.combined_summary_json)
    combined_summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined_summary_path.write_text(
        json.dumps(relativize_value(combined, root=ROOT_DIR), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(combined, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
