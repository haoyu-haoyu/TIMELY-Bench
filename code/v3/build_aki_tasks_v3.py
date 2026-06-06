#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.bq_utils import make_bq_client, quota_project_of
from v3.constants import (
    DEFAULT_DIAGNOSIS_PATHWAY_EVENTS,
    DEFAULT_HOURLY_STATE_GRID,
    DEFAULT_PROCEDURE_EVENTS,
    ROOT_DIR,
    V3_PROCESSED_DIR,
    V3_RESULTS_DIR,
    V3_SOURCE_COHORT_FILE,
)
from v3.io_utils import iter_table_chunks, relativize_value, write_table


try:
    from google.cloud import bigquery
except Exception:  # pragma: no cover
    bigquery = None


RRT_EVENT_PATTERNS = ("dialysis", "rrt", "crrt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 AKI task instances.")
    p.add_argument("--billing-project", default="timely-bench-mimic")
    p.add_argument(
        "--kdigo-table",
        default="physionet-data.mimiciv_3_1_derived.kdigo_stages",
    )
    p.add_argument(
        "--icustays-table",
        default="physionet-data.mimiciv_3_1_icu.icustays",
    )
    p.add_argument("--cohort-v3", default=str(V3_SOURCE_COHORT_FILE))
    p.add_argument(
        "--diagnosis-pathway-events",
        default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS),
    )
    p.add_argument(
        "--procedure-events",
        default=str(DEFAULT_PROCEDURE_EVENTS),
    )
    p.add_argument(
        "--hourly-grid",
        default=str(DEFAULT_HOURLY_STATE_GRID),
    )
    p.add_argument("--max-hour", type=int, default=168)
    p.add_argument("--stage1-onset-max-hour", type=int, default=48)
    p.add_argument("--prediction-interval", type=int, default=4)
    p.add_argument("--lookahead-hours", type=int, default=24)
    p.add_argument("--rrt-lookahead-hours", type=int, default=72)
    p.add_argument("--bq-batch-size", type=int, default=5000)
    p.add_argument(
        "--out-dir",
        default=str(V3_PROCESSED_DIR / "aki" / "tasks"),
    )
    p.add_argument(
        "--summary-json",
        default=str(V3_RESULTS_DIR / "aki" / "aki_task_build_summary.json"),
    )
    return p.parse_args()


def _stay_id_batches(stay_ids: List[int], batch_size: int) -> List[List[int]]:
    return [stay_ids[i : i + batch_size] for i in range(0, len(stay_ids), batch_size)]


def _extract_onsets_from_bigquery(args: argparse.Namespace, stay_ids: List[int]) -> tuple[pd.DataFrame, int, str | None]:
    if bigquery is None:
        raise RuntimeError("google-cloud-bigquery is not installed in this environment.")
    client = make_bq_client(args.billing_project)
    bt = "`"
    onset_frames: List[pd.DataFrame] = []
    total_rows = 0
    for idx, batch in enumerate(_stay_id_batches(stay_ids, int(args.bq_batch_size)), start=1):
        stay_list = ", ".join(str(int(x)) for x in batch)
        sql = (
            "WITH target_stays AS ("
            f"  SELECT stay_id FROM UNNEST([{stay_list}]) AS stay_id"
            "), kdigo_hours AS ("
            "  SELECT "
            "    k.stay_id, "
            "    CAST(DATETIME_DIFF(DATETIME(k.charttime), DATETIME(i.intime), HOUR) AS INT64) AS hour, "
            "    CAST(k.aki_stage AS INT64) AS aki_stage "
            f"  FROM {bt}{args.kdigo_table}{bt} k "
            f"  JOIN {bt}{args.icustays_table}{bt} i "
            "  ON k.stay_id = i.stay_id "
            "  JOIN target_stays t "
            "  ON k.stay_id = t.stay_id "
            "  WHERE k.aki_stage IS NOT NULL "
            f"  AND DATETIME_DIFF(DATETIME(k.charttime), DATETIME(i.intime), HOUR) BETWEEN 0 AND {int(args.max_hour)}"
            ") "
            "SELECT "
            "  stay_id, "
            "  MIN(CASE WHEN aki_stage = 1 THEN hour END) AS stage1_onset_hour, "
            "  MIN(CASE WHEN aki_stage >= 2 THEN hour END) AS stage2plus_onset_hour, "
            "  MIN(CASE WHEN aki_stage >= 3 THEN hour END) AS stage3_onset_hour, "
            "  COUNT(*) AS kdigo_rows "
            "FROM kdigo_hours "
            "GROUP BY stay_id"
        )
        frame = client.query(sql).result().to_dataframe(create_bqstorage_client=False)
        if not frame.empty:
            frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce").astype("Int64")
            frame["stage1_onset_hour"] = pd.to_numeric(frame["stage1_onset_hour"], errors="coerce")
            frame["stage2plus_onset_hour"] = pd.to_numeric(frame["stage2plus_onset_hour"], errors="coerce")
            frame["stage3_onset_hour"] = pd.to_numeric(frame["stage3_onset_hour"], errors="coerce")
            frame["kdigo_rows"] = pd.to_numeric(frame["kdigo_rows"], errors="coerce").fillna(0).astype("int64")
            total_rows += int(frame["kdigo_rows"].sum())
            frame = frame.dropna(subset=["stay_id", "stage1_onset_hour"]).copy()
            frame["stay_id"] = frame["stay_id"].astype("int64")
            frame = frame[frame["stage1_onset_hour"] <= int(args.stage1_onset_max_hour)].copy()
            onset_frames.append(frame[["stay_id", "stage1_onset_hour", "stage2plus_onset_hour", "stage3_onset_hour"]])
        print(f"[bq] batch {idx} / {math.ceil(len(stay_ids) / int(args.bq_batch_size))} rows={len(frame)}")
        del frame
        gc.collect()
    if not onset_frames:
        return pd.DataFrame(columns=["stay_id", "stage1_onset_hour", "stage2plus_onset_hour", "stage3_onset_hour"]), total_rows, quota_project_of(client)
    onsets = pd.concat(onset_frames, ignore_index=True)
    onsets = onsets.sort_values(["stay_id"], kind="mergesort").drop_duplicates(subset=["stay_id"], keep="first")
    return onsets, total_rows, quota_project_of(client)


def _load_main_cohort(path: Path) -> pd.DataFrame:
    cohort = pd.read_csv(path, low_memory=False)
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")
    if "has_aki_final" in cohort.columns:
        cohort["has_aki_final"] = pd.to_numeric(cohort["has_aki_final"], errors="coerce").fillna(0).astype("int64")
        cohort = cohort[cohort["has_aki_final"] == 1].copy()
    return cohort


def _extract_rrt_from_procedure_events(path: Path, stay_ids: set[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for chunk in iter_table_chunks(path):
        if "stay_id" not in chunk.columns or "event_start_hour" not in chunk.columns:
            continue
        df = chunk.loc[chunk["stay_id"].isin(stay_ids)].copy()
        if df.empty:
            continue
        text_cols = [c for c in ["event_name", "value", "valueuom", "source"] if c in df.columns]
        if not text_cols:
            continue
        mask = pd.Series(False, index=df.index)
        for col in text_cols:
            mask = mask | df[col].astype(str).str.lower().str.contains("|".join(RRT_EVENT_PATTERNS), na=False)
        df = df.loc[mask, ["stay_id", "event_start_hour"]].copy()
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stay_id", "rrt_proxy_hour", "rrt_time_source"])
    merged = pd.concat(frames, ignore_index=True)
    merged["stay_id"] = pd.to_numeric(merged["stay_id"], errors="coerce").astype("Int64")
    merged["event_start_hour"] = pd.to_numeric(merged["event_start_hour"], errors="coerce")
    merged = merged.dropna(subset=["stay_id", "event_start_hour"]).copy()
    if merged.empty:
        return pd.DataFrame(columns=["stay_id", "rrt_proxy_hour", "rrt_time_source"])
    merged["stay_id"] = merged["stay_id"].astype("int64")
    merged["event_start_hour"] = merged["event_start_hour"].astype(float)
    out = (
        merged.groupby("stay_id", sort=False)["event_start_hour"]
        .min()
        .reset_index()
        .rename(columns={"event_start_hour": "rrt_proxy_hour"})
    )
    out["rrt_time_source"] = "procedure_events"
    return out


def _extract_rrt_from_hourly_grid(path: Path, stay_ids: set[int]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for chunk in iter_table_chunks(path):
        if "stay_id" not in chunk.columns or "hour" not in chunk.columns:
            continue
        keep_cols = [c for c in ["stay_id", "hour", "rrt_active", "rrt"] if c in chunk.columns]
        if len(keep_cols) < 2:
            continue
        df = chunk.loc[chunk["stay_id"].isin(stay_ids), keep_cols].copy()
        if df.empty:
            continue
        if "rrt_active" in df.columns:
            df["rrt_active"] = pd.to_numeric(df["rrt_active"], errors="coerce").fillna(0)
        else:
            df["rrt_active"] = 0
        if "rrt" in df.columns:
            df["rrt"] = pd.to_numeric(df["rrt"], errors="coerce").fillna(0)
        else:
            df["rrt"] = 0
        df = df.loc[(df["rrt_active"] > 0) | (df["rrt"] > 0), ["stay_id", "hour"]].copy()
        if df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["stay_id", "rrt_proxy_hour", "rrt_time_source"])
    merged = pd.concat(frames, ignore_index=True)
    merged["stay_id"] = pd.to_numeric(merged["stay_id"], errors="coerce").astype("Int64")
    merged["hour"] = pd.to_numeric(merged["hour"], errors="coerce")
    merged = merged.dropna(subset=["stay_id", "hour"]).copy()
    if merged.empty:
        return pd.DataFrame(columns=["stay_id", "rrt_proxy_hour", "rrt_time_source"])
    merged["stay_id"] = merged["stay_id"].astype("int64")
    merged["hour"] = merged["hour"].astype(float)
    out = (
        merged.groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "rrt_proxy_hour"})
    )
    out["rrt_time_source"] = "hourly_grid_rrt"
    return out


def _extract_rrt_proxy(procedure_path: Path, hourly_grid_path: Path, stay_ids: set[int]) -> pd.DataFrame:
    proc = _extract_rrt_from_procedure_events(procedure_path, stay_ids)
    proc_ids = set(proc["stay_id"].tolist()) if not proc.empty else set()
    missing = stay_ids - proc_ids
    if missing:
        grid = _extract_rrt_from_hourly_grid(hourly_grid_path, missing)
        if not grid.empty:
            proc = pd.concat([proc, grid], ignore_index=True)
    if proc.empty:
        return pd.DataFrame(columns=["stay_id", "rrt_proxy_hour", "rrt_time_source"])
    proc = proc.sort_values(["stay_id", "rrt_proxy_hour"], kind="mergesort").drop_duplicates(subset=["stay_id"], keep="first")
    return proc.reset_index(drop=True)


def _build_primary_instances(cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: List[tuple] = []
    p_int = int(args.prediction_interval)
    lookahead = int(args.lookahead_hours)
    max_hour = int(args.max_hour)
    for row in cohort.itertuples(index=False):
        sid = int(row.stay_id)
        onset1 = int(row.stage1_onset_hour)
        onset2 = getattr(row, "stage2plus_onset_hour")
        onset2 = float(onset2) if pd.notna(onset2) else math.inf
        upper_t = min(max_hour, int(onset2) - 1 if math.isfinite(onset2) else max_hour)
        t = onset1 + p_int
        while t <= upper_t:
            label = int(math.isfinite(onset2) and onset2 > t and onset2 <= t + lookahead)
            records.append(
                (
                    sid,
                    "AKI-T1",
                    "temporal_progression",
                    int(t),
                    lookahead,
                    onset1,
                    float(onset2) if math.isfinite(onset2) else np.nan,
                    label,
                    True,
                    "stage1_onset_lte_48h",
                )
            )
            t += p_int
    return pd.DataFrame.from_records(
        records,
        columns=[
            "stay_id",
            "task_id",
            "task_family",
            "anchor_hour",
            "horizon_hours",
            "stage1_onset_hour",
            "stage2plus_onset_hour",
            "label",
            "eligible",
            "eligibility_reason",
        ],
    )


def _build_rrt_instances(cohort: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    records: List[tuple] = []
    p_int = int(args.prediction_interval)
    max_hour = int(args.max_hour)
    lookahead = int(args.rrt_lookahead_hours)
    for row in cohort.itertuples(index=False):
        sid = int(row.stay_id)
        onset1 = int(row.stage1_onset_hour)
        rrt_hour = getattr(row, "rrt_proxy_hour")
        rrt_hour = float(rrt_hour) if pd.notna(rrt_hour) else math.inf
        rrt_source = getattr(row, "rrt_time_source", np.nan)
        rrt_source = str(rrt_source) if pd.notna(rrt_source) else "unknown"
        upper_t = min(max_hour, int(rrt_hour) - 1 if math.isfinite(rrt_hour) else max_hour)
        t = onset1 + p_int
        while t <= upper_t:
            label = int(math.isfinite(rrt_hour) and rrt_hour > t and rrt_hour <= t + lookahead)
            records.append(
                (
                    sid,
                    "AKI-S1",
                    "support_escalation_proxy",
                    int(t),
                    lookahead,
                    onset1,
                    float(rrt_hour) if math.isfinite(rrt_hour) else np.nan,
                    label,
                    True,
                    "stage1_onset_lte_48h",
                    rrt_source,
                )
            )
            t += p_int
    return pd.DataFrame.from_records(
        records,
        columns=[
            "stay_id",
            "task_id",
            "task_family",
            "anchor_hour",
            "horizon_hours",
            "stage1_onset_hour",
            "rrt_proxy_hour",
            "label",
            "eligible",
            "eligibility_reason",
            "label_source",
        ],
    )


def main() -> None:
    args = parse_args()

    cohort_path = Path(args.cohort_v3)
    pathway_path = Path(args.diagnosis_pathway_events)
    procedure_path = Path(args.procedure_events)
    hourly_grid_path = Path(args.hourly_grid)
    out_dir = Path(args.out_dir)
    summary_path = Path(args.summary_json)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    cohort = _load_main_cohort(cohort_path)
    cohort_ids = set(cohort["stay_id"].tolist())

    onset_cohort, kdigo_rows, quota_project = _extract_onsets_from_bigquery(args, stay_ids=sorted(cohort_ids))
    print(f"[post-bq] onset_cohort_rows={len(onset_cohort)}")
    kdigo_out = out_dir / "aki_kdigo_onsets_v3.parquet"
    write_table(onset_cohort, kdigo_out, index=False)
    pathway = pd.read_parquet(pathway_path, columns=["stay_id", "event_time_hour", "event_name", "event_source", "is_proxy", "details_json"])
    print(f"[pathway] rows={len(pathway)}")
    rrt_proxy = _extract_rrt_proxy(procedure_path, hourly_grid_path, cohort_ids)
    print(f"[pathway] rrt_proxy_stays={len(rrt_proxy)}")
    onset_cohort = onset_cohort.merge(rrt_proxy, on="stay_id", how="left")

    primary = _build_primary_instances(onset_cohort, args)
    print(f"[primary] rows={len(primary)}")
    if primary.empty:
        raise RuntimeError("No AKI primary task instances were generated.")
    secondary = _build_rrt_instances(onset_cohort, args)
    print(f"[secondary] rows={len(secondary)}")

    cohort_out = out_dir / "aki_stage1_cohort_v3.parquet"
    primary_out = out_dir / "aki_stage2plus_instances.parquet"
    secondary_out = out_dir / "aki_rrt_proxy_instances.parquet"
    write_table(onset_cohort, cohort_out, index=False)
    write_table(primary, primary_out, index=False)
    write_table(secondary, secondary_out, index=False)

    pos_rate = float(primary["label"].mean())
    stays_with_positive = int(primary.groupby("stay_id", sort=False)["label"].max().sum())
    secondary_pos_rate = float(secondary["label"].mean()) if not secondary.empty else None
    secondary_positive_stays = int(secondary.groupby("stay_id", sort=False)["label"].max().sum()) if not secondary.empty else 0
    flags: List[str] = []
    if secondary.empty:
        flags.append("secondary_rrt_proxy_task_empty")
    if int(len(rrt_proxy)) == 0:
        flags.append("rrt_proxy_events_missing")

    summary: Dict[str, object] = {
        "condition": "aki",
        "cohort": {
            "cohort_source_rows": int(len(cohort)),
            "kdigo_rows": int(kdigo_rows),
            "kdigo_stays": int(onset_cohort["stay_id"].nunique()),
            "stage1_onset_lte_48h_stays": int(len(onset_cohort)),
            "stage2plus_any_stays": int(onset_cohort["stage2plus_onset_hour"].notna().sum()),
            "rrt_proxy_any_stays": int(onset_cohort["rrt_proxy_hour"].notna().sum()),
        },
        "tasks": {
            "AKI-T1": {
                "rows": int(len(primary)),
                "unique_stays": int(primary["stay_id"].nunique()),
                "positive_rate": pos_rate,
                "stays_with_any_positive": stays_with_positive,
                "label_definition": "stage2plus within (T, T+24]",
            },
            "AKI-S1": {
                "rows": int(len(secondary)),
                "unique_stays": int(secondary["stay_id"].nunique()) if not secondary.empty else 0,
                "positive_rate": secondary_pos_rate,
                "stays_with_any_positive": secondary_positive_stays,
                "label_definition": "rrt_proxy within (T, T+72]",
                "label_source": "procedure_events_then_hourly_grid_rrt",
                "rrt_time_source_counts": {
                    str(k): int(v)
                    for k, v in onset_cohort["rrt_time_source"].dropna().value_counts().sort_index().items()
                },
            },
        },
        "outputs": relativize_value(
            {
                "kdigo_staged": kdigo_out,
                "aki_stage1_cohort": cohort_out,
                "primary_instances": primary_out,
                "secondary_rrt_proxy_instances": secondary_out,
            },
            root=ROOT_DIR,
        ),
        "settings": {
            "billing_project": args.billing_project,
            "quota_project": quota_project,
            "prediction_interval": int(args.prediction_interval),
            "lookahead_hours": int(args.lookahead_hours),
            "rrt_lookahead_hours": int(args.rrt_lookahead_hours),
            "max_hour": int(args.max_hour),
            "stage1_onset_max_hour": int(args.stage1_onset_max_hour),
        },
        "flags": flags,
    }
    summary_path.write_text(json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
