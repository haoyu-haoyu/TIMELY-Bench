#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import DEFAULT_DIAGNOSIS_PATHWAY_EVENTS, DEFAULT_HOURLY_STATE_GRID, ROOT_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR, V3_SOURCE_COHORT_FILE  # type: ignore
from v3.io_utils import iter_table_chunks, relativize_value, write_table  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 sepsis Phase 3 task instances.")
    p.add_argument("--cohort-csv", default=str(V3_SOURCE_COHORT_FILE))
    p.add_argument("--diagnosis-pathway", default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS))
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--prediction-interval", type=int, default=4)
    p.add_argument("--lookahead-hours", type=int, default=12)
    p.add_argument("--clearance-window-hours", type=int, default=6)
    p.add_argument("--max-hour", type=int, default=168)
    p.add_argument("--map-threshold", type=float, default=65.0)
    p.add_argument("--lactate-threshold", type=float, default=2.0)
    p.add_argument("--metadata-out", default=str(V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_stay_metadata.parquet"))
    p.add_argument("--shock-hourly-out", default=str(V3_PROCESSED_DIR / "sepsis" / "tasks" / "septic_shock_hourly_v3.parquet"))
    p.add_argument("--primary-out", default=str(V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_shock_instances.parquet"))
    p.add_argument("--secondary-out", default=str(V3_PROCESSED_DIR / "sepsis" / "tasks" / "sepsis_lactate_clearance_instances.parquet"))
    p.add_argument("--summary-json", default=str(V3_RESULTS_DIR / "sepsis" / "sepsis_task_build_summary.json"))
    return p.parse_args()


def _build_sepsis_cohort(cohort_csv: str | Path, diagnosis_pathway: str | Path) -> pd.DataFrame:
    cohort = pd.read_csv(cohort_csv, usecols=["stay_id", "sepsis3"])
    cohort["sepsis3"] = cohort["sepsis3"].astype(bool)
    cohort = cohort.loc[cohort["sepsis3"], ["stay_id"]].copy()
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype("int64")

    pathway = pd.read_parquet(diagnosis_pathway, columns=["stay_id", "event_time_hour", "event_name"])
    onset_mask = pathway["event_name"].astype(str).str.contains("sepsis", case=False, na=False) & pathway["event_name"].astype(str).str.contains("onset", case=False, na=False)
    onset = (
        pathway.loc[onset_mask, ["stay_id", "event_time_hour"]]
        .groupby("stay_id", sort=False)["event_time_hour"]
        .min()
        .reset_index()
        .rename(columns={"event_time_hour": "sepsis_onset_hour"})
    )
    out = cohort.merge(onset, on="stay_id", how="left")
    out["onset_source"] = np.where(out["sepsis_onset_hour"].notna(), "diagnosis_pathway", "fallback_zero")
    out["onset_confidence"] = np.where(out["sepsis_onset_hour"].notna(), "high", "low")
    out["sepsis_onset_hour"] = out["sepsis_onset_hour"].fillna(0).astype(int)
    return out.sort_values("stay_id", kind="mergesort").reset_index(drop=True)


def _load_sepsis_grid(hourly_grid: str | Path, stay_ids: set[int]) -> pd.DataFrame:
    keep_cols = [
        "stay_id",
        "hour",
        "mbp",
        "map_merged",
        "lactate",
        "vasopressors_active",
        "vasopressors",
        "vasopressor_dose_norepi_equiv",
        "sofa_total",
        "fluid_balance",
    ]
    frames: list[pd.DataFrame] = []
    for idx, chunk in enumerate(iter_table_chunks(hourly_grid), start=1):
        cols = [c for c in keep_cols if c in chunk.columns]
        df = chunk.loc[chunk["stay_id"].isin(stay_ids), cols].copy()
        if len(df):
            frames.append(df)
        print(f"[grid] chunk {idx} kept_rows={len(df)}")
    if not frames:
        raise RuntimeError("No hourly grid rows found for sepsis cohort.")
    out = pd.concat(frames, ignore_index=True)
    out["stay_id"] = pd.to_numeric(out["stay_id"], errors="coerce").astype("Int64")
    out["hour"] = pd.to_numeric(out["hour"], errors="coerce")
    out = out.dropna(subset=["stay_id", "hour"]).copy()
    out["stay_id"] = out["stay_id"].astype("int64")
    out["hour"] = out["hour"].astype("int64")
    for col in ["mbp", "map_merged", "lactate", "vasopressors_active", "vasopressors", "vasopressor_dose_norepi_equiv", "sofa_total", "fluid_balance"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["map_used"] = out["map_merged"].where(out["map_merged"].notna(), out["mbp"])
    active = out["vasopressors_active"] if "vasopressors_active" in out.columns else pd.Series(np.nan, index=out.index)
    raw_vaso = out["vasopressors"] if "vasopressors" in out.columns else pd.Series(np.nan, index=out.index)
    out["vaso_active"] = np.where(active.notna(), active.fillna(0), raw_vaso.fillna(0))
    out["vaso_active"] = pd.to_numeric(out["vaso_active"], errors="coerce").fillna(0).astype(int)
    return out.sort_values(["stay_id", "hour"], kind="mergesort").reset_index(drop=True)


def _compute_shock(grid_df: pd.DataFrame, map_threshold: float, lactate_threshold: float) -> pd.DataFrame:
    shock = (grid_df["map_used"] < float(map_threshold)) & (grid_df["vaso_active"] > 0)
    lactate_present = grid_df["lactate"].notna()
    shock = shock & (~lactate_present | (grid_df["lactate"] > float(lactate_threshold)))
    out = grid_df.copy()
    out["is_septic_shock"] = shock.astype(int)
    return out


def _build_metadata(cohort_df: pd.DataFrame, shock_df: pd.DataFrame) -> pd.DataFrame:
    first_shock_any = (
        shock_df.loc[shock_df["is_septic_shock"] == 1, ["stay_id", "hour"]]
        .groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "shock_first_hour_any"})
    )
    shock_after = shock_df.loc[shock_df["is_septic_shock"] == 1, ["stay_id", "hour"]].merge(
        cohort_df[["stay_id", "sepsis_onset_hour"]],
        on="stay_id",
        how="inner",
    )
    shock_after = shock_after.loc[shock_after["hour"] > shock_after["sepsis_onset_hour"], ["stay_id", "hour"]]
    first_shock_after = (
        shock_after.groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "shock_onset_hour"})
    )
    first_lactate = (
        shock_df.loc[shock_df["lactate"].notna(), ["stay_id", "hour"]]
        .groupby("stay_id", sort=False)["hour"]
        .min()
        .reset_index()
        .rename(columns={"hour": "first_lactate_hour"})
    )
    agg = shock_df.groupby("stay_id", sort=False).agg(
        n_hours=("hour", "size"),
        has_map=("map_used", lambda s: int(s.notna().any())),
        has_lactate=("lactate", lambda s: int(s.notna().any())),
        has_vasopressor=("vaso_active", lambda s: int((s.fillna(0) > 0).any())),
        has_sofa_total=("sofa_total", lambda s: int(s.notna().any())),
    ).reset_index()
    meta = (
        cohort_df.merge(first_shock_any, on="stay_id", how="left")
        .merge(first_shock_after, on="stay_id", how="left")
        .merge(agg, on="stay_id", how="left")
        .merge(first_lactate, on="stay_id", how="left")
    )
    meta["first_lactate_hour"] = pd.to_numeric(meta["first_lactate_hour"], errors="coerce")
    meta["shock_first_hour_any"] = pd.to_numeric(meta["shock_first_hour_any"], errors="coerce")
    meta["shock_onset_hour"] = pd.to_numeric(meta["shock_onset_hour"], errors="coerce")
    meta["has_any_shock"] = meta["shock_first_hour_any"].notna().astype(int)
    meta["has_progression_shock"] = meta["shock_onset_hour"].notna().astype(int)
    meta["shock_before_sepsis_onset"] = (
        meta["shock_first_hour_any"].notna() & (meta["shock_first_hour_any"] < meta["sepsis_onset_hour"])
    ).astype(int)
    meta["shock_at_sepsis_onset"] = (
        meta["shock_first_hour_any"].notna() & (meta["shock_first_hour_any"] == meta["sepsis_onset_hour"])
    ).astype(int)
    return meta.sort_values("stay_id", kind="mergesort").reset_index(drop=True)


def _build_primary_instances(meta_df: pd.DataFrame, prediction_interval: int, lookahead_hours: int, max_hour: int) -> pd.DataFrame:
    records: list[tuple] = []
    for row in meta_df.itertuples(index=False):
        onset = int(row.sepsis_onset_hour)
        shock_any_hour = float(row.shock_first_hour_any) if pd.notna(row.shock_first_hour_any) else math.inf
        shock_hour = float(row.shock_onset_hour) if pd.notna(row.shock_onset_hour) else math.inf
        upper_t = min(int(max_hour), int(shock_any_hour) - 1 if math.isfinite(shock_any_hour) else int(max_hour))
        t = onset + int(prediction_interval)
        while t <= upper_t:
            label = int(math.isfinite(shock_hour) and (shock_hour > t) and (shock_hour <= t + int(lookahead_hours)))
            records.append(
                (
                    "SEP-T1",
                    "sepsis",
                    int(row.stay_id),
                    int(t),
                    int(onset),
                    int(lookahead_hours),
                    int(label),
                    row.onset_source,
                    int(row.has_lactate) if pd.notna(row.has_lactate) else 0,
                    int(row.has_vasopressor) if pd.notna(row.has_vasopressor) else 0,
                    int(row.has_sofa_total) if pd.notna(row.has_sofa_total) else 0,
                    float(row.shock_onset_hour) if pd.notna(row.shock_onset_hour) else np.nan,
                    float(row.shock_first_hour_any) if pd.notna(row.shock_first_hour_any) else np.nan,
                )
            )
            t += int(prediction_interval)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "task_id",
            "condition",
            "stay_id",
            "prediction_hour",
            "sepsis_onset_hour",
            "horizon_hours",
            "label",
            "onset_source",
            "has_lactate",
            "has_vasopressor",
            "has_sofa_total",
            "shock_onset_hour",
            "shock_first_hour_any",
        ],
    )


def _build_secondary_instances(shock_df: pd.DataFrame, meta_df: pd.DataFrame, prediction_interval: int, clearance_window_hours: int, max_hour: int) -> pd.DataFrame:
    records: list[tuple] = []
    for stay_id, g in shock_df.groupby("stay_id", sort=False):
        meta = meta_df.loc[meta_df["stay_id"] == stay_id].iloc[0]
        onset = int(meta["sepsis_onset_hour"])
        lactate_series = g.set_index("hour")["lactate"].sort_index()
        t = onset + int(prediction_interval)
        upper_t = min(int(max_hour), int(g["hour"].max()))
        while t <= upper_t:
            baseline = lactate_series.get(t, np.nan)
            if pd.notna(baseline) and baseline > 0:
                future = lactate_series.loc[(lactate_series.index > t) & (lactate_series.index <= t + int(clearance_window_hours))]
                label = 0
                if not future.empty:
                    future_min = pd.to_numeric(future, errors="coerce").dropna()
                    if not future_min.empty:
                        label = int(float(future_min.min()) <= 0.9 * float(baseline))
                records.append(
                    (
                        "SEP-S1",
                        "sepsis",
                        int(stay_id),
                        int(t),
                        int(onset),
                        int(clearance_window_hours),
                        int(label),
                        float(baseline),
                        meta["onset_source"],
                    )
                )
            t += int(prediction_interval)
    return pd.DataFrame.from_records(
        records,
        columns=[
            "task_id",
            "condition",
            "stay_id",
            "prediction_hour",
            "sepsis_onset_hour",
            "horizon_hours",
            "label",
            "baseline_lactate",
            "onset_source",
        ],
    )


def _task_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "unique_stays": int(df["stay_id"].nunique()) if len(df) else 0,
        "positive_rate": float(df["label"].mean()) if len(df) else None,
        "stays_with_any_positive": int(df.loc[df["label"] == 1, "stay_id"].nunique()) if len(df) else 0,
    }


def main() -> None:
    args = parse_args()
    cohort_df = _build_sepsis_cohort(args.cohort_csv, args.diagnosis_pathway)
    print(f"[cohort] sepsis3_positive_stays={len(cohort_df)}")
    grid_df = _load_sepsis_grid(args.hourly_grid, set(cohort_df["stay_id"].tolist()))
    print(f"[grid] total_rows={len(grid_df)}")
    shock_df = _compute_shock(grid_df, map_threshold=args.map_threshold, lactate_threshold=args.lactate_threshold)
    print(f"[shock] shock_hours={int(shock_df['is_septic_shock'].sum())}")
    meta_df = _build_metadata(cohort_df, shock_df)
    print(f"[metadata] stays_with_any_shock={int(meta_df['has_any_shock'].sum())}")
    primary_df = _build_primary_instances(meta_df, args.prediction_interval, args.lookahead_hours, args.max_hour)
    print(f"[primary] rows={len(primary_df)}")
    secondary_df = _build_secondary_instances(shock_df, meta_df, args.prediction_interval, args.clearance_window_hours, args.max_hour)
    print(f"[secondary] rows={len(secondary_df)}")

    primary_stays = set(primary_df["stay_id"].tolist())
    early_shock_excluded = int(
        (
            meta_df["shock_first_hour_any"].notna()
            & (meta_df["shock_first_hour_any"] <= (meta_df["sepsis_onset_hour"] + int(args.prediction_interval)))
            & (~meta_df["stay_id"].isin(primary_stays))
        ).sum()
    )

    metadata_path = write_table(meta_df, args.metadata_out, index=False)
    shock_hourly_path = write_table(shock_df, args.shock_hourly_out, index=False)
    primary_path = write_table(primary_df, args.primary_out, index=False)
    secondary_path = write_table(secondary_df, args.secondary_out, index=False)

    onset_fallback = int((meta_df["onset_source"] == "fallback_zero").sum())
    shock_before = int(meta_df["shock_before_sepsis_onset"].sum())
    shock_at = int(meta_df["shock_at_sepsis_onset"].sum())
    summary = {
        "condition": "sepsis",
        "cohort": {
            "sepsis3_positive_stays": int(len(cohort_df)),
            "onset_from_pathway_stays": int((meta_df["onset_source"] == "diagnosis_pathway").sum()),
            "onset_fallback_zero_stays": onset_fallback,
            "onset_confidence_high_stays": int((meta_df["onset_confidence"] == "high").sum()),
            "onset_confidence_low_stays": int((meta_df["onset_confidence"] == "low").sum()),
            "stays_with_any_shock": int(meta_df["has_any_shock"].sum()),
            "stays_with_progression_shock": int(meta_df["has_progression_shock"].sum()),
            "shock_before_sepsis_onset_stays": shock_before,
            "shock_at_sepsis_onset_stays": shock_at,
            "primary_excluded_due_shock_at_or_before_first_anchor": early_shock_excluded,
        },
        "support_coverage": {
            "stays_with_map": int(meta_df["has_map"].sum()),
            "stays_with_lactate": int(meta_df["has_lactate"].sum()),
            "stays_with_vasopressor": int(meta_df["has_vasopressor"].sum()),
            "stays_with_sofa_total": int(meta_df["has_sofa_total"].sum()),
        },
        "tasks": {
            "SEP-T1": {
                **_task_summary(primary_df),
                "label_definition": "shock within (T, T+12]",
            },
            "SEP-S1": {
                **_task_summary(secondary_df),
                "label_definition": "lactate clearance >10% within (T, T+6]",
            },
        },
        "outputs": relativize_value(
            {
                "metadata": str(metadata_path),
                "shock_hourly": str(shock_hourly_path),
                "primary_instances": str(primary_path),
                "secondary_instances": str(secondary_path),
            },
            root=ROOT_DIR,
        ),
        "settings": {
            "prediction_interval": int(args.prediction_interval),
            "lookahead_hours": int(args.lookahead_hours),
            "clearance_window_hours": int(args.clearance_window_hours),
            "max_hour": int(args.max_hour),
            "map_threshold": float(args.map_threshold),
            "lactate_threshold": float(args.lactate_threshold),
        },
        "flags": (
            ([f"onset_fallback_zero_for_{onset_fallback}_stays"] if onset_fallback else [])
            + ([f"shock_before_sepsis_onset_for_{shock_before}_stays"] if shock_before else [])
            + ([f"shock_at_sepsis_onset_for_{shock_at}_stays"] if shock_at else [])
            + (
                [f"primary_excluded_due_shock_at_or_before_first_anchor_for_{early_shock_excluded}_stays"]
                if early_shock_excluded
                else []
            )
        ),
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
