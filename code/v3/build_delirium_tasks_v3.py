#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import ROOT_DIR, V3_HOURLY_FEATURES_DIR, V3_PROCESSED_DIR, V3_RESULTS_DIR  # type: ignore
from v3.io_utils import read_table, relativize_value, write_table  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 delirium Phase 3 task instances.")
    p.add_argument("--delirium-cohort", default=str(V3_PROCESSED_DIR / "delirium" / "delirium_cohort_v3.parquet"))
    p.add_argument("--delirium-labels", default=str(V3_PROCESSED_DIR / "delirium" / "delirium_labels_v3.parquet"))
    p.add_argument("--delirium-neuro-hourly", default=str(V3_HOURLY_FEATURES_DIR / "delirium_neuro_hourly_v3.parquet"))
    p.add_argument("--gcs-hourly", default=str(V3_HOURLY_FEATURES_DIR / "gcs_hourly_v3.parquet"))
    p.add_argument("--restraint-hourly", default=str(V3_HOURLY_FEATURES_DIR / "restraint_hourly_v3.parquet"))
    p.add_argument("--metadata-out", default=str(V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_stay_metadata.parquet"))
    p.add_argument("--persistence-out", default=str(V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_persistence_instances.parquet"))
    p.add_argument("--resolution-out", default=str(V3_PROCESSED_DIR / "delirium" / "tasks" / "delirium_resolution_instances.parquet"))
    p.add_argument("--summary-json", default=str(V3_RESULTS_DIR / "delirium" / "delirium_task_build_summary.json"))
    p.add_argument("--lookahead-hours", type=int, default=24)
    return p.parse_args()


def _support_flag(df: pd.DataFrame, stay_ids: set[int], value_col: str, positive_only: bool = False) -> pd.Series:
    frame = df.loc[df["stay_id"].isin(stay_ids), ["stay_id", value_col]].copy()
    if positive_only:
        frame = frame.loc[pd.to_numeric(frame[value_col], errors="coerce").fillna(0) > 0]
    else:
        frame = frame.loc[frame[value_col].notna()]
    if frame.empty:
        return pd.Series(dtype="int64")
    return frame.groupby("stay_id").size().rename(value_col).astype("int64")


def _build_metadata(cohort_df: pd.DataFrame, neuro_df: pd.DataFrame, gcs_df: pd.DataFrame, restraint_df: pd.DataFrame) -> pd.DataFrame:
    stay_ids = set(cohort_df["stay_id"].astype(int).tolist())
    metadata = cohort_df.copy()
    metadata["stay_id"] = metadata["stay_id"].astype(int)

    rass_counts = _support_flag(neuro_df, stay_ids, "rass", positive_only=False)
    gcs_counts = _support_flag(gcs_df, stay_ids, "gcs_total", positive_only=False)
    restraint_counts = _support_flag(restraint_df, stay_ids, "restraint_active", positive_only=True)

    metadata = metadata.merge(rass_counts.rename("n_rass_hours").reset_index(), on="stay_id", how="left")
    metadata = metadata.merge(gcs_counts.rename("n_gcs_hours").reset_index(), on="stay_id", how="left")
    metadata = metadata.merge(restraint_counts.rename("n_restraint_hours").reset_index(), on="stay_id", how="left")
    for col in ["n_rass_hours", "n_gcs_hours", "n_restraint_hours"]:
        metadata[col] = metadata[col].fillna(0).astype(int)
    metadata["has_rass_support"] = (metadata["n_rass_hours"] > 0).astype(int)
    metadata["has_gcs_support"] = (metadata["n_gcs_hours"] > 0).astype(int)
    metadata["has_restraint_support"] = (metadata["n_restraint_hours"] > 0).astype(int)
    return metadata.sort_values("stay_id", kind="mergesort").reset_index(drop=True)


def _build_task_instances(labels_df: pd.DataFrame, metadata_df: pd.DataFrame, label_col: str, task_id: str) -> pd.DataFrame:
    inst = labels_df.copy()
    inst["stay_id"] = inst["stay_id"].astype(int)
    inst["prediction_hour"] = inst["prediction_hour"].astype(int)
    inst["delirium_onset_hour"] = inst["delirium_onset_hour"].astype(int)
    inst["left_censored"] = inst["left_censored"].astype(int)
    inst["label"] = pd.to_numeric(inst[label_col], errors="coerce").fillna(0).astype(int)
    inst["task_id"] = task_id
    inst["condition"] = "delirium"
    inst["horizon_hours"] = 24
    inst["eligible"] = 1
    inst = inst.merge(
        metadata_df[
            [
                "stay_id",
                "n_positive_hours",
                "n_negative_hours",
                "n_uta_hours",
                "n_rass_hours",
                "n_gcs_hours",
                "n_restraint_hours",
                "has_rass_support",
                "has_gcs_support",
                "has_restraint_support",
            ]
        ],
        on="stay_id",
        how="left",
    )
    return inst[
        [
            "task_id",
            "condition",
            "stay_id",
            "prediction_hour",
            "delirium_onset_hour",
            "horizon_hours",
            "left_censored",
            "eligible",
            "label",
            "n_positive_hours",
            "n_negative_hours",
            "n_uta_hours",
            "n_rass_hours",
            "n_gcs_hours",
            "n_restraint_hours",
            "has_rass_support",
            "has_gcs_support",
            "has_restraint_support",
        ]
    ].sort_values(["stay_id", "prediction_hour"], kind="mergesort").reset_index(drop=True)


def _task_summary(df: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": int(len(df)),
        "unique_stays": int(df["stay_id"].nunique()),
        "positive_rate": float(df["label"].mean()) if len(df) else None,
        "stays_with_any_positive": int(df.loc[df["label"] == 1, "stay_id"].nunique()),
        "left_censored_rate": float(df["left_censored"].mean()) if len(df) else None,
    }


def main() -> None:
    args = parse_args()
    cohort_df = read_table(args.delirium_cohort)
    labels_df = read_table(args.delirium_labels)
    neuro_df = read_table(args.delirium_neuro_hourly)
    gcs_df = read_table(args.gcs_hourly)
    restraint_df = read_table(args.restraint_hourly)

    metadata_df = _build_metadata(cohort_df, neuro_df, gcs_df, restraint_df)
    persistence_df = _build_task_instances(labels_df, metadata_df, "label_persistent_delirium", "DEL-T1")
    resolution_df = _build_task_instances(labels_df, metadata_df, "label_resolution", "DEL-S1")

    metadata_path = write_table(metadata_df, args.metadata_out, index=False)
    persistence_path = write_table(persistence_df, args.persistence_out, index=False)
    resolution_path = write_table(resolution_df, args.resolution_out, index=False)

    summary = {
        "condition": "delirium",
        "cohort": {
            "delirium_cohort_stays": int(len(cohort_df)),
            "left_censored_rate": float(cohort_df["left_censored"].mean()) if len(cohort_df) else None,
            "median_onset_hour": float(cohort_df["delirium_onset_hour"].median()) if len(cohort_df) else None,
            "median_positive_hours": float(cohort_df["n_positive_hours"].median()) if len(cohort_df) else None,
        },
        "support_coverage": {
            "stays_with_rass_support": int(metadata_df.loc[metadata_df["has_rass_support"] == 1, "stay_id"].nunique()),
            "stays_with_gcs_support": int(metadata_df.loc[metadata_df["has_gcs_support"] == 1, "stay_id"].nunique()),
            "stays_with_restraint_support": int(metadata_df.loc[metadata_df["has_restraint_support"] == 1, "stay_id"].nunique()),
        },
        "tasks": {
            "DEL-T1": {
                **_task_summary(persistence_df),
                "label_definition": "persistent delirium within (T, T+24]",
            },
            "DEL-S1": {
                **_task_summary(resolution_df),
                "label_definition": "resolution within (T, T+24] via sustained 24h non-positive window",
            },
        },
        "outputs": relativize_value(
            {
                "metadata": str(metadata_path),
                "persistence_instances": str(persistence_path),
                "resolution_instances": str(resolution_path),
            },
            root=ROOT_DIR,
        ),
        "settings": {
            "lookahead_hours": int(args.lookahead_hours),
        },
        "flags": [],
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
