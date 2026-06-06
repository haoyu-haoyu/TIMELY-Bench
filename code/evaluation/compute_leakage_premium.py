#!/usr/bin/env python3
"""Compute 2x2 leakage premium decomposition from experiment JSON results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute leakage premium decomposition (A/B/C/D).")
    parser.add_argument(
        "--results_dir",
        type=Path,
        default=Path("results/note_centered/core_experiments"),
        help="Directory that contains experiment result JSON files.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=Path("results/note_centered/leakage_premium_decomposition.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="mean_auroc",
        choices=["mean_auroc", "mean_auprc", "mean_ece"],
        help="Metric used for decomposition.",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="fusion",
        help="Optional modality filter (default: fusion). Use empty string to disable.",
    )
    parser.add_argument(
        "--fusion_strategy",
        type=str,
        default="early",
        help="Optional fusion strategy filter (default: early). Use empty string to disable.",
    )
    return parser.parse_args()


def iter_result_records(results_dir: Path) -> Iterable[Dict]:
    for path in sorted(results_dir.rglob("*.json")):
        try:
            with path.open("r") as fh:
                obj = json.load(fh)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        obj["_file"] = str(path)
        yield obj


def to_dataframe(records: Iterable[Dict]) -> pd.DataFrame:
    rows: List[Dict] = []
    for rec in records:
        rows.append(
            {
                "task": rec.get("task"),
                "model": rec.get("model"),
                "modality": rec.get("modality"),
                "fusion_strategy": rec.get("fusion_strategy", "none"),
                "window": rec.get("window"),
                "text_method": rec.get("text_method"),
                "resolved_text_method": rec.get("resolved_text_method", rec.get("text_method")),
                "note_ablation": rec.get("note_ablation", "none"),
                "mean_auroc": rec.get("mean_auroc"),
                "mean_auprc": rec.get("mean_auprc"),
                "mean_ece": rec.get("mean_ece"),
                "split_source": rec.get("split_source"),
                "_file": rec.get("_file"),
            }
        )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("mean_auroc", "mean_auprc", "mean_ece"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def cell_value(group: pd.DataFrame, metric: str, window: str, text_method: str | None = None) -> float | None:
    q = group[group["window"] == window]
    if text_method is not None:
        q = q[q["resolved_text_method"] == text_method]
    if q.empty:
        return None
    return float(q[metric].mean())


def format_num(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x:.4f}"


def main() -> None:
    args = parse_args()
    if not args.results_dir.exists():
        raise FileNotFoundError(f"results_dir does not exist: {args.results_dir}")

    df = to_dataframe(iter_result_records(args.results_dir))
    if df.empty:
        raise RuntimeError(f"No JSON result records found under: {args.results_dir}")

    # Only compare standard settings; ablation runs are analyzed separately.
    df = df[df["note_ablation"].fillna("none") == "none"].copy()

    if args.modality:
        df = df[df["modality"] == args.modality].copy()
    if args.fusion_strategy:
        df = df[df["fusion_strategy"] == args.fusion_strategy].copy()

    if df.empty:
        raise RuntimeError("No records left after filtering.")

    out_rows: List[Dict] = []
    group_cols = ["task", "model", "modality", "fusion_strategy"]
    for keys, group in df.groupby(group_cols, dropna=False):
        a_val = cell_value(group, args.metric, "leaked", "original")
        b_val = cell_value(group, args.metric, "leaked", "weighted_no_after")
        c_val = cell_value(group, args.metric, "W24", "original")
        d_val = cell_value(group, args.metric, "clean", None)

        missing = []
        if a_val is None:
            missing.append("A(leaked,original)")
        if b_val is None:
            missing.append("B(leaked,weighted_no_after)")
        if c_val is None:
            missing.append("C(W24,original)")
        if d_val is None:
            missing.append("D(clean)")

        row = {
            "task": keys[0],
            "model": keys[1],
            "modality": keys[2],
            "fusion_strategy": keys[3],
            "metric": args.metric,
            "A_full_leaked": a_val,
            "B_struct_only_leak": b_val,
            "C_text_only_leak": c_val,
            "D_clean": d_val,
            "premium_total_A_minus_D": None if a_val is None or d_val is None else a_val - d_val,
            "premium_struct_B_minus_D": None if b_val is None or d_val is None else b_val - d_val,
            "premium_text_C_minus_D": None if c_val is None or d_val is None else c_val - d_val,
            "premium_interaction": None,
            "missing_cells": ";".join(missing),
            "n_records_in_group": int(len(group)),
        }
        if (
            row["premium_total_A_minus_D"] is not None
            and row["premium_struct_B_minus_D"] is not None
            and row["premium_text_C_minus_D"] is not None
        ):
            row["premium_interaction"] = (
                row["premium_total_A_minus_D"]
                - row["premium_struct_B_minus_D"]
                - row["premium_text_C_minus_D"]
            )
        out_rows.append(row)

    out_df = pd.DataFrame(out_rows).sort_values(["task", "model"]).reset_index(drop=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output_csv, index=False)

    print(f"saved_csv={args.output_csv}")
    print(f"rows={len(out_df)}")
    for _, row in out_df.iterrows():
        print(
            f"[{row['task']}/{row['model']}] "
            f"A={format_num(row['A_full_leaked'])} "
            f"B={format_num(row['B_struct_only_leak'])} "
            f"C={format_num(row['C_text_only_leak'])} "
            f"D={format_num(row['D_clean'])} "
            f"total={format_num(row['premium_total_A_minus_D'])} "
            f"struct={format_num(row['premium_struct_B_minus_D'])} "
            f"text={format_num(row['premium_text_C_minus_D'])} "
            f"inter={format_num(row['premium_interaction'])} "
            f"missing={row['missing_cells'] or 'none'}"
        )


if __name__ == "__main__":
    main()
