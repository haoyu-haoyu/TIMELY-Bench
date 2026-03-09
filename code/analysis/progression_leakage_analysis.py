#!/usr/bin/env python3
"""
Progression-task leakage decomposition (Phase C helper).

Primary text leakage definition:
  premium_text = C - D
where
  C = W24 struct + original text
  D = W24 struct + clean text
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
OUTDIR = ROOT_DIR / "results" / "note_centered" / "progression_tasks"


def load_auroc(task: str, modality: str, model: str, window: str, text_method: str) -> float:
    pattern = OUTDIR / f"{task}_{modality}_{model}_{window}_{text_method}.json"
    files = glob.glob(str(pattern))
    if not files:
        return float("nan")
    with open(files[0], "r", encoding="utf-8") as f:
        return float(json.load(f)["mean_auroc"])


def compute_task_row(task: str) -> Dict[str, float]:
    A = load_auroc(task, "fusion", "xgb", "leaked", "original")
    B = load_auroc(task, "fusion", "xgb", "leaked", "weighted_no_after")
    C = load_auroc(task, "fusion", "xgb", "W24", "original")
    D = load_auroc(task, "fusion", "xgb", "clean", "weighted_no_after")

    premium_total = A - D
    premium_struct = B - D
    premium_text_cd = C - D
    premium_text_ab = A - B
    interaction = A - B - C + D
    text_share = premium_text_cd / premium_total * 100 if abs(premium_total) > 1e-6 else 0.0

    return {
        "task": task,
        "A_both_leaked": round(A, 4),
        "B_struct_only": round(B, 4),
        "C_text_only": round(C, 4),
        "D_clean": round(D, 4),
        "premium_total": round(premium_total, 4),
        "premium_struct": round(premium_struct, 4),
        "premium_text_C_D": round(premium_text_cd, 4),
        "premium_text_A_B": round(premium_text_ab, 4),
        "text_share_pct": round(text_share, 1),
        "interaction": round(interaction, 4),
    }


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []

    # WP1 reference rows
    rows.extend(
        [
            {
                "task": "mortality",
                "A_both_leaked": 0.9232,
                "B_struct_only": 0.9231,
                "C_text_only": 0.9079,
                "D_clean": 0.9079,
                "premium_total": 0.0154,
                "premium_struct": 0.0153,
                "premium_text_C_D": 0.0,
                "premium_text_A_B": -0.0,
                "text_share_pct": 0.0,
                "interaction": 0.0001,
            },
            {
                "task": "prolonged_los",
                "A_both_leaked": 0.9368,
                "B_struct_only": 0.9370,
                "C_text_only": 0.8856,
                "D_clean": 0.8860,
                "premium_total": 0.0508,
                "premium_struct": 0.0510,
                "premium_text_C_D": -0.0004,
                "premium_text_A_B": -0.0004,
                "text_share_pct": 0.0,
                "interaction": 0.0002,
            },
        ]
    )

    # Progression tasks
    for task in ["aki_progression", "sepsis_shock"]:
        rows.append(compute_task_row(task))

    out_df = pd.DataFrame(rows)
    out_csv = OUTDIR / "cross_task_leakage_decomposition.csv"
    out_df.to_csv(out_csv, index=False)

    print("=" * 80)
    print("CROSS-TASK LEAKAGE DECOMPOSITION (premium_text = C - D)")
    print("=" * 80)
    cols = ["task", "D_clean", "premium_total", "premium_struct", "premium_text_C_D", "text_share_pct"]
    print(out_df[cols].to_string(index=False))
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
