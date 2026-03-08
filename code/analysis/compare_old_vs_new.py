from __future__ import annotations

import pandas as pd

from common import TABLES_DIR, find_one, load_experiments


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    records = load_experiments(deduplicate=True)

    new_struct_lr = find_one(records, task="mortality", modality="structured", model="lr", window="W24", text_method="original").data["mean_auroc"]
    new_struct_xgb = find_one(records, task="mortality", modality="structured", model="xgb", window="W24", text_method="original").data["mean_auroc"]
    new_text_lr = find_one(records, task="mortality", modality="text_only", model="lr", window="W24", text_method="original").data["mean_auroc"]
    new_early_xgb = find_one(records, task="mortality", modality="fusion", fusion_strategy="early", model="xgb", window="W24", text_method="original").data["mean_auroc"]
    new_late_lr = find_one(records, task="mortality", modality="fusion", fusion_strategy="late_stacking", model="lr", window="W24", text_method="original").data["mean_auroc"]

    rows = [
        {
            "model": "Structured LR",
            "old_auroc": 0.8481,
            "new_auroc": new_struct_lr,
            "explanation": "Change reflects stronger 42-feature set and note-centered alignment protocol.",
        },
        {
            "model": "Structured XGBoost",
            "old_auroc": 0.8677,
            "new_auroc": new_struct_xgb,
            "explanation": "+0.036 largely attributable to expanded 42-feature set (SOFA, blood gas, ventilator, vasopressor dose, sedatives).",
        },
        {
            "model": "Text ClinicalBERT LR",
            "old_auroc": 0.8318,
            "new_auroc": new_text_lr,
            "explanation": "Text baseline is now note-centered lookback W24; direct comparability remains partial.",
        },
        {
            "model": "Early Fusion XGBoost",
            "old_auroc": 0.8848,
            "new_auroc": new_early_xgb,
            "explanation": "+0.023 from combined effects of feature expansion and alignment redesign.",
        },
        {
            "model": "Late Fusion (stacking)",
            "old_auroc": 0.8805,
            "new_auroc": new_late_lr,
            "explanation": "Late stacking improves under note-centered setup but gains are smaller than early fusion.",
        },
        {
            "model": "GRU",
            "old_auroc": 0.8419,
            "new_auroc": pd.NA,
            "explanation": "Not re-run in note-centered Phase 4; no directly matched new value in this release.",
        },
    ]

    df = pd.DataFrame(rows)
    df["delta"] = df["new_auroc"].astype("Float64") - df["old_auroc"]
    out = TABLES_DIR / "table5_old_vs_new_comparison.csv"
    df.to_csv(out, index=False)

    print(out)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
