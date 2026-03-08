from __future__ import annotations

import pandas as pd

from common import RESULTS_DIR, TABLES_DIR, find_one, load_experiments, load_raw_records

TASKS = ["mortality", "prolonged_los"]
WINDOWS_CORE = ["D0", "W6", "W12", "W24"]


def _model_label(model: str) -> str:
    return {"lr": "LR", "xgb": "XGBoost"}.get(model, model)


def _round_df(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].astype(float).round(4)
    return out


def build_table1(records):
    rows = []
    for task in TASKS:
        for model in ["lr", "xgb"]:
            row = {"task": task, "model": _model_label(model)}
            for w in WINDOWS_CORE:
                rec = find_one(records, task=task, modality="structured", model=model, window=w, text_method="original")
                row[w] = rec.data["mean_auroc"]
            rows.append(row)
    df = _round_df(pd.DataFrame(rows), ["D0", "W6", "W12", "W24"])
    path = TABLES_DIR / "table1_structured_baselines.csv"
    df.to_csv(path, index=False)
    return path, df


def build_table2():
    src = RESULTS_DIR / "leakage_premium_decomposition.csv"
    df = pd.read_csv(src)
    df["struct_share_pct"] = (df["premium_struct_B_minus_D"] / df["premium_total_A_minus_D"]) * 100.0
    df["text_share_pct"] = (df["premium_text_C_minus_D"] / df["premium_total_A_minus_D"]) * 100.0
    keep = [
        "task",
        "A_full_leaked",
        "B_struct_only_leak",
        "C_text_only_leak",
        "D_clean",
        "premium_total_A_minus_D",
        "premium_struct_B_minus_D",
        "premium_text_C_minus_D",
        "premium_interaction",
        "struct_share_pct",
        "text_share_pct",
    ]
    out = _round_df(df[keep], [
        "A_full_leaked",
        "B_struct_only_leak",
        "C_text_only_leak",
        "D_clean",
        "premium_total_A_minus_D",
        "premium_struct_B_minus_D",
        "premium_text_C_minus_D",
        "premium_interaction",
        "struct_share_pct",
        "text_share_pct",
    ])
    path = TABLES_DIR / "table2_leakage_decomposition.csv"
    out.to_csv(path, index=False)
    return path, out


def build_table3(records):
    rows = []
    method_map = {"mean": "original", "typed": "original_typed"}
    for task in TASKS:
        for text_type, text_method in method_map.items():
            row = {"task": task, "text_type": text_type}
            for w in ["D0", "W6", "W12", "W24", "leaked", "clean"]:
                rec = find_one(
                    records,
                    task=task,
                    modality="text_only",
                    model="lr",
                    window=w,
                    text_method=text_method,
                )
                row[w] = rec.data["mean_auroc"]
            rows.append(row)
    df = _round_df(pd.DataFrame(rows), ["D0", "W6", "W12", "W24", "leaked", "clean"])
    path = TABLES_DIR / "table3_text_baselines.csv"
    df.to_csv(path, index=False)
    return path, df


def build_table4(records):
    tabular = find_one(
        records,
        task="mortality",
        modality="structured",
        model="xgb",
        window="W24",
        text_method="original",
    ).data["mean_auroc"]

    nursing = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="W24",
        text_method="original_typed",
        note_ablation="nursing",
    ).data["mean_auroc"]

    radiology = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="W24",
        text_method="original_typed",
        note_ablation="radiology",
    ).data["mean_auroc"]

    lab = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="W24",
        text_method="original_typed",
        note_ablation="lab",
    ).data["mean_auroc"]

    all_typed = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="W24",
        text_method="original_typed",
    ).data["mean_auroc"]

    all_mean = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="W24",
        text_method="original",
    ).data["mean_auroc"]

    rows = [
        ("No text (tabular only)", tabular),
        ("Nursing only", nursing),
        ("Radiology only", radiology),
        ("Lab only", lab),
        ("All notes (typed pool)", all_typed),
        ("All notes (mean pool)", all_mean),
    ]

    out_rows = []
    for condition, val in rows:
        if condition == "No text (tabular only)":
            delta = "-"
        else:
            delta = f"{val - tabular:+.4f}"
        out_rows.append({
            "condition": condition,
            "mortality_auroc": round(float(val), 4),
            "delta_vs_tabular": delta,
        })

    df = pd.DataFrame(out_rows)
    path = TABLES_DIR / "table4_note_ablation.csv"
    df.to_csv(path, index=False)
    return path, df


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    raw_records = load_raw_records()
    records = load_experiments(deduplicate=True)
    print(f"raw_json={len(raw_records)} dedup_json={len(records)}")

    p1, t1 = build_table1(records)
    p2, t2 = build_table2()
    p3, t3 = build_table3(records)
    p4, t4 = build_table4(records)

    print("written:")
    print(p1)
    print(p2)
    print(p3)
    print(p4)

    print("\nTable 1 preview")
    print(t1.to_string(index=False))
    print("\nTable 2 preview")
    print(t2.to_string(index=False))
    print("\nTable 3 preview")
    print(t3.to_string(index=False))
    print("\nTable 4 preview")
    print(t4.to_string(index=False))


if __name__ == "__main__":
    main()
