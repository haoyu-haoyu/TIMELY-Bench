from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from common import ANALYSIS_DIR, RESULTS_DIR, ROOT_DIR, find_one, load_experiments


W24_PARQUET = ROOT_DIR / "data" / "processed" / "note_centered" / "note_window_structured_W24.parquet"
D0_PARQUET = ROOT_DIR / "data" / "processed" / "note_centered" / "note_window_structured_D0.parquet"


def _aggregate_by_bins(parquet_path: Path, bucket_col: str, bins: list[float], labels: list[str], *, include_n_measure=False):
    schema = pq.read_schema(parquet_path)
    missing_cols = [c for c in schema.names if c.endswith("_missing_rate")]
    n_measure_cols = [c for c in schema.names if c.endswith("_n_measurements")]

    cols = [bucket_col] + missing_cols
    if include_n_measure:
        cols += n_measure_cols

    stats = {label: {"count": 0, "sum_comp": 0.0, "sum_n_measure": 0.0} for label in labels}

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=cols, batch_size=100_000):
        df = batch.to_pandas()
        bucket_values = df[bucket_col].to_numpy(dtype=np.float64, copy=False)
        missing_matrix = df[missing_cols].to_numpy(dtype=np.float32, copy=False)
        completeness = 1.0 - np.nanmean(missing_matrix, axis=1)

        if include_n_measure:
            n_measure_matrix = df[n_measure_cols].to_numpy(dtype=np.float32, copy=False)
            n_measure_mean = np.nanmean(n_measure_matrix, axis=1)
        else:
            n_measure_mean = None

        idx = np.digitize(bucket_values, bins, right=False) - 1
        for i, label in enumerate(labels):
            mask = idx == i
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            stats[label]["count"] += cnt
            stats[label]["sum_comp"] += float(completeness[mask].sum())
            if include_n_measure and n_measure_mean is not None:
                stats[label]["sum_n_measure"] += float(n_measure_mean[mask].sum())

    total = sum(v["count"] for v in stats.values())
    rows = []
    for label in labels:
        count = stats[label]["count"]
        row = {
            "bucket": label,
            "n_notes": count,
            "pct_notes": (100.0 * count / total) if total else 0.0,
            "mean_feature_completeness": (stats[label]["sum_comp"] / count) if count else math.nan,
        }
        if include_n_measure:
            row["mean_n_measurements"] = (stats[label]["sum_n_measure"] / count) if count else math.nan
        rows.append(row)

    return pd.DataFrame(rows)


def build_truncation_analysis() -> Path:
    bins = [0, 6, 12, 18, 24, 48]
    labels = ["0-6h", "6-12h", "12-18h", "18-24h", "24-48h"]
    df = _aggregate_by_bins(W24_PARQUET, "chart_hour", bins, labels, include_n_measure=False)
    out = ANALYSIS_DIR / "truncation_analysis.csv"
    df.to_csv(out, index=False)
    return out


def build_d0_boundary_analysis() -> Path:
    bins = [0, 2, 6, 12, 24]
    labels = ["0-2h", "2-6h", "6-12h", "12-24h"]
    df = _aggregate_by_bins(D0_PARQUET, "window_hours_actual", bins, labels, include_n_measure=True)
    out = ANALYSIS_DIR / "d0_boundary_analysis.csv"
    df.to_csv(out, index=False)
    return out


def write_findings_md(records) -> Path:
    tables_dir = RESULTS_DIR / "tables"
    t1 = pd.read_csv(tables_dir / "table1_structured_baselines.csv")
    t2 = pd.read_csv(tables_dir / "table2_leakage_decomposition.csv")
    t3 = pd.read_csv(tables_dir / "table3_text_baselines.csv")
    t4 = pd.read_csv(tables_dir / "table4_note_ablation.csv")

    trunc = pd.read_csv(ANALYSIS_DIR / "truncation_analysis.csv")
    d0 = pd.read_csv(ANALYSIS_DIR / "d0_boundary_analysis.csv")

    mort_struct_xgb = t1[(t1.task == "mortality") & (t1.model == "XGBoost")].iloc[0]
    los_struct_xgb = t1[(t1.task == "prolonged_los") & (t1.model == "XGBoost")].iloc[0]

    los_text_mean = t3[(t3.task == "prolonged_los") & (t3.text_type == "mean")].iloc[0]
    los_text_typed = t3[(t3.task == "prolonged_los") & (t3.text_type == "typed")].iloc[0]

    mort_dec = t2[t2.task == "mortality"].iloc[0]
    los_dec = t2[t2.task == "prolonged_los"].iloc[0]

    tabular = float(t4[t4.condition == "No text (tabular only)"]["mortality_auroc"].iloc[0])
    all_mean = float(t4[t4.condition == "All notes (mean pool)"]["mortality_auroc"].iloc[0])

    text_clean_mean = t3[(t3.task == "mortality") & (t3.text_type == "mean")]["clean"].iloc[0]
    text_clean_typed = t3[(t3.task == "mortality") & (t3.text_type == "typed")]["clean"].iloc[0]

    late_clean_mean = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="late_stacking",
        model="lr",
        window="clean",
        text_method="original",
    ).data["mean_auroc"]
    late_clean_typed = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="late_stacking",
        model="lr",
        window="clean",
        text_method="original_typed",
    ).data["mean_auroc"]

    mort_clean = find_one(
        records,
        task="mortality",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="clean",
        text_method="weighted_no_after",
    ).data["mean_auroc"]
    los_clean = find_one(
        records,
        task="prolonged_los",
        modality="fusion",
        fusion_strategy="early",
        model="xgb",
        window="clean",
        text_method="weighted_no_after",
    ).data["mean_auroc"]

    severe_d0 = d0[d0.bucket == "0-2h"].iloc[0]

    md = f"""# Phase 5 Analysis Findings

## Q1. Window Effect
- Mortality structured XGBoost shows monotonic lookback behavior: W6={mort_struct_xgb['W6']:.4f} < W12={mort_struct_xgb['W12']:.4f} < W24={mort_struct_xgb['W24']:.4f}.
- Mortality D0 remains stronger than W6: D0={mort_struct_xgb['D0']:.4f} vs W6={mort_struct_xgb['W6']:.4f}.
- LOS structured XGBoost is flat across lookback windows and best at D0: D0={los_struct_xgb['D0']:.4f}, W6={los_struct_xgb['W6']:.4f}, W12={los_struct_xgb['W12']:.4f}, W24={los_struct_xgb['W24']:.4f}.
- LOS text-only has anti-monotonic behavior: mean W6={los_text_mean['W6']:.4f} > W24={los_text_mean['W24']:.4f} (delta={los_text_mean['W24']-los_text_mean['W6']:+.4f}); typed W6={los_text_typed['W6']:.4f} > W24={los_text_typed['W24']:.4f} (delta={los_text_typed['W24']-los_text_typed['W6']:+.4f}).

Interpretation: mortality benefits from longer context, while LOS appears trajectory-type dominated with weaker gain from longer lookback.

## Q2. Leakage Decomposition
- Mortality: A={mort_dec['A_full_leaked']:.4f}, B={mort_dec['B_struct_only_leak']:.4f}, C={mort_dec['C_text_only_leak']:.4f}, D={mort_dec['D_clean']:.4f}.
  premium_total={mort_dec['premium_total_A_minus_D']:+.4f}, premium_struct={mort_dec['premium_struct_B_minus_D']:+.4f} ({mort_dec['struct_share_pct']:.1f}%), premium_text={mort_dec['premium_text_C_minus_D']:+.4f}, interaction={mort_dec['premium_interaction']:+.4f}.
- Prolonged LOS: A={los_dec['A_full_leaked']:.4f}, B={los_dec['B_struct_only_leak']:.4f}, C={los_dec['C_text_only_leak']:.4f}, D={los_dec['D_clean']:.4f}.
  premium_total={los_dec['premium_total_A_minus_D']:+.4f}, premium_struct={los_dec['premium_struct_B_minus_D']:+.4f} ({los_dec['struct_share_pct']:.1f}%), premium_text={los_dec['premium_text_C_minus_D']:+.4f}, interaction={los_dec['premium_interaction']:+.4f}.

Interpretation: leakage premium is dominated by structural leakage; text AFTER leakage is approximately zero with note-level ClinicalBERT pooling.

## Q3. Note-Type Contribution
- No text baseline (tabular only): {tabular:.4f}
- Nursing only: {float(t4[t4.condition=='Nursing only']['mortality_auroc'].iloc[0]):.4f}
- Radiology only: {float(t4[t4.condition=='Radiology only']['mortality_auroc'].iloc[0]):.4f}
- Lab only: {float(t4[t4.condition=='Lab only']['mortality_auroc'].iloc[0]):.4f}
- All typed: {float(t4[t4.condition=='All notes (typed pool)']['mortality_auroc'].iloc[0]):.4f}
- All mean: {all_mean:.4f}

Interpretation: text adds only marginal AUROC over the strong 42-feature structured baseline.

## Q4. Typed vs Mean Pooling
- Mortality text-only clean: mean={text_clean_mean:.4f}, typed={text_clean_typed:.4f}, delta={text_clean_typed-text_clean_mean:+.4f}.
- Mortality late fusion clean: mean={late_clean_mean:.4f}, typed={late_clean_typed:.4f}, delta={late_clean_typed-late_clean_mean:+.4f}.
- Mortality all-notes ablation: mean={all_mean:.4f}, typed={float(t4[t4.condition=='All notes (typed pool)']['mortality_auroc'].iloc[0]):.4f}, delta={float(t4[t4.condition=='All notes (typed pool)']['mortality_auroc'].iloc[0])-all_mean:+.4f}.

Interpretation: mean pooling is equal or better than typed pooling in this setup, suggesting dimensionality overhead without measurable gain.

## Q5. LOS vs Mortality Comparison
- Clean early fusion AUROC: mortality={mort_clean:.4f}, prolonged_los={los_clean:.4f}.
- Mortality text marginal gain (all mean vs no text): {all_mean - tabular:+.4f}.

Interpretation: text contribution remains small for both tasks once 42 structured variables are available.

## Q6. Truncation Impact (W24)
Source: `truncation_analysis.csv`

{trunc.to_string(index=False)}

Interpretation: feature completeness varies only mildly across chart-hour buckets, indicating truncation does not induce severe quality collapse for note-window features.

## Q7. D0 Boundary Effect
Source: `d0_boundary_analysis.csv`

{d0.to_string(index=False)}

Interpretation: severe D0 truncation (<2h) accounts for {severe_d0['pct_notes']:.2f}% of notes in this calculation and does not prevent D0 from being competitive (or best on LOS structured baselines).
"""

    out = ANALYSIS_DIR / "analysis_findings.md"
    out.write_text(md, encoding="utf-8")
    return out


def main():
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    trunc_path = ANALYSIS_DIR / "truncation_analysis.csv"
    d0_path = ANALYSIS_DIR / "d0_boundary_analysis.csv"

    if trunc_path.exists():
        print(f"Using existing {trunc_path}")
    else:
        print(f"Computing truncation analysis from {W24_PARQUET} ...")
        trunc_path = build_truncation_analysis()
        print(trunc_path)

    if d0_path.exists():
        print(f"Using existing {d0_path}")
    else:
        print(f"Computing D0 boundary analysis from {D0_PARQUET} ...")
        d0_path = build_d0_boundary_analysis()
        print(d0_path)

    records = load_experiments(deduplicate=True)
    md_path = write_findings_md(records)
    print(md_path)


if __name__ == "__main__":
    main()
