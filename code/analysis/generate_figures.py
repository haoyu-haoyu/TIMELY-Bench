from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from common import FIGURES_DIR, RESULTS_DIR, find_one, load_experiments

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "sans-serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    }
)


def _auc(records, *, task, modality, model, window, text_method, fusion_strategy=None, note_ablation=None):
    rec = find_one(
        records,
        task=task,
        modality=modality,
        model=model,
        window=window,
        text_method=text_method,
        fusion_strategy=fusion_strategy,
        note_ablation=note_ablation,
    )
    return float(rec.data["mean_auroc"])


def fig1_robustness(records):
    tasks = ["mortality", "prolonged_los"]
    x_labels = ["D0", "W6", "W12", "W24", "leaked"]
    x = np.arange(len(x_labels))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=False)
    for ax, task in zip(axes, tasks):
        structured = [_auc(records, task=task, modality="structured", model="xgb", window=w, text_method="original") for w in x_labels[:-1]] + [np.nan]
        text_mean = [_auc(records, task=task, modality="text_only", model="lr", window=w, text_method="original") for w in x_labels]
        early_orig = [_auc(records, task=task, modality="fusion", fusion_strategy="early", model="xgb", window=w, text_method="original") for w in x_labels]
        late_orig = [
            _auc(records, task=task, modality="fusion", fusion_strategy="late_stacking", model="lr", window="D0", text_method="original"),
            np.nan,
            np.nan,
            _auc(records, task=task, modality="fusion", fusion_strategy="late_stacking", model="lr", window="W24", text_method="original"),
            _auc(records, task=task, modality="fusion", fusion_strategy="late_stacking", model="lr", window="leaked", text_method="original"),
        ]

        clean_ref = _auc(records, task=task, modality="fusion", fusion_strategy="early", model="xgb", window="clean", text_method="weighted_no_after")
        leaked_val = early_orig[-1]
        premium = leaked_val - clean_ref

        ax.plot(x, structured, marker="o", lw=2, label="Structured XGBoost")
        ax.plot(x, text_mean, marker="o", lw=2, label="Text-only mean")
        ax.plot(x, early_orig, marker="o", lw=2, label="Early fusion original")
        ax.plot(x, late_orig, marker="o", lw=2, label="Late stacking original")
        ax.axhline(clean_ref, color="tab:red", ls="--", lw=1.5, label="Early fusion clean")

        ax.annotate(
            f"Leakage Premium = {premium:+.4f}",
            xy=(x[-1], leaked_val),
            xytext=(x[-1] - 1.6, leaked_val + 0.01),
            arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "black"},
        )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=45, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_title("Mortality" if task == "mortality" else "Prolonged LOS")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=5,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    out = FIGURES_DIR / "fig1_robustness_curves.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig2_heatmap():
    decomp = pd.read_csv(RESULTS_DIR / "leakage_premium_decomposition.csv")
    tasks = ["mortality", "prolonged_los"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        row = decomp.loc[decomp["task"] == task].iloc[0]
        A = float(row["A_full_leaked"])
        B = float(row["B_struct_only_leak"])
        C = float(row["C_text_only_leak"])
        D = float(row["D_clean"])
        matrix = np.array([[D, C], [B, A]])

        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Text: weighted_no_after", "Text: original"], rotation=15, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Struct: W24", "Struct: leaked"])
        ax.set_title("Mortality" if task == "mortality" else "Prolonged LOS")

        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:.4f}", ha="center", va="center", color="black")

        txt = (
            f"Struct={row['premium_struct_B_minus_D']:+.4f} | "
            f"Text={row['premium_text_C_minus_D']:+.4f} | "
            f"Interact={row['premium_interaction']:+.4f}"
        )
        ax.text(0.5, -0.28, txt, transform=ax.transAxes, ha="center", va="top", fontsize=9)

    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="AUROC")
    out = FIGURES_DIR / "fig2_leakage_heatmap.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig3_ablation(records):
    conditions = [
        ("No text", _auc(records, task="mortality", modality="structured", model="xgb", window="W24", text_method="original")),
        ("Nursing", _auc(records, task="mortality", modality="fusion", fusion_strategy="early", model="xgb", window="W24", text_method="original_typed", note_ablation="nursing")),
        ("Radiology", _auc(records, task="mortality", modality="fusion", fusion_strategy="early", model="xgb", window="W24", text_method="original_typed", note_ablation="radiology")),
        ("Lab", _auc(records, task="mortality", modality="fusion", fusion_strategy="early", model="xgb", window="W24", text_method="original_typed", note_ablation="lab")),
        ("All typed", _auc(records, task="mortality", modality="fusion", fusion_strategy="early", model="xgb", window="W24", text_method="original_typed")),
        ("All mean", _auc(records, task="mortality", modality="fusion", fusion_strategy="early", model="xgb", window="W24", text_method="original")),
    ]

    names = [x[0] for x in conditions]
    vals = [x[1] for x in conditions]

    fig, ax = plt.subplots(figsize=(10, 4.8), constrained_layout=True)
    x = np.arange(len(names))
    colors = ["#4e79a7", "#59a14f", "#f28e2b", "#e15759", "#76b7b2", "#edc948"]
    ax.scatter(x, vals, s=80, c=colors, zorder=3)
    ax.plot(x, vals, color="#9c9c9c", lw=1.0, alpha=0.8, zorder=2)
    ref = vals[0]
    ax.axhline(ref, color="black", ls="--", lw=1.2, label=f"No text baseline ({ref:.4f})")
    ax.set_ylabel("AUROC")
    ax.set_title("Note-Type Ablation (Mortality, W24, Early Fusion XGBoost)")
    ax.set_ylim(0.895, 0.912)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    for idx, v in enumerate(vals):
        ax.text(idx, v + 0.00045, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    # Visual cue that y-axis is zoomed (not from zero).
    ax.text(-0.03, 0.02, "//", transform=ax.transAxes, fontsize=12, fontweight="bold")

    out = FIGURES_DIR / "fig3_note_type_ablation.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig4_typed_vs_mean(records):
    conds = ["W24", "leaked", "clean"]
    x = np.arange(len(conds))
    width = 0.35

    text_mean = [_auc(records, task="mortality", modality="text_only", model="lr", window=w, text_method="original") for w in conds]
    text_typed = [_auc(records, task="mortality", modality="text_only", model="lr", window=w, text_method="original_typed") for w in conds]

    late_mean = [_auc(records, task="mortality", modality="fusion", fusion_strategy="late_stacking", model="lr", window=w, text_method="original") for w in conds]
    late_typed = [_auc(records, task="mortality", modality="fusion", fusion_strategy="late_stacking", model="lr", window=w, text_method="original_typed") for w in conds]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for ax, mean_vals, typed_vals, title in [
        (axes[0], text_mean, text_typed, "Text-Only (Mortality)"),
        (axes[1], late_mean, late_typed, "Late Fusion (Mortality)"),
    ]:
        ax.bar(x - width / 2, mean_vals, width=width, label="mean", color="#4e79a7")
        ax.bar(x + width / 2, typed_vals, width=width, label="typed", color="#f28e2b")
        ax.set_xticks(x)
        ax.set_xticklabels(conds)
        ax.set_ylabel("AUROC")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(frameon=False)
    axes[0].set_ylim(0.82, 0.86)
    axes[1].set_ylim(0.88, 0.90)

    out = FIGURES_DIR / "fig4_typed_vs_mean.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def fig5_window_effect(records):
    tasks = ["mortality", "prolonged_los"]
    windows = ["D0", "W6", "W12", "W24"]
    x = np.arange(len(windows))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, task in zip(axes, tasks):
        struct_lr = [_auc(records, task=task, modality="structured", model="lr", window=w, text_method="original") for w in windows]
        struct_xgb = [_auc(records, task=task, modality="structured", model="xgb", window=w, text_method="original") for w in windows]
        early_orig = [_auc(records, task=task, modality="fusion", fusion_strategy="early", model="xgb", window=w, text_method="original") for w in windows]
        early_weighted = [_auc(records, task=task, modality="fusion", fusion_strategy="early", model="xgb", window=w, text_method="weighted_typed") for w in windows]

        ax.plot(x, struct_lr, marker="o", lw=2, label="Structured LR")
        ax.plot(x, struct_xgb, marker="o", lw=2, label="Structured XGBoost")
        ax.plot(x, early_orig, marker="o", lw=2, label="Early Fusion original")
        ax.plot(x, early_weighted, marker="o", lw=2, label="Early Fusion weighted_typed")

        ax.set_xticks(x)
        ax.set_xticklabels(windows)
        ax.set_ylabel("AUROC")
        ax.set_title("Mortality" if task == "mortality" else "Prolonged LOS")
        ax.grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    out = FIGURES_DIR / "fig5_window_effect_by_task.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    records = load_experiments(deduplicate=True)

    outputs = [
        fig1_robustness(records),
        fig2_heatmap(),
        fig3_ablation(records),
        fig4_typed_vs_mean(records),
        fig5_window_effect(records),
    ]

    for p in outputs:
        print(f"{p} size={p.stat().st_size}")


if __name__ == "__main__":
    main()
