#!/usr/bin/env python3
"""
Compute calibration metrics for fusion baselines (Text-only / Structured / Early / Late).

This script is intentionally aligned with the TIMELY-Bench baseline artifacts:
- It reuses saved per-row prediction files from `results/fusion_baselines/predictions/`
  where available.
- Late fusion probabilities are reconstructed from structured/text predictions using
  the tuned alpha recorded in `results/standardized/late_fusion_sanity_*.json`.
- Early fusion predictions are not always persisted by the training pipeline, so
  this script can (optionally) train the Early Fusion XGBoost model and export a
  prediction CSV in the same schema as other prediction files.

Output:
- Updates `results/calibration/calibration_summary.csv` (upsert by key).
- Writes `results/calibration/calibration_fusion_summary.csv` for auditability.
- Optionally writes reliability diagrams under `results/calibration/reliability_diagrams/`.

Performance note:
- If all prediction CSVs already exist (including Early Fusion), this script only reads
  the saved predictions and runs in seconds.
- It only loads feature matrices / trains Early Fusion when an Early Fusion prediction
  file is missing or `--regen-early` is passed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from evaluation.calibration_metrics import (
    compute_brier_score,
    compute_ece,
    compute_hosmer_lemeshow,
    compute_mce,
    plot_reliability_diagram,
)
from config import ROOT_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE, USE_HOLDOUT_TEST


def _read_pred_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"stay_id", "subject_id", "label", "pred", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return df


def _calib_row(window: str, task: str, cohort: str, model: str, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int) -> Dict:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    # Defensive clamp (rare numeric edge cases)
    y_prob = np.clip(y_prob, 0.0, 1.0)

    ece = float(compute_ece(y_true, y_prob, n_bins))
    mce = float(compute_mce(y_true, y_prob, n_bins))
    brier = float(compute_brier_score(y_true, y_prob))
    hl, hl_p, hl_groups, hl_df = compute_hosmer_lemeshow(y_true, y_prob, n_groups=n_bins)
    pos_rate = float(np.mean(y_true)) if len(y_true) else 0.0
    mean_pred = float(np.mean(y_prob)) if len(y_prob) else 0.0

    return {
        "window": window,
        "task": task,
        "cohort": cohort,
        "model": model,
        "n_samples": int(len(y_true)),
        "n_test": int(len(y_true)),
        "ece": ece,
        "mce": mce,
        "brier_score": brier,
        "hl_statistic": float(hl),
        "hl_p_value": float(hl_p),
        "hl_groups": int(hl_groups),
        "hl_df": int(hl_df),
        "positive_rate": pos_rate,
        "mean_predicted_prob": mean_pred,
        "n_bins": int(n_bins),
    }


def _export_early_fusion_preds(
    out_path: Path,
    stay_ids: np.ndarray,
    subject_ids: np.ndarray,
    y: np.ndarray,
    X_struct: np.ndarray,
    X_text: np.ndarray,
    splits: np.ndarray,
    folds: np.ndarray,
) -> Path:
    """
    Train Early Fusion XGBoost and export per-row predictions for:
    - CV validation rows (split=val, fold in 1..N_FOLDS)
    - holdout test rows (split=test, fold=-1)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    X = np.concatenate([X_struct, X_text], axis=1)

    # Use the canonical split assignment already encoded in the structured/text
    # prediction files (which are derived from predefined_splits.csv).
    splits = np.asarray(splits)
    folds = np.asarray(folds, dtype=int)
    dev_idx = np.where(splits == "val")[0]
    test_idx = np.where(splits == "test")[0]

    # CV preds on dev set
    val_pred = np.full(len(y), np.nan, dtype=float)
    fold_ids = np.full(len(y), -1, dtype=int)

    for fold in range(1, N_FOLDS + 1):
        tr_idx = dev_idx[folds[dev_idx] != fold]
        val_idx = dev_idx[folds[dev_idx] == fold]
        if len(tr_idx) == 0 or len(val_idx) == 0:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_tr, y[tr_idx])
        pred = model.predict_proba(X_val)[:, 1]

        val_pred[val_idx] = pred
        fold_ids[val_idx] = int(fold)

    # Holdout test preds
    test_pred = np.array([], dtype=float)
    if len(test_idx) > 0:
        scaler = StandardScaler()
        X_dev_s = scaler.fit_transform(X[dev_idx])
        X_test_s = scaler.transform(X[test_idx])

        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=RANDOM_STATE,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(X_dev_s, y[dev_idx])
        test_pred = model.predict_proba(X_test_s)[:, 1]

    # Build prediction CSV (match existing schema)
    rows = []
    for idx in dev_idx:
        if fold_ids[idx] < 1:
            continue
        rows.append(
            {
                "stay_id": int(stay_ids[idx]),
                "subject_id": int(subject_ids[idx]),
                "label": int(y[idx]),
                "pred": float(val_pred[idx]),
                "split": "val",
                "fold": float(fold_ids[idx]),
            }
        )
    for j, idx in enumerate(test_idx.tolist()):
        rows.append(
            {
                "stay_id": int(stay_ids[idx]),
                "subject_id": int(subject_ids[idx]),
                "label": int(y[idx]),
                "pred": float(test_pred[j]),
                "split": "test",
                "fold": -1.0,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return out_path


def _load_alpha_final(path: Path) -> float:
    payload = json.loads(path.read_text())
    alpha = payload.get("alpha_final", None)
    if alpha is None:
        raise ValueError(f"Missing alpha_final in {path}")
    return float(alpha)


def _late_fusion_pred(struct_df: pd.DataFrame, text_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    # Align by stay_id and carry split/fold from the structured file (they should match).
    merged = struct_df.merge(
        text_df[["stay_id", "pred"]].rename(columns={"pred": "pred_text"}),
        on="stay_id",
        how="inner",
    )
    merged["pred_fusion"] = alpha * merged["pred"].astype(float) + (1.0 - alpha) * merged["pred_text"].astype(float)
    return merged


def _upsert_csv(path: Path, new_rows: pd.DataFrame, key_cols: Tuple[str, ...]) -> None:
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_rows], ignore_index=True)
        # Keep last occurrence per key (new rows should win).
        combined["_key"] = combined[list(key_cols)].astype(str).agg("|".join, axis=1)
        combined = combined.drop_duplicates(subset=["_key"], keep="last").drop(columns=["_key"])
        combined.to_csv(path, index=False)
    else:
        new_rows.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", default="24h")
    ap.add_argument("--cohort", default="all")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--no-plots", action="store_true", help="Do not write reliability diagrams")
    ap.add_argument("--regen-early", action="store_true", help="Regenerate Early Fusion prediction CSVs even if present")
    args = ap.parse_args()

    root = Path(ROOT_DIR)
    pred_dir = root / "results" / "fusion_baselines" / "predictions"
    std_dir = root / "results" / "standardized"
    calib_dir = root / "results" / "calibration"
    diag_dir = calib_dir / "reliability_diagrams"
    diag_dir.mkdir(parents=True, exist_ok=True)

    window = args.window
    cohort = args.cohort
    n_bins = int(args.n_bins)

    tasks = ["mortality", "prolonged_los"]
    rows = []

    def maybe_plot(model_name: str, task_name: str, y_true: np.ndarray, y_prob: np.ndarray):
        if args.no_plots:
            return
        fname = f"reliability_{model_name}_{task_name}_{window}_{cohort}.png".replace(" ", "_").replace("[", "").replace("]", "")
        plot_reliability_diagram(
            y_true, y_prob, n_bins,
            title=f"{model_name} - {task_name} ({window}, {cohort})",
            save_path=diag_dir / fname,
        )

    for task in tasks:
        # -------------------------
        # A) Annotation feature set
        # -------------------------
        # Structured/Text-only calibration from saved preds
        struct_pred_path = pred_dir / f"structured_xgb_{window}_{cohort}_{task}.csv"
        text_pred_path = pred_dir / f"text_only_xgb_{window}_{cohort}_{task}.csv"
        struct_pred_df = _read_pred_csv(struct_pred_path)
        text_pred_df = _read_pred_csv(text_pred_path)

        for model_name, df_in, prob_col in [
            ("Structured_XGBoost", struct_pred_df, "pred"),
            ("TextOnly_XGBoost", text_pred_df, "pred"),
        ]:
            test_df = df_in[df_in["split"] == "test"]
            row = _calib_row(window, task, cohort, model_name, test_df["label"].values, test_df[prob_col].values, n_bins)
            rows.append(row)
            maybe_plot(model_name, task, test_df["label"].values, test_df[prob_col].values)

        # Late fusion (reconstruct from preds + alpha)
        alpha_path = std_dir / f"late_fusion_sanity_xgb_{window}_{cohort}_{task}.json"
        alpha = _load_alpha_final(alpha_path)
        late_df = _late_fusion_pred(struct_pred_df, text_pred_df, alpha=alpha)
        late_test = late_df[late_df["split"] == "test"]
        model_name = "LateFusion_XGBoost"
        rows.append(_calib_row(window, task, cohort, model_name, late_test["label"].values, late_test["pred_fusion"].values, n_bins))
        maybe_plot(model_name, task, late_test["label"].values, late_test["pred_fusion"].values)

        # Early fusion (train if needed, then read back)
        early_pred_path = pred_dir / f"early_fusion_xgb_{window}_{cohort}_{task}.csv"
        if args.regen_early or (not early_pred_path.exists()):
            # Import lazily: only needed when we have to train Early Fusion.
            from baselines.train_fusion import (
                load_structured_features,
                load_text_features,
                _align_struct_text,
            )

            struct_df = load_structured_features(window=window, cohort=cohort, task=task)
            text_df = load_text_features(task=task)
            struct_a, text_a, _ = _align_struct_text(struct_df, text_df)

            struct_cols = [c for c in struct_a.columns if c not in ["stay_id", "subject_id", "label"]]
            text_cols = [c for c in text_a.columns if c not in ["stay_id", "subject_id", "label"]]

            stay_ids = struct_a["stay_id"].values
            subject_ids = struct_a["subject_id"].values
            y = struct_a["label"].values.astype(int)
            X_struct = np.nan_to_num(struct_a[struct_cols].values, nan=0.0)
            X_text = np.nan_to_num(text_a[text_cols].values, nan=0.0)

            # Reuse canonical partition assignment from the structured prediction CSV.
            # Note: the saved structured prediction files do not necessarily have a fold for the test split
            # (often empty/NaN); folds are only required for CV dev/val rows.
            part = struct_pred_df[["stay_id", "split", "fold"]].drop_duplicates(subset=["stay_id"]).copy()
            part["stay_id"] = pd.to_numeric(part["stay_id"], errors="coerce")
            part = part.dropna(subset=["stay_id"]).copy()
            part["stay_id"] = part["stay_id"].astype(int)
            part["split"] = part["split"].astype(str)
            part["fold"] = pd.to_numeric(part["fold"], errors="coerce")
            part = part.set_index("stay_id")

            splits = struct_a["stay_id"].map(part["split"])
            folds = struct_a["stay_id"].map(part["fold"])

            if splits.isna().any():
                missing = int(splits.isna().sum())
                raise ValueError(
                    f"Early Fusion split mapping failed for {missing} rows "
                    f"(stay_id not found in structured preds; check split artifacts consistency)"
                )

            is_test = splits == "test"
            folds = folds.copy()
            folds[is_test] = folds[is_test].fillna(-1)

            bad_val = (splits == "val") & (folds.isna() | (folds < 1) | (folds > N_FOLDS))
            if bad_val.any():
                n_bad = int(bad_val.sum())
                raise ValueError(
                    f"Early Fusion fold mapping failed for {n_bad} val rows "
                    f"(expected 1..{N_FOLDS}; check structured preds fold column)"
                )

            folds = folds.fillna(-1).astype(int)

            _export_early_fusion_preds(
                out_path=early_pred_path,
                stay_ids=stay_ids,
                subject_ids=subject_ids,
                y=y,
                X_struct=X_struct,
                X_text=X_text,
                splits=splits.values,
                folds=folds.values,
            )
        early_df = _read_pred_csv(early_pred_path)
        early_test = early_df[early_df["split"] == "test"]
        model_name = "EarlyFusion_XGBoost"
        rows.append(_calib_row(window, task, cohort, model_name, early_test["label"].values, early_test["pred"].values, n_bins))
        maybe_plot(model_name, task, early_test["label"].values, early_test["pred"].values)

        # -------------------------
        # B) ClinicalBERT feature set
        # -------------------------
        struct_pred_path_b = pred_dir / f"structured_xgb_clinicalbert_{window}_{cohort}_{task}.csv"
        text_pred_path_b = pred_dir / f"text_only_xgb_clinicalbert_{window}_{cohort}_{task}.csv"
        try:
            struct_pred_df_b = _read_pred_csv(struct_pred_path_b)
            text_pred_df_b = _read_pred_csv(text_pred_path_b)
        except Exception as e:
            print(f"[WARN] ClinicalBERT calibration skipped for task={task}: {e}")
            continue

        for model_name, df_in in [
            ("Structured_XGBoost_ClinicalBERT", struct_pred_df_b),
            ("TextOnly_XGBoost_ClinicalBERT", text_pred_df_b),
        ]:
            test_df = df_in[df_in["split"] == "test"]
            rows.append(_calib_row(window, task, cohort, model_name, test_df["label"].values, test_df["pred"].values, n_bins))
            maybe_plot(model_name, task, test_df["label"].values, test_df["pred"].values)

        # Optional: Logistic Regression on ClinicalBERT embeddings (text-only).
        lr_pred_path_b = pred_dir / f"text_only_lr_clinicalbert_{window}_{cohort}_{task}.csv"
        try:
            lr_pred_df_b = _read_pred_csv(lr_pred_path_b)
            lr_test = lr_pred_df_b[lr_pred_df_b["split"] == "test"]
            model_name = "TextOnly_LogisticRegression_ClinicalBERT"
            rows.append(_calib_row(window, task, cohort, model_name, lr_test["label"].values, lr_test["pred"].values, n_bins))
            maybe_plot(model_name, task, lr_test["label"].values, lr_test["pred"].values)
        except Exception as e:
            print(f"[WARN] ClinicalBERT LR calibration skipped for task={task}: {e}")

        alpha_path_b = std_dir / f"late_fusion_sanity_xgb_clinicalbert_{window}_{cohort}_{task}.json"
        alpha_b = _load_alpha_final(alpha_path_b)
        late_df_b = _late_fusion_pred(struct_pred_df_b, text_pred_df_b, alpha=alpha_b)
        late_test_b = late_df_b[late_df_b["split"] == "test"]
        model_name = "LateFusion_XGBoost_ClinicalBERT"
        rows.append(_calib_row(window, task, cohort, model_name, late_test_b["label"].values, late_test_b["pred_fusion"].values, n_bins))
        maybe_plot(model_name, task, late_test_b["label"].values, late_test_b["pred_fusion"].values)

        early_pred_path_b = pred_dir / f"early_fusion_xgb_clinicalbert_{window}_{cohort}_{task}.csv"
        if args.regen_early or (not early_pred_path_b.exists()):
            try:
                # Import lazily: only needed when we have to train Early Fusion.
                from baselines.train_fusion import (
                    load_structured_features,
                    load_text_embeddings,
                    _align_struct_text_embeddings,
                )

                struct_df = load_structured_features(window=window, cohort=cohort, task=task)
                emb_meta_df, emb = load_text_embeddings(task=task)
                struct_b, emb_meta_b, _ = _align_struct_text_embeddings(struct_df, emb_meta_df)

                struct_cols_b = [c for c in struct_b.columns if c not in ["stay_id", "subject_id", "label"]]
                stay_ids_b = struct_b["stay_id"].values
                subject_ids_b = struct_b["subject_id"].values
                y_b = struct_b["label"].values.astype(int)

                X_struct_b = np.nan_to_num(struct_b[struct_cols_b].values, nan=0.0)
                X_emb = np.asarray(emb[emb_meta_b["emb_idx"].values], dtype=np.float32)

                part_b = struct_pred_df_b[["stay_id", "split", "fold"]].drop_duplicates(subset=["stay_id"]).copy()
                part_b["stay_id"] = pd.to_numeric(part_b["stay_id"], errors="coerce")
                part_b = part_b.dropna(subset=["stay_id"]).copy()
                part_b["stay_id"] = part_b["stay_id"].astype(int)
                part_b["split"] = part_b["split"].astype(str)
                part_b["fold"] = pd.to_numeric(part_b["fold"], errors="coerce")
                part_b = part_b.set_index("stay_id")

                splits_b = struct_b["stay_id"].map(part_b["split"])
                folds_b = struct_b["stay_id"].map(part_b["fold"])

                if splits_b.isna().any():
                    missing = int(splits_b.isna().sum())
                    raise ValueError(
                        f"Early Fusion ClinicalBERT split mapping failed for {missing} rows "
                        f"(stay_id not found in structured preds; check split artifacts consistency)"
                    )

                is_test_b = splits_b == "test"
                folds_b = folds_b.copy()
                folds_b[is_test_b] = folds_b[is_test_b].fillna(-1)

                bad_val_b = (splits_b == "val") & (folds_b.isna() | (folds_b < 1) | (folds_b > N_FOLDS))
                if bad_val_b.any():
                    n_bad = int(bad_val_b.sum())
                    raise ValueError(
                        f"Early Fusion ClinicalBERT fold mapping failed for {n_bad} val rows "
                        f"(expected 1..{N_FOLDS}; check structured preds fold column)"
                    )

                folds_b = folds_b.fillna(-1).astype(int)

                _export_early_fusion_preds(
                    out_path=early_pred_path_b,
                    stay_ids=stay_ids_b,
                    subject_ids=subject_ids_b,
                    y=y_b,
                    X_struct=X_struct_b,
                    X_text=X_emb,
                    splits=splits_b.values,
                    folds=folds_b.values,
                )
            except Exception as e:
                print(f"[WARN] Early Fusion ClinicalBERT regeneration failed for task={task}: {e}")

        if early_pred_path_b.exists():
            early_df_b = _read_pred_csv(early_pred_path_b)
            early_test_b = early_df_b[early_df_b["split"] == "test"]
            model_name = "EarlyFusion_XGBoost_ClinicalBERT"
            rows.append(_calib_row(window, task, cohort, model_name, early_test_b["label"].values, early_test_b["pred"].values, n_bins))
            maybe_plot(model_name, task, early_test_b["label"].values, early_test_b["pred"].values)

        # -------------------------
        # C) MedCAT concept features (text-only)
        # -------------------------
        for model_name, path in [
            ("TextOnly_LogisticRegression_MedCAT", pred_dir / f"text_only_lr_medcat_{window}_{cohort}_{task}.csv"),
            ("TextOnly_XGBoost_MedCAT", pred_dir / f"text_only_xgb_medcat_{window}_{cohort}_{task}.csv"),
        ]:
            try:
                df_in = _read_pred_csv(path)
                test_df = df_in[df_in["split"] == "test"]
                rows.append(_calib_row(window, task, cohort, model_name, test_df["label"].values, test_df["pred"].values, n_bins))
                maybe_plot(model_name, task, test_df["label"].values, test_df["pred"].values)
            except Exception as e:
                print(f"[WARN] MedCAT calibration skipped for model={model_name}, task={task}: {e}")

    out_df = pd.DataFrame(rows)
    fusion_out = calib_dir / "calibration_fusion_summary.csv"
    out_df.to_csv(fusion_out, index=False)
    print(f"Wrote {fusion_out}")

    # Upsert into global calibration summary
    summary_path = calib_dir / "calibration_summary.csv"
    _upsert_csv(summary_path, out_df, key_cols=("window", "task", "cohort", "model"))
    print(f"Updated {summary_path}")


if __name__ == "__main__":
    main()
