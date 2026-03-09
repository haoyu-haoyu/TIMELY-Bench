#!/usr/bin/env python3
"""
Train progression-task baselines for:
- Task A: AKI stage progression
- Task B: Sepsis -> Septic shock progression

Output JSON filename format:
  {task}_{modality}_{model}_{window}_{text_method}.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import COHORT_FILE, PREDEFINED_SPLITS_FILE, ROOT_DIR  # type: ignore
from utils.predefined_split import load_predefined_split_index  # type: ignore


LABEL_FILE_MAP = {
    "aki_progression": "labels_aki_progression.csv",
    "sepsis_shock": "labels_sepsis_shock.csv",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train progression baselines")
    p.add_argument("--task", choices=["aki_progression", "sepsis_shock"], required=True)
    p.add_argument("--modality", choices=["structured", "text_only", "fusion"], required=True)
    p.add_argument("--model", choices=["lr", "xgb"], default="xgb")
    p.add_argument("--window", choices=["W6", "W12", "W24", "leaked", "clean"], required=True)
    p.add_argument("--text_method", default="original")
    p.add_argument("--output_dir", default="results/note_centered/progression_tasks")
    p.add_argument("--n-jobs", type=int, default=8)
    p.add_argument("--xgb-n-estimators", type=int, default=300)
    p.add_argument("--xgb-max-depth", type=int, default=6)
    p.add_argument("--xgb-learning-rate", type=float, default=0.1)
    p.add_argument("--lr-max-iter", type=int, default=300)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def safe_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    if len(y_true) == 0:
        return 0.5, 0.0, 0.25
    if np.unique(y_true).size < 2:
        auroc = 0.5
        auprc = float(np.mean(y_true))
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        auprc = float(average_precision_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    return auroc, auprc, brier


def resolve_clean(window: str, text_method: str, modality: str) -> Tuple[str, str]:
    if modality == "structured":
        if window == "clean":
            return "W24", "none"
        return window, "none"

    if window == "clean":
        return "W24", "weighted_no_after"
    return window, text_method


def load_labels(task: str) -> pd.DataFrame:
    path = ROOT_DIR / "data" / "processed" / LABEL_FILE_MAP[task]
    if not path.exists():
        raise FileNotFoundError(f"Missing labels file: {path}")
    labels = pd.read_csv(path)
    req = {"stay_id", "prediction_hour", "label"}
    miss = req - set(labels.columns)
    if miss:
        raise ValueError(f"Missing required label columns {sorted(miss)} in {path}")
    labels["stay_id"] = pd.to_numeric(labels["stay_id"], errors="coerce").astype("Int64")
    labels["prediction_hour"] = pd.to_numeric(labels["prediction_hour"], errors="coerce")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["stay_id", "prediction_hour", "label"]).copy()
    labels["stay_id"] = labels["stay_id"].astype(np.int64)
    labels["prediction_hour"] = labels["prediction_hour"].astype(np.int16)
    labels["label"] = labels["label"].astype(np.int8)
    labels = labels.sort_values(["stay_id", "prediction_hour"], kind="mergesort")
    labels = labels.drop_duplicates(subset=["stay_id", "prediction_hour"], keep="first")
    return labels


def load_feature_frame(modality: str, struct_window: str, text_method: str) -> pd.DataFrame:
    feature_dir = ROOT_DIR / "data" / "processed" / "progression_features"
    struct_path = feature_dir / f"structured_{struct_window}.parquet"
    text_path = feature_dir / f"text_W24_{text_method}.parquet"

    if modality == "structured":
        if not struct_path.exists():
            raise FileNotFoundError(f"Missing structured feature file: {struct_path}")
        features = pd.read_parquet(struct_path)
    elif modality == "text_only":
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text feature file: {text_path}")
        features = pd.read_parquet(text_path)
    else:
        if not struct_path.exists():
            raise FileNotFoundError(f"Missing structured feature file: {struct_path}")
        if not text_path.exists():
            raise FileNotFoundError(f"Missing text feature file: {text_path}")
        struct = pd.read_parquet(struct_path)
        text = pd.read_parquet(text_path)
        features = struct.merge(text, on=["stay_id", "prediction_hour"], how="inner")

    for c in ["stay_id", "prediction_hour"]:
        if c not in features.columns:
            raise ValueError(f"Missing key column `{c}` in feature frame")
    features["stay_id"] = pd.to_numeric(features["stay_id"], errors="coerce").astype("Int64")
    features["prediction_hour"] = pd.to_numeric(features["prediction_hour"], errors="coerce")
    features = features.dropna(subset=["stay_id", "prediction_hour"]).copy()
    features["stay_id"] = features["stay_id"].astype(np.int64)
    features["prediction_hour"] = features["prediction_hour"].astype(np.int16)
    features = features.sort_values(["stay_id", "prediction_hour"], kind="mergesort")
    features = features.drop_duplicates(subset=["stay_id", "prediction_hour"], keep="first")
    return features


def load_subject_mapping() -> pd.DataFrame:
    cohort_path = Path(COHORT_FILE)
    if not cohort_path.exists():
        fallback = ROOT_DIR / "data" / "processed" / "merge_output" / "cohort_final.csv"
        cohort_path = fallback if fallback.exists() else cohort_path
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort file: {cohort_path}")

    cohort = pd.read_csv(cohort_path, usecols=["stay_id", "subject_id"])
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="coerce").astype("Int64")
    cohort = cohort.dropna(subset=["stay_id"]).copy()
    cohort["stay_id"] = cohort["stay_id"].astype(np.int64)
    cohort["subject_id"] = cohort["subject_id"].fillna(-1).astype(np.int64)
    cohort = cohort.drop_duplicates(subset=["stay_id"], keep="first")
    return cohort


def build_dataset(task: str, modality: str, struct_window: str, text_method: str) -> pd.DataFrame:
    labels = load_labels(task)
    features = load_feature_frame(modality, struct_window, text_method)
    data = features.merge(labels, on=["stay_id", "prediction_hour"], how="inner")
    data = data.merge(load_subject_mapping(), on="stay_id", how="left")
    data["subject_id"] = data["subject_id"].fillna(-1).astype(np.int64)
    data["label"] = data["label"].astype(np.int8)
    return data


def prepare_matrix(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    non_feature = {
        "stay_id",
        "prediction_hour",
        "label",
        "subject_id",
        "stage1_onset_hour",
        "stage2_onset_hour",
        "sepsis_onset_hour",
        "shock_onset_hour",
    }
    feature_cols = [
        c
        for c in data.columns
        if c not in non_feature and pd.api.types.is_numeric_dtype(data[c])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found.")

    X = (
        data[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .astype(np.float32)
        .to_numpy()
    )
    y = data["label"].astype(np.int8).to_numpy()
    stay_ids = data["stay_id"].astype(np.int64).to_numpy()
    return X, y, stay_ids, feature_cols


def resolve_dev_folds(stay_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    split_idx = load_predefined_split_index()
    aligned = split_idx.reindex(stay_ids)

    split_vals = aligned["split"].astype(str).fillna("missing").to_numpy()
    fold_vals = pd.to_numeric(aligned["fold_id"], errors="coerce").fillna(-1).astype(int).to_numpy()

    dev_mask = split_vals == "dev"
    valid_fold = (fold_vals >= 1) & (fold_vals <= 5)
    keep = dev_mask & valid_fold
    if keep.sum() == 0:
        raise ValueError("No rows mapped to dev folds in predefined_splits.csv")
    return keep, fold_vals


def fit_and_predict(
    model_name: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    if model_name == "xgb":
        pos = float((y_tr == 1).sum())
        neg = float((y_tr == 0).sum())
        spw = (neg / pos) if pos > 0 else 1.0
        model = XGBClassifier(
            n_estimators=int(args.xgb_n_estimators),
            max_depth=int(args.xgb_max_depth),
            learning_rate=float(args.xgb_learning_rate),
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=int(args.random_state),
            n_jobs=max(1, int(args.n_jobs)),
            eval_metric="aucpr",
            tree_method="hist",
            scale_pos_weight=spw,
            verbosity=0,
        )
        model.fit(X_tr, y_tr)
        return model.predict_proba(X_te)[:, 1]

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    model = LogisticRegression(
        max_iter=int(args.lr_max_iter),
        solver="saga",
        class_weight="balanced",
        n_jobs=max(1, int(args.n_jobs // 2) or 1),
        random_state=int(args.random_state),
        tol=1e-3,
    )
    model.fit(X_tr_s, y_tr)
    return model.predict_proba(X_te_s)[:, 1]


def main() -> None:
    args = parse_args()
    req_text = str(args.text_method or "original")
    struct_window, actual_text_method = resolve_clean(args.window, req_text, args.modality)

    data = build_dataset(args.task, args.modality, struct_window, actual_text_method)
    X, y, stay_ids, feature_cols = prepare_matrix(data)
    keep, fold_vals = resolve_dev_folds(stay_ids)

    X = X[keep]
    y = y[keep]
    stay_ids = stay_ids[keep]
    fold_vals = fold_vals[keep]

    print(
        f"Task={args.task} Modality={args.modality} Model={args.model} "
        f"Window={args.window} StructWindow={struct_window} TextMethod={actual_text_method}"
    )
    print(
        f"Rows={len(y)} PosRate={float(y.mean()):.4f} "
        f"Features={len(feature_cols)} UniqueStays={len(np.unique(stay_ids))}"
    )

    fold_results: List[Dict[str, float]] = []
    for fold in range(1, 6):
        te = fold_vals == fold
        tr = fold_vals != fold
        if te.sum() == 0 or tr.sum() == 0:
            raise ValueError(f"Fold {fold}: empty train/test after split resolution.")

        y_prob = fit_and_predict(args.model, X[tr], y[tr], X[te], args)
        auroc, auprc, brier = safe_metrics(y[te], y_prob)
        fold_results.append(
            {
                "fold": int(fold),
                "auroc": float(auroc),
                "auprc": float(auprc),
                "brier": float(brier),
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "pos_rate_train": float(y[tr].mean()),
                "pos_rate_test": float(y[te].mean()),
            }
        )
        print(
            f"  Fold {fold}: AUROC={auroc:.4f} AUPRC={auprc:.4f} "
            f"n_train={int(tr.sum())} n_test={int(te.sum())}"
        )

    out = {
        "task": args.task,
        "modality": args.modality,
        "model": args.model,
        "window": args.window,
        "struct_window": struct_window,
        "requested_text_method": req_text,
        "text_method": actual_text_method,
        "split_source": Path(PREDEFINED_SPLITS_FILE).name,
        "fold_results": fold_results,
        "mean_auroc": float(np.mean([r["auroc"] for r in fold_results])),
        "std_auroc": float(np.std([r["auroc"] for r in fold_results])),
        "mean_auprc": float(np.mean([r["auprc"] for r in fold_results])),
        "std_auprc": float(np.std([r["auprc"] for r in fold_results])),
        "mean_brier": float(np.mean([r["brier"] for r in fold_results])),
        "std_brier": float(np.std([r["brier"] for r in fold_results])),
        "n_samples": int(len(y)),
        "n_positive": int(y.sum()),
        "positive_rate": float(y.mean()),
        "n_features": int(len(feature_cols)),
        "feature_columns": feature_cols,
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.task}_{args.modality}_{args.model}_{args.window}_{actual_text_method}.json"
    out_file.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
