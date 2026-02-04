"""
Permutation sanity test for structured mortality baseline.
Permutes training labels to verify AUROC≈0.5 (no leakage).
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from config import (
    COHORT_FILE,
    RESULTS_DIR,
    RANDOM_STATE,
    N_FOLDS,
    TEST_SIZE,
    USE_HOLDOUT_TEST,
    get_features_file,
)


def filter_cohort(cohort_df: pd.DataFrame, cohort_name: str) -> np.ndarray:
    if cohort_name == "all":
        return cohort_df["stay_id"].values
    if cohort_name == "sepsis":
        return cohort_df[cohort_df["has_sepsis_final"] == 1]["stay_id"].values
    if cohort_name == "aki":
        return cohort_df[cohort_df["has_aki_final"] == 1]["stay_id"].values
    if cohort_name == "sepsis_aki":
        mask = (cohort_df["has_sepsis_final"] == 1) & (cohort_df["has_aki_final"] == 1)
        return cohort_df[mask]["stay_id"].values
    raise ValueError(f"Unknown cohort: {cohort_name}")


def run_permutation_test(window: str, cohort_name: str, n_permutations: int) -> pd.DataFrame:
    features_file = get_features_file(window)
    if not Path(features_file).exists():
        raise FileNotFoundError(f"Missing features file: {features_file}")

    cohort = pd.read_csv(COHORT_FILE)
    cohort["stay_id"] = cohort["stay_id"].astype(int)

    features_df = pd.read_csv(features_file)
    features_df["stay_id"] = features_df["stay_id"].astype(int)

    cohort_ids = filter_cohort(cohort, cohort_name)

    df = features_df[features_df["stay_id"].isin(cohort_ids)].merge(
        cohort[["stay_id", "subject_id", "label_mortality"]],
        on="stay_id",
        how="inner",
    )
    df = df[df["label_mortality"].notna()].copy()

    feature_cols = [c for c in df.columns if c not in ["stay_id", "subject_id", "label_mortality"]]
    X = df[feature_cols].values
    y = df["label_mortality"].values
    groups = df["subject_id"].values

    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X, y, groups=groups))
        X_tv, X_test = X[train_val_idx], X[test_idx]
        y_tv, y_test = y[train_val_idx], y[test_idx]
        groups_tv = groups[train_val_idx]
    else:
        X_tv, y_tv, groups_tv = X, y, groups
        X_test, y_test = None, None

    gkf = GroupKFold(n_splits=N_FOLDS)

    rows = []
    for perm_id in range(n_permutations):
        rng = np.random.default_rng(RANDOM_STATE + perm_id)
        y_tv_perm = rng.permutation(y_tv)

        fold_metrics = []
        for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tv, y_tv, groups=groups_tv), start=1):
            X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
            y_tr_perm = y_tv_perm[tr_idx]
            y_val = y_tv[val_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
            X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(X_tr, y_tr_perm)
            preds = model.predict_proba(X_val)[:, 1]

            auroc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
            auprc = average_precision_score(y_val, preds) if len(np.unique(y_val)) > 1 else float(np.mean(y_val))
            fold_metrics.append((auroc, auprc))

        aurocs = [m[0] for m in fold_metrics]
        auprcs = [m[1] for m in fold_metrics]

        test_auroc = None
        test_auprc = None
        if X_test is not None:
            scaler = StandardScaler()
            X_tv_s = scaler.fit_transform(X_tv)
            X_test_s = scaler.transform(X_test)
            X_tv_s = np.nan_to_num(X_tv_s, nan=0.0, posinf=0.0, neginf=0.0)
            X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
            model.fit(X_tv_s, y_tv_perm)
            test_pred = model.predict_proba(X_test_s)[:, 1]
            test_auroc = roc_auc_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else 0.5
            test_auprc = average_precision_score(y_test, test_pred) if len(np.unique(y_test)) > 1 else float(np.mean(y_test))

        rows.append({
            "step": "permutation_sanity",
            "task": "mortality",
            "model": "LogisticRegression",
            "cohort": cohort_name,
            "window": window,
            "permutation_id": perm_id + 1,
            "cv_auroc_mean": float(np.mean(aurocs)),
            "cv_auroc_std": float(np.std(aurocs)),
            "cv_auprc_mean": float(np.mean(auprcs)),
            "cv_auprc_std": float(np.std(auprcs)),
            "test_auroc": float(test_auroc) if test_auroc is not None else None,
            "test_auprc": float(test_auprc) if test_auprc is not None else None,
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="24h")
    parser.add_argument("--cohort", default="all")
    parser.add_argument("--n-permutations", type=int, default=5)
    args = parser.parse_args()

    df = run_permutation_test(args.window, args.cohort, args.n_permutations)

    out_dir = RESULTS_DIR / "standardized"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "permutation_structured_mortality.csv"
    json_path = out_dir / "permutation_structured_mortality.json"

    df.to_csv(csv_path, index=False)

    summary = {
        "step": "permutation_sanity",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": RANDOM_STATE,
        "window": args.window,
        "cohort": args.cohort,
        "n_permutations": args.n_permutations,
        "cv_auroc_mean": float(df["cv_auroc_mean"].mean()),
        "cv_auroc_std": float(df["cv_auroc_mean"].std()),
        "cv_auprc_mean": float(df["cv_auprc_mean"].mean()),
        "cv_auprc_std": float(df["cv_auprc_mean"].std()),
        "test_auroc_mean": float(df["test_auroc"].mean()),
        "test_auroc_std": float(df["test_auroc"].std()),
        "test_auprc_mean": float(df["test_auprc"].mean()),
        "test_auprc_std": float(df["test_auprc"].std()),
    }
    payload = {
        "summary": summary,
        "results": df.to_dict(orient="records"),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
