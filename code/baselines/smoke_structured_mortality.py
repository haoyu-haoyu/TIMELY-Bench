"""
Quick smoke test for structured mortality baseline.
Writes standardized results with AUROC/AUPRC, fold stats, seed, and input paths.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    RESULTS_DIR,
    COHORT_FILE,
    RANDOM_STATE,
    N_FOLDS,
    TEST_SIZE,
    USE_HOLDOUT_TEST,
    get_features_file,
)


WINDOW = "24h"
COHORT_NAME = "all"
MODEL_NAME = "LogisticRegression"
OUTPUT_DIR = RESULTS_DIR / "standardized"
OUTPUT_JSON = OUTPUT_DIR / "smoke_structured_mortality.json"
OUTPUT_CSV = OUTPUT_DIR / "smoke_structured_mortality.csv"


def main():
    features_file = get_features_file(WINDOW)
    if not Path(features_file).exists():
        raise FileNotFoundError(f"Missing features file: {features_file}")

    cohort = pd.read_csv(COHORT_FILE)
    cohort["stay_id"] = cohort["stay_id"].astype(int)

    features_df = pd.read_csv(features_file)
    features_df["stay_id"] = features_df["stay_id"].astype(int)

    df = features_df.merge(
        cohort[["stay_id", "subject_id", "label_mortality"]],
        on="stay_id",
        how="inner",
    ).dropna(subset=["label_mortality"])

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
    fold_details = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tv, y_tv, groups=groups_tv), start=1):
        X_tr, X_val = X_tv[tr_idx], X_tv[val_idx]
        y_tr, y_val = y_tv[tr_idx], y_tv[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_tr, y_tr)
        preds = model.predict_proba(X_val)[:, 1]

        auroc = roc_auc_score(y_val, preds) if len(np.unique(y_val)) > 1 else 0.5
        auprc = average_precision_score(y_val, preds) if len(np.unique(y_val)) > 1 else y_val.mean()
        fold_details.append({"fold": fold, "auroc": float(auroc), "auprc": float(auprc)})

    aurocs = [r["auroc"] for r in fold_details]
    auprcs = [r["auprc"] for r in fold_details]

    test_auroc = None
    test_auprc = None
    if X_test is not None:
        scaler = StandardScaler()
        X_tv_s = scaler.fit_transform(X_tv)
        X_test_s = scaler.transform(X_test)
        X_tv_s = np.nan_to_num(X_tv_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_test_s = np.nan_to_num(X_test_s, nan=0.0, posinf=0.0, neginf=0.0)

        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_tv_s, y_tv)
        test_pred = model.predict_proba(X_test_s)[:, 1]
        test_auroc = float(roc_auc_score(y_test, test_pred)) if len(np.unique(y_test)) > 1 else 0.5
        test_auprc = float(average_precision_score(y_test, test_pred)) if len(np.unique(y_test)) > 1 else float(y_test.mean())

    result = {
        "step": "smoke_structured_mortality",
        "task": "mortality",
        "model": MODEL_NAME,
        "cohort": COHORT_NAME,
        "window": WINDOW,
        "n_samples": int(len(df)),
        "positive_rate": float(y.mean()) if len(y) > 0 else 0.0,
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "test_auroc": test_auroc,
        "test_auprc": test_auprc,
        "fold_details": fold_details,
        "seed": RANDOM_STATE,
        "input_paths": json.dumps({
            "features_file": str(features_file),
            "cohort_file": str(COHORT_FILE),
        }, ensure_ascii=True),
    }

    payload = {
        "step": "smoke_structured_mortality",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": RANDOM_STATE,
        "inputs": {
            "features_file": str(features_file),
            "cohort_file": str(COHORT_FILE),
        },
        "results": [result],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    pd.DataFrame([result]).to_csv(OUTPUT_CSV, index=False)

    print(f"Smoke test completed. Results saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
