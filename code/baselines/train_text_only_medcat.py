"""
Text-only baselines using MedCAT concept extraction features.

This implements the "text windows -> concept extraction (UMLS/MedCAT) -> bag-of-concepts"
baseline requested by the project spec in `作业要求.md`.

Input:
- data/processed/medcat_full/medcat_has_concepts_24h.csv (stay-level binary features)
- data/processed/merge_output/cohort_final.csv (labels + subject_id grouping)

Output:
- results/text_only_baselines/text_only_medcat_results_folds.json
- results/text_only_baselines/text_only_medcat_results.csv
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import (
    COHORT_FILE,
    PROCESSED_DIR,
    RESULTS_DIR,
    N_FOLDS,
    RANDOM_STATE,
    USE_HOLDOUT_TEST,
)
from utils.predefined_split import resolve_predefined_partition


MEDCAT_FILE = PROCESSED_DIR / "medcat_full" / "medcat_has_concepts_24h.csv"

OUTPUT_DIR = RESULTS_DIR / "text_only_baselines"
OUTPUT_JSON = OUTPUT_DIR / "text_only_medcat_results_folds.json"
OUTPUT_CSV = OUTPUT_DIR / "text_only_medcat_results.csv"
FUSION_PRED_DIR = RESULTS_DIR / "fusion_baselines" / "predictions"


def _safe_metrics(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return 0.5, float(np.mean(y_true))
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)


def load_dataset(task: str):
    if not MEDCAT_FILE.exists():
        raise FileNotFoundError(f"Missing MedCAT features: {MEDCAT_FILE}")
    if not COHORT_FILE.exists():
        raise FileNotFoundError(f"Missing cohort file: {COHORT_FILE}")

    med = pd.read_csv(MEDCAT_FILE)
    med["stay_id"] = med["stay_id"].astype(int)

    cohort = pd.read_csv(COHORT_FILE)
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    cohort["subject_id"] = cohort["subject_id"].astype(int)

    label_col = {
        "mortality": "label_mortality",
        "prolonged_los": "prolonged_los_7d",
    }[task]
    if label_col not in cohort.columns:
        raise ValueError(f"Missing label column {label_col} in {COHORT_FILE}")

    labels = cohort[["stay_id", "subject_id", label_col]].rename(columns={label_col: "label"})
    labels = labels[labels["label"].notna()].copy()
    labels["label"] = labels["label"].astype(int)

    df = med.merge(labels, on="stay_id", how="inner")
    feature_cols = [c for c in df.columns if c not in ["stay_id", "subject_id", "label"]]

    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = df["label"].values
    groups = df["subject_id"].values
    stay_ids = df["stay_id"].values
    return X, y, groups, stay_ids, feature_cols


def train_and_evaluate(
    X,
    y,
    groups,
    stay_ids,
    model_name: str,
    export_pred_path: Path | None = None,
):
    if USE_HOLDOUT_TEST:
        train_val_idx, test_idx, fold_ids, split_info = resolve_predefined_partition(stay_ids)
        fold_train_val = fold_ids[train_val_idx]
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        stay_ids_train_val = stay_ids[train_val_idx]
        stay_ids_test = stay_ids[test_idx]
        subject_ids_train_val = groups[train_val_idx]
        subject_ids_test = groups[test_idx]
    else:
        split_info = {"source": "runtime_groupkfold"}
        X_train_val, y_train_val, groups_train_val = X, y, groups
        stay_ids_train_val = stay_ids
        X_test, y_test = None, None

    fold_details = []
    oof_pred = np.full(len(y_train_val), np.nan, dtype=float)
    oof_fold = np.full(len(y_train_val), -1, dtype=int)

    if USE_HOLDOUT_TEST:
        fold_iter = []
        for fold in range(1, N_FOLDS + 1):
            tr_idx = np.where(fold_train_val != fold)[0]
            val_idx = np.where(fold_train_val == fold)[0]
            if len(tr_idx) == 0 or len(val_idx) == 0:
                continue
            fold_iter.append((fold, tr_idx, val_idx))
    else:
        gkf = GroupKFold(n_splits=N_FOLDS)
        fold_iter = []
        for fold, (tr_idx, val_idx) in enumerate(
            gkf.split(X_train_val, y_train_val, groups=groups_train_val), start=1
        ):
            fold_iter.append((fold, tr_idx, val_idx))

    for fold, tr_idx, val_idx in fold_iter:
        X_tr, X_val = X_train_val[tr_idx], X_train_val[val_idx]
        y_tr, y_val = y_train_val[tr_idx], y_train_val[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        if model_name.startswith("XGBoost"):
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            )
        else:
            model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)

        model.fit(X_tr, y_tr)
        pred = model.predict_proba(X_val)[:, 1]
        oof_pred[val_idx] = pred
        oof_fold[val_idx] = fold
        auroc, auprc = _safe_metrics(y_val, pred)
        fold_details.append({"fold": fold, "auroc": float(auroc), "auprc": float(auprc)})

    test_result = None
    if X_test is not None:
        scaler = StandardScaler()
        X_tv = scaler.fit_transform(X_train_val)
        X_te = scaler.transform(X_test)

        if model_name.startswith("XGBoost"):
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=-1,
            )
        else:
            model = LogisticRegression(max_iter=5000, random_state=RANDOM_STATE, n_jobs=-1)

        model.fit(X_tv, y_train_val)
        pred = model.predict_proba(X_te)[:, 1]
        auroc, auprc = _safe_metrics(y_test, pred)
        test_result = {"auroc": float(auroc), "auprc": float(auprc)}

        if export_pred_path is not None:
            FUSION_PRED_DIR.mkdir(parents=True, exist_ok=True)
            out = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "stay_id": stay_ids_train_val.astype(int),
                            "subject_id": subject_ids_train_val.astype(int),
                            "label": y_train_val.astype(int),
                            "pred": oof_pred.astype(float),
                            "split": "val",
                            "fold": oof_fold.astype(int),
                        }
                    ),
                    pd.DataFrame(
                        {
                            "stay_id": stay_ids_test.astype(int),
                            "subject_id": subject_ids_test.astype(int),
                            "label": y_test.astype(int),
                            "pred": pred.astype(float),
                            "split": "test",
                            "fold": np.nan,
                        }
                    ),
                ],
                ignore_index=True,
            )
            out.to_csv(export_pred_path, index=False)
            print(f"Saved predictions: {export_pred_path}")

    aurocs = [r["auroc"] for r in fold_details]
    auprcs = [r["auprc"] for r in fold_details]
    return (
        fold_details,
        test_result,
        float(np.mean(aurocs)),
        float(np.std(aurocs)),
        float(np.mean(auprcs)),
        float(np.std(auprcs)),
        split_info["source"],
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for task in ["mortality", "prolonged_los"]:
        X, y, groups, stay_ids, _feat_cols = load_dataset(task)
        n_samples = int(len(y))
        pos_rate = float(np.mean(y)) if n_samples > 0 else 0.0

        for model_name in ["XGBoost (MedCAT)", "LogisticRegression (MedCAT)"]:
            export_path = None
            if model_name.startswith("XGBoost"):
                export_path = FUSION_PRED_DIR / f"text_only_xgb_medcat_24h_all_{task}.csv"
            else:
                export_path = FUSION_PRED_DIR / f"text_only_lr_medcat_24h_all_{task}.csv"
            fold_details, test_result, m_auc, s_auc, m_ap, s_ap, split_source = train_and_evaluate(
                X, y, groups, stay_ids, model_name, export_pred_path=export_path
            )
            results.append(
                {
                    "task": task,
                    "model": model_name,
                    "n_samples": n_samples,
                    "positive_rate": pos_rate,
                    "cv_auroc_mean": m_auc,
                    "cv_auroc_std": s_auc,
                    "cv_auprc_mean": m_ap,
                    "cv_auprc_std": s_ap,
                    "test_auroc": test_result["auroc"] if test_result else None,
                    "test_auprc": test_result["auprc"] if test_result else None,
                    "fold_details": fold_details,
                    "split_source": split_source,
                }
            )

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_CSV, index=False)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": RANDOM_STATE,
        "input_paths": {
            "cohort_file": str(COHORT_FILE),
            "medcat_features_file": str(MEDCAT_FILE),
        },
        "results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"Saved: {OUTPUT_JSON}")
    print(f"Saved: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
