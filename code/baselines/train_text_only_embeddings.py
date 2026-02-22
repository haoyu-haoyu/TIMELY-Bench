"""
Text-only baselines using precomputed ClinicalBERT embeddings.

This implements the "text windows -> sentence embeddings" baseline requested by the
project spec in `作业要求.md`.

Notes:
- Embeddings are precomputed from Episode notes within the first 24 hours via
  `code/data_processing/extract_bert_embeddings.py` and stored under:
    data/processed/text_embeddings/clinical_bert_embeddings.npy
    data/processed/text_embeddings/embedding_stay_ids.csv
- We train simple tabular classifiers on the embeddings (LR / XGBoost) with
  subject-level predefined splits (dev/test + fold_id), matching the rest of
  the benchmark.
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
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
    RESULTS_DIR,
    N_FOLDS,
    RANDOM_STATE,
    USE_HOLDOUT_TEST,
    ROOT_DIR,
)
from utils.predefined_split import resolve_predefined_partition


EMB_DIR = ROOT_DIR / "data" / "processed" / "text_embeddings"
EMB_FILE = EMB_DIR / "clinical_bert_embeddings.npy"
EMB_IDS_FILE = EMB_DIR / "embedding_stay_ids.csv"

OUTPUT_DIR = RESULTS_DIR / "text_only_baselines"
OUTPUT_JSON = OUTPUT_DIR / "text_only_embeddings_results_folds.json"
OUTPUT_CSV = OUTPUT_DIR / "text_only_embeddings_results.csv"

FUSION_PRED_DIR = RESULTS_DIR / "fusion_baselines" / "predictions"


def _safe_metrics(y_true, y_pred):
    """Return (AUROC, AUPRC); handle degenerate folds."""
    if len(np.unique(y_true)) <= 1:
        return 0.5, float(np.mean(y_true))
    return roc_auc_score(y_true, y_pred), average_precision_score(y_true, y_pred)


def load_embeddings():
    if not EMB_FILE.exists() or not EMB_IDS_FILE.exists():
        raise FileNotFoundError(
            f"Missing ClinicalBERT embeddings. Expected:\n- {EMB_FILE}\n- {EMB_IDS_FILE}"
        )

    emb = np.load(EMB_FILE)
    ids = pd.read_csv(EMB_IDS_FILE)
    if "stay_id" not in ids.columns:
        raise ValueError(f"Missing stay_id column in {EMB_IDS_FILE}")

    stay_ids = ids["stay_id"].astype(int).tolist()
    if len(stay_ids) != emb.shape[0]:
        raise ValueError(
            f"Embeddings mismatch: {EMB_FILE} has {emb.shape[0]} rows but "
            f"{EMB_IDS_FILE} has {len(stay_ids)} stay_ids."
        )

    id_to_idx = {sid: i for i, sid in enumerate(stay_ids)}
    return emb, id_to_idx


def build_dataset(task: str, emb: np.ndarray, id_to_idx: dict):
    cohort = pd.read_csv(COHORT_FILE)
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    cohort["subject_id"] = cohort["subject_id"].astype(int)

    label_col = {
        "mortality": "label_mortality",
        "prolonged_los": "prolonged_los_7d",
    }[task]
    if label_col not in cohort.columns:
        raise ValueError(f"Missing label column {label_col} in {COHORT_FILE}")

    cohort = cohort[["stay_id", "subject_id", label_col]].rename(columns={label_col: "label"})
    cohort = cohort[cohort["label"].notna()].copy()
    cohort["label"] = cohort["label"].astype(int)

    # Keep only stays with embeddings.
    mask = cohort["stay_id"].isin(id_to_idx.keys())
    cohort = cohort[mask].copy()

    idx = cohort["stay_id"].map(id_to_idx).astype(int).values
    X = emb[idx]
    y = cohort["label"].values
    groups = cohort["subject_id"].values
    stay_ids = cohort["stay_id"].values
    return X, y, groups, stay_ids


def train_and_evaluate(
    X,
    y,
    groups,
    stay_ids,
    model_name: str,
    task: str,
    export_pred_path: Path | None = None,
):
    if USE_HOLDOUT_TEST:
        train_val_idx, test_idx, fold_ids, split_info = resolve_predefined_partition(stay_ids)
        X_train_val, X_test = X[train_val_idx], X[test_idx]
        y_train_val, y_test = y[train_val_idx], y[test_idx]
        groups_train_val = groups[train_val_idx]
        stay_ids_train_val = stay_ids[train_val_idx]
        stay_ids_test = stay_ids[test_idx]
        subject_ids_train_val = groups[train_val_idx]
        subject_ids_test = groups[test_idx]
        fold_train_val = fold_ids[train_val_idx]
    else:
        split_info = {"source": "runtime_groupkfold"}
        X_train_val, y_train_val, groups_train_val = X, y, groups
        X_test, y_test = None, None

    fold_details = []
    oof_pred = np.full(len(y_train_val), np.nan, dtype=float)
    oof_fold = np.full(len(y_train_val), -1, dtype=int)

    if USE_HOLDOUT_TEST:
        fold_iter = []
        for fold in range(1, N_FOLDS + 1):
            train_idx = np.where(fold_train_val != fold)[0]
            val_idx = np.where(fold_train_val == fold)[0]
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            fold_iter.append((fold, train_idx, val_idx))
    else:
        gkf = GroupKFold(n_splits=N_FOLDS)
        fold_iter = []
        for fold, (train_idx, val_idx) in enumerate(
            gkf.split(X_train_val, y_train_val, groups=groups_train_val),
            start=1,
        ):
            fold_iter.append((fold, train_idx, val_idx))

    for fold, train_idx, val_idx in fold_iter:
        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
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
            model = LogisticRegression(
                max_iter=5000,
                solver="saga",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        model.fit(X_train, y_train)
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
            model = LogisticRegression(
                max_iter=5000,
                solver="saga",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )

        model.fit(X_tv, y_train_val)
        pred = model.predict_proba(X_te)[:, 1]
        auroc, auprc = _safe_metrics(y_test, pred)
        test_result = {
            "auroc": float(auroc),
            "auprc": float(auprc),
            "n_test": int(len(y_test)),
        }

        # Export per-stay predictions in the canonical fusion prediction schema so
        # compute_fusion_calibration.py can upsert calibration into
        # results/calibration/calibration_summary.csv.
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
        split_info,
    )


def main():
    ap = argparse.ArgumentParser(description="Text-only baselines using ClinicalBERT embeddings.")
    ap.add_argument(
        "--models",
        default="xgb,lr",
        help="Comma-separated list: xgb,lr (default: xgb,lr).",
    )
    ap.add_argument(
        "--export-preds",
        action="store_true",
        help="Export per-stay prediction CSVs for calibration (LR only; written to results/fusion_baselines/predictions).",
    )
    ap.add_argument(
        "--fast-export",
        action="store_true",
        help="When exporting preds, skip CV and only fit once on train_val to export holdout test predictions.",
    )
    args = ap.parse_args()
    models_req = {m.strip().lower() for m in args.models.split(",") if m.strip()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    emb, id_to_idx = load_embeddings()
    results = []

    for task in ["mortality", "prolonged_los"]:
        X, y, groups, _stay_ids = build_dataset(task, emb, id_to_idx)
        n_samples = int(len(y))
        pos_rate = float(np.mean(y)) if n_samples > 0 else 0.0

        model_names = []
        if "xgb" in models_req:
            model_names.append("XGBoost (ClinicalBERT)")
        if "lr" in models_req:
            model_names.append("LogisticRegression (ClinicalBERT)")
        if not model_names:
            raise ValueError("--models must include at least one of: xgb,lr")

        # Fast path: only export holdout test predictions (no CV) for calibration.
        if args.export_preds and args.fast_export:
            if not USE_HOLDOUT_TEST:
                raise ValueError("--fast-export requires USE_HOLDOUT_TEST=True")
            train_val_idx, test_idx, _fold_ids, _split_info = resolve_predefined_partition(_stay_ids)
            X_train_val, X_test = X[train_val_idx], X[test_idx]
            y_train_val, y_test = y[train_val_idx], y[test_idx]

            stay_ids_test = _stay_ids[test_idx]
            subject_ids_test = groups[test_idx]

            scaler = StandardScaler()
            X_tv = scaler.fit_transform(X_train_val)
            X_te = scaler.transform(X_test)

            for model_name in model_names:
                if not model_name.startswith("LogisticRegression"):
                    continue
                export_path = FUSION_PRED_DIR / f"text_only_lr_clinicalbert_24h_all_{task}.csv"

                model = LogisticRegression(
                    max_iter=5000,
                    solver="saga",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                )
                model.fit(X_tv, y_train_val)
                pred = model.predict_proba(X_te)[:, 1]

                FUSION_PRED_DIR.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    {
                        "stay_id": stay_ids_test.astype(int),
                        "subject_id": subject_ids_test.astype(int),
                        "label": y_test.astype(int),
                        "pred": pred.astype(float),
                        "split": "test",
                        "fold": np.nan,
                    }
                ).to_csv(export_path, index=False)
                auroc, auprc = _safe_metrics(y_test, pred)
                print(f"Saved predictions: {export_path} (AUROC={auroc:.4f}, AUPRC={auprc:.4f})")

            # No results CSV updates in fast-export mode.
            continue

        for model_name in model_names:
            export_path = None
            if args.export_preds and model_name.startswith("LogisticRegression"):
                export_path = FUSION_PRED_DIR / f"text_only_lr_clinicalbert_24h_all_{task}.csv"
            fold_details, test_result, m_auc, s_auc, m_ap, s_ap, split_info = train_and_evaluate(
                X, y, groups, _stay_ids, model_name, task, export_pred_path=export_path
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
                    "split_source": split_info["source"],
                }
            )

    # In fast-export mode we intentionally do not touch the canonical results CSV/JSON.
    if args.export_preds and args.fast_export:
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)

    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": RANDOM_STATE,
        "input_paths": {
            "cohort_file": str(COHORT_FILE),
            "embeddings_file": str(EMB_FILE),
            "embedding_stay_ids": str(EMB_IDS_FILE),
        },
        "results": results,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
