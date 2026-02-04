"""
Sanity check for late fusion alignment on mortality.
Runs alpha extremes (1.0/0.0) and reports alignment diagnostics.
Matches train_fusion.py feature definitions so results are comparable.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import xgboost as xgb

from config import (
    COHORT_FILE,
    RESULTS_DIR,
    RANDOM_STATE,
    N_FOLDS,
    TEST_SIZE,
    USE_HOLDOUT_TEST,
)

EPISODES_DIR = Path(__file__).parent.parent.parent / "episodes" / "episodes_enhanced"


def load_tabular_features() -> pd.DataFrame:
    """Match train_fusion.py timeseries feature extraction."""
    episode_files = list(EPISODES_DIR.glob("TIMELY_v2_*.json"))
    vitals_cols = ["heart_rate", "sbp", "dbp", "mbp", "resp_rate", "temperature", "spo2"]
    features = []
    for ep_file in episode_files:
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            feat = {"stay_id": ep.get("stay_id")}
            ts = ep.get("timeseries", {})
            vitals = ts.get("vitals", [])
            if vitals:
                df = pd.DataFrame(vitals)
                for col in vitals_cols:
                    if col in df.columns:
                        values = pd.to_numeric(df[col], errors="coerce").dropna()
                        if len(values) > 0:
                            feat[f"{col}_mean"] = values.mean()
                            feat[f"{col}_std"] = values.std() if len(values) > 1 else 0
                            feat[f"{col}_min"] = values.min()
                            feat[f"{col}_max"] = values.max()
            features.append(feat)
        except Exception:
            continue
    return pd.DataFrame(features)


def load_annotation_features() -> pd.DataFrame:
    """Match train_fusion.py annotation features."""
    episode_files = list(EPISODES_DIR.glob("TIMELY_v2_*.json"))
    annotations = []
    for ep_file in episode_files:
        try:
            with open(ep_file) as f:
                ep = json.load(f)
            stay_id = ep.get("stay_id")
            reasoning = ep.get("reasoning", {})
            n_supportive = reasoning.get("n_supportive", 0)
            n_contradictory = reasoning.get("n_contradictory", 0)
            n_alignments = reasoning.get("n_alignments", 0)
            total_annot = n_supportive + n_contradictory
            supportive_ratio = n_supportive / total_annot if total_annot > 0 else 0.5
            annotation_density = total_annot / n_alignments if n_alignments > 0 else 0
            annotations.append({
                "stay_id": stay_id,
                "n_supportive": n_supportive,
                "n_contradictory": n_contradictory,
                "supportive_ratio": supportive_ratio,
                "annotation_density": annotation_density,
            })
        except Exception:
            continue
    return pd.DataFrame(annotations)


def check_alignment(ts_df: pd.DataFrame, annot_df: pd.DataFrame, cohort: pd.DataFrame) -> dict:
    info = {}
    info["cohort_rows"] = int(len(cohort))
    info["ts_rows"] = int(len(ts_df))
    info["annot_rows"] = int(len(annot_df))

    info["ts_dup_stay"] = int(ts_df["stay_id"].duplicated().sum())
    info["annot_dup_stay"] = int(annot_df["stay_id"].duplicated().sum())

    merged = cohort.merge(ts_df, on="stay_id", how="inner").merge(annot_df, on="stay_id", how="inner")
    info["merged_rows"] = int(len(merged))
    info["merged_dup_stay"] = int(merged["stay_id"].duplicated().sum())

    cohort_only = set(cohort["stay_id"]) - set(merged["stay_id"])
    ts_only = set(ts_df["stay_id"]) - set(merged["stay_id"])
    annot_only = set(annot_df["stay_id"]) - set(merged["stay_id"])
    info["cohort_only"] = int(len(cohort_only))
    info["ts_only"] = int(len(ts_only))
    info["annot_only"] = int(len(annot_only))
    return info, merged


def fit_predict_xgb(X_train, y_train, X_val, max_depth, random_state):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=max_depth,
        learning_rate=0.1,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    pred = model.predict_proba(X_val)[:, 1]
    return pred, model, scaler


def compute_metrics(y_true, y_pred):
    auroc = roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.5
    auprc = average_precision_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    return auroc, auprc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", default="24h")
    parser.add_argument("--cohort", default="all")
    parser.add_argument("--alpha-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    args = parser.parse_args()

    alpha_grid = [float(x) for x in args.alpha_grid.split(",")]

    cohort = pd.read_csv(COHORT_FILE)
    cohort["stay_id"] = cohort["stay_id"].astype(int)
    cohort = cohort[["stay_id", "subject_id", "label_mortality"]].dropna()

    ts_df = load_tabular_features()
    annot_df = load_annotation_features()

    info, merged = check_alignment(ts_df, annot_df, cohort)

    merged = merged.rename(columns={"label_mortality": "label"})

    ts_cols = [c for c in ts_df.columns if c != "stay_id"]
    annot_cols = ["n_supportive", "n_contradictory", "supportive_ratio", "annotation_density"]

    X_ts = merged[ts_cols].values
    X_ts = np.nan_to_num(X_ts, nan=0.0)
    X_annot = merged[annot_cols].values
    y = merged["label"].values
    groups = merged["subject_id"].values

    if USE_HOLDOUT_TEST:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train_val_idx, test_idx = next(gss.split(X_ts, y, groups=groups))
        Xs_tv, Xs_test = X_ts[train_val_idx], X_ts[test_idx]
        Xt_tv, Xt_test = X_annot[train_val_idx], X_annot[test_idx]
        y_tv, y_test = y[train_val_idx], y[test_idx]
        groups_tv = groups[train_val_idx]
    else:
        Xs_tv, Xt_tv, y_tv, groups_tv = X_ts, X_annot, y, groups
        Xs_test, Xt_test, y_test = None, None, None

    gkf = GroupKFold(n_splits=N_FOLDS)

    fold_rows = []
    best_alphas = []

    for fold, (tr_idx, val_idx) in enumerate(gkf.split(Xs_tv, y_tv, groups=groups_tv), start=1):
        # structured model (match baseline XGBoost settings)
        pred_s, model_s, scaler_s = fit_predict_xgb(Xs_tv[tr_idx], y_tv[tr_idx], Xs_tv[val_idx], 6, RANDOM_STATE)
        # annot model
        pred_t, model_t, scaler_t = fit_predict_xgb(Xt_tv[tr_idx], y_tv[tr_idx], Xt_tv[val_idx], 4, RANDOM_STATE)

        # alpha extremes
        auroc_s, auprc_s = compute_metrics(y_tv[val_idx], pred_s)
        auroc_t, auprc_t = compute_metrics(y_tv[val_idx], pred_t)

        # best alpha search
        best_alpha = None
        best_score = -1
        best_pred = None
        for alpha in alpha_grid:
            pred = alpha * pred_s + (1 - alpha) * pred_t
            score, _ = compute_metrics(y_tv[val_idx], pred)
            if score > best_score:
                best_score = score
                best_alpha = alpha
                best_pred = pred

        auroc_best, auprc_best = compute_metrics(y_tv[val_idx], best_pred)

        fold_rows.append({
            "fold": fold,
            "val_auroc_alpha1": auroc_s,
            "val_auprc_alpha1": auprc_s,
            "val_auroc_alpha0": auroc_t,
            "val_auprc_alpha0": auprc_t,
            "best_alpha": best_alpha,
            "best_val_auroc": auroc_best,
            "best_val_auprc": auprc_best,
        })
        best_alphas.append(best_alpha)

    alpha_final = float(np.mean(best_alphas)) if best_alphas else 0.5

    test_metrics = {}
    if Xs_test is not None:
        # train on all train_val, predict test
        pred_s_test, model_s, scaler_s = fit_predict_xgb(Xs_tv, y_tv, Xs_test, 6, RANDOM_STATE)
        pred_t_test, model_t, scaler_t = fit_predict_xgb(Xt_tv, y_tv, Xt_test, 4, RANDOM_STATE)

        test_metrics["test_alpha1"] = compute_metrics(y_test, pred_s_test)
        test_metrics["test_alpha0"] = compute_metrics(y_test, pred_t_test)
        pred_best = alpha_final * pred_s_test + (1 - alpha_final) * pred_t_test
        test_metrics["test_best"] = compute_metrics(y_test, pred_best)

    output = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "window": args.window,
        "cohort": args.cohort,
        "alpha_grid": alpha_grid,
        "alignment_info": info,
        "alpha_final": alpha_final,
        "folds": fold_rows,
        "test_metrics": test_metrics,
    }

    out_dir = RESULTS_DIR / "standardized"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "late_fusion_sanity.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=True)

    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
