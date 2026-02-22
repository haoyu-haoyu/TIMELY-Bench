"""
Note-category ablation using alignment-derived note-type features.

Requirement reference: `作业要求.md` -> A4 includes "ablation by note category".

This script builds per-stay note-category features from
`temporal_textual_alignment.csv` (canonical 24h alignment source), then runs:
- all included note types
- only one note type
- leave-one-out by note type

Canonical policy:
- discharge notes excluded by default
- patient-level split from predefined_splits.csv (dev/test + fold_id)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from config import (
    COHORT_FILE,
    N_FOLDS,
    PREDEFINED_SPLITS_FILE,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)


ALIGNMENT_FILE = PROCESSED_DIR / "temporal_alignment" / "temporal_textual_alignment.csv"

OUTPUT_DIR = RESULTS_DIR / "note_ablation"
OUTPUT_CSV = OUTPUT_DIR / "note_ablation_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "note_ablation_results.json"

DEFAULT_INCLUDED_TYPES = ("nursing", "radiology", "lab_comment")
DEFAULT_NOTE_TYPES = ("nursing", "radiology", "lab_comment")
QUALITY_SCORE = {"low": 0.0, "medium": 1.0, "high": 2.0}


def _safe_metrics(y_true: np.ndarray, y_prob: np.ndarray):
    if len(np.unique(y_true)) <= 1:
        return 0.5, float(np.mean(y_true))
    return roc_auc_score(y_true, y_prob), average_precision_score(y_true, y_prob)


def _load_base_frame() -> pd.DataFrame:
    if not COHORT_FILE.exists():
        raise FileNotFoundError(COHORT_FILE)
    if not PREDEFINED_SPLITS_FILE.exists():
        raise FileNotFoundError(PREDEFINED_SPLITS_FILE)

    cohort = pd.read_csv(COHORT_FILE)
    cohort["stay_id"] = pd.to_numeric(cohort["stay_id"], errors="coerce").astype("Int64")
    cohort["subject_id"] = pd.to_numeric(cohort["subject_id"], errors="coerce").astype("Int64")

    splits = pd.read_csv(PREDEFINED_SPLITS_FILE)
    splits["stay_id"] = pd.to_numeric(splits["stay_id"], errors="coerce").astype("Int64")
    splits["fold_id"] = pd.to_numeric(splits["fold_id"], errors="coerce").astype("Int64")

    df = cohort.merge(splits[["stay_id", "split", "fold_id"]], on="stay_id", how="inner")
    keep_cols = [
        "stay_id",
        "subject_id",
        "label_mortality",
        "prolonged_los_7d",
        "split",
        "fold_id",
    ]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in merged cohort/splits: {missing}")

    df = df[keep_cols].copy()
    df = df.dropna(subset=["stay_id", "subject_id", "split", "fold_id"])
    df["stay_id"] = df["stay_id"].astype(int)
    df["subject_id"] = df["subject_id"].astype(int)
    df["fold_id"] = df["fold_id"].astype(int)
    df["label_mortality"] = pd.to_numeric(df["label_mortality"], errors="coerce")
    df["prolonged_los_7d"] = pd.to_numeric(df["prolonged_los_7d"], errors="coerce")
    return df


def _train_eval_predefined(df: pd.DataFrame, feature_cols, label_col: str, model_name: str):
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)

    dev = df[df["split"] == "dev"].copy()
    test = df[df["split"] == "test"].copy()
    if len(dev) == 0 or len(test) == 0:
        raise ValueError("Split file must contain both dev and test rows")

    X_dev = dev[feature_cols].to_numpy(dtype=float)
    y_dev = dev[label_col].to_numpy(dtype=int)
    fold_dev = dev["fold_id"].to_numpy(dtype=int)

    fold_details = []
    for fold in range(1, N_FOLDS + 1):
        tr = fold_dev != fold
        va = fold_dev == fold
        if not va.any() or not tr.any():
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_dev[tr])
        X_va = scaler.transform(X_dev[va])

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

        model.fit(X_tr, y_dev[tr])
        pred = model.predict_proba(X_va)[:, 1]
        auroc, auprc = _safe_metrics(y_dev[va], pred)
        fold_details.append({"fold": fold, "auroc": float(auroc), "auprc": float(auprc)})

    scaler = StandardScaler()
    X_tv = scaler.fit_transform(X_dev)
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
    model.fit(X_tv, y_dev)

    X_te = test[feature_cols].to_numpy(dtype=float)
    y_te = test[label_col].to_numpy(dtype=int)
    X_te = scaler.transform(X_te)
    pred_te = model.predict_proba(X_te)[:, 1]
    test_auroc, test_auprc = _safe_metrics(y_te, pred_te)

    aurocs = [r["auroc"] for r in fold_details]
    auprcs = [r["auprc"] for r in fold_details]
    return {
        "cv_auroc_mean": float(np.mean(aurocs)) if aurocs else None,
        "cv_auroc_std": float(np.std(aurocs)) if aurocs else None,
        "cv_auprc_mean": float(np.mean(auprcs)) if auprcs else None,
        "cv_auprc_std": float(np.std(auprcs)) if auprcs else None,
        "test_auroc": float(test_auroc),
        "test_auprc": float(test_auprc),
        "fold_details": fold_details,
        "n_samples": int(len(df)),
        "positive_rate": float(np.mean(df[label_col])),
    }


def _build_alignment_note_features(
    alignment_file: Path,
    stay_ids: list[int],
    note_types: list[str],
    chunksize: int,
) -> tuple[pd.DataFrame, dict]:
    if not alignment_file.exists():
        raise FileNotFoundError(alignment_file)

    stay_id_to_index = {sid: i for i, sid in enumerate(stay_ids)}
    n_stays = len(stay_ids)

    n_align = {nt: np.zeros(n_stays, dtype=np.int32) for nt in note_types}
    sum_abs_delta = {nt: np.zeros(n_stays, dtype=np.float64) for nt in note_types}
    high_count = {nt: np.zeros(n_stays, dtype=np.float64) for nt in note_types}
    sum_quality = {nt: np.zeros(n_stays, dtype=np.float64) for nt in note_types}

    usecols = ["stay_id", "note_type", "time_delta_hours", "alignment_quality"]
    for chunk in pd.read_csv(alignment_file, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
        chunk = chunk.dropna(subset=["stay_id", "note_type"])
        if chunk.empty:
            continue

        chunk["stay_id"] = chunk["stay_id"].astype(int)
        chunk["note_type"] = chunk["note_type"].astype(str).str.lower()
        chunk = chunk[chunk["note_type"].isin(note_types)]
        if chunk.empty:
            continue

        chunk["time_delta_hours"] = pd.to_numeric(chunk["time_delta_hours"], errors="coerce").fillna(0.0)
        chunk["alignment_quality"] = chunk["alignment_quality"].astype(str).str.lower()
        chunk["quality_score"] = chunk["alignment_quality"].map(QUALITY_SCORE).fillna(0.0)
        chunk["high_flag"] = (chunk["alignment_quality"] == "high").astype(float)
        chunk["stay_idx"] = chunk["stay_id"].map(stay_id_to_index)
        chunk = chunk.dropna(subset=["stay_idx"])
        if chunk.empty:
            continue
        chunk["stay_idx"] = chunk["stay_idx"].astype(int)

        for note_type in note_types:
            part = chunk[chunk["note_type"] == note_type]
            if part.empty:
                continue
            idx = part["stay_idx"].to_numpy(dtype=int)
            np.add.at(n_align[note_type], idx, 1)
            np.add.at(sum_abs_delta[note_type], idx, part["time_delta_hours"].abs().to_numpy(dtype=float))
            np.add.at(high_count[note_type], idx, part["high_flag"].to_numpy(dtype=float))
            np.add.at(sum_quality[note_type], idx, part["quality_score"].to_numpy(dtype=float))

    feat = pd.DataFrame({"stay_id": stay_ids})
    coverage = {}
    for note_type in note_types:
        cnt = n_align[note_type].astype(float)
        feat[f"{note_type}_n_alignments"] = cnt
        feat[f"{note_type}_mean_abs_delta_hours"] = np.divide(
            sum_abs_delta[note_type], cnt, out=np.zeros_like(cnt), where=cnt > 0
        )
        feat[f"{note_type}_high_quality_ratio"] = np.divide(
            high_count[note_type], cnt, out=np.zeros_like(cnt), where=cnt > 0
        )
        feat[f"{note_type}_mean_quality_score"] = np.divide(
            sum_quality[note_type], cnt, out=np.zeros_like(cnt), where=cnt > 0
        )
        coverage[note_type] = float((cnt > 0).mean())
    return feat, coverage


def _feature_cols_for_types(note_types: list[str]) -> list[str]:
    cols = []
    for t in note_types:
        cols.extend(
            [
                f"{t}_n_alignments",
                f"{t}_mean_abs_delta_hours",
                f"{t}_high_quality_ratio",
                f"{t}_mean_quality_score",
            ]
        )
    return cols


def _experiment_nonzero_rate(df: pd.DataFrame, note_types: list[str]) -> float:
    if not note_types:
        return 0.0
    count_cols = [f"{t}_n_alignments" for t in note_types]
    active = (df[count_cols].sum(axis=1) > 0).astype(float)
    return float(active.mean())


def main():
    parser = argparse.ArgumentParser(description="Note-category ablation from temporal alignment features")
    parser.add_argument(
        "--include-discharge",
        action="store_true",
        help="Opt-in non-canonical mode to include discharge notes.",
    )
    parser.add_argument(
        "--alignment-file",
        default=str(ALIGNMENT_FILE),
        help="Path to temporal_textual_alignment.csv",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=250000,
        help="CSV chunksize for alignment feature build.",
    )
    parser.add_argument(
        "--low-coverage-threshold",
        type=float,
        default=0.01,
        help="Flag note-type coverage below this stay-level non-zero rate.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base = _load_base_frame()
    stay_ids = base["stay_id"].tolist()

    note_types = list(DEFAULT_NOTE_TYPES)
    if args.include_discharge:
        note_types.append("discharge")

    align_feat, coverage_by_type = _build_alignment_note_features(
        alignment_file=Path(args.alignment_file),
        stay_ids=stay_ids,
        note_types=note_types,
        chunksize=args.chunksize,
    )

    base_feat = base.merge(align_feat, on="stay_id", how="left").fillna(0.0)

    tasks = {
        "mortality": "label_mortality",
        "prolonged_los": "prolonged_los_7d",
    }
    models = ["XGBoost (AlignmentStats)", "LogisticRegression (AlignmentStats)"]

    experiments = []
    experiments.append(("all_included", list(DEFAULT_INCLUDED_TYPES)))
    for t in note_types:
        experiments.append((f"only_{t}", [t]))
    for t in DEFAULT_INCLUDED_TYPES:
        keep = [x for x in DEFAULT_INCLUDED_TYPES if x != t]
        experiments.append((f"exclude_{t}", keep))

    low_coverage_note_types = [k for k, v in coverage_by_type.items() if v < args.low_coverage_threshold]

    results_rows = []
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": int(RANDOM_STATE),
        "input_paths": {
            "cohort_file": str(COHORT_FILE),
            "splits_file": str(PREDEFINED_SPLITS_FILE),
            "alignment_file": str(Path(args.alignment_file)),
        },
        "feature_source": "temporal_textual_alignment",
        "feature_definition": [
            "n_alignments",
            "mean_abs_delta_hours",
            "high_quality_ratio",
            "mean_quality_score",
        ],
        "note_types": note_types,
        "default_included_types": list(DEFAULT_INCLUDED_TYPES),
        "canonical_policy": {
            "discharge_excluded": not args.include_discharge,
            "include_discharge_override": bool(args.include_discharge),
            "low_coverage_threshold": float(args.low_coverage_threshold),
        },
        "feature_coverage_by_note_type": coverage_by_type,
        "low_coverage_note_types": low_coverage_note_types,
        "tasks": list(tasks.keys()),
        "results": [],
    }

    for exp_name, exp_types in experiments:
        feature_cols = _feature_cols_for_types(exp_types)
        if not feature_cols:
            continue
        exp_nonzero_rate = _experiment_nonzero_rate(base_feat, exp_types)

        for task_name, label_col in tasks.items():
            for model_name in models:
                metrics = _train_eval_predefined(base_feat, feature_cols, label_col, model_name)
                row = {
                    "ablation": exp_name,
                    "note_types": ",".join(exp_types),
                    "task": task_name,
                    "model": model_name,
                    "n_samples": metrics["n_samples"],
                    "positive_rate": metrics["positive_rate"],
                    "cv_auroc_mean": metrics["cv_auroc_mean"],
                    "cv_auroc_std": metrics["cv_auroc_std"],
                    "cv_auprc_mean": metrics["cv_auprc_mean"],
                    "cv_auprc_std": metrics["cv_auprc_std"],
                    "test_auroc": metrics["test_auroc"],
                    "test_auprc": metrics["test_auprc"],
                    "feature_nonzero_rate": exp_nonzero_rate,
                    "low_coverage_flag": exp_nonzero_rate < args.low_coverage_threshold,
                }
                results_rows.append(row)
                payload["results"].append({**row, "fold_details": metrics["fold_details"]})

    out_df = pd.DataFrame(results_rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=True))

    print(f"Saved: {OUTPUT_CSV}")
    print(f"Saved: {OUTPUT_JSON}")
    if low_coverage_note_types:
        print("Warning: low note-type coverage detected: " + ",".join(low_coverage_note_types))


if __name__ == "__main__":
    main()
