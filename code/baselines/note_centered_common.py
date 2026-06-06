"""Shared helpers for note-centered baseline/fusion experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from utils.feature_masks import apply_feature_mask


WINDOW_CHOICES = ["W6", "W12", "W24", "D0", "leaked", "clean"]
TEXT_METHOD_CHOICES = [
    "original",
    "hard",
    "weighted",
    "original_typed",
    "hard_typed",
    "weighted_typed",
    "weighted_no_after",
    "weighted_typed_no_after",
]

NUM_FOLDS = 5
RANDOM_STATE = 42


@dataclass(frozen=True)
class WindowResolution:
    requested_window: str
    requested_text_method: str
    structured_window: str
    text_window: str
    actual_text_method: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_features_dir() -> Path:
    return project_root() / "data" / "processed" / "note_centered" / "stay_level"


def default_results_dir() -> Path:
    return project_root() / "results" / "note_centered"


def resolve_window(window: str, text_method: str) -> WindowResolution:
    if window not in WINDOW_CHOICES:
        raise ValueError(f"Unsupported window={window}; expected one of {WINDOW_CHOICES}")
    if text_method not in TEXT_METHOD_CHOICES:
        raise ValueError(
            f"Unsupported text_method={text_method}; expected one of {TEXT_METHOD_CHOICES}"
        )

    structured_window = "W24" if window == "clean" else window
    text_window = "W24" if window in ("leaked", "clean") else window
    if window == "clean":
        typed_request = bool(text_method) and ("typed" in text_method)
        actual_text_method = "weighted_typed_no_after" if typed_request else "weighted_no_after"
    else:
        actual_text_method = text_method
    return WindowResolution(
        requested_window=window,
        requested_text_method=text_method,
        structured_window=structured_window,
        text_window=text_window,
        actual_text_method=actual_text_method,
    )


def load_labels(task: str, root: Path | None = None) -> pd.DataFrame:
    if task not in {"mortality", "prolonged_los"}:
        raise ValueError("task must be one of: mortality, prolonged_los")

    root = root or project_root()
    candidates = [
        root / "data" / "processed" / "merge_output" / f"labels_{task}.csv",
        root / "data" / "processed" / f"labels_{task}.csv",
    ]

    labels = None
    for path in candidates:
        if path.exists():
            labels = pd.read_csv(path)
            break
    if labels is None:
        raise FileNotFoundError(
            f"Cannot find labels for task={task}. Tried: {', '.join(str(p) for p in candidates)}"
        )

    if "label" not in labels.columns:
        if task == "mortality" and "label_mortality" in labels.columns:
            labels = labels.rename(columns={"label_mortality": "label"})
        elif task == "prolonged_los" and "prolonged_los_7d" in labels.columns:
            labels = labels.rename(columns={"prolonged_los_7d": "label"})
        else:
            raise ValueError(
                f"Label file for task={task} is missing a usable label column. "
                f"Columns={list(labels.columns)}"
            )

    labels = labels[["stay_id", "label"]].copy()
    labels["stay_id"] = pd.to_numeric(labels["stay_id"], errors="coerce").astype("Int64")
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce")
    labels = labels.dropna(subset=["stay_id", "label"]).copy()
    labels["stay_id"] = labels["stay_id"].astype(int)
    labels["label"] = labels["label"].astype(int)
    return labels.drop_duplicates(subset=["stay_id"], keep="first")


def load_predefined_splits(root: Path | None = None) -> pd.DataFrame:
    root = root or project_root()
    split_path = root / "data" / "splits" / "predefined_splits.csv"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing canonical split file: {split_path}")

    splits = pd.read_csv(split_path)
    required = {"stay_id", "split", "fold_id"}
    missing = required - set(splits.columns)
    if missing:
        raise ValueError(f"predefined_splits.csv missing required columns: {sorted(missing)}")

    splits["stay_id"] = pd.to_numeric(splits["stay_id"], errors="coerce").astype("Int64")
    splits["fold_id"] = pd.to_numeric(splits["fold_id"], errors="coerce").astype("Int64")
    splits["split"] = splits["split"].astype(str)
    splits = splits.dropna(subset=["stay_id"]).copy()
    splits["stay_id"] = splits["stay_id"].astype(int)
    return splits.drop_duplicates(subset=["stay_id"], keep="first")


def _safe_auc_ap(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) <= 1:
        return 0.5, float(np.mean(y_true))
    return float(roc_auc_score(y_true, y_prob)), float(average_precision_score(y_true, y_prob))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0

    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def build_model(model_name: str, y_train: np.ndarray):
    model_name = model_name.lower()
    if model_name == "lr":
        return LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            solver="liblinear",
            random_state=RANDOM_STATE,
        )
    if model_name == "xgb":
        neg = max(1, int((y_train == 0).sum()))
        pos = max(1, int((y_train == 1).sum()))
        return xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=neg / pos,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
        )
    raise ValueError("model_name must be one of: lr, xgb")


def fit_predict(model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_eval: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_eval_scaled = scaler.transform(X_eval)

    model = build_model(model_name, y_train)
    model.fit(X_train_scaled, y_train)
    return model.predict_proba(X_eval_scaled)[:, 1]


def evaluate_predefined_cv(
    X: np.ndarray,
    y: np.ndarray,
    stay_ids: np.ndarray,
    split_df: pd.DataFrame,
    model_name: str,
) -> Dict:
    split_index = split_df.set_index("stay_id")[["split", "fold_id"]]
    aligned = split_index.reindex(stay_ids)

    valid_mask = aligned["split"].isin(["dev", "test"]).to_numpy()
    X = X[valid_mask]
    y = y[valid_mask]
    stay_ids = stay_ids[valid_mask]
    aligned = aligned.iloc[np.where(valid_mask)[0]].copy()

    split_vals = aligned["split"].to_numpy()
    fold_ids = pd.to_numeric(aligned["fold_id"], errors="coerce").fillna(-1).astype(int).to_numpy()

    dev_idx = np.where(split_vals == "dev")[0]
    test_idx = np.where(split_vals == "test")[0]

    fold_results = []
    for fold in range(1, NUM_FOLDS + 1):
        val_idx = np.where((split_vals == "dev") & (fold_ids == fold))[0]
        train_idx = np.where((split_vals == "dev") & (fold_ids != fold) & (fold_ids >= 1) & (fold_ids <= NUM_FOLDS))[0]
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        y_prob = fit_predict(model_name, X[train_idx], y[train_idx], X[val_idx])
        auroc, auprc = _safe_auc_ap(y[val_idx], y_prob)
        ece = expected_calibration_error(y[val_idx], y_prob)
        fold_results.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "auroc": float(auroc),
                "auprc": float(auprc),
                "ece": float(ece),
            }
        )

    if not fold_results:
        raise ValueError("No valid dev folds found under predefined_splits.csv")

    test_metrics = None
    if len(dev_idx) > 0 and len(test_idx) > 0:
        test_prob = fit_predict(model_name, X[dev_idx], y[dev_idx], X[test_idx])
        test_auroc, test_auprc = _safe_auc_ap(y[test_idx], test_prob)
        test_ece = expected_calibration_error(y[test_idx], test_prob)
        test_metrics = {
            "n_test": int(len(test_idx)),
            "auroc": float(test_auroc),
            "auprc": float(test_auprc),
            "ece": float(test_ece),
        }

    aurocs = np.array([r["auroc"] for r in fold_results], dtype=float)
    auprcs = np.array([r["auprc"] for r in fold_results], dtype=float)
    eces = np.array([r["ece"] for r in fold_results], dtype=float)

    return {
        "split_source": "predefined_splits.csv",
        "fold_results": fold_results,
        "mean_auroc": float(np.mean(aurocs)),
        "std_auroc": float(np.std(aurocs)),
        "mean_auprc": float(np.mean(auprcs)),
        "std_auprc": float(np.std(auprcs)),
        "mean_ece": float(np.mean(eces)),
        "std_ece": float(np.std(eces)),
        "test_metrics": test_metrics,
    }


def _numeric_feature_columns(df: pd.DataFrame, blocked_exact: List[str]) -> List[str]:
    cols = []
    for col in df.columns:
        if col in blocked_exact:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def _read_structured(features_dir: Path, structured_window: str, task: str) -> pd.DataFrame:
    path = features_dir / f"structured_{structured_window}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing structured feature file: {path}")
    df = pd.read_parquet(path)

    labels = load_labels(task)
    df = df.merge(labels, on="stay_id", how="inner")

    id_like = {
        "stay_id",
        "label",
        "note_idx",
        "window_start",
        "window_end",
        "window_id",
        "note_type",
    }
    protected = [c for c in df.columns if c in id_like]
    feature_part = df.drop(columns=["label"]) if "label" in df.columns else df.copy()
    masked = apply_feature_mask(feature_part, task)
    if "label" in df.columns:
        masked["label"] = df["label"].values

    feature_cols = _numeric_feature_columns(masked, blocked_exact=["stay_id", "label"])  # metadata dropped later
    # Keep full frame for downstream logic; numerical selection happens in caller.
    info_cols = [c for c in protected if c in masked.columns]
    if info_cols:
        masked[info_cols] = df[info_cols]
    return masked


def _read_text(features_dir: Path, text_window: str, text_method: str, task: str) -> pd.DataFrame:
    path = features_dir / f"text_{text_window}_{text_method}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing text feature file: {path}")
    df = pd.read_parquet(path)
    labels = load_labels(task)
    return df.merge(labels, on="stay_id", how="inner")


def load_dataset_for_modality(
    modality: str,
    window: str,
    text_method: str,
    task: str,
    features_dir: Path,
) -> Tuple[pd.DataFrame, Dict]:
    modality = modality.lower()
    resolution = resolve_window(window, text_method)

    meta = {
        "requested_window": resolution.requested_window,
        "requested_text_method": resolution.requested_text_method,
        "resolved_structured_window": resolution.structured_window,
        "resolved_text_window": resolution.text_window,
        "resolved_text_method": resolution.actual_text_method,
    }

    if modality == "structured":
        struct = _read_structured(features_dir, resolution.structured_window, task)
        meta["structured_path"] = str(features_dir / f"structured_{resolution.structured_window}.parquet")
        return struct, meta

    if modality == "text_only":
        text = _read_text(features_dir, resolution.text_window, resolution.actual_text_method, task)
        meta["text_path"] = str(
            features_dir / f"text_{resolution.text_window}_{resolution.actual_text_method}.parquet"
        )
        pre = len(text)
        if "text_has_notes" in text.columns:
            text = text[text["text_has_notes"].astype(bool)].copy()
        meta["rows_before_no_note_filter"] = int(pre)
        meta["rows_after_no_note_filter"] = int(len(text))
        return text, meta

    if modality == "fusion":
        struct = _read_structured(features_dir, resolution.structured_window, task)
        text = _read_text(features_dir, resolution.text_window, resolution.actual_text_method, task)
        merged = struct.merge(text, on=["stay_id", "label"], how="inner", suffixes=("_struct", "_text"))
        meta["structured_path"] = str(features_dir / f"structured_{resolution.structured_window}.parquet")
        meta["text_path"] = str(
            features_dir / f"text_{resolution.text_window}_{resolution.actual_text_method}.parquet"
        )
        return merged, meta

    raise ValueError("modality must be one of: structured, text_only, fusion")


def frame_to_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    if "stay_id" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain stay_id and label columns")

    blocked = {
        "stay_id",
        "label",
        "subject_id",
        "note_idx",
        "window_start",
        "window_end",
        "window_id",
        "note_type",
        "chart_hour",
    }
    feature_cols = _numeric_feature_columns(df, blocked_exact=list(blocked))
    if not feature_cols:
        raise ValueError("No numeric feature columns found after filtering")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = df["label"].to_numpy(dtype=int)
    stay_ids = df["stay_id"].to_numpy(dtype=int)
    return X, y, stay_ids, feature_cols


def load_structured_text_frames_for_fusion(
    window: str,
    text_method: str,
    task: str,
    features_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Load aligned structured/text tables for late-fusion training.

    Keeps all stays present in both tables, including no-note stays with zero vectors.
    """
    resolution = resolve_window(window, text_method)
    struct = _read_structured(features_dir, resolution.structured_window, task)
    text = _read_text(features_dir, resolution.text_window, resolution.actual_text_method, task)

    merged = struct[["stay_id", "label"]].merge(
        text[["stay_id", "label"]], on=["stay_id", "label"], how="inner"
    )
    keep_ids = set(merged["stay_id"].tolist())
    struct = struct[struct["stay_id"].isin(keep_ids)].copy()
    text = text[text["stay_id"].isin(keep_ids)].copy()

    struct = struct.sort_values("stay_id").reset_index(drop=True)
    text = text.sort_values("stay_id").reset_index(drop=True)

    if not np.array_equal(struct["stay_id"].to_numpy(), text["stay_id"].to_numpy()):
        raise ValueError("Failed to align structured/text by stay_id")

    meta = {
        "requested_window": resolution.requested_window,
        "requested_text_method": resolution.requested_text_method,
        "resolved_structured_window": resolution.structured_window,
        "resolved_text_window": resolution.text_window,
        "resolved_text_method": resolution.actual_text_method,
        "structured_path": str(features_dir / f"structured_{resolution.structured_window}.parquet"),
        "text_path": str(features_dir / f"text_{resolution.text_window}_{resolution.actual_text_method}.parquet"),
    }
    return struct, text, meta
