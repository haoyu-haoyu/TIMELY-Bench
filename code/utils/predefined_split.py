"""Utilities for canonical predefined split usage.

Canonical split source:
- data/splits/predefined_splits.csv

Columns:
- stay_id
- split: dev/test
- fold_id: 1..N_FOLDS for dev, null for test
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import N_FOLDS, PREDEFINED_SPLITS_FILE, USE_HOLDOUT_TEST


_SPLIT_CACHE = None


def load_predefined_split_index() -> pd.DataFrame:
    global _SPLIT_CACHE
    if _SPLIT_CACHE is not None:
        return _SPLIT_CACHE

    split_path = Path(PREDEFINED_SPLITS_FILE)
    if not split_path.exists():
        raise FileNotFoundError(
            f"Missing canonical split file: {split_path}. "
            "Run code/data_processing/generate_predefined_splits.py first."
        )

    df = pd.read_csv(split_path)
    required = {"stay_id", "split", "fold_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split file missing required columns: {sorted(missing)}")

    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df["fold_id"] = pd.to_numeric(df["fold_id"], errors="coerce").astype("Int64")
    df["split"] = df["split"].astype(str)
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = df["stay_id"].astype(int)
    df = df.drop_duplicates(subset=["stay_id"], keep="first")
    _SPLIT_CACHE = df.set_index("stay_id")[["split", "fold_id"]]
    return _SPLIT_CACHE


def resolve_predefined_partition(stay_ids) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Resolve canonical dev/test/fold assignment for an array-like of stay_ids.

    Returns:
    - train_val_idx: indices in input stay_ids assigned to dev (or all valid if no holdout)
    - test_idx: indices in input stay_ids assigned to test (empty if USE_HOLDOUT_TEST=False)
    - fold_ids: per-row fold id, -1 for non-dev rows
    - split_info: summary metadata
    """

    stay_ids = np.asarray(stay_ids, dtype=int)
    split_idx = load_predefined_split_index()
    aligned = split_idx.reindex(stay_ids)

    split_vals = aligned["split"].to_numpy(dtype=object)
    fold_vals = pd.to_numeric(aligned["fold_id"], errors="coerce").fillna(-1).astype(int).to_numpy()

    valid = np.isin(split_vals, ["dev", "test"])
    missing_count = int((~valid).sum())

    dev_mask = valid & (split_vals == "dev")
    test_mask = valid & (split_vals == "test")

    if USE_HOLDOUT_TEST:
        train_val_idx = np.where(dev_mask)[0]
        test_idx = np.where(test_mask)[0]
    else:
        train_val_idx = np.where(valid)[0]
        test_idx = np.array([], dtype=int)

    if len(train_val_idx) == 0:
        raise ValueError("No dev rows available after applying predefined splits")
    if USE_HOLDOUT_TEST and len(test_idx) == 0:
        raise ValueError("No test rows available after applying predefined splits")

    fold_ids = np.full(len(stay_ids), -1, dtype=int)
    fold_ids[train_val_idx] = fold_vals[train_val_idx]

    bad_dev = train_val_idx[(fold_ids[train_val_idx] < 1) | (fold_ids[train_val_idx] > N_FOLDS)]
    if len(bad_dev) > 0:
        raise ValueError(f"Invalid fold_id for {len(bad_dev)} dev rows (expected 1..{N_FOLDS})")

    split_info = {
        "source": "predefined_splits.csv",
        "n_rows_total": int(len(stay_ids)),
        "n_rows_valid": int(valid.sum()),
        "n_rows_dev": int(len(train_val_idx)),
        "n_rows_test": int(len(test_idx)),
        "n_rows_dropped_missing_split": missing_count,
    }
    return train_val_idx, test_idx, fold_ids, split_info
