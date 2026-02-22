"""
Generate and save canonical predefined splits for TIMELY-Bench.

Why:
- Prevent patient leakage: split is subject-level (groups=subject_id).
- Publish deterministic splits that are reproducible from cohort + config only.

Split design (matches PROVENANCE.json):
- Holdout test: GroupShuffleSplit(test_size=TEST_SIZE, random_state=RANDOM_STATE)
- CV folds (on the dev set): GroupKFold(n_splits=N_FOLDS), deterministic order
  by sorting (subject_id, stay_id) before fold assignment.

File semantics:
- split:
  - "test": holdout test set (never used for model selection).
  - "dev": development set used for GroupKFold CV.
- fold_id:
  - For split=="dev": integer in [1..N_FOLDS]. For a given fold k, validation
    set is rows with fold_id==k, and training set is rows with fold_id!=k.
  - For split=="test": -1

Outputs (kept in sync by this script):
- data/splits/predefined_splits.csv (canonical source of truth)
- data/processed/predefined_splits.csv (backwards-compatible copy used by QA)
- final_release/predefined_splits.csv (release artifact)
- data/splits/split_summary.json (human-readable metadata)
- final_release/evidence/split_inventory.json (audit evidence)
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import COHORT_FILE, PROCESSED_DIR, ROOT_DIR, SPLITS_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE


def _assign_folds_deterministic(dev_stay_ids: np.ndarray, dev_subject_ids: np.ndarray, n_folds: int) -> np.ndarray:
    """
    Deterministic GroupKFold fold assignment independent of the original cohort row order.
    We sort by (subject_id, stay_id) before folding so repeated runs are stable.
    """
    order = np.lexsort((dev_stay_ids, dev_subject_ids))
    dev_groups = dev_subject_ids[order]
    fold_out = np.full(len(dev_stay_ids), -1, dtype=int)
    gkf = GroupKFold(n_splits=n_folds)
    X = np.zeros((len(dev_stay_ids), 1))
    for fold, (_tr_rel, val_rel) in enumerate(gkf.split(X[order], groups=dev_groups), start=1):
        fold_out[order[val_rel]] = fold
    return fold_out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 72)
    print("Generate canonical predefined splits (holdout test + CV folds)")
    print("=" * 72)

    cohort_path = Path(COHORT_FILE)
    if not cohort_path.exists():
        raise FileNotFoundError(f"Missing cohort file: {cohort_path}")

    cohort = pd.read_csv(cohort_path)
    required_cols = {"stay_id", "subject_id"}
    missing_cols = required_cols - set(cohort.columns)
    if missing_cols:
        raise ValueError(f"cohort file missing required columns: {sorted(missing_cols)}")

    stay_ids = cohort["stay_id"].astype(int).values
    subject_ids = cohort["subject_id"].astype(int).values

    n_total = len(stay_ids)
    print(f"Cohort: {n_total:,} stays, {cohort['subject_id'].nunique():,} subjects")
    print(f"Config: TEST_SIZE={TEST_SIZE}, N_FOLDS={N_FOLDS}, RANDOM_STATE={RANDOM_STATE}")

    # Holdout test split (subject-level)
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    dev_idx, test_idx = next(gss.split(np.zeros((n_total, 1)), groups=subject_ids))

    dev_subjects = set(subject_ids[dev_idx].tolist())
    test_subjects = set(subject_ids[test_idx].tolist())
    overlap = dev_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"Subject leakage in holdout split: {len(overlap)} overlapping subjects")

    split = np.full(n_total, "dev", dtype=object)
    split[test_idx] = "test"

    fold_id = np.full(n_total, -1, dtype=int)

    # CV folds on dev set (subject-level), deterministic from raw cohort + config only.
    fold_source = "deterministic_groupkfold_sorted"
    dev_fold = _assign_folds_deterministic(
        dev_stay_ids=stay_ids[dev_idx],
        dev_subject_ids=subject_ids[dev_idx],
        n_folds=N_FOLDS,
    )
    fold_id[dev_idx] = dev_fold

    if (fold_id[dev_idx] < 1).any():
        missing = int((fold_id[dev_idx] < 1).sum())
        raise RuntimeError(f"Fold assignment incomplete: {missing} dev rows missing fold_id")

    out_df = pd.DataFrame(
        {
            "stay_id": stay_ids,
            "subject_id": subject_ids,
            "split": split,
            "fold_id": fold_id,
        }
    ).sort_values(["split", "fold_id", "subject_id", "stay_id"], kind="mergesort")

    # Validate subject -> single fold mapping (within dev)
    dev_df = out_df[out_df["split"] == "dev"]
    multi_fold = dev_df.groupby("subject_id")["fold_id"].nunique()
    n_multi = int((multi_fold > 1).sum())
    if n_multi:
        raise RuntimeError(f"{n_multi} subjects appear in multiple folds (should not happen)")

    # Paths
    splits_dir = Path(SPLITS_DIR)
    processed_dir = Path(PROCESSED_DIR)
    final_release_dir = Path(ROOT_DIR) / "final_release"
    evidence_dir = final_release_dir / "evidence"
    for d in (splits_dir, processed_dir, final_release_dir, evidence_dir):
        d.mkdir(parents=True, exist_ok=True)

    canonical_path = splits_dir / "predefined_splits.csv"
    processed_copy_path = processed_dir / "predefined_splits.csv"
    release_path = final_release_dir / "predefined_splits.csv"

    out_df.to_csv(canonical_path, index=False)
    processed_copy_path.write_bytes(canonical_path.read_bytes())
    release_path.write_bytes(canonical_path.read_bytes())

    print("\nWrote split files:")
    print(f"  - {canonical_path}")
    print(f"  - {processed_copy_path} (copy)")
    print(f"  - {release_path} (release)")

    # Summary stats (core labels if present)
    def _rate(series):
        s = pd.to_numeric(series, errors="coerce")
        s = s.dropna()
        if len(s) == 0:
            return None
        return float(s.mean())

    mortality_col = "label_mortality" if "label_mortality" in cohort.columns else None
    plos_col = "prolonged_los_7d" if "prolonged_los_7d" in cohort.columns else None

    def _split_stats(mask):
        sub = cohort.loc[mask]
        stats = {
            "n_episodes": int(len(sub)),
            "n_subjects": int(sub["subject_id"].nunique()),
        }
        if mortality_col:
            stats["mortality_rate"] = _rate(sub[mortality_col])
        if plos_col:
            stats["prolonged_los_rate"] = _rate(sub[plos_col])
        return stats

    is_test = out_df["split"].values == "test"
    is_dev = ~is_test

    summary = {
        "version": "2.3_canonical_holdout_cv",
        "created": datetime.now(timezone.utc).isoformat(),
        "random_state": int(RANDOM_STATE),
        "test_size": float(TEST_SIZE),
        "n_folds": int(N_FOLDS),
        "split_method": "subject_id_grouped_holdout_plus_cv",
        "fold_assignment_source": fold_source,
        "groups_column": "subject_id",
        "total_episodes": int(n_total),
        "statistics": {
            "dev": _split_stats(is_dev),
            "test": _split_stats(is_test),
        },
        "folds": {},
    }

    # Fold stats (dev only)
    for f in range(1, N_FOLDS + 1):
        m = (out_df["split"].values == "dev") & (out_df["fold_id"].values == f)
        summary["folds"][str(f)] = _split_stats(m)

    summary_path = splits_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"\nWrote {summary_path}")

    # Inventory evidence for final_release
    sha = _sha256_file(release_path)
    inventory = {
        "check_name": "SPLIT_INVENTORY",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "split_files": [
            {
                "path": release_path.name,
                "n_rows": int(len(out_df)),
                "columns": out_df.columns.tolist(),
                "sha256": sha,
            }
        ],
        "config_params": {
            "TEST_SIZE": str(TEST_SIZE),
            "N_FOLDS": str(N_FOLDS),
            "RANDOM_STATE": str(RANDOM_STATE),
        },
        "canonical_split": release_path.name,
        "verdict": "PASS",
    }
    inv_path = evidence_dir / "split_inventory.json"
    inv_path.write_text(json.dumps(inventory, indent=2, ensure_ascii=True))
    print(f"Wrote {inv_path}")


if __name__ == "__main__":
    main()
