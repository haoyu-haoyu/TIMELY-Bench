"""
Final QA checks for leakage and label consistency.
Run after regenerating datasets.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import COHORT_FILE, TEMPORAL_ALIGNMENT_DIR, ROOT_DIR, TIMESERIES_FILE, N_FOLDS


EPISODES_DIR = ROOT_DIR / 'episodes' / 'episodes_enhanced'


def check_cohort_labels(path: Path) -> bool:
    ok = True
    if not path.exists():
        print(f"[WARN] Missing cohort file: {path}")
        return False

    df = pd.read_csv(path)
    if 'label_mortality' not in df.columns or 'readmission_30d' not in df.columns:
        print("[WARN] cohort file missing required columns for label QA.")
        return False

    conflict = df[(df['label_mortality'] == 1) & (df['readmission_30d'] == 1)]
    if len(conflict) > 0:
        print(f"[FAIL] mortality=1 & readmission_30d=1: {len(conflict)} rows")
        ok = False
    else:
        print("[PASS] no mortality/readmission conflicts")

    return ok


def check_alignment_files(alignment_dir: Path, max_hour: int = 24) -> bool:
    ok = True
    if not alignment_dir.exists():
        print(f"[WARN] Missing alignment dir: {alignment_dir}")
        return False

    csv_files = sorted(alignment_dir.glob("*.csv"))
    if not csv_files:
        print(f"[WARN] No alignment CSV files found in {alignment_dir}")
        return False

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"[WARN] Failed to read {csv_path}: {exc}")
            ok = False
            continue

        hour_cols = [c for c in ['note_hour', 'chart_hour', 'hour_offset'] if c in df.columns]
        for col in hour_cols:
            over = df[df[col] >= max_hour]
            if len(over) > 0:
                print(f"[FAIL] {csv_path.name}: {len(over)} rows with {col} >= {max_hour}")
                ok = False

        if 'note_type' in df.columns:
            discharge = df[df['note_type'].astype(str).str.lower().str.contains('discharge', na=False)]
            if len(discharge) > 0:
                print(f"[FAIL] {csv_path.name}: {len(discharge)} discharge notes present")
                ok = False

    if ok:
        print("[PASS] alignment files within window and no discharge notes")

    return ok


def check_episode_windows(episodes_dir: Path, max_hour: int = 24, sample: int = None) -> bool:
    ok = True
    if not episodes_dir.exists():
        print(f"[WARN] Missing episodes dir: {episodes_dir}")
        return False

    episode_files = sorted(episodes_dir.glob("TIMELY_v2_*.json"))
    if sample:
        episode_files = episode_files[:sample]
        print(f"[INFO] Sampling {len(episode_files)} episodes for QA")
    else:
        print(f"[INFO] Scanning {len(episode_files)} episodes for QA")

    for ep_file in episode_files:
        try:
            with open(ep_file) as f:
                ep = json.load(f)
        except Exception as exc:
            print(f"[WARN] Failed to read {ep_file.name}: {exc}")
            ok = False
            continue

        vitals = ep.get('timeseries', {}).get('vitals', [])
        labs = ep.get('timeseries', {}).get('labs', [])
        notes = ep.get('clinical_text', {}).get('notes', [])

        for item in vitals:
            hour = item.get('hour')
            if hour is not None and hour >= max_hour:
                print(f"[FAIL] {ep_file.name}: vitals hour {hour} >= {max_hour}")
                ok = False
                break

        for item in labs:
            hour = item.get('hour')
            if hour is not None and hour >= max_hour:
                print(f"[FAIL] {ep_file.name}: labs hour {hour} >= {max_hour}")
                ok = False
                break

        for note in notes:
            hour = note.get('chart_hour')
            note_type = str(note.get('note_type', '')).lower()
            if hour is not None and hour >= max_hour:
                print(f"[FAIL] {ep_file.name}: note hour {hour} >= {max_hour}")
                ok = False
                break
            if 'discharge' in note_type:
                print(f"[FAIL] {ep_file.name}: discharge note present")
                ok = False
                break

    if ok:
        print("[PASS] episodes within window and no discharge notes")

    return ok


def check_timeseries_order(path: Path, max_rows: int = None) -> bool:
    if max_rows is None:
        print("[SKIP] timeseries order check (use --timeseries-rows to enable)")
        return True
    if not path.exists():
        print(f"[WARN] Missing timeseries file: {path}")
        return False

    usecols = ["stay_id", "hour"]
    df = pd.read_csv(path, usecols=usecols, nrows=max_rows)
    if df.empty:
        print("[WARN] timeseries file empty")
        return False

    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

    ordered = df.sort_values(["stay_id", "hour"], kind="mergesort")
    if not ordered[usecols].reset_index(drop=True).equals(df[usecols].reset_index(drop=True)):
        print(f"[FAIL] timeseries order not sorted in {path.name}")
        return False

    print("[PASS] timeseries order sorted by stay_id/hour")
    return True


def check_subject_split(root_dir: Path, cohort_path: Path) -> bool:
    """检查 train/val/test 的 subject_id 是否有交集"""
    ok = True

    if not cohort_path.exists():
        print(f"[WARN] Missing cohort file for split check: {cohort_path}")
        return False

    cohort = pd.read_csv(cohort_path, usecols=['stay_id', 'subject_id'])
    cohort['stay_id'] = pd.to_numeric(cohort['stay_id'], errors='coerce')
    cohort['subject_id'] = pd.to_numeric(cohort['subject_id'], errors='coerce')

    # Prefer canonical predefined split file (data/splits).
    predefined = root_dir / 'data' / 'splits' / 'predefined_splits.csv'
    if not predefined.exists():
        predefined = root_dir / 'data' / 'processed' / 'predefined_splits.csv'
    if predefined.exists():
        splits_df = pd.read_csv(predefined)
        if 'stay_id' not in splits_df.columns or 'split' not in splits_df.columns:
            print("[WARN] predefined_splits.csv missing stay_id/split columns")
            return False

        # If subject_id not present, join from cohort (required for leakage check).
        if 'subject_id' not in splits_df.columns:
            merged = splits_df.merge(cohort, on='stay_id', how='left')
        else:
            merged = splits_df.copy()

        # Canonical schema: split in {dev,test} with fold_id for dev rows.
        split_values = set(merged['split'].astype(str).str.lower().unique().tolist())
        if split_values.issubset({'dev', 'test'}) and 'fold_id' in merged.columns:
            dev = merged[merged['split'].astype(str).str.lower() == 'dev']
            test = merged[merged['split'].astype(str).str.lower() == 'test']

            dev_subjects = set(dev['subject_id'].dropna())
            test_subjects = set(test['subject_id'].dropna())
            inter = dev_subjects & test_subjects
            if inter:
                print(f"[FAIL] subject_id overlap between dev/test: {len(inter)}")
                ok = False

            # Each subject must map to a single fold_id.
            fold_nunique = dev.groupby('subject_id')['fold_id'].nunique()
            n_multi = int((fold_nunique > 1).sum())
            if n_multi > 0:
                print(f"[FAIL] {n_multi} subjects appear in multiple fold_id values")
                ok = False

            # fold_id range sanity
            bad_fold = dev[(dev['fold_id'] < 1) | (dev['fold_id'] > N_FOLDS)]
            if len(bad_fold) > 0:
                print(f"[FAIL] {len(bad_fold)} dev rows have invalid fold_id (expected 1..{N_FOLDS})")
                ok = False

            if ok:
                print("[PASS] subject_id split check (predefined_splits.csv: dev/test + fold_id)")
            return ok

    print("[WARN] No predefined_splits.csv found for subject_id check")
    return False


def check_split_summary_consistency(root_dir: Path) -> bool:
    """
    Ensure the canonical split summary and final_release copy are consistent.
    This prevents 2.2/2.3 style metadata drift.
    """
    canonical = root_dir / "data" / "splits" / "split_summary.json"
    release = root_dir / "final_release" / "evidence" / "split_summary.json"

    if not canonical.exists():
        print(f"[WARN] Missing canonical split summary: {canonical}")
        return False
    if not release.exists():
        print(f"[WARN] Missing release split summary: {release}")
        return False

    try:
        c = json.loads(canonical.read_text())
        r = json.loads(release.read_text())
    except Exception as exc:
        print(f"[WARN] Failed to parse split summary JSON: {exc}")
        return False

    keys = [
        "version",
        "test_size",
        "n_folds",
        "split_method",
        "fold_assignment_source",
        "groups_column",
        "total_episodes",
    ]
    mismatches = []
    for key in keys:
        if c.get(key) != r.get(key):
            mismatches.append((key, c.get(key), r.get(key)))

    if mismatches:
        print("[FAIL] split_summary mismatch between canonical and final_release:")
        for key, lhs, rhs in mismatches:
            print(f"   - {key}: canonical={lhs} | release={rhs}")
        return False

    print("[PASS] split_summary canonical/release are consistent")
    return True


def main():
    parser = argparse.ArgumentParser(description="Final QA checks for TIMELY-Bench")
    parser.add_argument("--sample", type=int, default=None, help="sample N episodes to speed up QA")
    parser.add_argument("--timeseries-rows", type=int, default=None, help="rows to sample for timeseries order check")
    args = parser.parse_args()

    ok = True
    ok &= check_cohort_labels(Path(COHORT_FILE))
    ok &= check_alignment_files(Path(TEMPORAL_ALIGNMENT_DIR))
    ok &= check_episode_windows(EPISODES_DIR, sample=args.sample)
    ok &= check_timeseries_order(Path(TIMESERIES_FILE), max_rows=args.timeseries_rows)
    ok &= check_subject_split(Path(ROOT_DIR), Path(COHORT_FILE))
    ok &= check_split_summary_consistency(Path(ROOT_DIR))

    if ok:
        print("\n[OK] Final QA checks passed")
    else:
        print("\n[WARN] Final QA checks found issues")


if __name__ == "__main__":
    main()
