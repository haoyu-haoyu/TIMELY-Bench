"""
Final QA checks for leakage and label consistency.
Run after regenerating datasets.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from config import COHORT_FILE, TEMPORAL_ALIGNMENT_DIR, ROOT_DIR, TIMESERIES_FILE


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

    # 优先检查 predefined_splits.csv
    predefined = root_dir / 'data' / 'processed' / 'predefined_splits.csv'
    if predefined.exists():
        splits_df = pd.read_csv(predefined)
        if 'stay_id' not in splits_df.columns or 'split' not in splits_df.columns:
            print("[WARN] predefined_splits.csv missing stay_id/split columns")
            return False

        merged = splits_df.merge(cohort, on='stay_id', how='left')
        train_subjects = set(merged[merged['split'] == 'train']['subject_id'].dropna())
        val_subjects = set(merged[merged['split'] == 'val']['subject_id'].dropna())
        test_subjects = set(merged[merged['split'] == 'test']['subject_id'].dropna())

        if train_subjects & val_subjects:
            print(f"[FAIL] subject_id overlap between train/val: {len(train_subjects & val_subjects)}")
            ok = False
        if train_subjects & test_subjects:
            print(f"[FAIL] subject_id overlap between train/test: {len(train_subjects & test_subjects)}")
            ok = False
        if val_subjects & test_subjects:
            print(f"[FAIL] subject_id overlap between val/test: {len(val_subjects & test_subjects)}")
            ok = False

        if ok:
            print("[PASS] subject_id split check (predefined_splits.csv)")
        return ok

    # 备选：data/splits/{train,val,test}.csv
    splits_dir = root_dir / 'data' / 'splits'
    train_path = splits_dir / 'train.csv'
    val_path = splits_dir / 'val.csv'
    test_path = splits_dir / 'test.csv'
    if train_path.exists() and val_path.exists() and test_path.exists():
        train = pd.read_csv(train_path)
        val = pd.read_csv(val_path)
        test = pd.read_csv(test_path)

        for df in (train, val, test):
            if 'stay_id' not in df.columns:
                print("[WARN] split file missing stay_id column")
                return False

        train_sub = set(cohort[cohort['stay_id'].isin(train['stay_id'])]['subject_id'].dropna())
        val_sub = set(cohort[cohort['stay_id'].isin(val['stay_id'])]['subject_id'].dropna())
        test_sub = set(cohort[cohort['stay_id'].isin(test['stay_id'])]['subject_id'].dropna())

        if train_sub & val_sub:
            print(f"[FAIL] subject_id overlap between train/val: {len(train_sub & val_sub)}")
            ok = False
        if train_sub & test_sub:
            print(f"[FAIL] subject_id overlap between train/test: {len(train_sub & test_sub)}")
            ok = False
        if val_sub & test_sub:
            print(f"[FAIL] subject_id overlap between val/test: {len(val_sub & test_sub)}")
            ok = False

        if ok:
            print("[PASS] subject_id split check (data/splits)")
        return ok

    print("[WARN] No split files found for subject_id check")
    return False


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

    if ok:
        print("\n[OK] Final QA checks passed")
    else:
        print("\n[WARN] Final QA checks found issues")


if __name__ == "__main__":
    main()
