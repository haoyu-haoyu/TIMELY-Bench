#!/usr/bin/env python3
"""
Full-population subject-level leakage audit.
Reproduces the exact GroupShuffleSplit + GroupKFold splits used in training
and verifies zero subject intersection between every train/val/test partition.
Also produces a subject multiplicity report.
"""
import json, csv, sys
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR, N_FOLDS, RANDOM_STATE, TEST_SIZE

from sklearn.model_selection import GroupKFold, GroupShuffleSplit

ROOT = ROOT_DIR
AUDIT_DIR = ROOT / "results" / "audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load cohort ──────────────────────────────────────────────────
cohort_path = ROOT / "data" / "processed" / "merge_output" / "cohort_final.csv"
print(f"Loading cohort from {cohort_path} ...")

rows = []
with open(cohort_path, newline="") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

stay_ids   = np.array([int(r["stay_id"])   for r in rows])
subject_ids = np.array([int(r["subject_id"]) for r in rows])

label_cols = {
    "mortality":     [int(r["label_mortality"]) for r in rows],
    "prolonged_los": [int(r.get("prolonged_los_3d", r.get("prolonged_los", "0"))) for r in rows],
}
# readmission_30d may be NaN for deceased patients
readm = []
for r in rows:
    val = r.get("readmission_30d", "")
    if val == "" or val is None:
        readm.append(-1)
    else:
        readm.append(int(float(val)))
label_cols["readmission_30d"] = readm

n_total = len(stay_ids)
unique_subjects = np.unique(subject_ids)
print(f"Cohort: {n_total} stays, {len(unique_subjects)} unique subjects")

# ── Subject multiplicity ─────────────────────────────────────────
subj_counts = Counter(subject_ids.tolist())
counts_dist = Counter(subj_counts.values())  # count_of_stays -> n_subjects

multiplicity = {
    "total_stays": n_total,
    "total_unique_subjects": len(unique_subjects),
    "n_subjects_with_1_stay": counts_dist.get(1, 0),
    "n_subjects_with_2plus_stays": sum(v for k, v in counts_dist.items() if k >= 2),
    "max_stays_per_subject": max(subj_counts.values()),
    "distribution": {str(k): v for k, v in sorted(counts_dist.items())},
    "top_20_subjects": sorted(subj_counts.items(), key=lambda x: -x[1])[:20],
}
multiplicity["top_20_subjects"] = [
    {"subject_id": int(sid), "stay_count": cnt}
    for sid, cnt in multiplicity["top_20_subjects"]
]

mult_path = AUDIT_DIR / "subject_multiplicity.json"
mult_path.write_text(json.dumps(multiplicity, indent=2))
print(f"Wrote {mult_path}")
_s1 = multiplicity['n_subjects_with_1_stay']
_s2 = multiplicity['n_subjects_with_2plus_stays']
_s3 = multiplicity['max_stays_per_subject']
print(f"  1-stay: {_s1}, 2+: {_s2}, max: {_s3}")

# ── Leakage audit ─────────────────────────────────────────────────
results = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "cohort_path": str(cohort_path),
    "n_total_stays": n_total,
    "n_unique_subjects": int(len(unique_subjects)),
    "split_config": {
        "holdout": "GroupShuffleSplit(n_splits=1, test_size={}, random_state={})".format(TEST_SIZE, RANDOM_STATE),
        "cv": "GroupKFold(n_splits={})".format(N_FOLDS),
        "groups": "subject_id",
    },
    "tasks": {},
    "global_max_intersection": 0,
    "verdict": "PENDING",
}

tasks_to_check = ["mortality", "prolonged_los", "readmission_30d"]

for task_name in tasks_to_check:
    y = np.array(label_cols[task_name])

    # Filter out -1 (missing labels)
    valid_mask = y >= 0
    y_valid = y[valid_mask]
    subj_valid = subject_ids[valid_mask]
    stay_valid = stay_ids[valid_mask]
    X_dummy = np.zeros((len(y_valid), 1))

    task_result = {
        "n_valid": int(len(y_valid)),
        "holdout_test": {},
        "cv_folds": [],
    }

    # Holdout split
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_val_idx, test_idx = next(gss.split(X_dummy, y_valid, groups=subj_valid))

    tv_subjects = set(subj_valid[train_val_idx].tolist())
    test_subjects = set(subj_valid[test_idx].tolist())
    holdout_inter = tv_subjects & test_subjects

    task_result["holdout_test"] = {
        "n_train_val_subjects": len(tv_subjects),
        "n_test_subjects": len(test_subjects),
        "intersection_count": len(holdout_inter),
    }
    results["global_max_intersection"] = max(results["global_max_intersection"], len(holdout_inter))

    # CV folds on train_val
    y_tv = y_valid[train_val_idx]
    subj_tv = subj_valid[train_val_idx]
    X_tv = X_dummy[train_val_idx]

    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold_i, (tr_rel, val_rel) in enumerate(gkf.split(X_tv, y_tv, groups=subj_tv)):
        tr_subj = set(subj_tv[tr_rel].tolist())
        val_subj = set(subj_tv[val_rel].tolist())
        inter = tr_subj & val_subj
        fold_info = {
            "fold": fold_i + 1,
            "n_train_subjects": len(tr_subj),
            "n_val_subjects": len(val_subj),
            "intersection_subjects_count": len(inter),
        }
        task_result["cv_folds"].append(fold_info)
        results["global_max_intersection"] = max(results["global_max_intersection"], len(inter))

    results["tasks"][task_name] = task_result

results["verdict"] = "PASS" if results["global_max_intersection"] == 0 else "FAIL"

leak_path = AUDIT_DIR / "subject_leakage_full.json"
leak_path.write_text(json.dumps(results, indent=2))
print(f"\nWrote {leak_path}")
_gmi = results['global_max_intersection']
print(f"Global max intersection: {_gmi}")
_vrd = results['verdict']
print(f"Verdict: {_vrd}")

for task, info in results["tasks"].items():
    ht = info["holdout_test"]
    _ic = ht['intersection_count']
    print(f"\n  {task}: holdout inter={_ic}")
    for f in info["cv_folds"]:
        _ff = f['fold']; _ft = f['n_train_subjects']; _fv = f['n_val_subjects']; _fi = f['intersection_subjects_count']
        print(f"    fold {_ff}: train={_ft}, val={_fv}, inter={_fi}")
