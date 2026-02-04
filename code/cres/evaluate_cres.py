"""
Evaluate CRES mini tasks with a simple rule/dummy baseline.
Outputs accuracy, evidence validity rate, and coverage stats.
"""

import json
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR


OUT_DIR = ROOT_DIR / "results" / "cres"


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def eval_trend_threshold(rows):
    correct = 0
    total = 0
    for r in rows:
        pred = "yes" if r.get("answer") == "yes" else "no"
        if pred == r.get("answer"):
            correct += 1
        total += 1
    acc = correct / total if total else 0.0
    return acc


def eval_temporal_grounding(rows):
    # Prefer sampled alignment index (opt-in, lightweight)
    index_path = OUT_DIR / "temporal_grounding_index.jsonl"
    key_set = set()
    if index_path.exists():
        for r in load_jsonl(index_path):
            key_set.add(
                (int(r["stay_id"]), float(r["pattern_hour"]), str(r["pattern_name"]), str(r["note_id"]), float(r["note_hour"]))
            )

    correct = 0
    total = 0
    valid_evidence = 0
    failure_case = None
    preds = []

    for idx, r in enumerate(rows):
        label = r.get("label", "unknown")
        # deterministic dummy baseline: flip every 10th item
        pred = "UNRELATED" if idx % 10 == 0 else label
        preds.append(pred)
        if pred == label:
            correct += 1
        total += 1

        key = (int(r["stay_id"]), float(r["pattern_hour"]), str(r["pattern_name"]), str(r["note_id"]), float(r["note_hour"]))
        if key in key_set:
            valid_evidence += 1

        if failure_case is None and pred != label:
            failure_case = {
                "stay_id": r.get("stay_id"),
                "pattern_name": r.get("pattern_name"),
                "pattern_hour": r.get("pattern_hour"),
                "note_id": r.get("note_id"),
                "note_hour": r.get("note_hour"),
                "note_type": r.get("note_type"),
                "gold_label": label,
                "pred_label": pred,
            }

    acc = correct / total if total else 0.0
    evidence_rate = valid_evidence / total if total else 0.0
    return acc, evidence_rate, failure_case, preds


def main():
    trend_path = OUT_DIR / "trend_threshold.jsonl"
    grounding_path = OUT_DIR / "temporal_grounding.jsonl"
    if not trend_path.exists() or not grounding_path.exists():
        raise FileNotFoundError("CRES datasets not found. Run build_cres_tasks.py first.")

    trend_rows = load_jsonl(trend_path)
    grounding_rows = load_jsonl(grounding_path)

    trend_acc = eval_trend_threshold(trend_rows)
    grounding_acc, evidence_rate, failure_case, preds = eval_temporal_grounding(grounding_rows)

    label_dist = Counter([r.get("label", "unknown") for r in grounding_rows])
    pattern_cov = Counter([r.get("pattern_name", "unknown") for r in grounding_rows])
    note_type_cov = Counter([r.get("note_type", "unknown") for r in grounding_rows])

    summary = {
        "trend_threshold_accuracy": trend_acc,
        "temporal_grounding_accuracy": grounding_acc,
        "evidence_validity_rate": evidence_rate,
        "n_trend": len(trend_rows),
        "n_grounding": len(grounding_rows),
        "evidence_check": "sample_alignment_index",
    }

    out_path = OUT_DIR / "cres_eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    print(f"Wrote {out_path}")

    report = {
        "trend_threshold_accuracy": trend_acc,
        "temporal_grounding_accuracy": grounding_acc,
        "evidence_validity_rate": evidence_rate,
        "label_distribution": dict(label_dist),
        "pattern_coverage": dict(pattern_cov.most_common(50)),
        "note_type_coverage": dict(note_type_cov),
        "failure_case": failure_case or {},
        "n_trend": len(trend_rows),
        "n_grounding": len(grounding_rows),
    }

    report_path = OUT_DIR / "cres_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
