"""
Evaluate CRES tasks from external model predictions.

Important:
- This script does NOT generate pseudo predictions.
- It only scores provided predictions against the CRES task files.
"""

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

from config import ROOT_DIR


OUT_DIR = ROOT_DIR / "results" / "cres"

TASK_TO_FILE = {
    "trend_threshold": OUT_DIR / "trend_threshold.jsonl",
    "temporal_grounding": OUT_DIR / "temporal_grounding.jsonl",
    "diagnostic_consistency": OUT_DIR / "diagnostic_consistency.jsonl",
    "contrastive_inference": OUT_DIR / "contrastive_inference.jsonl",
}

TASK_LABEL_KEY = {
    "trend_threshold": "answer",
    "temporal_grounding": "label",
    "diagnostic_consistency": "label",
    "contrastive_inference": "label",
}


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _classification_metrics(labels, preds):
    if not labels:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "balanced_accuracy": 0.0,
            "confusion_matrix": {},
        }

    acc = sum(int(y == p) for y, p in zip(labels, preds)) / len(labels)
    uniq = sorted(set(labels) | set(preds))
    macro_f1 = float(f1_score(labels, preds, labels=uniq, average="macro", zero_division=0))
    bal_acc = float(balanced_accuracy_score(labels, preds))
    cm = confusion_matrix(labels, preds, labels=uniq).tolist()
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "confusion_matrix": {"labels": uniq, "matrix": cm},
    }


def _entropy(dist: Counter):
    total = sum(dist.values()) or 1
    return -sum((count / total) * math.log(count / total, 2) for count in dist.values() if count > 0)


def _load_predictions(pred_path: Path):
    rows = load_jsonl(pred_path)
    pred_map = {}

    model_names = set()
    model_versions = set()
    prompt_shas = set()

    for row in rows:
        rid = str(row.get("id", "")).strip()
        pred_label = str(row.get("pred_label", "")).strip()
        if not rid or not pred_label:
            continue

        task = str(row.get("task", "")).strip() or None
        pred_map[rid] = {
            "pred_label": pred_label,
            "task": task,
            "raw": row,
        }

        if row.get("model_name"):
            model_names.add(str(row.get("model_name")))
        if row.get("model_version"):
            model_versions.add(str(row.get("model_version")))
        if row.get("prompt_sha"):
            prompt_shas.add(str(row.get("prompt_sha")))

    metadata = {
        "n_rows_raw": int(len(rows)),
        "n_rows_valid": int(len(pred_map)),
        "model_names": sorted(model_names),
        "model_versions": sorted(model_versions),
        "prompt_shas": sorted(prompt_shas),
    }
    return pred_map, metadata


def _evaluate_task(task_name: str, gold_rows: list, pred_map: dict):
    label_key = TASK_LABEL_KEY[task_name]

    y_true = []
    y_pred = []
    missing_ids = []
    task_mismatch_ids = []

    for row in gold_rows:
        rid = str(row.get("id", "")).strip()
        if not rid:
            continue

        pred = pred_map.get(rid)
        if pred is None:
            missing_ids.append(rid)
            continue

        pred_task = pred.get("task")
        if pred_task is not None and pred_task != task_name:
            task_mismatch_ids.append(rid)
            continue

        y_true.append(str(row.get(label_key, "")))
        y_pred.append(pred["pred_label"])

    metrics = _classification_metrics(y_true, y_pred)
    gold_label_dist = Counter([str(r.get(label_key, "")) for r in gold_rows])
    pred_label_dist = Counter(y_pred)

    result = {
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "confusion_matrix": metrics["confusion_matrix"],
        "n_gold": int(len(gold_rows)),
        "n_scored": int(len(y_true)),
        "coverage": float(len(y_true) / len(gold_rows)) if gold_rows else 0.0,
        "n_missing_predictions": int(len(missing_ids)),
        "n_task_mismatch": int(len(task_mismatch_ids)),
        "missing_id_examples": missing_ids[:20],
        "task_mismatch_examples": task_mismatch_ids[:20],
        "gold_label_distribution": dict(gold_label_dist),
        "pred_label_distribution": dict(pred_label_dist),
    }
    return result


def _temporal_grounding_evidence_rate(rows):
    index_path = OUT_DIR / "temporal_grounding_index.jsonl"
    if not index_path.exists() or not rows:
        return 0.0

    key_set = set()
    for row in load_jsonl(index_path):
        key_set.add(
            (
                int(row["stay_id"]),
                float(row["pattern_hour"]),
                str(row["pattern_name"]),
                str(row["note_id"]),
                float(row["note_hour"]),
            )
        )

    valid_evidence = 0
    for row in rows:
        key = (
            int(row["stay_id"]),
            float(row["pattern_hour"]),
            str(row["pattern_name"]),
            str(row["note_id"]),
            float(row["note_hour"]),
        )
        if key in key_set:
            valid_evidence += 1
    return float(valid_evidence / len(rows))


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRES from external predictions")
    parser.add_argument(
        "--predictions-jsonl",
        required=True,
        help="Path to model predictions JSONL. Required fields per row: id, pred_label.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run identifier. Defaults to timestamp.",
    )
    parser.add_argument(
        "--write-canonical",
        action="store_true",
        help="Also write summary/report to results/cres root for backward compatibility.",
    )
    args = parser.parse_args()

    pred_path = Path(args.predictions_jsonl)
    if not pred_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {pred_path}")

    for task, path in TASK_TO_FILE.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing CRES task file ({task}): {path}")

    pred_map, pred_meta = _load_predictions(pred_path)

    gold = {task: load_jsonl(path) for task, path in TASK_TO_FILE.items()}
    eval_results = {task: _evaluate_task(task, gold_rows, pred_map) for task, gold_rows in gold.items()}

    grounding_label_dist = Counter([str(r.get("label", "")) for r in gold["temporal_grounding"]])
    contrastive_label_dist = Counter([str(r.get("label", "")) for r in gold["contrastive_inference"]])
    note_type_cov = Counter([str(r.get("note_type", "unknown")) for r in gold["temporal_grounding"]])

    label_total = sum(grounding_label_dist.values()) or 1
    note_total = sum(note_type_cov.values()) or 1
    dominant_label_ratio = max(grounding_label_dist.values()) / label_total if grounding_label_dist else 0.0
    dominant_note_type_ratio = max(note_type_cov.values()) / note_total if note_type_cov else 0.0
    label_entropy = _entropy(grounding_label_dist)

    warning_flags = []
    if eval_results["temporal_grounding"]["coverage"] < 0.95:
        warning_flags.append("grounding_coverage_below_95pct")
    if eval_results["diagnostic_consistency"]["coverage"] < 0.95:
        warning_flags.append("diagnostic_coverage_below_95pct")
    if eval_results["contrastive_inference"]["coverage"] < 0.95:
        warning_flags.append("contrastive_coverage_below_95pct")

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / "model_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "evaluation_mode": "external_predictions",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predictions_file": str(pred_path),
        "prediction_metadata": pred_meta,
        "trend_threshold_accuracy": eval_results["trend_threshold"]["accuracy"],
        "temporal_grounding_accuracy": eval_results["temporal_grounding"]["accuracy"],
        "temporal_grounding_macro_f1": eval_results["temporal_grounding"]["macro_f1"],
        "temporal_grounding_balanced_accuracy": eval_results["temporal_grounding"]["balanced_accuracy"],
        "diagnostic_consistency_accuracy": eval_results["diagnostic_consistency"]["accuracy"],
        "diagnostic_consistency_macro_f1": eval_results["diagnostic_consistency"]["macro_f1"],
        "contrastive_inference_accuracy": eval_results["contrastive_inference"]["accuracy"],
        "contrastive_inference_macro_f1": eval_results["contrastive_inference"]["macro_f1"],
        "contrastive_tie_rate": float(
            contrastive_label_dist.get("tie", 0) / max(1, sum(contrastive_label_dist.values()))
        ),
        "evidence_validity_rate": _temporal_grounding_evidence_rate(gold["temporal_grounding"]),
        "n_trend": int(len(gold["trend_threshold"])),
        "n_grounding": int(len(gold["temporal_grounding"])),
        "n_diagnostic": int(len(gold["diagnostic_consistency"])),
        "n_contrastive": int(len(gold["contrastive_inference"])),
        "n_trend_scored": eval_results["trend_threshold"]["n_scored"],
        "n_grounding_scored": eval_results["temporal_grounding"]["n_scored"],
        "n_diagnostic_scored": eval_results["diagnostic_consistency"]["n_scored"],
        "n_contrastive_scored": eval_results["contrastive_inference"]["n_scored"],
        "label_unique_count": len(grounding_label_dist),
        "label_entropy": label_entropy,
        "dominant_label_ratio": dominant_label_ratio,
        "dominant_note_type_ratio": dominant_note_type_ratio,
        "warning_flags": warning_flags,
    }

    report = {
        "summary": summary,
        "task_results": eval_results,
    }

    summary_path = run_dir / "cres_eval_summary.json"
    report_path = run_dir / "cres_evaluation_report.json"
    pred_meta_path = run_dir / "predictions_manifest.json"

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True))
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True))
    pred_meta_path.write_text(json.dumps(pred_meta, indent=2, ensure_ascii=True))

    if args.write_canonical:
        (OUT_DIR / "cres_eval_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True))
        (OUT_DIR / "cres_evaluation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True))

    print(f"Wrote {summary_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {pred_meta_path}")


if __name__ == "__main__":
    main()
