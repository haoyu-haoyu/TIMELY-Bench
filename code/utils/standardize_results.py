"""
Standardize results across training steps.
Outputs JSON + CSV with AUROC/AUPRC, fold stats, seed, and input paths.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    ROOT_DIR,
    RESULTS_DIR,
    BENCHMARK_RESULTS_DIR,
    RANDOM_STATE,
    COHORT_FILE,
    TIMESERIES_FILE,
    NOTE_TIME_FILE,
    LLM_FEATURES_FILE,
    get_features_file,
)


STANDARD_DIR = RESULTS_DIR / "standardized"
FUSION_LATE_XGB_JSON = RESULTS_DIR / "fusion_baselines" / "fusion_results_late_xgb.json"


def _dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _dump_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def standardize_structured():
    input_path = BENCHMARK_RESULTS_DIR / "benchmark_results_full.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing structured results: {input_path}")

    df = pd.read_csv(input_path)
    results = []
    for _, row in df.iterrows():
        window = row.get("window")
        input_paths = {
            "features_file": str(get_features_file(str(window))),
            "cohort_file": str(COHORT_FILE),
        }
        results.append({
            "step": "structured",
            "task": row.get("task"),
            "model": row.get("model"),
            "cohort": row.get("cohort"),
            "window": window,
            "n_samples": row.get("n_samples"),
            "positive_rate": row.get("positive_rate"),
            "auroc_mean": row.get("auroc_mean"),
            "auroc_std": row.get("auroc_std"),
            "auprc_mean": row.get("auprc_mean"),
            "auprc_std": row.get("auprc_std"),
            "test_auroc": row.get("test_auroc"),
            "test_auprc": row.get("test_auprc"),
            "seed": RANDOM_STATE,
            "input_paths": json.dumps(input_paths, ensure_ascii=True),
        })

    payload = {
        "step": "structured",
        "timestamp": _now_iso(),
        "seed": RANDOM_STATE,
        "inputs": {
            "cohort_file": str(COHORT_FILE),
        },
        "results": results,
    }

    _dump_json(STANDARD_DIR / "structured_results.json", payload)
    _dump_csv(STANDARD_DIR / "structured_results.csv", results)


def standardize_text():
    input_path = RESULTS_DIR / "text_only_baselines" / "text_only_results_folds.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing text results: {input_path}")

    with open(input_path) as f:
        payload = json.load(f)

    results = []
    for row in payload.get("results", []):
        results.append({
            "step": "text",
            "task": row.get("task"),
            "model": row.get("model"),
            "n_samples": row.get("n_samples"),
            "positive_rate": row.get("positive_rate"),
            "auroc_mean": row.get("cv_auroc_mean"),
            "auroc_std": row.get("cv_auroc_std"),
            "auprc_mean": row.get("cv_auprc_mean"),
            "auprc_std": row.get("cv_auprc_std"),
            "test_auroc": row.get("test_auroc"),
            "test_auprc": row.get("test_auprc"),
            "fold_details": json.dumps(row.get("fold_details", []), ensure_ascii=True),
            "seed": RANDOM_STATE,
            "input_paths": json.dumps(payload.get("input_paths", {}), ensure_ascii=True),
        })

    output_payload = {
        "step": "text",
        "timestamp": _now_iso(),
        "seed": RANDOM_STATE,
        "inputs": payload.get("input_paths", {}),
        "results": results,
    }

    _dump_json(STANDARD_DIR / "text_results.json", output_payload)
    _dump_csv(STANDARD_DIR / "text_results.csv", results)


def standardize_fusion():
    input_path = RESULTS_DIR / "fusion_baselines" / "fusion_results_folds.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing fusion results: {input_path}")

    with open(input_path) as f:
        payload = json.load(f)

    results = []
    for row in payload.get("results", []):
        results.append({
            "step": "fusion",
            "task": row.get("task"),
            "model": row.get("model"),
            "n_samples": row.get("n_samples"),
            "positive_rate": row.get("positive_rate"),
            "auroc_mean": row.get("cv_auroc_mean"),
            "auroc_std": row.get("cv_auroc_std"),
            "auprc_mean": row.get("cv_auprc_mean"),
            "auprc_std": row.get("cv_auprc_std"),
            "test_auroc": row.get("test_auroc"),
            "test_auprc": row.get("test_auprc"),
            "fold_details": json.dumps(row.get("fold_details", []), ensure_ascii=True),
            "seed": RANDOM_STATE,
            "input_paths": json.dumps(payload.get("input_paths", {}), ensure_ascii=True),
        })

    output_payload = {
        "step": "fusion",
        "timestamp": _now_iso(),
        "seed": RANDOM_STATE,
        "inputs": payload.get("input_paths", {}),
        "results": results,
    }

    _dump_json(STANDARD_DIR / "fusion_results.json", output_payload)
    _dump_csv(STANDARD_DIR / "fusion_results.csv", results)


def standardize_fusion_late_xgb():
    if not FUSION_LATE_XGB_JSON.exists():
        return

    with open(FUSION_LATE_XGB_JSON) as f:
        payload = json.load(f)

    results = []
    for row in payload.get("results", []):
        results.append({
            "step": "fusion",
            "task": row.get("task"),
            "model": row.get("model"),
            "cohort": row.get("cohort"),
            "window": row.get("window"),
            "n_samples": row.get("n_samples"),
            "positive_rate": row.get("positive_rate"),
            "auroc_mean": row.get("cv_auroc_mean"),
            "auroc_std": row.get("cv_auroc_std"),
            "auprc_mean": row.get("cv_auprc_mean"),
            "auprc_std": row.get("cv_auprc_std"),
            "test_auroc": row.get("test_auroc"),
            "test_auprc": row.get("test_auprc"),
            "fold_details": json.dumps(row.get("fold_details", []), ensure_ascii=True),
            "seed": RANDOM_STATE,
            "input_paths": json.dumps(payload.get("input_paths", {}), ensure_ascii=True),
        })

    output_payload = {
        "step": "fusion",
        "timestamp": _now_iso(),
        "seed": RANDOM_STATE,
        "inputs": payload.get("input_paths", {}),
        "results": results,
    }

    _dump_json(STANDARD_DIR / "fusion_results_late_xgb.json", output_payload)
    _dump_csv(STANDARD_DIR / "fusion_results_late_xgb.csv", results)


def standardize_gru():
    input_path = RESULTS_DIR / "Output_temporal_gru" / "training_results.json"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing GRU results: {input_path}")

    with open(input_path) as f:
        raw = json.load(f)

    cv = raw.get("cross_validation", {})
    test = raw.get("test", {})
    fold_details = cv.get("fold_details", [])

    input_paths = {
        "timeseries_file": str(TIMESERIES_FILE),
        "note_time_file": str(NOTE_TIME_FILE),
        "llm_features_file": str(LLM_FEATURES_FILE),
        "cohort_file": str(COHORT_FILE),
    }

    results = [{
        "step": "gru",
        "task": "mortality",
        "model": "temporal_gru_v2",
        "n_samples": raw.get("data", {}).get("total_samples"),
        "auroc_mean": cv.get("mean_auroc"),
        "auroc_std": cv.get("std_auroc"),
        "auprc_mean": cv.get("mean_auprc"),
        "auprc_std": cv.get("std_auprc"),
        "test_auroc": test.get("test_auroc"),
        "test_auprc": test.get("test_auprc"),
        "fold_details": json.dumps(fold_details, ensure_ascii=True),
        "seed": RANDOM_STATE,
        "input_paths": json.dumps(input_paths, ensure_ascii=True),
    }]

    output_payload = {
        "step": "gru",
        "timestamp": _now_iso(),
        "seed": RANDOM_STATE,
        "inputs": input_paths,
        "results": results,
    }

    _dump_json(STANDARD_DIR / "gru_results.json", output_payload)
    _dump_csv(STANDARD_DIR / "gru_results.csv", results)


def build_results_summary():
    standard_files = [
        "structured_results.csv",
        "text_results.csv",
        "fusion_results.csv",
        "fusion_results_late_xgb.csv",
        "gru_results.csv",
    ]
    frames = []
    for fname in standard_files:
        path = STANDARD_DIR / fname
        if path.exists():
            df = pd.read_csv(path)
            df["source_file"] = fname
            frames.append(df)

    if not frames:
        return

    summary = pd.concat(frames, ignore_index=True)

    # If late fusion XGB results exist, drop old late fusion rows from fusion_results.csv
    if (STANDARD_DIR / "fusion_results_late_xgb.csv").exists():
        mask_old_late = (summary["source_file"] == "fusion_results.csv") & (
            summary["model"].astype(str).str.contains("Late Fusion")
        )
        summary = summary[~mask_old_late].copy()

    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "NA"

    summary["auroc_mean_std"] = summary.apply(
        lambda r: f"{fmt(r.get('auroc_mean'))} +/- {fmt(r.get('auroc_std'))}", axis=1
    )
    summary["auprc_mean_std"] = summary.apply(
        lambda r: f"{fmt(r.get('auprc_mean'))} +/- {fmt(r.get('auprc_std'))}", axis=1
    )
    summary["test_auroc_fmt"] = summary["test_auroc"].apply(fmt)
    summary["test_auprc_fmt"] = summary["test_auprc"].apply(fmt)

    cols = [
        "step", "task", "model", "cohort", "window",
        "auroc_mean", "auroc_std", "auprc_mean", "auprc_std",
        "test_auroc", "test_auprc",
        "auroc_mean_std", "auprc_mean_std",
        "test_auroc_fmt", "test_auprc_fmt",
        "source_file",
    ]
    for col in cols:
        if col not in summary.columns:
            summary[col] = "NA"

    summary = summary[cols]
    summary.to_csv(STANDARD_DIR / "results_summary.csv", index=False)

    md_cols = [
        "step", "task", "model", "cohort", "window",
        "auroc_mean_std", "auprc_mean_std", "test_auroc", "test_auprc"
    ]
    md = summary[md_cols].copy()
    md["test_auroc"] = md["test_auroc"].apply(fmt)
    md["test_auprc"] = md["test_auprc"].apply(fmt)
    (STANDARD_DIR / "results_summary.md").write_text(md.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", required=True, choices=["structured", "text", "fusion", "gru"])
    args = parser.parse_args()

    if args.step == "structured":
        standardize_structured()
    elif args.step == "text":
        standardize_text()
    elif args.step == "fusion":
        standardize_fusion()
        standardize_fusion_late_xgb()
    elif args.step == "gru":
        standardize_gru()

    build_results_summary()
    print(f"Standardized results saved to: {STANDARD_DIR}")


if __name__ == "__main__":
    main()
