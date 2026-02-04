"""
Build CRES mini tasks (opt-in).
Task 1: Trend/Threshold reasoning on structured summaries.
Task 2: Temporal grounding + evidence attribution from alignment data.
"""

import argparse
import json
import random
import hashlib
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR


EPISODES_DIR = ROOT_DIR / "episodes" / "episodes_enhanced"
OUT_DIR = ROOT_DIR / "results" / "cres"


VITALS_FEATURES = ["heart_rate", "sbp", "mbp", "resp_rate", "temperature", "spo2"]
THRESHOLDS = {
    "heart_rate": 100,
    "sbp": 90,
    "mbp": 70,
    "resp_rate": 20,
    "temperature": 38.0,
    "spo2": 94,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_episode_vitals(ep_path: Path):
    with ep_path.open() as f:
        ep = json.load(f)
    vitals = ep.get("timeseries", {}).get("vitals", [])
    if not vitals:
        return None
    df = pd.DataFrame(vitals)
    if "hour" not in df.columns and "hour_offset" in df.columns:
        df["hour"] = df["hour_offset"]
    return ep, df


def build_trend_threshold(n_samples):
    examples = []
    files = list(EPISODES_DIR.glob("TIMELY_v2_*.json"))
    random.shuffle(files)
    for ep_path in files:
        if len(examples) >= n_samples:
            break
        loaded = load_episode_vitals(ep_path)
        if loaded is None:
            continue
        ep, df = loaded
        stay_id = ep.get("stay_id")
        if stay_id is None:
            continue

        for feat in VITALS_FEATURES:
            if feat not in df.columns:
                continue
            values = pd.to_numeric(df[feat], errors="coerce").dropna()
            if values.empty:
                continue

            # threshold question
            thr = THRESHOLDS.get(feat)
            if thr is None:
                continue
            mean_val = float(values.mean())
            answer = "yes" if mean_val >= thr else "no"
            examples.append({
                "id": f"{stay_id}_{feat}_threshold",
                "task": "trend_threshold",
                "subtask": "threshold",
                "stay_id": int(stay_id),
                "feature": feat,
                "question": f"Is mean {feat} >= {thr} in 0-24h?",
                "value": mean_val,
                "threshold": thr,
                "answer": answer,
                "evidence": {"feature": feat, "mean": mean_val, "threshold": thr}
            })
            if len(examples) >= n_samples:
                break

            # trend question (compare early vs late)
            if "hour" in df.columns:
                early = df[df["hour"] < 6]
                late = df[df["hour"] >= 18]
                if not early.empty and not late.empty:
                    early_mean = float(pd.to_numeric(early[feat], errors="coerce").dropna().mean())
                    late_mean = float(pd.to_numeric(late[feat], errors="coerce").dropna().mean())
                    if not (pd.isna(early_mean) or pd.isna(late_mean)):
                        trend = "up" if late_mean > early_mean else "down"
                        examples.append({
                            "id": f"{stay_id}_{feat}_trend",
                            "task": "trend_threshold",
                            "subtask": "trend",
                            "stay_id": int(stay_id),
                            "feature": feat,
                            "question": f"Is {feat} increasing from 0-6h to 18-24h?",
                            "value": {"early_mean": early_mean, "late_mean": late_mean},
                            "answer": "yes" if trend == "up" else "no",
                            "evidence": {"early_mean": early_mean, "late_mean": late_mean}
                        })
            if len(examples) >= n_samples:
                break

    return examples


def build_temporal_grounding(n_samples, max_rows=200000):
    alignment_path = TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"
    if not alignment_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {alignment_path}")

    # Reservoir sampling over a streaming reader to avoid heavy pandas loads
    sample = []
    total_seen = 0
    with alignment_path.open() as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            if max_rows and idx > max_rows:
                break
            note_type = str(row.get("note_type", "")).lower()
            if note_type == "discharge":
                continue
            try:
                note_hour = float(row.get("note_hour", "nan"))
                pattern_hour = float(row.get("pattern_hour", "nan"))
                time_delta = float(row.get("time_delta_hours", "nan"))
            except ValueError:
                continue
            if not (0 <= note_hour < 24):
                continue
            if time_delta > 0:
                continue

            total_seen += 1
            if len(sample) < n_samples:
                sample.append(row)
            else:
                j = random.randint(1, total_seen)
                if j <= n_samples:
                    sample[j - 1] = row

            if len(sample) >= n_samples and total_seen >= n_samples * 5:
                break

    examples = []
    for row in sample:
        examples.append({
            "id": f"{row.get('stay_id')}_{row.get('note_id')}_{row.get('pattern_name')}",
            "task": "temporal_grounding",
            "stay_id": int(row.get("stay_id")),
            "pattern_hour": float(row.get("pattern_hour")),
            "pattern_name": row.get("pattern_name"),
            "note_id": str(row.get("note_id")),
            "note_hour": float(row.get("note_hour")),
            "note_type": row.get("note_type"),
            "note_text_relevant": row.get("note_text_relevant", ""),
            "time_delta_hours": float(row.get("time_delta_hours")),
            "label": row.get("alignment_quality", "unknown"),
        })

    return examples


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trend", type=int, default=2000)
    parser.add_argument("--n-grounding", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=200000)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    trend_rows = build_trend_threshold(args.n_trend)
    grounding_rows = build_temporal_grounding(args.n_grounding, args.max_rows)

    trend_path = OUT_DIR / "trend_threshold.jsonl"
    grounding_path = OUT_DIR / "temporal_grounding.jsonl"
    grounding_index_path = OUT_DIR / "temporal_grounding_index.jsonl"

    write_jsonl(trend_path, trend_rows)
    write_jsonl(grounding_path, grounding_rows)

    # write alignment index subset for evidence validation
    with grounding_index_path.open("w") as f:
        for r in grounding_rows:
            key = {
                "stay_id": r["stay_id"],
                "pattern_hour": r["pattern_hour"],
                "pattern_name": r["pattern_name"],
                "note_id": r["note_id"],
                "note_hour": r["note_hour"],
            }
            f.write(json.dumps(key, ensure_ascii=True) + "\n")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "trend_threshold": {"n": len(trend_rows), "path": str(trend_path)},
        "temporal_grounding": {"n": len(grounding_rows), "path": str(grounding_path)},
        "temporal_grounding_index": {"n": len(grounding_rows), "path": str(grounding_index_path)},
        "alignment_source": str(TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"),
        "seed": args.seed,
        "max_rows": args.max_rows,
    }

    meta_path = OUT_DIR / "cres_build_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=True))

    alignment_path = Path(meta["alignment_source"])
    inputs = []
    if alignment_path.exists():
        inputs.append({"path": str(alignment_path), "sha256": sha256_file(alignment_path)})

    episodes_tar = ROOT_DIR / "episodes_enhanced.tar.gz"
    if episodes_tar.exists():
        inputs.append({"path": str(episodes_tar), "sha256": sha256_file(episodes_tar)})

    outputs = [
        {"path": str(trend_path), "sha256": sha256_file(trend_path)},
        {"path": str(grounding_path), "sha256": sha256_file(grounding_path)},
        {"path": str(grounding_index_path), "sha256": sha256_file(grounding_index_path)},
        {"path": str(meta_path), "sha256": sha256_file(meta_path)},
    ]

    manifest = {
        "timestamp": meta["timestamp"],
        "n_trend": len(trend_rows),
        "n_grounding": len(grounding_rows),
        "seed": args.seed,
        "max_rows": args.max_rows,
        "inputs": inputs,
        "outputs": outputs,
    }
    manifest_path = OUT_DIR / "cres_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    print(f"Wrote {trend_path}")
    print(f"Wrote {grounding_path}")
    print(f"Wrote {meta_path}")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
