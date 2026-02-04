"""
Build a stratified LLM annotation sample set (opt-in).
Outputs sample CSV, prompts, and rule-based (or LLM) annotation JSONL + metadata.
"""

import argparse
import json
import hashlib
import csv
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TEMPORAL_ALIGNMENT_DIR, ROOT_DIR
from data_processing.rule_based_annotation import annotate_record, rule_config_hash, RULE_CONFIG


OUT_DIR = ROOT_DIR / "results" / "llm_annotations"
PROMPT_TEMPLATE_PATH = ROOT_DIR / "code" / "data_processing" / "prompt_templates" / "llm_annotation_prompt.txt"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(t: str) -> str:
    """Lowercase, strip, collapse spaces, remove trailing punctuation."""
    import re as _re
    t = t.lower().strip()
    t = _re.sub(r"\s+", " ", t)
    t = t.rstrip(".,;:!?")
    return t


def bucket_time_delta(x):
    if x <= -12:
        return "<= -12h"
    if x <= -6:
        return "-12h~-6h"
    return "-6h~0h"


def stratified_sample(
    alignment_path: Path,
    n_per_stratum: int,
    seed: int,
    max_chunks: int = 50,
    chunk_size: int = 50000,
    use_pandas: bool = False,
    max_rows: int = 50000,
    dedup_mode: str = "none",
):
    random.seed(seed)
    stratified_sample._seen_nursing = set()
    strata_samples = {}
    cols = [
        "stay_id", "pattern_hour", "pattern_name", "pattern_severity",
        "note_id", "note_hour", "note_type", "note_text_relevant", "time_delta_hours"
    ]
    if use_pandas:
        for idx, chunk in enumerate(pd.read_csv(alignment_path, usecols=lambda c: c in cols, chunksize=chunk_size), start=1):
            chunk = chunk[chunk["note_type"].astype(str).str.lower() != "discharge"]
            chunk = chunk[chunk["note_hour"] < 24]
            chunk = chunk[chunk["time_delta_hours"] <= 0]

            for _, row in chunk.iterrows():
                note_type = str(row.get("note_type", "unknown")).lower()
                time_bucket = bucket_time_delta(float(row.get("time_delta_hours")))
                severity = str(row.get("pattern_severity", "unknown")).lower()
                key = (note_type, time_bucket, severity)
                if key not in strata_samples:
                    strata_samples[key] = []
                buf = strata_samples[key]
                if len(buf) < n_per_stratum:
                    buf.append(row)
                else:
                    j = random.randint(1, len(buf) + 1)
                    if j <= n_per_stratum:
                        buf[j - 1] = row

            if idx >= max_chunks:
                break
    else:
        with alignment_path.open() as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader, start=1):
                if max_rows and idx > max_rows:
                    break
                note_type = str(row.get("note_type", "unknown")).lower()
                if note_type == "discharge":
                    continue
                try:
                    note_hour = float(row.get("note_hour", "nan"))
                    time_delta = float(row.get("time_delta_hours", "nan"))
                except ValueError:
                    continue
                if note_hour >= 24 or note_hour < 0:
                    continue
                if time_delta > 0:
                    continue

                # Opt-in nursing dedup
                if dedup_mode != "none" and note_type == "nursing":
                    note_text = str(row.get("note_text_relevant", ""))
                    stay_id = str(row.get("stay_id", ""))
                    if dedup_mode == "exact":
                        _dedup_key = note_text.strip()
                    else:  # normalized
                        _dedup_key = normalize_text(note_text)
                    _full_key = (stay_id, _dedup_key)
                    if not hasattr(stratified_sample, '_seen_nursing'):
                        stratified_sample._seen_nursing = set()
                    if _full_key in stratified_sample._seen_nursing:
                        continue
                    stratified_sample._seen_nursing.add(_full_key)

                time_bucket = bucket_time_delta(time_delta)
                severity = str(row.get("pattern_severity", "unknown")).lower()
                key = (note_type, time_bucket, severity)
                if key not in strata_samples:
                    strata_samples[key] = []
                buf = strata_samples[key]
                if len(buf) < n_per_stratum:
                    buf.append(row)
                else:
                    j = random.randint(1, len(buf) + 1)
                    if j <= n_per_stratum:
                        buf[j - 1] = row

    samples = []
    for _, rows in strata_samples.items():
        samples.extend(rows)
    return samples


def load_prompt_template() -> str:
    if not PROMPT_TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Missing prompt template: {PROMPT_TEMPLATE_PATH}")
    return PROMPT_TEMPLATE_PATH.read_text()


def build_prompt(template: str, rec: dict) -> str:
    return template.format(
        pattern_name=rec["pattern_name"],
        pattern_severity=rec.get("pattern_severity", "unknown"),
        pattern_hour=rec["pattern_hour"],
        note_type=rec["note_type"],
        note_hour=rec["note_hour"],
        note_text_relevant=rec.get("note_text_relevant", ""),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-stratum", type=int, default=50)
    parser.add_argument("--max-chunks", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--use-pandas", action="store_true")
    parser.add_argument("--max-rows", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["rule_based", "llm"], default="rule_based")
    parser.add_argument("--run-llm", action="store_true")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model-name", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff", type=str, default="exponential")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--dedup-nursing", choices=["exact", "normalized", "none"], default="none",
                        help="Opt-in: deduplicate nursing notes within each stay before sampling. "
                             "'exact' removes exact text duplicates, 'normalized' removes case/whitespace-normalized duplicates.")
    args = parser.parse_args()

    alignment_path = TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv"
    if not alignment_path.exists():
        raise FileNotFoundError(f"Missing alignment file: {alignment_path}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    method = "llm" if args.run_llm else args.method
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    dedup_mode = getattr(args, 'dedup_nursing', 'none') or 'none'
    samples = stratified_sample(
        alignment_path,
        args.n_per_stratum,
        args.seed,
        args.max_chunks,
        args.chunk_size,
        args.use_pandas,
        args.max_rows,
        dedup_mode=dedup_mode,
    )
    rows = []
    prompts = []
    prompt_template = load_prompt_template()

    for row in samples:
        rec = {
            "stay_id": int(row.get("stay_id")),
            "pattern_hour": float(row.get("pattern_hour")),
            "pattern_name": row.get("pattern_name"),
            "pattern_severity": row.get("pattern_severity"),
            "note_id": str(row.get("note_id")),
            "note_hour": float(row.get("note_hour")),
            "note_type": row.get("note_type"),
            "note_text_relevant": row.get("note_text_relevant", ""),
            "time_delta_hours": float(row.get("time_delta_hours")),
        }
        rows.append(rec)

        prompt = {
            "id": f"{rec['stay_id']}_{rec['note_id']}_{rec['pattern_name']}",
            "prompt": build_prompt(prompt_template, rec),
        }
        prompts.append(prompt)

    # Deduplicate by key to avoid duplicate injections
    dedup = {}
    for rec in rows:
        key = (rec["stay_id"], rec["pattern_hour"], rec["pattern_name"], rec["note_id"])
        dedup[key] = rec
    rows = list(dedup.values())

    sample_path = OUT_DIR / "llm_annotation_set.csv"
    pd.DataFrame(rows).to_csv(sample_path, index=False)

    prompt_path = OUT_DIR / "llm_annotation_prompts.jsonl"
    with prompt_path.open("w") as f:
        for p in prompts:
            f.write(json.dumps(p, ensure_ascii=True) + "\n")

    annotations_path = OUT_DIR / f"annotations_{method}_{run_id}.jsonl"
    with annotations_path.open("w") as f:
        for rec in rows:
            ann = {
                "stay_id": rec["stay_id"],
                "pattern_name": rec["pattern_name"],
                "pattern_hour": rec["pattern_hour"],
                "note_id": rec["note_id"],
                "note_hour": rec["note_hour"],
                "note_type": rec["note_type"],
                "method": method,
            }
            if method == "rule_based":
                label_info = annotate_record(rec)
                ann.update(label_info)
            else:
                ann.update({
                    "label": None,
                    "evidence_span": "",
                    "evidence_note": "pending_llm",
                    "model_name": args.model_name,
                })
            f.write(json.dumps(ann, ensure_ascii=True) + "\n")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_samples": len(rows),
        "sample_path": str(sample_path),
        "prompt_path": str(prompt_path),
        "annotations_path": str(annotations_path),
        "alignment_source": str(alignment_path),
        "alignment_sha256": sha256_file(alignment_path),
        "method": method,
        "run_id": run_id,
        "prompt_template_path": str(PROMPT_TEMPLATE_PATH),
        "prompt_template_sha256": sha256_file(PROMPT_TEMPLATE_PATH),
        "sampling": {
            "dedup_nursing": dedup_mode,
            "n_per_stratum": args.n_per_stratum,
            "max_chunks": args.max_chunks,
            "chunk_size": args.chunk_size,
            "use_pandas": args.use_pandas,
            "max_rows": args.max_rows,
            "seed": args.seed,
        },
        "inputs": {
            "annotation_set_sha256": sha256_file(sample_path),
        },
        "outputs": {
            "annotations_sha256": sha256_file(annotations_path),
        },
    }

    if method == "llm":
        meta.update({
            "provider": args.provider,
            "model_name": args.model_name,
            "generation_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_tokens": args.max_tokens,
                "seed": args.seed,
            },
            "concurrency": args.concurrency,
            "retry_policy": {
                "max_retries": args.max_retries,
                "retry_backoff": args.retry_backoff,
            },
        })
    else:
        meta.update({
            "rule_source_path": str(Path(__file__).resolve()),
            "rule_config_hash": rule_config_hash(),
            "rule_config": RULE_CONFIG,
        })

    (OUT_DIR / "llm_annotation_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True))

    annotation_metadata = {
        "method": method,
        "run_timestamp": meta["timestamp"],
        "run_id": run_id,
        "annotation_set_path": str(sample_path),
        "annotation_set_sha256": meta["inputs"]["annotation_set_sha256"],
        "annotations_path": str(annotations_path),
        "annotations_sha256": meta["outputs"]["annotations_sha256"],
        "alignment_source": str(alignment_path),
        "alignment_sha256": meta["alignment_sha256"],
        "prompt_template_path": meta["prompt_template_path"],
        "prompt_template_sha256": meta["prompt_template_sha256"],
    }
    if method == "llm":
        annotation_metadata.update({
            "provider": args.provider,
            "model_name": args.model_name,
            "generation_params": meta["generation_params"],
            "concurrency": args.concurrency,
            "retry_policy": meta["retry_policy"],
        })
    else:
        annotation_metadata.update({
            "rule_source_path": meta["rule_source_path"],
            "rule_config_hash": meta["rule_config_hash"],
        })

    (OUT_DIR / "ANNOTATION_METADATA.json").write_text(json.dumps(annotation_metadata, indent=2, ensure_ascii=True))

    print(f"Wrote {sample_path}")
    print(f"Wrote {prompt_path}")
    print(f"Wrote {annotations_path}")


if __name__ == "__main__":
    main()
