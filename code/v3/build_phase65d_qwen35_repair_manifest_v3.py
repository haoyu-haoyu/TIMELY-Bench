#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_ROOT = "."
DEFAULT_OUTPUT_DIR = "results/cres_v3/phase65d_tier1b_full"
DEFAULT_FULL_MANIFEST = "results/cres_v3/phase65d_tier1b_full/phase65d_full_manifest_full_multimodal.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Qwen targeted repair manifest from parse-failed prompts")
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--full-manifest", default=DEFAULT_FULL_MANIFEST)
    p.add_argument("--provider", default="qwen35")
    p.add_argument("--repair-manifest-name", default="phase65d_qwen35_repair_manifest.jsonl")
    p.add_argument("--summary-name", default="phase65d_qwen35_repair_manifest_summary.json")
    return p.parse_args()


def jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed jsonl row in {path.as_posix()}:{line_num}", flush=True)


def load_preferred_rows(output_dir: Path, provider: str) -> dict[str, dict]:
    def row_rank(row: dict | None) -> tuple[int, int]:
        if row is None:
            return (-1, -1)
        if row.get("status") == "ok" and row.get("parse_success") is True:
            return (2, 1)
        if row.get("status") == "ok":
            return (1, 1)
        return (0, 0)

    preferred: dict[str, dict] = {}
    paths = sorted(
        output_dir.glob(f"{provider}_responses_shard*.jsonl"),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    for path in paths:
        for row in jsonl_iter(path):
            prompt_id = row["prompt_id"]
            existing = preferred.get(prompt_id)
            if row_rank(row) >= row_rank(existing):
                preferred[prompt_id] = row
    return preferred


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    full_manifest_path = (root / args.full_manifest).resolve()
    repair_manifest_path = output_dir / args.repair_manifest_name
    summary_path = output_dir / args.summary_name

    latest = load_preferred_rows(output_dir, args.provider)
    failed_prompt_ids = {
        prompt_id for prompt_id, row in latest.items()
        if not (row.get("status") == "ok" and row.get("parse_success") is True)
    }

    repair_rows: list[dict] = []
    for row in jsonl_iter(full_manifest_path):
        if row["prompt_id"] in failed_prompt_ids:
            repair_rows.append(row)

    by_task_dim = Counter((row["task_id"], row["dimension_id"], row["variant_id"]) for row in repair_rows)
    by_error = Counter(
        ((row.get("parse_error") or "<none>").split(":")[0])
        for prompt_id, row in latest.items()
        if prompt_id in failed_prompt_ids
    )

    write_jsonl(repair_manifest_path, repair_rows)
    summary = {
        "provider": args.provider,
        "repair_rows": len(repair_rows),
        "repair_manifest_path": repair_manifest_path.as_posix(),
        "failed_prompt_ids": len(failed_prompt_ids),
        "error_types": dict(sorted(by_error.items())),
        "by_task_dim_variant": {
            f"{task}::{dimension}::{variant}": count
            for (task, dimension, variant), count in sorted(by_task_dim.items())
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
