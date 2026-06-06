#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


JSON_ONLY_SYSTEM_PROMPT = (
    "Return exactly one valid JSON object and nothing else. "
    "Do not output markdown, prose, or analysis outside the JSON object. "
    "The user prompt already specifies the required JSON fields and format."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit OpenBio retry manifest for context budget safety")
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--summary-path", required=True)
    p.add_argument("--model-name", default="aaditya/Llama3-OpenBioLLM-70B")
    p.add_argument("--tokenizer-path", default="")
    p.add_argument("--context-limit", type=int, default=8192)
    p.add_argument("--safety-margin", type=int, default=64)
    p.add_argument("--default-max-tokens", type=int, default=512)
    return p.parse_args()


def jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def budget_bucket(available_tokens: int) -> str:
    if available_tokens < 128:
        return "lt128_compaction"
    if available_tokens < 256:
        return "128_255_low_budget"
    if available_tokens < 512:
        return "256_511_dynamic"
    return "ge512_default512"


def count_input_tokens(tokenizer, messages: list[dict]) -> int:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    output_path = Path(args.output_path)
    summary_path = Path(args.summary_path)

    tokenizer_source = args.tokenizer_path or args.model_name
    tokenizer_kwargs = {"local_files_only": True} if args.tokenizer_path else {}
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)

    audited_rows = []
    bucket_counts = Counter()
    task_bucket_counts = defaultdict(Counter)
    task_rows = Counter()
    input_token_values = []
    available_token_values = []

    for row in jsonl_iter(manifest_path):
        messages = [
            {"role": "system", "content": JSON_ONLY_SYSTEM_PROMPT},
            {"role": "user", "content": row["prompt_text"]},
        ]
        input_tokens_est = count_input_tokens(tokenizer, messages)
        available_tokens_est = max(0, args.context_limit - input_tokens_est - args.safety_margin)
        max_tokens_override = min(args.default_max_tokens, available_tokens_est)
        needs_compaction = available_tokens_est < 128
        bucket = budget_bucket(available_tokens_est)

        audited = dict(row)
        audited.update(
            {
                "input_tokens_est": input_tokens_est,
                "available_tokens_est": available_tokens_est,
                "max_tokens_override": max_tokens_override,
                "needs_compaction": needs_compaction,
                "budget_bucket": bucket,
            }
        )
        audited_rows.append(audited)
        bucket_counts[bucket] += 1
        task_bucket_counts[row["task_id"]][bucket] += 1
        task_rows[row["task_id"]] += 1
        input_token_values.append(input_tokens_est)
        available_token_values.append(available_tokens_est)

    write_jsonl(output_path, audited_rows)

    def summarize(values: list[int]) -> dict:
        values = sorted(values)
        if not values:
            return {}
        def q(p: float) -> int:
            return values[int((len(values) - 1) * p)]
        return {
            "mean": round(sum(values) / len(values), 2),
            "median": q(0.5),
            "p75": q(0.75),
            "p90": q(0.9),
            "p95": q(0.95),
            "p99": q(0.99),
            "min": values[0],
            "max": values[-1],
        }

    summary = {
        "manifest_rows": len(audited_rows),
        "model_name": args.model_name,
        "context_limit": args.context_limit,
        "safety_margin": args.safety_margin,
        "default_max_tokens": args.default_max_tokens,
        "input_tokens_est_summary": summarize(input_token_values),
        "available_tokens_est_summary": summarize(available_token_values),
        "budget_bucket_counts": dict(sorted(bucket_counts.items())),
        "needs_compaction_rows": bucket_counts["lt128_compaction"],
        "eligible_without_compaction_rows": len(audited_rows) - bucket_counts["lt128_compaction"],
        "tasks": {
            task_id: {
                "rows": task_rows[task_id],
                "budget_buckets": dict(sorted(task_bucket_counts[task_id].items())),
            }
            for task_id in sorted(task_rows)
        },
        "output_path": output_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
