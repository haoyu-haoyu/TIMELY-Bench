#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

from run_phase65d_tier1b_v3 import extract_json_blob, jsonl_iter, normalize_confidence, run_curl_json


DEFAULT_OUTPUT_DIR = "results/cres_v3/phase65e_tier2_full"
DEFAULT_PILOT_OUTPUT_DIR = "results/cres_v3/phase65e_tier2_pilot"
DEFAULT_FULL_MANIFEST = "results/cres_v3/phase65d_tier1b_full/phase65d_full_manifest_full_multimodal.jsonl"
DEFAULT_PILOT_MANIFEST = "results/cres_v3/phase65d_tier1b_pilot/phase65d_pilot100_manifest.jsonl"
PROVIDERS = [
    "openbiollm70b",
    "aloe70b",
    "aloe7b",
    "me_llama",
    "meditron3_8b",
    "medgemma15_4b",
]
JSON_ONLY_SYSTEM_PROMPT = (
    "Return exactly one valid JSON object and nothing else. "
    "Do not output markdown, prose, or analysis outside the JSON object. "
    "The user prompt already specifies the required JSON fields and format."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5E Tier 2 local/openai-compatible runner")
    p.add_argument("--mode", choices=["run_provider_shard", "summarize", "summarize_manifest_subset"], required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--manifest-path", default="")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pilot-output-dir", default=DEFAULT_PILOT_OUTPUT_DIR)
    p.add_argument("--provider", choices=PROVIDERS)
    p.add_argument("--providers", nargs="+", default=PROVIDERS, choices=PROVIDERS)
    p.add_argument("--model-name", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2600)
    p.add_argument("--timeout-seconds", type=int, default=300)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=6)
    p.add_argument("--summary-path", default="")
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def provider_shard_path(output_dir: Path, provider: str, shard_index: int, num_shards: int) -> Path:
    return output_dir / f"{provider}_responses_shard{shard_index:02d}_of_{num_shards:02d}.jsonl"


def provider_shard_paths(output_dir: Path, provider: str) -> List[Path]:
    return sorted(output_dir.glob(f"{provider}_responses_shard*.jsonl"))


def load_latest_rows(path: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if not path.exists():
        return latest
    for row in jsonl_iter(path):
        latest[row["prompt_id"]] = row
    return latest


def load_preferred_rows(paths: List[Path]) -> Dict[str, dict]:
    def row_rank(row: dict | None) -> tuple[int, int]:
        if row is None:
            return (-1, -1)
        if row.get("status") == "ok" and row.get("parse_success") is True:
            return (2, 1)
        if row.get("status") == "ok":
            return (1, 1)
        return (0, 0)

    preferred: Dict[str, dict] = {}
    ordered_paths = sorted(paths, key=lambda path: (path.stat().st_mtime, path.name))
    for path in ordered_paths:
        if not path.exists():
            continue
        for row in jsonl_iter(path):
            prompt_id = row["prompt_id"]
            existing = preferred.get(prompt_id)
            if row_rank(row) >= row_rank(existing):
                preferred[prompt_id] = row
    return preferred


def load_done_prompt_ids(output_dir: Path, provider: str, model_name: str) -> set[str]:
    done: set[str] = set()
    for path in provider_shard_paths(output_dir, provider):
        latest = load_latest_rows(path)
        for prompt_id, row in latest.items():
            if (
                row.get("status") == "ok"
                and row.get("parse_success") is True
                and row.get("model_name") == model_name
            ):
                done.add(prompt_id)
    return done


def iter_manifest_rows(manifest_path: Path, shard_index: int, num_shards: int) -> Iterator[dict]:
    for idx, row in enumerate(jsonl_iter(manifest_path)):
        if idx % num_shards == shard_index:
            yield row


def provider_env_prefix(provider: str) -> str:
    return f"TIER2_{provider.upper()}"


def resolve_provider_runtime(provider: str, model_name: str) -> dict:
    prefix = provider_env_prefix(provider)
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    api_mode = os.environ.get(f"{prefix}_API_MODE", "openai_chat").strip()
    endpoint = os.environ.get(f"{prefix}_ENDPOINT", "/chat/completions").strip()
    extra_body_env = os.environ.get(f"{prefix}_EXTRA_BODY_JSON", "").strip()
    extra_body = {}
    if not base_url or not api_key:
        raise SystemExit(f"{prefix}_BASE_URL and {prefix}_API_KEY are required")
    if not model_name:
        raise SystemExit("--model-name is required")
    if api_mode != "openai_chat":
        raise SystemExit(f"Unsupported API mode for Tier 2 runner: {api_mode}")
    if extra_body_env:
        try:
            parsed = json.loads(extra_body_env)
            if isinstance(parsed, dict):
                extra_body.update(parsed)
            else:
                raise SystemExit(f"{prefix}_EXTRA_BODY_JSON must decode to an object")
        except json.JSONDecodeError as exc:  # noqa: BLE001
            raise SystemExit(f"Invalid JSON in {prefix}_EXTRA_BODY_JSON: {exc}") from exc
    return {
        "base_url": base_url.rstrip("/"),
        "api_key": api_key,
        "api_mode": api_mode,
        "endpoint": endpoint,
        "extra_body": extra_body,
        "json_system_prompt": os.environ.get(f"{prefix}_JSON_SYSTEM_PROMPT", JSON_ONLY_SYSTEM_PROMPT).strip(),
        "use_json_system_prompt": os.environ.get(f"{prefix}_USE_JSON_SYSTEM_PROMPT", "1").strip().lower()
        not in {"0", "false", "no"},
    }


def invoke_provider(
    *,
    provider: str,
    model_name: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[dict, str, dict]:
    runtime = resolve_provider_runtime(provider, model_name)
    url = runtime["base_url"] + runtime["endpoint"]
    messages = [{"role": "user", "content": prompt_text}]
    if runtime["use_json_system_prompt"] and runtime["json_system_prompt"]:
        messages = [
            {"role": "system", "content": runtime["json_system_prompt"]},
            {"role": "user", "content": prompt_text},
        ]
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if runtime["extra_body"]:
        payload.update(runtime["extra_body"])
    response = run_curl_json(
        url=url,
        headers=[f"Authorization: Bearer {runtime['api_key']}"],
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(
            json.dumps({"http_status": 200, "body": json.dumps(response, ensure_ascii=False)[:4000]}, ensure_ascii=False)
        )
    choice = ((choices[0] or {}).get("message") or {}).get("content", "")
    if isinstance(choice, list):
        texts = []
        for item in choice:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        choice = "\n".join(text for text in texts if text).strip()
    elif not isinstance(choice, str):
        choice = str(choice)
    usage = response.get("usage", {}) or {}
    return response, choice, {
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
    }


def maybe_reduce_max_tokens(error_text: str, current_max_tokens: int) -> int | None:
    context_match = re.search(r"maximum context length is (\d+) tokens", error_text)
    input_match = re.search(r"prompt contains at least (\d+) input tokens", error_text)
    if not context_match or not input_match:
        return None

    context_limit = int(context_match.group(1))
    input_tokens = int(input_match.group(1))
    if input_tokens >= context_limit:
        return None

    # vLLM reports input length as a lower bound ("at least ... tokens"), and
    # the exact tokenized prompt can drift upward between retries. Keep a much
    # wider safety margin so retries do not keep landing back on the limit.
    safe_max_tokens = max(64, context_limit - input_tokens - 2048)
    if safe_max_tokens >= current_max_tokens:
        return None
    return safe_max_tokens


def run_one_request(
    *,
    row: dict,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    max_retries: int,
) -> dict:
    manifest_max_tokens = row.get("max_tokens_override")
    if not isinstance(manifest_max_tokens, int) or manifest_max_tokens <= 0:
        manifest_max_tokens = max_tokens
    result = {
        "provider": provider,
        "model_name": model_name,
        "prompt_id": row["prompt_id"],
        "instance_id": row["instance_id"],
        "task_id": row["task_id"],
        "dimension_id": row["dimension_id"],
        "variant_id": row["variant_id"],
        "status": "ok",
        "parse_success": False,
        "manifest_max_tokens": manifest_max_tokens,
    }
    for optional_key in ("input_tokens_est", "available_tokens_est", "budget_bucket", "needs_compaction"):
        if optional_key in row:
            result[optional_key] = row[optional_key]
    current_max_tokens = manifest_max_tokens
    for attempt in range(max_retries):
        started = time.time()
        try:
            response, choice, usage = invoke_provider(
                provider=provider,
                model_name=model_name,
                prompt_text=row["prompt_text"],
                temperature=temperature,
                max_tokens=current_max_tokens,
                timeout_seconds=timeout_seconds,
            )
            parsed, parse_error = extract_json_blob(choice)
            result.update(
                {
                    "status": "ok",
                    "latency_seconds": round(time.time() - started, 3),
                    "attempts": attempt + 1,
                    "requested_max_tokens": current_max_tokens,
                    **usage,
                    "raw_content": choice,
                    "parse_success": parsed is not None,
                    "parse_error": parse_error,
                    "parsed_response": parsed,
                    "confidence_value": normalize_confidence(parsed),
                    "provider_response_meta": {
                        "finish_reason": ((response.get("choices") or [{}])[0] or {}).get("finish_reason")
                    },
                }
            )
            return result
        except Exception as exc:  # noqa: BLE001
            error_text = str(exc)
            http_status = None
            status = "exception"
            if error_text.startswith("{") and error_text.endswith("}"):
                try:
                    parsed_error = json.loads(error_text)
                    http_status = parsed_error.get("http_status")
                    error_text = parsed_error.get("body", error_text)
                    status = "http_error"
                except Exception:  # noqa: BLE001
                    pass
            result.update(
                {
                    "status": status,
                    "latency_seconds": round(time.time() - started, 3),
                    "attempts": attempt + 1,
                    "requested_max_tokens": current_max_tokens,
                    "http_status": http_status,
                    "error": error_text[:4000],
                }
            )
            adjusted_max_tokens = None
            if http_status == 400:
                adjusted_max_tokens = maybe_reduce_max_tokens(error_text, current_max_tokens)
                if adjusted_max_tokens is not None:
                    current_max_tokens = adjusted_max_tokens
            retriable = (
                status == "exception"
                or http_status in {408, 409, 429, 500, 502, 503, 504}
                or adjusted_max_tokens is not None
            )
            if not retriable or attempt + 1 >= max_retries:
                return result
        time.sleep(min(20.0, 3.0 * (attempt + 1)))
    return result


def run_provider_shard(
    *,
    manifest_path: Path,
    output_dir: Path,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    num_shards: int,
    shard_index: int,
    max_workers: int,
    max_retries: int,
) -> Path:
    shard_path = provider_shard_path(output_dir, provider, shard_index, num_shards)
    done_prompt_ids = load_done_prompt_ids(output_dir, provider, model_name)
    pending_rows = [
        row for row in iter_manifest_rows(manifest_path, shard_index, num_shards) if row["prompt_id"] not in done_prompt_ids
    ]
    if not pending_rows:
        return shard_path

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        row_iter = iter(pending_rows)
        while True:
            while len(futures) < max_workers:
                try:
                    row = next(row_iter)
                except StopIteration:
                    break
                fut = pool.submit(
                    run_one_request,
                    row=row,
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                    max_retries=max_retries,
                )
                futures[fut] = row["prompt_id"]

            if not futures:
                break

            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut, None)
                row = fut.result()
                append_jsonl(shard_path, row)

    return shard_path


def summarize_provider(output_dir: Path, provider: str) -> dict:
    preferred_by_prompt = load_preferred_rows(provider_shard_paths(output_dir, provider))
    rows = list(preferred_by_prompt.values())
    merged_path = output_dir / f"{provider}_full_responses.jsonl"
    write_jsonl(merged_path, rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    parse_rows = [r for r in ok_rows if r.get("parse_success")]
    task_counts = Counter(r["task_id"] for r in rows)
    task_parse = Counter(r["task_id"] for r in parse_rows)
    conf_counts = Counter(r.get("confidence_value") or "missing" for r in parse_rows)

    summary = {
        "provider": provider,
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "parse_success_rows": len(parse_rows),
        "parse_success_rate": (len(parse_rows) / len(rows)) if rows else 0.0,
        "avg_latency_seconds": (
            sum((r.get("latency_seconds") or 0.0) for r in rows) / len(rows)
        ) if rows else 0.0,
        "usage_total_tokens": sum(r.get("usage_total_tokens") or 0 for r in ok_rows),
        "tasks": dict(sorted(task_counts.items())),
        "task_parse_success_rate": {
            task: (task_parse.get(task, 0) / count) if count else 0.0 for task, count in sorted(task_counts.items())
        },
        "confidence_distribution": dict(sorted(conf_counts.items())),
        "confidence_distribution_rate": {
            key: (value / len(parse_rows)) if parse_rows else 0.0 for key, value in sorted(conf_counts.items())
        },
        "merged_output_path": str(merged_path.as_posix()),
    }
    (output_dir / f"{provider}_full_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def summarize_manifest_subset(manifest_path: Path, output_dir: Path, provider: str, model_name: str) -> dict:
    requested_rows = list(jsonl_iter(manifest_path))
    preferred_rows = load_preferred_rows(provider_shard_paths(output_dir, provider))
    rows: List[dict] = []
    for row in requested_rows:
        result = preferred_rows.get(row["prompt_id"])
        if result is None or result.get("model_name") != model_name:
            rows.append(
                {
                    "provider": provider,
                    "model_name": model_name,
                    "prompt_id": row["prompt_id"],
                    "instance_id": row["instance_id"],
                    "task_id": row["task_id"],
                    "dimension_id": row["dimension_id"],
                    "variant_id": row["variant_id"],
                    "status": "missing",
                    "parse_success": False,
                }
            )
        else:
            rows.append(result)

    found_rows = [row for row in rows if row.get("status") != "missing"]
    ok_rows = [row for row in found_rows if row.get("status") == "ok"]
    parse_rows = [row for row in ok_rows if row.get("parse_success") is True]
    missing_rows = [row for row in rows if row.get("status") == "missing"]
    status_counts = Counter(row.get("status", "missing") for row in rows)
    parse_error_counts = Counter(row.get("parse_error", "")[:200] for row in ok_rows if row.get("parse_success") is not True)
    summary = {
        "provider": provider,
        "model_name": model_name,
        "manifest_path": str(manifest_path.as_posix()),
        "rows_requested": len(requested_rows),
        "rows_found": len(found_rows),
        "missing_rows": len(missing_rows),
        "ok_rows": len(ok_rows),
        "parse_success_rows": len(parse_rows),
        "ok_rate": (len(ok_rows) / len(requested_rows)) if requested_rows else 0.0,
        "parse_success_rate": (len(parse_rows) / len(requested_rows)) if requested_rows else 0.0,
        "avg_latency_seconds": (
            sum((row.get("latency_seconds") or 0.0) for row in found_rows) / len(found_rows)
        ) if found_rows else 0.0,
        "usage_total_tokens": sum(row.get("usage_total_tokens") or 0 for row in ok_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "top_parse_errors": dict(parse_error_counts.most_common(10)),
    }
    return summary


def summarize_all(output_dir: Path, providers: List[str]) -> None:
    summary = {
        "providers": {provider: summarize_provider(output_dir, provider) for provider in providers},
    }
    (output_dir / "phase65e_full_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    pilot_output_dir = (root / args.pilot_output_dir).resolve()
    ensure_dir(output_dir)
    ensure_dir(pilot_output_dir)

    if args.mode == "run_provider_shard":
        if not args.provider or not args.model_name:
            raise SystemExit("--provider and --model-name are required for run_provider_shard")
        manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else (root / DEFAULT_FULL_MANIFEST).resolve()
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        run_provider_shard(
            manifest_path=manifest_path,
            output_dir=output_dir,
            provider=args.provider,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            timeout_seconds=args.timeout_seconds,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            max_workers=args.max_workers,
            max_retries=args.max_retries,
        )
        return

    if args.mode == "summarize_manifest_subset":
        if not args.provider or not args.model_name or not args.manifest_path:
            raise SystemExit("--provider, --model-name, and --manifest-path are required for summarize_manifest_subset")
        manifest_path = Path(args.manifest_path).resolve()
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        summary = summarize_manifest_subset(
            manifest_path=manifest_path,
            output_dir=output_dir,
            provider=args.provider,
            model_name=args.model_name,
        )
        if args.summary_path:
            summary_path = Path(args.summary_path).resolve()
            ensure_dir(summary_path.parent)
            summary_path.write_text(json.dumps(summary, indent=2))
        else:
            print(json.dumps(summary, indent=2))
        return

    summarize_all(output_dir, args.providers)


if __name__ == "__main__":
    main()
