#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


DEFAULT_OUTPUT_DIR = "results/cres_v3/phase65d_tier1b_full"
DEFAULT_PILOT_OUTPUT_DIR = "results/cres_v3/phase65d_tier1b_pilot"
DEFAULT_PROMPTS_PATH = "data/processed/v3/cres/cres_eval_prompts_12k.jsonl"
DEFAULT_PHASE65B_SUMMARY = "results/cres_v3/phase65b_prompt_build_summary.json"
PRIMARY_PROVIDERS = ["deepseek_chat", "qwen35", "gemma4_26b"]
ALL_PROVIDERS = PRIMARY_PROVIDERS + ["deepseek_v32_thinking"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5D Tier 1B runner")
    p.add_argument(
        "--mode",
        choices=[
            "build_manifest",
            "build_pilot_manifest",
            "build_remaining_manifest",
            "run_provider_shard",
            "summarize",
            "summarize_manifest_subset",
        ],
        required=True,
    )
    p.add_argument("--root", default=".")
    p.add_argument("--prompts-path", default=DEFAULT_PROMPTS_PATH)
    p.add_argument("--phase65b-summary", default=DEFAULT_PHASE65B_SUMMARY)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pilot-output-dir", default=DEFAULT_PILOT_OUTPUT_DIR)
    p.add_argument("--manifest-path", default="")
    p.add_argument("--manifest-output-path", default="")
    p.add_argument("--summary-path", default="")
    p.add_argument("--variant-id", default="full_multimodal")
    p.add_argument("--pilot-size", type=int, default=100)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--provider", choices=ALL_PROVIDERS)
    p.add_argument("--providers", nargs="+", default=PRIMARY_PROVIDERS, choices=ALL_PROVIDERS)
    p.add_argument("--model-name", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=6)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def jsonl_iter(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_num, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] skipping malformed jsonl row in {path.as_posix()}:{line_num}", flush=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def reasoning_audit_dir(output_dir: Path) -> Path:
    return output_dir.parent / f"{output_dir.name}_reasoning_audit"


def reasoning_shard_path(output_dir: Path, provider: str, shard_index: int, num_shards: int) -> Path:
    audit_dir = reasoning_audit_dir(output_dir)
    ensure_dir(audit_dir)
    return audit_dir / f"{provider}_reasoning_shard{shard_index:02d}_of_{num_shards:02d}.jsonl"


def reasoning_shard_paths(output_dir: Path, provider: str) -> List[Path]:
    audit_dir = reasoning_audit_dir(output_dir)
    if not audit_dir.exists():
        return []
    return sorted(audit_dir.glob(f"{provider}_reasoning_shard*.jsonl"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def extract_json_blob(text: str) -> Tuple[dict | None, str | None]:
    text = text.strip()
    candidates: List[str] = []
    if text:
        candidates.append(text)
    if "```json" in text:
        after = text.split("```json", 1)[1]
        candidates.append(after.split("```", 1)[0].strip())
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            candidates.append(parts[1].strip())

    last_err: str | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception as exc:  # noqa: BLE001
            return None, f"{type(exc).__name__}: {exc}"
    return None, last_err or "no_json_object_found"


def manifest_name(prefix: str, variant_id: str) -> str:
    return f"{prefix}_{variant_id}.jsonl"


def build_manifest(prompts_path: Path, output_dir: Path, variant_id: str) -> Path:
    selected: List[dict] = []
    task_counts: Counter[str] = Counter()
    dimension_counts: Counter[str] = Counter()
    for row in jsonl_iter(prompts_path):
        if row.get("variant_id") != variant_id:
            continue
        selected.append(row)
        task_counts[row["task_id"]] += 1
        dimension_counts[row["dimension_id"]] += 1

    selected.sort(key=lambda r: (r["task_id"], r["instance_id"], r["dimension_id"], r["prompt_id"]))
    out_path = output_dir / manifest_name("phase65d_full_manifest", variant_id)
    write_jsonl(out_path, selected)
    summary = {
        "variant_id": variant_id,
        "manifest_rows": len(selected),
        "unique_instances": len({row["instance_id"] for row in selected}),
        "tasks": dict(sorted(task_counts.items())),
        "dimensions": dict(sorted(dimension_counts.items())),
        "output_path": str(out_path.as_posix()),
    }
    (output_dir / "phase65d_full_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    return out_path


def largest_remainder_quotas(targets: Dict[str, int], total: int) -> Dict[str, int]:
    weight_sum = sum(targets.values())
    raw = {k: (v / weight_sum) * total for k, v in targets.items()}
    quotas = {k: int(math.floor(x)) for k, x in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(targets, key=lambda k: (raw[k] - quotas[k], targets[k]), reverse=True)
    for key in order[:remainder]:
        quotas[key] += 1
    return quotas


def build_pilot_manifest(prompts_path: Path, phase65b_summary_path: Path, output_dir: Path, pilot_size: int) -> Path:
    summary = load_json(phase65b_summary_path)
    quotas = largest_remainder_quotas(summary["task_sample_targets"], pilot_size)

    pair_candidates: Dict[str, Dict[Tuple[str, str], dict]] = {}
    extras: Dict[str, List[dict]] = {}
    for row in jsonl_iter(prompts_path):
        task_id = row["task_id"]
        pair_candidates.setdefault(task_id, {})
        extras.setdefault(task_id, [])
        pair = (row["variant_id"], row["dimension_id"])
        if pair not in pair_candidates[task_id]:
            pair_candidates[task_id][pair] = row
        elif len(extras[task_id]) < 1024:
            extras[task_id].append(row)

    selected: List[dict] = []
    for task_id, quota in sorted(quotas.items()):
        primary = sorted(
            pair_candidates.get(task_id, {}).values(),
            key=lambda r: (r["variant_id"], r["dimension_id"], r["prompt_id"]),
        )
        fallback = sorted(
            extras.get(task_id, []),
            key=lambda r: (r["variant_id"], r["dimension_id"], r["prompt_id"]),
        )
        rows = (primary + fallback)[:quota]
        if len(rows) < quota:
            raise RuntimeError(f"Pilot quota unmet for {task_id}: need {quota}, found {len(rows)}")
        selected.extend(rows)

    selected.sort(key=lambda r: (r["task_id"], r["variant_id"], r["dimension_id"], r["prompt_id"]))
    out_path = output_dir / f"phase65d_pilot{pilot_size}_manifest.jsonl"
    write_jsonl(out_path, selected)
    build_summary = {
        "pilot_size": len(selected),
        "tasks": {task: quotas[task] for task in sorted(quotas)},
        "output_path": str(out_path.as_posix()),
        "source_prompts_path": str(prompts_path.as_posix()),
    }
    (output_dir / "phase65d_pilot_manifest_summary.json").write_text(json.dumps(build_summary, indent=2))
    return out_path


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


def run_curl_json(*, url: str, headers: List[str], payload: dict, timeout_seconds: int) -> dict:
    marker = "__HTTP_STATUS__:"
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        "Content-Type: application/json",
    ]
    for header in headers:
        cmd.extend(["-H", header])
    cmd.extend(
        [
            "--connect-timeout",
            "20",
            "--max-time",
            str(timeout_seconds),
            "--data-binary",
            "@-",
            "-w",
            f"\n{marker}%{{http_code}}",
        ]
    )
    completed = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "curl_failed").strip()[:4000])

    body, sep, status_text = completed.stdout.rpartition(marker)
    if not sep:
        raise RuntimeError(f"missing_http_status_marker: {(completed.stdout or '')[:4000]}")
    try:
        http_status = int(status_text.strip())
    except ValueError as exc:  # noqa: BLE001
        raise RuntimeError(f"invalid_http_status_marker: {status_text!r}") from exc

    response_body = body.strip()
    if http_status >= 400:
        raise RuntimeError(json.dumps({"http_status": http_status, "body": response_body[:4000]}))
    return json.loads(response_body)


def extract_openai_content(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(
            json.dumps(
                {
                    "http_status": 200,
                    "body": json.dumps(response, ensure_ascii=False)[:4000],
                },
                ensure_ascii=False,
            )
        )
    choice = ((choices[0] or {}).get("message") or {}).get("content", "")
    if isinstance(choice, str):
        return choice
    if isinstance(choice, list):
        texts = []
        for item in choice:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(text for text in texts if text).strip()
    return str(choice)


def extract_openai_reasoning(response: dict) -> str:
    def normalize_reasoning(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            text = value.strip()
            if text.lower() in {"", "none", "null"}:
                return ""
            return text
        if isinstance(value, list):
            texts = []
            for item in value:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = str(item.get("text", "")).strip()
                    if text:
                        texts.append(text)
            return "\n".join(texts).strip()
        if isinstance(value, dict):
            text = value.get("text")
            if isinstance(text, str):
                return text.strip()
        return ""

    choices = response.get("choices") or []
    if not choices:
        return ""
    message = (choices[0] or {}).get("message") or {}
    for key in ("reasoning_content", "reasoning"):
        reasoning = normalize_reasoning(message.get(key))
        if reasoning:
            return reasoning
    return ""


def extract_google_content(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
    return "\n".join(texts).strip()


def provider_env_prefix(provider: str) -> str:
    return f"TIER1B_{provider.upper()}"


def resolve_provider_runtime(provider: str, model_name: str) -> dict:
    prefix = provider_env_prefix(provider)
    base_url = os.environ.get(f"{prefix}_BASE_URL")
    api_key = os.environ.get(f"{prefix}_API_KEY")
    api_mode = os.environ.get(f"{prefix}_API_MODE", "openai_chat").strip()
    endpoint = os.environ.get(f"{prefix}_ENDPOINT", "").strip()
    if not base_url or not api_key:
        raise SystemExit(f"{prefix}_BASE_URL and {prefix}_API_KEY are required")
    if not model_name:
        raise SystemExit("--model-name is required")
    if api_mode not in {"openai_chat", "google_generate_content"}:
        raise SystemExit(f"Unsupported API mode: {api_mode}")

    if not endpoint:
        endpoint = "/chat/completions" if api_mode == "openai_chat" else "/v1beta/models/{model_name}:generateContent"
    extra_body = {}
    if provider == "qwen35":
        extra_body["enable_thinking"] = False
        extra_body["response_format"] = {"type": "json_object"}
    extra_body_env = os.environ.get(f"{prefix}_EXTRA_BODY_JSON", "").strip()
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
    }


def invoke_provider(
    *,
    provider: str,
    model_name: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[dict, str, str, dict]:
    runtime = resolve_provider_runtime(provider, model_name)
    api_mode = runtime["api_mode"]
    base_url = runtime["base_url"]
    api_key = runtime["api_key"]
    endpoint = runtime["endpoint"].format(model_name=model_name)
    extra_body = runtime.get("extra_body") or {}
    url = base_url + endpoint

    if api_mode == "openai_chat":
        messages = [{"role": "user", "content": prompt_text}]
        if provider == "qwen35":
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object and nothing else. "
                        "Do not output markdown, prose, analysis, or reasoning outside the JSON object. "
                        "The user prompt already specifies the required JSON fields and format."
                    ),
                },
                {"role": "user", "content": prompt_text},
            ]
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        response = run_curl_json(
            url=url,
            headers=[f"Authorization: Bearer {api_key}"],
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        usage = response.get("usage", {})
        return response, extract_openai_content(response), extract_openai_reasoning(response), {
            "usage_prompt_tokens": usage.get("prompt_tokens"),
            "usage_completion_tokens": usage.get("completion_tokens"),
            "usage_total_tokens": usage.get("total_tokens"),
        }

    response = run_curl_json(
        url=url,
        headers=[f"x-goog-api-key: {api_key}"],
        payload={
            "contents": [{"parts": [{"text": prompt_text}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        },
        timeout_seconds=timeout_seconds,
    )
    usage = response.get("usageMetadata", {})
    return response, extract_google_content(response), "", {
        "usage_prompt_tokens": usage.get("promptTokenCount"),
        "usage_completion_tokens": usage.get("candidatesTokenCount"),
        "usage_total_tokens": usage.get("totalTokenCount"),
    }


def normalize_confidence(parsed: dict | None) -> str | None:
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("confidence")
    if value is None:
        return None
    return str(value).strip().lower()


CONTEXT_LENGTH_ERROR_RE = re.compile(
    r"maximum context length is (\d+) tokens.*?requested (\d+) output tokens and your prompt contains at least (\d+) input tokens",
    re.IGNORECASE | re.DOTALL,
)


def derive_context_retry_max_tokens(error_text: str, requested_max_tokens: int) -> int | None:
    match = CONTEXT_LENGTH_ERROR_RE.search(error_text)
    if not match:
        return None
    max_context = int(match.group(1))
    input_tokens = int(match.group(3))
    retry_max_tokens = max_context - input_tokens
    if retry_max_tokens <= 0 or retry_max_tokens >= requested_max_tokens:
        return None
    return retry_max_tokens


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
    request_max_tokens = max_tokens
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
    }
    for attempt in range(max_retries):
        started = time.time()
        try:
            response, choice, raw_reasoning, usage = invoke_provider(
                provider=provider,
                model_name=model_name,
                prompt_text=row["prompt_text"],
                temperature=temperature,
                max_tokens=request_max_tokens,
                timeout_seconds=timeout_seconds,
            )
            parsed, parse_error = extract_json_blob(choice)
            result.update(
                {
                    "status": "ok",
                    "latency_seconds": round(time.time() - started, 3),
                    "attempts": attempt + 1,
                    **usage,
                    "raw_content": choice,
                    "raw_reasoning": raw_reasoning or None,
                    "parse_success": parsed is not None,
                    "parse_error": parse_error,
                    "parsed_response": parsed,
                    "confidence_value": normalize_confidence(parsed),
                    "requested_max_tokens": request_max_tokens,
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
                    "http_status": http_status,
                    "error": error_text[:4000],
                    "requested_max_tokens": request_max_tokens,
                }
            )
            retry_max_tokens = None
            if status == "http_error" and http_status == 400:
                retry_max_tokens = derive_context_retry_max_tokens(error_text, request_max_tokens)
            if retry_max_tokens is not None:
                request_max_tokens = retry_max_tokens
                continue
            retriable = status == "exception" or http_status in {408, 409, 429, 500, 502, 503, 504}
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
    reasoning_path = reasoning_shard_path(output_dir, provider, shard_index, num_shards)
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
                raw_reasoning = row.pop("raw_reasoning", None)
                if raw_reasoning:
                    append_jsonl(
                        reasoning_path,
                        {
                            "provider": row["provider"],
                            "model_name": row["model_name"],
                            "prompt_id": row["prompt_id"],
                            "instance_id": row["instance_id"],
                            "task_id": row["task_id"],
                            "dimension_id": row["dimension_id"],
                            "variant_id": row["variant_id"],
                            "status": row["status"],
                            "parse_success": row["parse_success"],
                            "attempts": row.get("attempts"),
                            "latency_seconds": row.get("latency_seconds"),
                            "usage_total_tokens": row.get("usage_total_tokens"),
                            "raw_reasoning": raw_reasoning,
                        },
                    )
                    row["has_reasoning"] = True
                else:
                    row["has_reasoning"] = False
                append_jsonl(shard_path, row)

    return shard_path


def summarize_provider(output_dir: Path, provider: str) -> dict:
    preferred_by_prompt = load_preferred_rows(provider_shard_paths(output_dir, provider))
    rows: List[dict] = []
    extracted_reasoning_rows: Dict[str, dict] = {}
    for prompt_id, row in preferred_by_prompt.items():
        row_copy = dict(row)
        raw_reasoning = row_copy.pop("raw_reasoning", None)
        if raw_reasoning:
            extracted_reasoning_rows[prompt_id] = {
                "provider": row_copy["provider"],
                "model_name": row_copy["model_name"],
                "prompt_id": row_copy["prompt_id"],
                "instance_id": row_copy["instance_id"],
                "task_id": row_copy["task_id"],
                "dimension_id": row_copy["dimension_id"],
                "variant_id": row_copy["variant_id"],
                "status": row_copy["status"],
                "parse_success": row_copy["parse_success"],
                "attempts": row_copy.get("attempts"),
                "latency_seconds": row_copy.get("latency_seconds"),
                "usage_total_tokens": row_copy.get("usage_total_tokens"),
                "raw_reasoning": raw_reasoning,
            }
            row_copy["has_reasoning"] = True
        rows.append(row_copy)
    merged_path = output_dir / f"{provider}_full_responses.jsonl"
    write_jsonl(merged_path, rows)

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    parse_rows = [r for r in ok_rows if r.get("parse_success")]
    task_counts = Counter(r["task_id"] for r in rows)
    task_parse = Counter(r["task_id"] for r in parse_rows)
    conf_counts = Counter(r.get("confidence_value") or "missing" for r in parse_rows)

    preferred_reasoning_by_prompt = load_preferred_rows(reasoning_shard_paths(output_dir, provider))
    preferred_reasoning_by_prompt.update(extracted_reasoning_rows)
    reasoning_rows = list(preferred_reasoning_by_prompt.values())
    reasoning_ok_rows = [r for r in reasoning_rows if r.get("raw_reasoning")]
    reasoning_merged_path = reasoning_audit_dir(output_dir) / f"{provider}_full_reasoning.jsonl"
    if reasoning_rows:
        write_jsonl(reasoning_merged_path, reasoning_rows)

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
        "reasoning_rows": len(reasoning_rows),
        "reasoning_nonempty_rows": len(reasoning_ok_rows),
        "reasoning_output_path": str(reasoning_merged_path.as_posix()) if reasoning_rows else None,
    }
    (output_dir / f"{provider}_full_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def build_remaining_manifest(
    manifest_path: Path,
    output_dir: Path,
    provider: str,
    model_name: str,
    manifest_output_path: Path,
    limit: int,
) -> dict:
    done_prompt_ids = load_done_prompt_ids(output_dir, provider, model_name)
    remaining_rows = [row for row in jsonl_iter(manifest_path) if row["prompt_id"] not in done_prompt_ids]
    if limit > 0:
        remaining_rows = remaining_rows[:limit]
    write_jsonl(manifest_output_path, remaining_rows)
    summary = {
        "provider": provider,
        "model_name": model_name,
        "source_manifest_path": str(manifest_path.as_posix()),
        "output_manifest_path": str(manifest_output_path.as_posix()),
        "done_prompt_ids": len(done_prompt_ids),
        "remaining_rows": len(remaining_rows),
        "limit": limit,
    }
    return summary


def summarize_manifest_subset(
    manifest_path: Path,
    output_dir: Path,
    provider: str,
    model_name: str,
) -> dict:
    requested_rows = list(jsonl_iter(manifest_path))
    requested_by_prompt = {row["prompt_id"]: row for row in requested_rows}
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
                    "has_reasoning": False,
                }
            )
        else:
            rows.append(result)

    found_rows = [row for row in rows if row.get("status") != "missing"]
    ok_rows = [row for row in found_rows if row.get("status") == "ok"]
    parse_rows = [row for row in ok_rows if row.get("parse_success") is True]
    missing_rows = [row for row in rows if row.get("status") == "missing"]
    has_reasoning_true = sum(1 for row in rows if row.get("has_reasoning") is True)

    http_error_counts = Counter(row.get("error", "")[:200] for row in rows if row.get("status") == "http_error")
    exception_error_counts = Counter(row.get("error", "")[:200] for row in rows if row.get("status") == "exception")
    parse_error_counts = Counter(row.get("parse_error", "")[:200] for row in ok_rows if row.get("parse_success") is not True)
    status_counts = Counter(row.get("status", "missing") for row in rows)
    task_counts = Counter(row["task_id"] for row in requested_rows)
    task_parse = Counter(row["task_id"] for row in parse_rows)

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
        "has_reasoning_true": has_reasoning_true,
        "status_counts": dict(sorted(status_counts.items())),
        "top_http_errors": dict(http_error_counts.most_common(10)),
        "top_exception_errors": dict(exception_error_counts.most_common(10)),
        "top_parse_errors": dict(parse_error_counts.most_common(10)),
        "tasks": dict(sorted(task_counts.items())),
        "task_parse_success_rate": {
            task: (task_parse.get(task, 0) / count) if count else 0.0 for task, count in sorted(task_counts.items())
        },
        "requested_prompt_ids": list(requested_by_prompt.keys()),
    }
    return summary


def summarize_all(output_dir: Path, providers: List[str], variant_id: str) -> None:
    summary = {
        "variant_id": variant_id,
        "providers": {provider: summarize_provider(output_dir, provider) for provider in providers},
    }
    (output_dir / "phase65d_full_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    pilot_output_dir = (root / args.pilot_output_dir).resolve()
    prompts_path = (root / args.prompts_path).resolve()
    phase65b_summary_path = (root / args.phase65b_summary).resolve()
    ensure_dir(output_dir)
    ensure_dir(pilot_output_dir)

    if args.mode == "build_manifest":
        build_manifest(prompts_path, output_dir, args.variant_id)
        return

    if args.mode == "build_pilot_manifest":
        build_pilot_manifest(prompts_path, phase65b_summary_path, pilot_output_dir, args.pilot_size)
        return

    if args.mode == "build_remaining_manifest":
        if not args.provider or not args.model_name or not args.manifest_output_path:
            raise SystemExit(
                "--provider, --model-name, and --manifest-output-path are required for build_remaining_manifest"
            )
        manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else output_dir / manifest_name(
            "phase65d_full_manifest", args.variant_id
        )
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        manifest_output_path = Path(args.manifest_output_path).resolve()
        ensure_dir(manifest_output_path.parent)
        summary = build_remaining_manifest(
            manifest_path=manifest_path,
            output_dir=output_dir,
            provider=args.provider,
            model_name=args.model_name,
            manifest_output_path=manifest_output_path,
            limit=args.limit,
        )
        if args.summary_path:
            summary_path = Path(args.summary_path).resolve()
            ensure_dir(summary_path.parent)
            summary_path.write_text(json.dumps(summary, indent=2))
        return

    if args.mode == "run_provider_shard":
        if not args.provider or not args.model_name:
            raise SystemExit("--provider and --model-name are required for run_provider_shard")
        manifest_path = Path(args.manifest_path).resolve() if args.manifest_path else output_dir / manifest_name(
            "phase65d_full_manifest", args.variant_id
        )
        if not manifest_path.exists():
            raise SystemExit(f"manifest not found: {manifest_path}")
        run_provider_shard(
            manifest_path=manifest_path,
            output_dir=manifest_path.parent,
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
        if not args.provider or not args.model_name:
            raise SystemExit("--provider and --model-name are required for summarize_manifest_subset")
        if not args.manifest_path:
            raise SystemExit("--manifest-path is required for summarize_manifest_subset")
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

    summarize_all(output_dir, args.providers, args.variant_id)


if __name__ == "__main__":
    main()
