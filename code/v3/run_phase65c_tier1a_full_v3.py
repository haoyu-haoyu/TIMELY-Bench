#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


DEFAULT_OUTPUT_DIR = "results/cres_v3/phase65c_tier1a_full"
DEFAULT_PROMPTS_PATH = "data/processed/v3/cres/cres_eval_prompts_12k.jsonl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5C-2 Tier 1A full run")
    p.add_argument("--mode", choices=["build_manifest", "run_provider_shard", "summarize"], required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--prompts-path", default=DEFAULT_PROMPTS_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--variant-id", default="full_multimodal")
    p.add_argument("--provider", choices=["gpt54", "gemini31pro"])
    p.add_argument("--model-name", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument("--num-shards", type=int, default=8)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=2)
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
                # A small number of provider shard files can contain truncated tail
                # fragments after interrupted writes. Skip those rows during merge.
                print(
                    f"[warn] skipping malformed jsonl row in {path.as_posix()}:{line_num}",
                    flush=True,
                )


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def manifest_name(variant_id: str) -> str:
    return f"phase65c_full_manifest_{variant_id}.jsonl"


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
    out_path = output_dir / manifest_name(variant_id)
    write_jsonl(out_path, selected)
    summary = {
        "variant_id": variant_id,
        "manifest_rows": len(selected),
        "unique_instances": len({row["instance_id"] for row in selected}),
        "tasks": dict(sorted(task_counts.items())),
        "dimensions": dict(sorted(dimension_counts.items())),
        "output_path": str(out_path.as_posix()),
    }
    (output_dir / "phase65c_full_manifest_summary.json").write_text(json.dumps(summary, indent=2))
    return out_path


def provider_shard_path(output_dir: Path, provider: str, shard_index: int, num_shards: int) -> Path:
    return output_dir / f"{provider}_responses_shard{shard_index:02d}_of_{num_shards:02d}.jsonl"


def provider_shard_paths(output_dir: Path, provider: str) -> List[Path]:
    return sorted(output_dir.glob(f"{provider}_responses_shard*.jsonl"))


def load_latest_rows(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    latest: Dict[str, dict] = {}
    for row in jsonl_iter(path):
        latest[row["prompt_id"]] = row
    return latest


def load_preferred_rows(paths: List[Path]) -> Dict[str, dict]:
    preferred: Dict[str, dict] = {}
    ordered_paths = sorted(paths, key=lambda path: (path.stat().st_mtime, path.name))
    for path in ordered_paths:
        if not path.exists():
            continue
        for row in jsonl_iter(path):
            prompt_id = row["prompt_id"]
            existing = preferred.get(prompt_id)
            row_ok = row.get("status") == "ok"
            existing_ok = existing is not None and existing.get("status") == "ok"
            if row_ok:
                preferred[prompt_id] = row
            elif not existing_ok:
                preferred[prompt_id] = row
    return preferred


def load_done_prompt_ids(output_dir: Path, provider: str, model_name: str) -> set[str]:
    done: set[str] = set()
    for path in provider_shard_paths(output_dir, provider):
        latest = load_latest_rows(path)
        for prompt_id, row in latest.items():
            if row.get("status") == "ok" and row.get("model_name") == model_name:
                done.add(prompt_id)
    return done


def iter_manifest_shard(manifest_path: Path, shard_index: int, num_shards: int) -> Iterator[dict]:
    for idx, row in enumerate(jsonl_iter(manifest_path)):
        if idx % num_shards == shard_index:
            yield row


def run_curl_json(
    *,
    url: str,
    headers: List[str],
    payload: dict,
    timeout_seconds: int,
) -> dict:
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


def extract_gpt_content(response: dict) -> str:
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
        parts = []
        for item in choice:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(part for part in parts if part).strip()
    return str(choice)


def extract_gemini_content(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part:
            texts.append(part["text"])
    return "\n".join(texts).strip()


def invoke_provider(
    *,
    provider: str,
    base_url: str,
    api_key: str,
    model_name: str,
    prompt_text: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[dict, str, dict]:
    base_root = base_url.rstrip("/")
    if provider == "gpt54":
        response = run_curl_json(
            url=base_root + "/v1/chat/completions",
            headers=[f"Authorization: Bearer {api_key}"],
            payload={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout_seconds=timeout_seconds,
        )
        usage = response.get("usage", {})
        return response, extract_gpt_content(response), {
            "usage_prompt_tokens": usage.get("prompt_tokens"),
            "usage_completion_tokens": usage.get("completion_tokens"),
            "usage_total_tokens": usage.get("total_tokens"),
        }

    if provider == "gemini31pro":
        # Some third-party Gemini routes expose an OpenAI-compatible chat endpoint.
        if "vibeapi.cn" in base_root:
            response = run_curl_json(
                url=base_root + "/v1/chat/completions",
                headers=[f"Authorization: Bearer {api_key}"],
                payload={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout_seconds=timeout_seconds,
            )
            usage = response.get("usage", {})
            return response, extract_gpt_content(response), {
                "usage_prompt_tokens": usage.get("prompt_tokens"),
                "usage_completion_tokens": usage.get("completion_tokens"),
                "usage_total_tokens": usage.get("total_tokens"),
            }
        response = run_curl_json(
            url=base_root + f"/v1beta/models/{model_name}:generateContent",
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
        return response, extract_gemini_content(response), {
            "usage_prompt_tokens": usage.get("promptTokenCount"),
            "usage_completion_tokens": usage.get("candidatesTokenCount"),
            "usage_total_tokens": usage.get("totalTokenCount"),
        }

    raise ValueError(f"unsupported provider: {provider}")


def normalize_confidence(parsed: dict | None) -> str | None:
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("confidence")
    if value is None:
        return None
    return str(value).strip().lower()


def run_one_request(
    *,
    row: dict,
    provider: str,
    model_name: str,
    base_url: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    max_retries: int,
) -> dict:
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
            response, choice, usage = invoke_provider(
                provider=provider,
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                prompt_text=row["prompt_text"],
                temperature=temperature,
                max_tokens=max_tokens,
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
                    "parse_success": parsed is not None,
                    "parse_error": parse_error,
                    "parsed_response": parsed,
                    "confidence_value": normalize_confidence(parsed),
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
                }
            )
            retriable = status == "exception" or http_status in {408, 409, 429, 500, 502, 503, 504}
            if not retriable:
                return result
            if attempt + 1 >= max_retries:
                return result
        time.sleep(min(20.0, 3.0 * (attempt + 1)))
    return result


def run_provider_shard(
    *,
    output_dir: Path,
    variant_id: str,
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
    base_url = os.environ.get("IKUNCODE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("IKUNCODE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not base_url or not api_key:
        raise SystemExit("IKUNCODE_BASE_URL and IKUNCODE_API_KEY (or OPENAI_* fallbacks) are required")
    manifest_path = output_dir / manifest_name(variant_id)
    shard_path = provider_shard_path(output_dir, provider, shard_index, num_shards)
    done_prompt_ids = load_done_prompt_ids(output_dir, provider, model_name)
    pending_rows = [
        row
        for row in iter_manifest_shard(manifest_path, shard_index, num_shards)
        if row["prompt_id"] not in done_prompt_ids
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
                    base_url=base_url,
                    api_key=api_key,
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
                append_jsonl(shard_path, fut.result())

    return shard_path


def summarize_provider(output_dir: Path, provider: str, num_shards: int) -> dict:
    shard_paths = provider_shard_paths(output_dir, provider)
    preferred_by_prompt = load_preferred_rows(shard_paths)
    rows = list(preferred_by_prompt.values())

    merged_path = output_dir / f"{provider}_full_responses.jsonl"
    write_jsonl(merged_path, rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
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
        "avg_latency_seconds": (sum(r.get("latency_seconds", 0.0) for r in rows) / len(rows)) if rows else 0.0,
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


def summarize_all(output_dir: Path, providers: List[str], num_shards: int, variant_id: str) -> None:
    summary = {
        "variant_id": variant_id,
        "providers": {provider: summarize_provider(output_dir, provider, num_shards) for provider in providers},
    }
    (output_dir / "phase65c_full_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    prompts_path = (root / args.prompts_path).resolve()
    ensure_dir(output_dir)

    if args.mode == "build_manifest":
        build_manifest(prompts_path, output_dir, args.variant_id)
    elif args.mode == "run_provider_shard":
        if not args.provider or not args.model_name:
            raise SystemExit("--provider and --model-name are required for run_provider_shard")
        run_provider_shard(
            output_dir=output_dir,
            variant_id=args.variant_id,
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
    else:
        summarize_all(output_dir, ["gpt54", "gemini31pro"], args.num_shards, args.variant_id)


if __name__ == "__main__":
    main()
