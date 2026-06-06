#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List
from urllib import error, request

from run_phase65d_tier1b_v3 import (
    extract_json_blob,
    extract_openai_content,
    extract_openai_reasoning,
    jsonl_iter,
    normalize_confidence,
    summarize_provider,
)


DEFAULT_ROOT = "."
DEFAULT_PROVIDER = "qwen35"
DEFAULT_MODEL_NAME = "qwen3.5-flash"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_FULL_MANIFEST = "results/cres_v3/phase65d_tier1b_full/phase65d_full_manifest_full_multimodal.jsonl"
DEFAULT_PILOT_MANIFEST = "results/cres_v3/phase65d_tier1b_pilot/phase65d_pilot100_manifest.jsonl"
DEFAULT_OUTPUT_DIR = "results/cres_v3/phase65d_tier1b_full"
DEFAULT_BATCH_WORK_DIR = "results/cres_v3/phase65d_tier1b_batch/qwen35"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5D Qwen batch runner")
    p.add_argument(
        "--mode",
        choices=["build_inputs", "submit", "poll", "summarize"],
        required=True,
    )
    p.add_argument("--root", default=DEFAULT_ROOT)
    p.add_argument("--manifest-path", default=DEFAULT_FULL_MANIFEST)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--batch-work-dir", default=DEFAULT_BATCH_WORK_DIR)
    p.add_argument("--provider", default=DEFAULT_PROVIDER)
    p.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL)
    p.add_argument("--endpoint", default="/v1/chat/completions")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=2600)
    p.add_argument("--completion-window", default="24h")
    p.add_argument("--max-rows-per-file", type=int, default=30000)
    p.add_argument("--max-bytes-per-file", type=int, default=450_000_000)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {"jobs": []}
    return json.loads(path.read_text())


def save_registry(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def prompt_lookup(manifest_path: Path) -> Dict[str, dict]:
    lookup: Dict[str, dict] = {}
    for row in jsonl_iter(manifest_path):
        lookup[row["prompt_id"]] = row
    return lookup


def build_batch_line(row: dict, model_name: str, temperature: float, max_tokens: int, endpoint: str) -> dict:
    return {
        "custom_id": row["prompt_id"],
        "method": "POST",
        "url": endpoint,
        "body": {
            "model": model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Return exactly one valid JSON object and nothing else. "
                        "Do not output markdown, prose, analysis, or reasoning outside the JSON object. "
                        "The user prompt already specifies the required JSON fields and format."
                    ),
                },
                {"role": "user", "content": row["prompt_text"]},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "response_format": {"type": "json_object"},
        },
    }


def iter_split_rows(
    manifest_path: Path,
    *,
    model_name: str,
    temperature: float,
    max_tokens: int,
    endpoint: str,
    max_rows_per_file: int,
    max_bytes_per_file: int,
) -> Iterator[List[dict]]:
    current: List[dict] = []
    current_bytes = 0
    for row in jsonl_iter(manifest_path):
        batch_row = build_batch_line(row, model_name, temperature, max_tokens, endpoint)
        encoded = (json.dumps(batch_row, ensure_ascii=False) + "\n").encode("utf-8")
        if current and (
            len(current) >= max_rows_per_file or current_bytes + len(encoded) > max_bytes_per_file
        ):
            yield current
            current = []
            current_bytes = 0
        current.append(batch_row)
        current_bytes += len(encoded)
    if current:
        yield current


def build_inputs(
    *,
    manifest_path: Path,
    batch_work_dir: Path,
    model_name: str,
    temperature: float,
    max_tokens: int,
    endpoint: str,
    max_rows_per_file: int,
    max_bytes_per_file: int,
) -> dict:
    inputs_dir = batch_work_dir / "inputs"
    ensure_dir(inputs_dir)
    parts_meta: List[dict] = []
    parts = list(
        iter_split_rows(
            manifest_path,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            endpoint=endpoint,
            max_rows_per_file=max_rows_per_file,
            max_bytes_per_file=max_bytes_per_file,
        )
    )
    total_parts = len(parts)
    for idx, rows in enumerate(parts):
        path = inputs_dir / f"qwen35_batch_input_part{idx:02d}_of_{total_parts:02d}.jsonl"
        write_jsonl(path, rows)
        parts_meta.append(
            {
                "part_index": idx,
                "total_parts": total_parts,
                "input_path": str(path.as_posix()),
                "row_count": len(rows),
                "byte_count": path.stat().st_size,
            }
        )
    registry = {
        "provider": DEFAULT_PROVIDER,
        "model_name": model_name,
        "base_url": DEFAULT_BASE_URL,
        "endpoint": endpoint,
        "manifest_path": str(manifest_path.as_posix()),
        "parts": parts_meta,
        "jobs": [],
    }
    save_registry(batch_work_dir / "qwen35_batch_registry.json", registry)
    return registry


def curl_json(cmd: List[str]) -> dict:
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "curl_failed").strip()[:4000])
    data = completed.stdout.strip()
    return json.loads(data) if data else {}


def upload_file(base_url: str, api_key: str, path: Path) -> str:
    data = curl_json(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{base_url}/files",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-F",
            "purpose=batch",
            "-F",
            f"file=@{path.as_posix()}",
        ]
    )
    return data["id"]


def create_batch(base_url: str, api_key: str, input_file_id: str, endpoint: str, completion_window: str) -> str:
    payload = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    data = curl_json(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            f"{base_url}/batches",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            json.dumps(payload, ensure_ascii=False),
        ]
    )
    return data["id"]


def http_json(method: str, url: str, api_key: str) -> dict:
    req = request.Request(url, method=method, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:4000]}") from exc


def http_download(url: str, api_key: str, path: Path) -> None:
    req = request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with request.urlopen(req, timeout=300) as resp:
        path.write_bytes(resp.read())


def submit_batches(batch_work_dir: Path, base_url: str, api_key: str, completion_window: str) -> dict:
    registry_path = batch_work_dir / "qwen35_batch_registry.json"
    registry = load_registry(registry_path)
    jobs: List[dict] = []
    registry["jobs"] = jobs
    save_registry(registry_path, registry)
    for part in registry["parts"]:
        input_path = Path(part["input_path"])
        input_file_id = upload_file(base_url, api_key, input_path)
        batch_id = create_batch(base_url, api_key, input_file_id, registry["endpoint"], completion_window)
        jobs.append(
            {
                "part_index": part["part_index"],
                "total_parts": part["total_parts"],
                "input_path": part["input_path"],
                "row_count": part["row_count"],
                "byte_count": part["byte_count"],
                "input_file_id": input_file_id,
                "batch_id": batch_id,
                "status": "submitted",
                "output_file_id": None,
                "error_file_id": None,
                "downloaded_output_path": None,
                "downloaded_error_path": None,
                "transformed": False,
            }
        )
        save_registry(registry_path, registry)
    return registry


def transform_output_line(raw: dict, prompt_meta: dict, provider: str, model_name: str) -> dict:
    body = ((raw.get("response") or {}).get("body")) or {}
    choice = extract_openai_content(body)
    raw_reasoning = extract_openai_reasoning(body)
    parsed, parse_error = extract_json_blob(choice)
    usage = body.get("usage", {})
    return {
        "provider": provider,
        "model_name": body.get("model") or model_name,
        "prompt_id": raw["custom_id"],
        "instance_id": prompt_meta["instance_id"],
        "task_id": prompt_meta["task_id"],
        "dimension_id": prompt_meta["dimension_id"],
        "variant_id": prompt_meta["variant_id"],
        "status": "ok",
        "parse_success": parsed is not None,
        "attempts": 1,
        "latency_seconds": None,
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
        "raw_content": choice,
        "has_reasoning": bool(raw_reasoning),
        "parse_error": parse_error,
        "parsed_response": parsed,
        "confidence_value": normalize_confidence(parsed),
    }


def transform_error_line(raw: dict, prompt_meta: dict, provider: str, model_name: str) -> dict:
    err = raw.get("error") or {}
    status_code = ((raw.get("response") or {}).get("status_code")) or err.get("status_code")
    msg = err.get("message") or err.get("error_message") or json.dumps(err, ensure_ascii=False)
    return {
        "provider": provider,
        "model_name": model_name,
        "prompt_id": raw["custom_id"],
        "instance_id": prompt_meta["instance_id"],
        "task_id": prompt_meta["task_id"],
        "dimension_id": prompt_meta["dimension_id"],
        "variant_id": prompt_meta["variant_id"],
        "status": "http_error" if status_code else "exception",
        "parse_success": False,
        "attempts": 1,
        "latency_seconds": None,
        "http_status": status_code,
        "error": msg[:4000],
    }


def transform_downloads(batch_work_dir: Path, output_dir: Path, registry: dict) -> None:
    manifest_lookup = prompt_lookup(Path(registry["manifest_path"]))
    total_parts = len(registry["jobs"])
    for job in registry["jobs"]:
        if not job.get("downloaded_output_path") and not job.get("downloaded_error_path"):
            continue
        shard_path = output_dir / f"qwen35_responses_shard{job['part_index']:02d}_of_{total_parts:02d}.jsonl"
        rows: Dict[str, dict] = {}
        output_path = job.get("downloaded_output_path")
        if output_path:
            for raw in jsonl_iter(Path(output_path)):
                prompt_id = raw["custom_id"]
                rows[prompt_id] = transform_output_line(raw, manifest_lookup[prompt_id], registry["provider"], registry["model_name"])
        error_path = job.get("downloaded_error_path")
        if error_path:
            for raw in jsonl_iter(Path(error_path)):
                prompt_id = raw["custom_id"]
                if prompt_id not in rows:
                    rows[prompt_id] = transform_error_line(raw, manifest_lookup[prompt_id], registry["provider"], registry["model_name"])
        write_jsonl(shard_path, rows.values())
        job["transformed"] = True


def poll_batches(batch_work_dir: Path, output_dir: Path, base_url: str, api_key: str) -> dict:
    registry_path = batch_work_dir / "qwen35_batch_registry.json"
    registry = load_registry(registry_path)
    downloads_dir = batch_work_dir / "downloads"
    ensure_dir(downloads_dir)

    for job in registry["jobs"]:
        batch = http_json("GET", f"{base_url}/batches/{job['batch_id']}", api_key)
        job["status"] = batch["status"]
        job["output_file_id"] = batch.get("output_file_id")
        job["error_file_id"] = batch.get("error_file_id")
        if batch["status"] == "completed":
            if job["output_file_id"] and not job.get("downloaded_output_path"):
                output_path = downloads_dir / f"qwen35_batch_output_part{job['part_index']:02d}.jsonl"
                http_download(f"{base_url}/files/{job['output_file_id']}/content", api_key, output_path)
                job["downloaded_output_path"] = str(output_path.as_posix())
            if job["error_file_id"] and not job.get("downloaded_error_path"):
                error_path = downloads_dir / f"qwen35_batch_error_part{job['part_index']:02d}.jsonl"
                http_download(f"{base_url}/files/{job['error_file_id']}/content", api_key, error_path)
                job["downloaded_error_path"] = str(error_path.as_posix())

    transform_downloads(batch_work_dir, output_dir, registry)
    save_registry(registry_path, registry)
    return registry


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest_path).resolve()
    output_dir = (root / args.output_dir).resolve()
    batch_work_dir = (root / args.batch_work_dir).resolve()
    ensure_dir(output_dir)
    ensure_dir(batch_work_dir)

    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip() or os.environ.get("TIER1B_QWEN35_API_KEY", "").strip()
    if args.mode in {"submit", "poll"} and not api_key:
        raise SystemExit("DASHSCOPE_API_KEY is required")

    if args.mode == "build_inputs":
        build_inputs(
            manifest_path=manifest_path,
            batch_work_dir=batch_work_dir,
            model_name=args.model_name,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            endpoint=args.endpoint,
            max_rows_per_file=args.max_rows_per_file,
            max_bytes_per_file=args.max_bytes_per_file,
        )
        return

    if args.mode == "submit":
        submit_batches(batch_work_dir, args.base_url.rstrip("/"), api_key, args.completion_window)
        return

    if args.mode == "poll":
        poll_batches(batch_work_dir, output_dir, args.base_url.rstrip("/"), api_key)
        return

    summarize_provider(output_dir, args.provider)


if __name__ == "__main__":
    main()
