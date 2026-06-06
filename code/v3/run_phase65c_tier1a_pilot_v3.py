#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_OUTPUT_DIR = "results/cres_v3/phase65c_tier1a_pilot"
DEFAULT_PROMPTS_PATH = "data/processed/v3/cres/cres_eval_prompts_12k.jsonl"
DEFAULT_SUMMARY_PATH = "results/cres_v3/phase65b_prompt_build_summary.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5C-1 Tier 1A pilot")
    p.add_argument("--mode", choices=["build_manifest", "run_provider", "summarize"], required=True)
    p.add_argument("--root", default=".")
    p.add_argument("--prompts-path", default=DEFAULT_PROMPTS_PATH)
    p.add_argument("--phase65b-summary", default=DEFAULT_SUMMARY_PATH)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--pilot-size", type=int, default=50)
    p.add_argument("--provider", choices=["gpt54mini", "gemini31pro"])
    p.add_argument("--model-name", default="")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout-seconds", type=int, default=180)
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def jsonl_iter(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def largest_remainder_quotas(targets: Dict[str, int], total: int) -> Dict[str, int]:
    weight_sum = sum(targets.values())
    raw = {k: (v / weight_sum) * total for k, v in targets.items()}
    quotas = {k: int(math.floor(x)) for k, x in raw.items()}
    remainder = total - sum(quotas.values())
    order = sorted(targets, key=lambda k: (raw[k] - quotas[k], targets[k]), reverse=True)
    for k in order[:remainder]:
        quotas[k] += 1
    return quotas


def build_pilot_manifest(root: Path, prompts_path: Path, phase65b_summary_path: Path, output_dir: Path, pilot_size: int) -> Path:
    summary = load_json(phase65b_summary_path)
    quotas = largest_remainder_quotas(summary["task_sample_targets"], pilot_size)

    pair_candidates: Dict[str, Dict[Tuple[str, str], dict]] = defaultdict(dict)
    extras: Dict[str, List[dict]] = defaultdict(list)
    for row in jsonl_iter(prompts_path):
        task_id = row["task_id"]
        pair = (row["variant_id"], row["dimension_id"])
        if pair not in pair_candidates[task_id]:
            pair_candidates[task_id][pair] = row
        elif len(extras[task_id]) < 512:
            extras[task_id].append(row)

    selected: List[dict] = []
    for task_id, quota in quotas.items():
        primary = sorted(
            pair_candidates[task_id].values(),
            key=lambda r: (r["variant_id"], r["dimension_id"], r["prompt_id"]),
        )
        fallback = sorted(
            extras[task_id],
            key=lambda r: (r["variant_id"], r["dimension_id"], r["prompt_id"]),
        )
        rows = (primary + fallback)[:quota]
        if len(rows) < quota:
            raise RuntimeError(f"Pilot quota for {task_id} unmet: need {quota}, found {len(rows)}")
        selected.extend(rows)

    selected = sorted(selected, key=lambda r: (r["task_id"], r["variant_id"], r["dimension_id"], r["prompt_id"]))
    manifest_path = output_dir / "phase65c_pilot50_manifest.jsonl"
    write_jsonl(manifest_path, selected)

    build_summary = {
        "pilot_size": len(selected),
        "tasks": {task: quotas[task] for task in sorted(quotas)},
        "output_path": str(manifest_path.as_posix()),
        "source_prompts_path": str(prompts_path.as_posix()),
    }
    (output_dir / "phase65c_pilot_manifest_summary.json").write_text(json.dumps(build_summary, indent=2))
    return manifest_path


def post_chat_completion(base_url: str, api_key: str, model_name: str, prompt_text: str, temperature: float, max_tokens: int, timeout_seconds: int) -> dict:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
        return json.loads(resp.read().decode("utf-8"))


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

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"
    # last resort: brace extraction
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
    return None, "no_json_object_found"


def run_provider(root: Path, output_dir: Path, provider: str, model_name: str, temperature: float, max_tokens: int, timeout_seconds: int) -> Path:
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    manifest_path = output_dir / "phase65c_pilot50_manifest.jsonl"
    rows_out: List[dict] = []
    for row in jsonl_iter(manifest_path):
        started = time.time()
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
        try:
            response = post_chat_completion(
                base_url=base_url,
                api_key=api_key,
                model_name=model_name,
                prompt_text=row["prompt_text"],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            choice = response["choices"][0]["message"]["content"]
            parsed, parse_error = extract_json_blob(choice)
            usage = response.get("usage", {})
            result.update(
                {
                    "latency_seconds": round(time.time() - started, 3),
                    "usage_prompt_tokens": usage.get("prompt_tokens"),
                    "usage_completion_tokens": usage.get("completion_tokens"),
                    "usage_total_tokens": usage.get("total_tokens"),
                    "raw_content": choice,
                    "parse_success": parsed is not None,
                    "parse_error": parse_error,
                    "parsed_response": parsed,
                }
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            result.update(
                {
                    "status": "http_error",
                    "latency_seconds": round(time.time() - started, 3),
                    "http_status": exc.code,
                    "error": detail[:4000],
                }
            )
        except Exception as exc:  # noqa: BLE001
            result.update(
                {
                    "status": "exception",
                    "latency_seconds": round(time.time() - started, 3),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        rows_out.append(result)

    output_path = output_dir / f"{provider}_pilot_responses.jsonl"
    write_jsonl(output_path, rows_out)
    return output_path


def summarize_provider(output_dir: Path, provider: str) -> dict:
    path = output_dir / f"{provider}_pilot_responses.jsonl"
    rows = list(jsonl_iter(path))
    ok_rows = [r for r in rows if r["status"] == "ok"]
    parse_rows = [r for r in ok_rows if r.get("parse_success")]
    return {
        "provider": provider,
        "rows": len(rows),
        "ok_rows": len(ok_rows),
        "parse_success_rows": len(parse_rows),
        "parse_success_rate": (len(parse_rows) / len(rows)) if rows else 0.0,
        "avg_latency_seconds": (sum(r.get("latency_seconds", 0.0) for r in rows) / len(rows)) if rows else 0.0,
    }


def summarize(output_dir: Path) -> None:
    providers = ["gpt54mini", "gemini31pro"]
    summary = {provider: summarize_provider(output_dir, provider) for provider in providers if (output_dir / f"{provider}_pilot_responses.jsonl").exists()}
    (output_dir / "phase65c_pilot_summary.json").write_text(json.dumps(summary, indent=2))


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    ensure_dir(output_dir)
    prompts_path = (root / args.prompts_path).resolve()
    phase65b_summary_path = (root / args.phase65b_summary).resolve()

    if args.mode == "build_manifest":
        build_pilot_manifest(root, prompts_path, phase65b_summary_path, output_dir, args.pilot_size)
    elif args.mode == "run_provider":
        if not args.provider or not args.model_name:
            raise SystemExit("--provider and --model-name are required for run_provider")
        run_provider(root, output_dir, args.provider, args.model_name, args.temperature, args.max_tokens, args.timeout_seconds)
    else:
        summarize(output_dir)


if __name__ == "__main__":
    main()
