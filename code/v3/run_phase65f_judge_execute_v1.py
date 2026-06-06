#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Iterable, Iterator, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "cres_v3" / "phase65f_frozen_eval"
DEFAULT_MANIFEST_PATH = DEFAULT_OUTPUT_DIR / "phase65f_judge500_manifest.jsonl"
DEFAULT_RUBRIC_PATH = DEFAULT_OUTPUT_DIR / "phase65f_judge_rubric.md"

REQUIRED_FIELDS = [
    "overall_quality_1to5",
    "clinical_correctness_1to5",
    "temporal_grounding_1to5_or_na",
    "evidence_grounding_1to5",
    "confidence_calibration_1to5",
    "brief_rationale",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Execute Phase 6.5F LLM-as-Judge runs with resumable JSONL outputs.")
    p.add_argument("--manifest-path", default=str(DEFAULT_MANIFEST_PATH))
    p.add_argument("--rubric-path", default=str(DEFAULT_RUBRIC_PATH))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--judge-role", default="primary")
    p.add_argument("--judge-provider", default="claude")
    p.add_argument("--judge-label", default="claude_opus_4_6")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--messages-path", default="/v1/messages")
    p.add_argument("--api-key-env", default="ANTHROPIC_API_KEY")
    p.add_argument("--user-agent", default="curl/8.7.1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-seconds", type=float, default=180.0)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--retry-backoff-seconds", type=float, default=3.0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def json_sanitize(value):
    if isinstance(value, dict):
        return {str(k): json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_sanitize(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```json"):
        stripped = stripped[len("```json") :].strip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    return stripped


def extract_json_object(text: str) -> Optional[dict]:
    stripped = strip_code_fence(text)
    candidates = [stripped]
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def coerce_score_1to5(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        iv = int(value)
        return iv if 1 <= iv <= 5 else None
    text = str(value).strip().lower()
    if text.isdigit():
        iv = int(text)
        return iv if 1 <= iv <= 5 else None
    return None


def normalize_temporal_score(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"na", "n/a", "not_applicable", "not applicable"}:
        return "na"
    numeric = coerce_score_1to5(value)
    return numeric


def normalize_judge_output(obj: dict) -> Optional[dict]:
    out = {}
    out["overall_quality_1to5"] = coerce_score_1to5(obj.get("overall_quality_1to5"))
    out["clinical_correctness_1to5"] = coerce_score_1to5(obj.get("clinical_correctness_1to5"))
    out["temporal_grounding_1to5_or_na"] = normalize_temporal_score(obj.get("temporal_grounding_1to5_or_na"))
    out["evidence_grounding_1to5"] = coerce_score_1to5(obj.get("evidence_grounding_1to5"))
    out["confidence_calibration_1to5"] = coerce_score_1to5(obj.get("confidence_calibration_1to5"))
    rationale = str(obj.get("brief_rationale", "") or "").strip()
    out["brief_rationale"] = rationale

    if any(out[key] is None for key in REQUIRED_FIELDS if key != "brief_rationale"):
        return None
    if not out["brief_rationale"]:
        return None
    return out


def build_user_prompt(*, rubric_text: str, row: dict, judge_role: str) -> str:
    packet = {
        "judge_role": judge_role,
        "judge_row_id": row.get("judge_row_id"),
        "prompt_id": row.get("prompt_id"),
        "provider": row.get("provider"),
        "tier": row.get("tier"),
        "condition": row.get("condition"),
        "task_id": row.get("task_id"),
        "dimension_id": row.get("dimension_id"),
        "selection_bucket": row.get("selection_bucket"),
        "selection_rationale": row.get("selection_rationale"),
        "judge_overlap_note": row.get("judge_overlap_note"),
        "prompt_text": row.get("prompt_text"),
        "contestant_raw_content": row.get("raw_content"),
        "contestant_parsed_response": row.get("parsed_response"),
    }
    serialized = json.dumps(json_sanitize(packet), ensure_ascii=False)
    return (
        "You are an expert clinical benchmark judge.\n"
        "Score the contestant response strictly according to the rubric.\n"
        "Return JSON only. Do not include markdown fences, prose, or extra keys.\n\n"
        f"{rubric_text}\n\n"
        "Return exactly this JSON schema:\n"
        "{\n"
        '  "overall_quality_1to5": 1,\n'
        '  "clinical_correctness_1to5": 1,\n'
        '  "temporal_grounding_1to5_or_na": 1,\n'
        '  "evidence_grounding_1to5": 1,\n'
        '  "confidence_calibration_1to5": 1,\n'
        '  "brief_rationale": "short rationale"\n'
        "}\n\n"
        "If temporal grounding is not applicable, set temporal_grounding_1to5_or_na to \"na\".\n\n"
        f"Judge packet:\n{serialized}\n"
    )


def anthropic_post_json(
    *,
    base_url: str,
    messages_path: str,
    api_key: str,
    model: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: float,
    user_agent: str,
) -> tuple[int, dict]:
    url = base_url.rstrip("/") + "/" + messages_path.lstrip("/")
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt_text}],
    }
    data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    req = urllib.request.Request(
        url=url,
        data=data,
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": user_agent,
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return resp.getcode(), json.loads(body)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except Exception:
            parsed = {"error": {"message": body}}
        return e.code, parsed


def extract_anthropic_text(resp_obj: dict) -> str:
    parts = resp_obj.get("content") or []
    texts = []
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "text":
            text = str(part.get("text", "") or "").strip()
            if text:
                texts.append(text)
    return "\n".join(texts).strip()


def load_existing_ok_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    for row in iter_jsonl(path):
        if row.get("status") == "ok" and row.get("judge_row_id"):
            done.add(str(row["judge_row_id"]))
    return done


def summarize_outputs(path: Path, *, total_expected: int, args: argparse.Namespace) -> dict:
    status_counts = {}
    usage_input = 0
    usage_output = 0
    latencies = []
    completed_ids = set()
    if path.exists():
        for row in iter_jsonl(path):
            status = str(row.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            judge_row_id = row.get("judge_row_id")
            if judge_row_id and status == "ok":
                completed_ids.add(str(judge_row_id))
            usage = row.get("usage") or {}
            usage_input += int(usage.get("input_tokens") or 0)
            usage_output += int(usage.get("output_tokens") or 0)
            latency = row.get("latency_seconds")
            if isinstance(latency, (int, float)):
                latencies.append(float(latency))
    summary = {
        "judge_role": args.judge_role,
        "judge_provider": args.judge_provider,
        "judge_label": args.judge_label,
        "model": args.model,
        "base_url": args.base_url,
        "messages_path": args.messages_path,
        "user_agent": args.user_agent,
        "total_expected_rows": total_expected,
        "completed_ok_rows": len(completed_ids),
        "status_counts": status_counts,
        "usage_input_tokens": usage_input,
        "usage_output_tokens": usage_output,
        "avg_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
        "output_jsonl": str(output_path_for(args)),
    }
    return summary


def output_stem_for(args: argparse.Namespace) -> str:
    return f"phase65f_judge_{args.judge_role}_{args.judge_label}"


def output_path_for(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve() / f"{output_stem_for(args)}_outputs.jsonl"


def summary_path_for(args: argparse.Namespace) -> Path:
    return Path(args.output_dir).resolve() / f"{output_stem_for(args)}_summary.json"


def run_one(
    row: dict,
    *,
    rubric_text: str,
    args: argparse.Namespace,
    api_key: str,
) -> dict:
    judge_row_id = str(row["judge_row_id"])
    prompt = build_user_prompt(rubric_text=rubric_text, row=row, judge_role=args.judge_role)
    started = time.time()
    last_error = None
    for attempt in range(1, args.max_retries + 1):
        try:
            code, resp_obj = anthropic_post_json(
                base_url=args.base_url,
                messages_path=args.messages_path,
                api_key=api_key,
                model=args.model,
                prompt_text=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                timeout_seconds=args.timeout_seconds,
                user_agent=args.user_agent,
            )
            latency = time.time() - started
            usage = resp_obj.get("usage") or {}
            usage_out = {
                "input_tokens": int(usage.get("input_tokens") or 0),
                "output_tokens": int(usage.get("output_tokens") or 0),
            }
            if 200 <= code < 300:
                raw_text = extract_anthropic_text(resp_obj)
                parsed = extract_json_object(raw_text)
                normalized = normalize_judge_output(parsed or {})
                if normalized is None:
                    return {
                        "judge_row_id": judge_row_id,
                        "status": "parse_error",
                        "attempts": attempt,
                        "latency_seconds": latency,
                        "usage": usage_out,
                        "raw_judge_content": raw_text,
                        "judge_output": parsed,
                    }
                return {
                    "judge_row_id": judge_row_id,
                    "status": "ok",
                    "attempts": attempt,
                    "latency_seconds": latency,
                    "usage": usage_out,
                    "judge_output": normalized,
                    "raw_judge_content": raw_text,
                    "provider": row.get("provider"),
                    "prompt_id": row.get("prompt_id"),
                    "condition": row.get("condition"),
                    "task_id": row.get("task_id"),
                    "dimension_id": row.get("dimension_id"),
                }
            message = ((resp_obj.get("error") or {}).get("message")) or str(resp_obj)
            last_error = f"http_{code}: {message}"
            if code in {408, 409, 429, 500, 502, 503, 504, 529} and attempt < args.max_retries:
                time.sleep(args.retry_backoff_seconds * attempt)
                continue
            return {
                "judge_row_id": judge_row_id,
                "status": "http_error",
                "attempts": attempt,
                "http_status": code,
                "latency_seconds": latency,
                "usage": usage_out,
                "error": last_error,
                "http_error_body": resp_obj,
            }
        except Exception as e:  # noqa: BLE001
            latency = time.time() - started
            last_error = str(e)
            if attempt < args.max_retries:
                time.sleep(args.retry_backoff_seconds * attempt)
                continue
            return {
                "judge_row_id": judge_row_id,
                "status": "exception",
                "attempts": attempt,
                "latency_seconds": latency,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "error": last_error,
            }
    raise RuntimeError(f"unreachable for judge_row_id={judge_row_id}")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is not set")

    manifest_path = Path(args.manifest_path).resolve()
    rubric_path = Path(args.rubric_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(args)
    summary_path = summary_path_for(args)

    rubric_text = rubric_path.read_text(encoding="utf-8")
    rows = list(iter_jsonl(manifest_path))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    done_ok_ids = load_existing_ok_ids(output_path)
    pending_rows = [row for row in rows if str(row.get("judge_row_id")) not in done_ok_ids]

    lock = threading.Lock()
    with output_path.open("a", encoding="utf-8") as out_fh:
        def write_result(result: dict) -> None:
            with lock:
                out_fh.write(json.dumps(json_sanitize(result), ensure_ascii=False) + "\n")
                out_fh.flush()

        if pending_rows:
            with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
                futures = {
                    executor.submit(run_one, row, rubric_text=rubric_text, args=args, api_key=api_key): row
                    for row in pending_rows[: max(1, args.workers)]
                }
                next_idx = len(futures)
                while futures:
                    done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                    for future in done:
                        _ = futures.pop(future)
                        result = future.result()
                        write_result(result)
                        if next_idx < len(pending_rows):
                            row = pending_rows[next_idx]
                            futures[executor.submit(run_one, row, rubric_text=rubric_text, args=args, api_key=api_key)] = row
                            next_idx += 1
                        summary = summarize_outputs(output_path, total_expected=len(rows), args=args)
                        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            summary = summarize_outputs(output_path, total_expected=len(rows), args=args)
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return

    summary = summarize_outputs(output_path, total_expected=len(rows), args=args)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
