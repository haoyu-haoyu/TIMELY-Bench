#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from run_phase65d_tier1b_v3 import (
    extract_json_blob,
    jsonl_iter,
    normalize_confidence,
    resolve_provider_runtime,
    run_curl_json,
)


PROMPT_ID = "AKI-T1::38301645::h46::D1::full_multimodal"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compact one-off repair for the final Qwen tail row")
    p.add_argument("--root", default=".")
    p.add_argument("--prompts-path", default="data/processed/v3/cres/cres_eval_prompts_12k.jsonl")
    p.add_argument("--provider", default="qwen35")
    p.add_argument("--model-name", default="qwen3.5-flash")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=1200)
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument(
        "--output-path",
        default="results/cres_v3/phase65d_tier1b_full/qwen35_responses_shard99_of_99.jsonl",
    )
    return p.parse_args()


def load_prompt_row(prompts_path: Path) -> dict:
    for row in jsonl_iter(prompts_path):
        if row.get("prompt_id") == PROMPT_ID:
            return row
    raise SystemExit(f"Prompt id not found: {PROMPT_ID}")


def invoke_compact_qwen(
    *,
    prompt_text: str,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
) -> tuple[dict, str, dict]:
    runtime = resolve_provider_runtime(provider, model_name)
    url = runtime["base_url"] + runtime["endpoint"].format(model_name=model_name)
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one valid JSON object and nothing else. "
                    "Follow the JSON schema in the user prompt exactly. "
                    "Use conservative AKI criteria: do not claim AKI Stage 2 unless the provided data explicitly meets KDIGO Stage 2. "
                    "If Stage 2 is not definitively documented by the anchor, say so explicitly in the answer. "
                    "Keep the output concise but not lossy: reasoning must be one short sentence under 35 words; "
                    "evidence must contain at most four objects and only the most relevant timestamps; "
                    "do not add extra keys, markdown, prose, or analysis outside the JSON."
                ),
            },
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    extra_body = runtime.get("extra_body") or {}
    if extra_body:
        payload.update(extra_body)
    response = run_curl_json(
        url=url,
        headers=[f"Authorization: Bearer {runtime['api_key']}"],
        payload=payload,
        timeout_seconds=timeout_seconds,
    )
    choice = (((response.get("choices") or [{}])[0] or {}).get("message") or {}).get("content")
    usage = response.get("usage") or {}
    return response, choice if isinstance(choice, str) else str(choice), {
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "usage_total_tokens": usage.get("total_tokens"),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    row = load_prompt_row((root / args.prompts_path).resolve())
    output_path = (root / args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    response, raw_content, usage = invoke_compact_qwen(
        prompt_text=row["prompt_text"],
        provider=args.provider,
        model_name=args.model_name,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout_seconds,
    )
    parsed, parse_error = extract_json_blob(raw_content)
    result = {
        "provider": args.provider,
        "model_name": args.model_name,
        "prompt_id": row["prompt_id"],
        "instance_id": row["instance_id"],
        "task_id": row["task_id"],
        "dimension_id": row["dimension_id"],
        "variant_id": row["variant_id"],
        "status": "ok",
        "parse_success": parsed is not None,
        "latency_seconds": round(time.time() - started, 3),
        "attempts": 1,
        **usage,
        "raw_content": raw_content,
        "parse_error": parse_error,
        "parsed_response": parsed,
        "confidence_value": normalize_confidence(parsed),
        "repair_mode": "compact_tail1_v2",
        "provider_response_meta": {
            "finish_reason": ((response.get("choices") or [{}])[0] or {}).get("finish_reason")
        },
    }
    with output_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(json.dumps({"output_path": str(output_path), "parse_success": result["parse_success"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
