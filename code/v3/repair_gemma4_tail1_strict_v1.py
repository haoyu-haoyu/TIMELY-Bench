#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from run_phase65d_tier1b_v3 import (
    extract_json_blob,
    jsonl_iter,
    normalize_confidence,
    resolve_provider_runtime,
    run_curl_json,
)


PROMPT_ID = "AKI-T1::37437370::h49::D1::full_multimodal"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strict one-off repair for final Gemma4 tail row")
    p.add_argument("--root", default=".")
    p.add_argument(
        "--manifest-path",
        default="results/cres_v3/phase65d_tier1b_full/phase65d_full_manifest_full_multimodal.jsonl",
    )
    p.add_argument("--provider", default="gemma4_26b")
    p.add_argument("--model-name", default="arc:lite")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout-seconds", type=int, default=180)
    p.add_argument(
        "--output-path",
        default="results/cres_v3/phase65d_tier1b_full/gemma4_26b_responses_shard99_of_99.jsonl",
    )
    return p.parse_args()


def load_prompt_row(manifest_path: Path) -> dict:
    for row in jsonl_iter(manifest_path):
        if row.get("prompt_id") == PROMPT_ID:
            return row
    raise SystemExit(f"Prompt id not found: {PROMPT_ID}")


def invoke_strict_gemma(
    *,
    prompt_text: str,
    provider: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout_seconds: int,
    system_prompt: str,
) -> tuple[dict, str, dict]:
    prefix = f"TIER1B_{provider.upper()}_EXTRA_BODY_JSON"
    extra_body = {}
    extra_body_env = os.environ.get(prefix, "").strip()
    if extra_body_env:
        try:
            parsed = json.loads(extra_body_env)
            if isinstance(parsed, dict):
                extra_body.update(parsed)
        except json.JSONDecodeError:
            pass
    extra_body["response_format"] = {"type": "json_object"}
    extra_body.setdefault("chat_template_kwargs", {})
    if isinstance(extra_body["chat_template_kwargs"], dict):
        extra_body["chat_template_kwargs"]["enable_thinking"] = False
    os.environ[prefix] = json.dumps(extra_body, ensure_ascii=False)

    runtime = resolve_provider_runtime(provider, model_name)
    url = runtime["base_url"] + runtime["endpoint"].format(model_name=model_name)
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    payload.update(runtime.get("extra_body") or {})
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
    row = load_prompt_row((root / args.manifest_path).resolve())
    output_path = (root / args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    strategies = [
        {
            "repair_mode": "strict_json_tail1_v1",
            "max_tokens": 1200,
            "system_prompt": (
                "Return exactly one valid JSON object and nothing else. "
                "Follow the JSON schema from the user prompt exactly. "
                "Do not output markdown fences. "
                "Keep reasoning concise and clinically faithful: at most 60 words. "
                "Evidence must contain at most 4 objects. "
                "No extra keys."
            ),
        },
        {
            "repair_mode": "strict_json_tail1_v2",
            "max_tokens": 1800,
            "system_prompt": (
                "Return exactly one valid JSON object and nothing else. "
                "Use the JSON schema from the user prompt exactly. "
                "Keep reasoning under 40 words. "
                "Evidence at most 4 objects. "
                "No markdown, no prose outside JSON, no repeated analysis."
            ),
        },
        {
            "repair_mode": "compact_json_tail1_v3",
            "max_tokens": 900,
            "system_prompt": (
                "Return exactly one valid JSON object and nothing else. "
                "Use concise clinical reasoning. "
                "reasoning must be one short sentence. "
                "evidence must have at most 3 items. "
                "No markdown or extra text."
            ),
        },
    ]

    last_result = None
    for strategy in strategies:
        started = time.time()
        try:
            response, raw_content, usage = invoke_strict_gemma(
                prompt_text=row["prompt_text"],
                provider=args.provider,
                model_name=args.model_name,
                temperature=args.temperature,
                max_tokens=strategy["max_tokens"],
                timeout_seconds=args.timeout_seconds,
                system_prompt=strategy["system_prompt"],
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
                "requested_max_tokens": strategy["max_tokens"],
                "repair_mode": strategy["repair_mode"],
                "provider_response_meta": {
                    "finish_reason": ((response.get("choices") or [{}])[0] or {}).get("finish_reason")
                },
            }
        except Exception as exc:  # noqa: BLE001
            result = {
                "provider": args.provider,
                "model_name": args.model_name,
                "prompt_id": row["prompt_id"],
                "instance_id": row["instance_id"],
                "task_id": row["task_id"],
                "dimension_id": row["dimension_id"],
                "variant_id": row["variant_id"],
                "status": "exception",
                "parse_success": False,
                "error": str(exc)[:4000],
                "repair_mode": strategy["repair_mode"],
            }

        last_result = result
        print(
            json.dumps(
                {
                    "repair_mode": strategy["repair_mode"],
                    "status": result.get("status"),
                    "parse_success": result.get("parse_success"),
                    "parse_error": result.get("parse_error"),
                    "raw_len": len(result.get("raw_content") or ""),
                    "usage_completion_tokens": result.get("usage_completion_tokens"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if result.get("status") == "ok" and result.get("parse_success") is True:
            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                json.dumps(
                    {
                        "output_path": str(output_path),
                        "repair_mode": strategy["repair_mode"],
                        "parse_success": True,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return

    if last_result is not None:
        with output_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(last_result, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "repair_mode": (last_result or {}).get("repair_mode"),
                "parse_success": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
