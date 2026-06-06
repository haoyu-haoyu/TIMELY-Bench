#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


PROMPT_IDS = [
    "AKI-T1::30020731::h11::D1::no_temporal_markers",
    "AKI-T1::30020731::h11::D1::shuffled_timeline",
    "S-T1::30065812::h6::D1::no_temporal_markers",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe token safety for Qwen tail prompts")
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--base-url", required=True)
    p.add_argument("--api-key", required=True)
    p.add_argument("--endpoint", default="/chat/completions")
    p.add_argument("--model-name", required=True)
    p.add_argument("--token-levels", nargs="+", type=int, default=[2200, 2600, 3200, 4000])
    return p.parse_args()


def load_prompts(manifest_path: Path) -> dict[str, str]:
    prompts: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            prompt_id = row.get("prompt_id")
            if prompt_id in PROMPT_IDS:
                prompts[prompt_id] = row["prompt_text"]
    missing = [prompt_id for prompt_id in PROMPT_IDS if prompt_id not in prompts]
    if missing:
        raise SystemExit(f"Missing prompt_ids in manifest: {missing}")
    return prompts


def extract_json_blob(text: str) -> tuple[bool, str | None]:
    text = (text or "").strip()
    candidates: list[str] = []
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
                return True, None
        except Exception as exc:  # noqa: BLE001
            last_err = f"{type(exc).__name__}: {exc}"

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return True, None
        except Exception as exc:  # noqa: BLE001
            return False, f"{type(exc).__name__}: {exc}"
    return False, last_err or "no_json_object_found"


def invoke(*, url: str, api_key: str, model_name: str, prompt_text: str, max_tokens: int) -> dict:
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
    }
    cmd = [
        "curl",
        "-sS",
        "-X",
        "POST",
        url,
        "-H",
        f"Authorization: Bearer {api_key}",
        "-H",
        "Content-Type: application/json",
        "--data-binary",
        "@-",
    ]
    completed = subprocess.run(
        cmd,
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError((completed.stderr or completed.stdout or "curl_failed").strip()[:4000])
    return json.loads(completed.stdout)


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(manifest_path)
    url = args.base_url.rstrip("/") + args.endpoint

    rows: list[dict] = []
    for prompt_id in PROMPT_IDS:
        prompt_text = prompts[prompt_id]
        for max_tokens in args.token_levels:
            response = invoke(
                url=url,
                api_key=args.api_key,
                model_name=args.model_name,
                prompt_text=prompt_text,
                max_tokens=max_tokens,
            )
            choice = ((response.get("choices") or [{}])[0].get("message") or {})
            content = choice.get("content")
            usage = response.get("usage") or {}
            parse_success, parse_error = extract_json_blob(content if isinstance(content, str) else "")
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "max_tokens": max_tokens,
                    "finish_reason": (response.get("choices") or [{}])[0].get("finish_reason"),
                    "content_is_none": content is None,
                    "content_len": len(content) if isinstance(content, str) else None,
                    "parse_success": parse_success,
                    "parse_error": parse_error,
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
            )

    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_path": str(output_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
