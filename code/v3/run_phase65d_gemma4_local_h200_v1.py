#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from run_phase65d_tier1b_v3 import (
    append_jsonl,
    extract_json_blob,
    iter_manifest_rows,
    load_done_prompt_ids,
    normalize_confidence,
    provider_shard_path,
)


JSON_ONLY_SYSTEM_PROMPT = (
    "Return exactly one valid JSON object and nothing else. "
    "Do not output markdown fences, prompt text, or any prose outside the JSON object. "
    "Do not repeat or quote the input sections. "
    "Keep the reasoning concise, ideally within 4 sentences, and include only the most relevant evidence items."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5D Gemma4 local H200 runner")
    p.add_argument("--root", default=".")
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--provider", default="gemma4_26b")
    p.add_argument("--model-name", default="arc:lite")
    p.add_argument("--backend-model-name", default="google/gemma-4-26B-A4B-it")
    p.add_argument("--num-shards", type=int, default=2)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-rows", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    return p.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def split_prompt_sections(prompt_text: str) -> dict[str, str]:
    text = prompt_text.strip()
    parts = re.split(r"(?m)^\[(SYSTEM|PATIENT CONTEXT|QUESTION|FORMAT)\]\s*$", text)
    sections: dict[str, str] = {}
    for idx in range(1, len(parts), 2):
        key = parts[idx].strip()
        value = parts[idx + 1].strip() if idx + 1 < len(parts) else ""
        sections[key] = value
    return sections


def build_messages(prompt_text: str, *, compact: bool = False) -> tuple[list[dict], str]:
    sections = split_prompt_sections(prompt_text)
    if not sections:
        user_text = prompt_text.strip() + "\n\nReturn only one valid JSON object."
        if compact:
            user_text += " Keep reasoning to at most 2 short sentences and evidence to at most 3 items."
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": JSON_ONLY_SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": user_text}],
            },
        ], user_text

    system_parts = []
    if sections.get("SYSTEM"):
        system_parts.append(sections["SYSTEM"])
    system_parts.append(JSON_ONLY_SYSTEM_PROMPT)
    system_parts.append("Begin directly with the JSON object.")
    if compact:
        system_parts.append("Keep reasoning to at most 2 short sentences and evidence to at most 3 items.")
    system_text = "\n\n".join(part for part in system_parts if part)

    user_parts = []
    if sections.get("PATIENT CONTEXT"):
        user_parts.append("Patient context:\n" + sections["PATIENT CONTEXT"])
    if sections.get("QUESTION"):
        user_parts.append("Question:\n" + sections["QUESTION"])
    if sections.get("FORMAT"):
        user_parts.append("Required JSON schema:\n" + sections["FORMAT"])
    user_parts.append("Use only the provided evidence. Return only the JSON object.")
    if compact:
        user_parts.append("Keep the reasoning brief and avoid repeating measurements.")
    user_text = "\n\n".join(part for part in user_parts if part)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_text}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        },
    ], user_text


def strip_prompt_echo(raw_content: str, prompt_text: str, user_text: str) -> str:
    cleaned = raw_content.strip()
    for prefix in (prompt_text.strip(), user_text.strip()):
        if prefix and cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].lstrip()
    return cleaned


def repair_common_json_escapes(raw_content: str) -> str:
    # Some local models emit Markdown-style escapes such as `map\_merged`,
    # which are invalid in JSON strings but semantically harmless.
    return re.sub(r"\\([_*`#\-\[\]\(\)])", r"\1", raw_content)


def move_to_device(batch: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def load_model_and_processor(model_name: str, torch_dtype: torch.dtype) -> tuple[AutoProcessor, AutoModelForImageTextToText]:
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        raise SystemExit("HF_TOKEN or HUGGING_FACE_HUB_TOKEN is required")

    processor = AutoProcessor.from_pretrained(model_name, token=hf_token)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None):
            tokenizer.pad_token = tokenizer.eos_token
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = "left"
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        token=hf_token,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    if tokenizer is not None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return processor, model


def run_one_local_request(
    *,
    row: dict,
    provider: str,
    model_name: str,
    backend_model_name: str,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: torch.device,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
    initial_compact_mode: bool = False,
    attempt_offset: int = 0,
) -> dict:
    result = {
        "provider": provider,
        "model_name": model_name,
        "backend_model_name": backend_model_name,
        "runtime_backend": "hf_local_h200",
        "prompt_id": row["prompt_id"],
        "instance_id": row["instance_id"],
        "task_id": row["task_id"],
        "dimension_id": row["dimension_id"],
        "variant_id": row["variant_id"],
        "status": "ok",
        "parse_success": False,
    }
    compact_mode = initial_compact_mode
    for local_attempt in range(max_retries):
        attempt = attempt_offset + local_attempt
        started = time.time()
        try:
            messages, user_text = build_messages(row["prompt_text"], compact=compact_mode)
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = move_to_device(inputs, device)
            input_len = int(inputs["input_ids"].shape[-1])
            generate_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "repetition_penalty": 1.08 if compact_mode else 1.04,
            }
            if temperature > 0.0:
                generate_kwargs["temperature"] = temperature
            with torch.inference_mode():
                generated_ids = model.generate(**inputs, **generate_kwargs)
            completion_ids = generated_ids[:, input_len:]
            raw_content = processor.batch_decode(completion_ids, skip_special_tokens=True)[0].strip()
            raw_content = strip_prompt_echo(raw_content, row["prompt_text"], user_text)
            parsed, parse_error = extract_json_blob(raw_content)
            if parsed is None:
                repaired_content = repair_common_json_escapes(raw_content)
                if repaired_content != raw_content:
                    parsed, parse_error = extract_json_blob(repaired_content)
                    if parsed is not None:
                        raw_content = repaired_content
            completion_tokens = int(completion_ids.shape[-1])
            result.update(
                {
                    "status": "ok",
                    "latency_seconds": round(time.time() - started, 3),
                    "attempts": attempt + 1,
                    "usage_prompt_tokens": input_len,
                    "usage_completion_tokens": completion_tokens,
                    "usage_total_tokens": input_len + completion_tokens,
                    "raw_content": raw_content,
                    "parse_success": parsed is not None,
                    "parse_error": parse_error,
                    "parsed_response": parsed,
                    "confidence_value": normalize_confidence(parsed),
                    "has_reasoning": False,
                }
            )
            if (
                parsed is None
                and completion_tokens >= max_new_tokens
                and local_attempt + 1 < max_retries
            ):
                compact_mode = True
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(min(20.0, 3.0 * (local_attempt + 1)))
                continue
            return result
        except Exception as exc:  # noqa: BLE001
            result.update(
                {
                    "status": "exception",
                    "latency_seconds": round(time.time() - started, 3),
                    "attempts": attempt + 1,
                    "error": str(exc)[:4000],
                }
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if local_attempt + 1 >= max_retries:
                return result
            time.sleep(min(20.0, 3.0 * (local_attempt + 1)))
    return result


def estimate_prompt_length(row: dict) -> int:
    input_tokens_est = row.get("input_tokens_est")
    if isinstance(input_tokens_est, int) and input_tokens_est > 0:
        return input_tokens_est
    return len(row.get("prompt_text", ""))


def iter_row_batches(rows: list[dict], batch_size: int) -> list[list[dict]]:
    if batch_size <= 1:
        return [[row] for row in rows]
    return [rows[idx : idx + batch_size] for idx in range(0, len(rows), batch_size)]


def run_local_batch_requests(
    *,
    batch_rows: list[dict],
    provider: str,
    model_name: str,
    backend_model_name: str,
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: torch.device,
    temperature: float,
    max_new_tokens: int,
    max_retries: int,
) -> list[dict]:
    attempted_batch_size = len(batch_rows)
    if len(batch_rows) <= 1:
        return [
            run_one_local_request(
                row=batch_rows[0],
                provider=provider,
                model_name=model_name,
                backend_model_name=backend_model_name,
                processor=processor,
                model=model,
                device=device,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
            )
        ]

    try:
        conversations = []
        user_texts = []
        for row in batch_rows:
            messages, user_text = build_messages(row["prompt_text"], compact=False)
            conversations.append(messages)
            user_texts.append(user_text)

        inputs = processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = move_to_device(inputs, device)
        attention_mask = inputs.get("attention_mask")
        padded_input_len = int(inputs["input_ids"].shape[-1])
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "repetition_penalty": 1.04,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature

        started = time.time()
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, **generate_kwargs)
        batch_latency = round(time.time() - started, 3)

        completion_ids = generated_ids[:, padded_input_len:]
        raw_contents = processor.batch_decode(completion_ids, skip_special_tokens=True)
        tokenizer = getattr(processor, "tokenizer", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None

        results: list[dict] = []
        for idx, row in enumerate(batch_rows):
            input_len = padded_input_len
            if attention_mask is not None:
                input_len = int(attention_mask[idx].sum().item())
            raw_content = strip_prompt_echo(raw_contents[idx].strip(), row["prompt_text"], user_texts[idx])
            parsed, parse_error = extract_json_blob(raw_content)
            if parsed is None:
                repaired_content = repair_common_json_escapes(raw_content)
                if repaired_content != raw_content:
                    parsed, parse_error = extract_json_blob(repaired_content)
                    if parsed is not None:
                        raw_content = repaired_content

            if pad_token_id is None:
                completion_tokens = int(completion_ids[idx].shape[-1])
            else:
                completion_tokens = int((completion_ids[idx] != pad_token_id).sum().item())

            result = {
                "provider": provider,
                "model_name": model_name,
                "backend_model_name": backend_model_name,
                "runtime_backend": "hf_local_h200",
                "prompt_id": row["prompt_id"],
                "instance_id": row["instance_id"],
                "task_id": row["task_id"],
                "dimension_id": row["dimension_id"],
                "variant_id": row["variant_id"],
                "status": "ok",
                "parse_success": parsed is not None,
                "latency_seconds": batch_latency,
                "attempts": 1,
                "usage_prompt_tokens": input_len,
                "usage_completion_tokens": completion_tokens,
                "usage_total_tokens": input_len + completion_tokens,
                "raw_content": raw_content,
                "parse_error": parse_error,
                "parsed_response": parsed,
                "confidence_value": normalize_confidence(parsed),
                "has_reasoning": False,
                "batch_size_used": len(batch_rows),
            }
            if parsed is None and completion_tokens >= max_new_tokens and max_retries > 1:
                result = run_one_local_request(
                    row=row,
                    provider=provider,
                    model_name=model_name,
                    backend_model_name=backend_model_name,
                    processor=processor,
                    model=model,
                    device=device,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    max_retries=max_retries - 1,
                    initial_compact_mode=True,
                    attempt_offset=1,
                )
                result["batch_size_used"] = len(batch_rows)
                result["batch_retry_origin"] = "batch_truncation_fallback"
            results.append(result)
        return results
    except Exception as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        fallback_reason = str(exc)[:400]
        fallback_results = [
            run_one_local_request(
                row=row,
                provider=provider,
                model_name=model_name,
                backend_model_name=backend_model_name,
                processor=processor,
                model=model,
                device=device,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                max_retries=max_retries,
            )
            for row in batch_rows
        ]
        for result in fallback_results:
            result["batch_attempt_size"] = attempted_batch_size
            result["batch_size_used"] = 1
            result["batch_fallback_reason"] = fallback_reason
        return fallback_results


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    manifest_path = (root / args.manifest_path).resolve() if not Path(args.manifest_path).is_absolute() else Path(args.manifest_path)
    output_dir = (root / args.output_dir).resolve() if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for local H200 run")

    device = torch.device(args.device)
    torch_dtype = resolve_dtype(args.dtype)
    torch.backends.cuda.matmul.allow_tf32 = True
    processor, model = load_model_and_processor(args.backend_model_name, torch_dtype)
    model = model.to(device)
    model.eval()

    shard_path = provider_shard_path(output_dir, args.provider, args.shard_index, args.num_shards)
    done_prompt_ids = load_done_prompt_ids(output_dir, args.provider, args.model_name)
    pending_rows = [
        row
        for row in iter_manifest_rows(manifest_path, args.shard_index, args.num_shards)
        if row["prompt_id"] not in done_prompt_ids
    ]
    if args.max_rows > 0:
        pending_rows = pending_rows[: args.max_rows]

    if args.batch_size > 1:
        pending_rows = sorted(pending_rows, key=estimate_prompt_length)

    for row_batch in iter_row_batches(pending_rows, max(1, args.batch_size)):
        results = run_local_batch_requests(
            batch_rows=row_batch,
            provider=args.provider,
            model_name=args.model_name,
            backend_model_name=args.backend_model_name,
            processor=processor,
            model=model,
            device=device,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            max_retries=args.max_retries,
        )
        for result in results:
            append_jsonl(shard_path, result)
            print(
                json.dumps(
                    {
                        "prompt_id": result["prompt_id"],
                        "status": result["status"],
                        "parse_success": result.get("parse_success"),
                        "latency_seconds": result.get("latency_seconds"),
                        "usage_total_tokens": result.get("usage_total_tokens"),
                        "batch_size_used": result.get("batch_size_used", 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
