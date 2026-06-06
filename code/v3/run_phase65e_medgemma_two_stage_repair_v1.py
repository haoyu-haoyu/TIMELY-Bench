#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Dict, Iterator, List

from run_phase65d_tier1b_v3 import extract_json_blob, jsonl_iter, normalize_confidence
from run_phase65e_tier2_v1 import invoke_provider


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-stage compact MedGemma repair runner")
    p.add_argument("--manifest-path", required=True)
    p.add_argument("--output-path", required=True)
    p.add_argument("--provider", default="medgemma15_4b")
    p.add_argument("--model-name", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--stage-a-max-tokens", type=int, default=900)
    p.add_argument("--stage-b-max-tokens", type=int, default=320)
    p.add_argument("--timeout-seconds", type=int, default=1200)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--max-retries", type=int, default=4)
    return p.parse_args()


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_latest_rows(path: Path) -> Dict[str, dict]:
    latest: Dict[str, dict] = {}
    if not path.exists():
        return latest
    for row in jsonl_iter(path):
        latest[row["prompt_id"]] = row
    return latest


def strip_format_block(prompt_text: str) -> str:
    marker = "\n[FORMAT]\n"
    idx = prompt_text.rfind(marker)
    if idx >= 0:
        return prompt_text[:idx].rstrip()
    return prompt_text.rstrip()


def extract_question(prompt_text: str) -> str:
    marker = "\n[QUESTION]\n"
    idx = prompt_text.rfind(marker)
    if idx < 0:
        return "Which clinical explanation best fits the observed evidence?"
    block = prompt_text[idx + len(marker) :]
    if "\n\n[FORMAT]" in block:
        block = block.split("\n\n[FORMAT]", 1)[0]
    return block.strip()


def build_stage_a_prompt(prompt_text: str, task_id: str | None = None) -> str:
    base = strip_format_block(prompt_text)
    answer_spec = '  "answer": "one concise answer, <= 120 characters",\n'
    evidence_rule = "- evidence: 1 to 3 items only\n"
    extra_rules = ""
    if task_id and task_id.startswith("S-R"):
        answer_spec = '  "answer": "main strategy summary, <= 80 characters",\n'
        evidence_rule = "- evidence: exactly 1 item only\n"
        extra_rules = (
            "- if asked for retrospective treatment strategy or mechanism, summarize the main strategy only\n"
            "- never output a medication list or repeat drug names\n"
            "- if many drugs appear, compress them into one therapy category\n"
            "- prefer an answer like 'Hyperosmolar therapy and BP control.' not a full list\n"
        )
    return (
        f"{base}\n\n"
        "[FORMAT]\n"
        "Return exactly one valid JSON object and nothing else.\n"
        "Do not restate the full chart or copy long note text.\n"
        "Compress aggressively and use the minimum text needed.\n\n"
        "{\n"
        f"{answer_spec}"
        '  "confidence": "high/medium/low",\n'
        '  "evidence": [\n'
        '    {"timestamp": "Hour X", "measurement": "...", "value": "..."}\n'
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        "- answer: exactly 1 short sentence\n"
        f"{evidence_rule}"
        "- keep only the most decisive evidence\n"
        "- each evidence item must be atomic and short\n"
        "- measurement: <= 24 characters\n"
        "- value: <= 40 characters\n"
        "- never copy a full note sentence or imaging sentence\n"
        "- compress long findings into a short phrase or number\n"
        "- deduplicate repeated evidence\n"
        f"{extra_rules}"
        "- no reasoning field\n"
        "- stop immediately after the closing brace\n"
    )


def build_stage_b_prompt(question: str, answer: str, evidence: List[dict]) -> str:
    evidence_lines = []
    for item in evidence[:3]:
        evidence_lines.append(
            f'- {item["timestamp"]}: {item["measurement"]} = {item["value"]}'
        )
    evidence_text = "\n".join(evidence_lines) if evidence_lines else "- no evidence"
    return (
        "[SYSTEM]\n"
        "You are writing a compact clinical rationale.\n"
        "Use only the answer and evidence below.\n"
        "Do not restate the full chart.\n"
        "Do not repeat phrases.\n"
        "Return exactly one valid JSON object and nothing else.\n\n"
        f"[QUESTION]\n{question}\n\n"
        f"[ANSWER]\n{answer}\n\n"
        f"[EVIDENCE]\n{evidence_text}\n\n"
        "[FORMAT]\n"
        '{\n  "reasoning": "exactly 1 short sentence, <= 140 characters total"\n}\n\n'
        "Rules:\n"
        "- reasoning must be compact and non-repetitive\n"
        "- reasoning must be exactly one sentence\n"
        "- do not repeat the same phrase twice\n"
        "- no bullet list inside the reasoning string\n"
        "- stop immediately after the closing brace\n"
    )


def normalize_evidence(evidence: object) -> List[dict]:
    if not isinstance(evidence, list):
        return []
    normalized: List[dict] = []
    for item in evidence[:4]:
        if not isinstance(item, dict):
            continue
        timestamp = str(item.get("timestamp", "")).strip()
        measurement = str(item.get("measurement", "")).strip()
        value = str(item.get("value", "")).strip()
        if timestamp and measurement and value:
            normalized.append(
                {
                    "timestamp": timestamp,
                    "measurement": measurement,
                    "value": value,
                }
            )
    return normalized


PARTIAL_STAGE_A_FIELD_RE = re.compile(
    r'"(?P<key>answer|confidence)"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"',
    re.DOTALL,
)
PARTIAL_STAGE_A_EVIDENCE_RE = re.compile(
    r'\{\s*"timestamp"\s*:\s*"(?P<timestamp>(?:\\.|[^"\\])*)"\s*,\s*"measurement"\s*:\s*"(?P<measurement>(?:\\.|[^"\\])*)"\s*,\s*"value"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)"\s*\}',
    re.DOTALL,
)
PARTIAL_STAGE_A_EVIDENCE_PREFIX_RE = re.compile(
    r'\{\s*"timestamp"\s*:\s*"(?P<timestamp>(?:\\.|[^"\\])*)"\s*,\s*"measurement"\s*:\s*"(?P<measurement>(?:\\.|[^"\\])*)"\s*,\s*"value"\s*:\s*"(?P<value>.*)$',
    re.DOTALL,
)
PARTIAL_STAGE_B_REASONING_RE = re.compile(
    r'"reasoning"\s*:\s*"(?P<value>(?:\\.|[^"\\])*)',
    re.DOTALL,
)


def _json_unescape(text: str) -> str:
    try:
        return json.loads(f'"{text}"')
    except Exception:  # noqa: BLE001
        return text


def salvage_stage_a_response(raw_content: object) -> dict | None:
    if not isinstance(raw_content, str) or not raw_content:
        return None

    fields = {}
    for match in PARTIAL_STAGE_A_FIELD_RE.finditer(raw_content):
        fields[match.group("key")] = _json_unescape(match.group("value")).strip()

    evidence = []
    for match in PARTIAL_STAGE_A_EVIDENCE_RE.finditer(raw_content):
        timestamp = _json_unescape(match.group("timestamp")).strip()
        measurement = _json_unescape(match.group("measurement")).strip()
        value = _json_unescape(match.group("value")).strip()
        if timestamp and measurement and value:
            evidence.append(
                {
                    "timestamp": timestamp,
                    "measurement": measurement[:24],
                    "value": value[:40],
                }
            )
        if len(evidence) >= 3:
            break

    if not evidence:
        partial_match = PARTIAL_STAGE_A_EVIDENCE_PREFIX_RE.search(raw_content)
        if partial_match:
            timestamp = _json_unescape(partial_match.group("timestamp")).strip()
            measurement = _json_unescape(partial_match.group("measurement")).strip()
            value = _json_unescape(partial_match.group("value")).strip()
            # Keep only a short leading phrase from a truncated evidence value.
            value = re.sub(r"\s+", " ", value).strip()
            if "." in value:
                value = value.split(".", 1)[0].strip()
            value = value[:40]
            if timestamp and measurement and value:
                evidence.append(
                    {
                        "timestamp": timestamp,
                        "measurement": measurement[:24],
                        "value": value,
                    }
                )

    answer = str(fields.get("answer", "")).strip()[:120]
    confidence = str(fields.get("confidence", "")).strip().lower()
    if confidence not in {"high", "medium", "low"}:
        return None
    if not answer or not evidence:
        return None
    return {
        "answer": answer,
        "confidence": confidence,
        "evidence": evidence,
    }


def salvage_stage_b_response(raw_content: object) -> dict | None:
    if not isinstance(raw_content, str) or not raw_content:
        return None
    match = PARTIAL_STAGE_B_REASONING_RE.search(raw_content)
    if not match:
        return None
    reasoning = _json_unescape(match.group("value")).strip()
    if not reasoning:
        return None
    # Keep only the first sentence to remove repetition cascades from truncation.
    sentence_match = re.search(r"(.+?[.!?])(?:\s|$)", reasoning)
    if sentence_match:
        reasoning = sentence_match.group(1).strip()
    reasoning = re.sub(r"\s+", " ", reasoning).strip()[:140]
    if len(reasoning) < 10:
        return None
    return {"reasoning": reasoning}


def call_json(
    *,
    provider: str,
    model_name: str,
    prompt_text: str,
    max_tokens: int,
    temperature: float,
    timeout_seconds: int,
    max_retries: int,
) -> dict:
    last_result: dict = {
        "status": "exception",
        "parse_success": False,
    }
    for attempt in range(max_retries):
        started = time.time()
        try:
            response, choice, usage = invoke_provider(
                provider=provider,
                model_name=model_name,
                prompt_text=prompt_text,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_seconds=timeout_seconds,
            )
            parsed, parse_error = extract_json_blob(choice)
            return {
                "status": "ok",
                "parse_success": parsed is not None,
                "parsed_response": parsed,
                "parse_error": parse_error,
                "raw_content": choice,
                "latency_seconds": round(time.time() - started, 3),
                "attempts": attempt + 1,
                "requested_max_tokens": max_tokens,
                "usage_prompt_tokens": usage.get("usage_prompt_tokens"),
                "usage_completion_tokens": usage.get("usage_completion_tokens"),
                "usage_total_tokens": usage.get("usage_total_tokens"),
                "finish_reason": ((response.get("choices") or [{}])[0] or {}).get("finish_reason"),
            }
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
            last_result = {
                "status": status,
                "parse_success": False,
                "error": error_text[:4000],
                "http_status": http_status,
                "latency_seconds": round(time.time() - started, 3),
                "attempts": attempt + 1,
                "requested_max_tokens": max_tokens,
            }
            if attempt + 1 >= max_retries:
                return last_result
            time.sleep(min(12.0, 2.0 * (attempt + 1)))
    return last_result


def run_two_stage_row(
    *,
    row: dict,
    provider: str,
    model_name: str,
    temperature: float,
    stage_a_max_tokens: int,
    stage_b_max_tokens: int,
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
        "status": "failed",
        "parse_success": False,
        "repair_method": "two_stage_compact_v2",
    }

    stage_a_prompt = build_stage_a_prompt(row["prompt_text"], task_id=row["task_id"])
    stage_a = call_json(
        provider=provider,
        model_name=model_name,
        prompt_text=stage_a_prompt,
        max_tokens=stage_a_max_tokens,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
    result["stage_a"] = stage_a
    if stage_a.get("status") == "ok" and not stage_a.get("parse_success"):
        salvaged = salvage_stage_a_response(stage_a.get("raw_content"))
        if salvaged is not None:
            stage_a["parse_success"] = True
            stage_a["parsed_response"] = salvaged
            stage_a["parse_error"] = "salvaged_partial_json"
    if stage_a.get("status") != "ok" or not stage_a.get("parse_success"):
        result["status"] = "stage_a_failed"
        return result

    parsed_a = stage_a.get("parsed_response") or {}
    answer = str(parsed_a.get("answer", "")).strip()
    evidence = normalize_evidence(parsed_a.get("evidence"))
    confidence = normalize_confidence(parsed_a)
    if not answer or not evidence or not confidence:
        result["status"] = "stage_a_invalid"
        return result

    stage_b_prompt = build_stage_b_prompt(
        question=extract_question(row["prompt_text"]),
        answer=answer,
        evidence=evidence,
    )
    stage_b = call_json(
        provider=provider,
        model_name=model_name,
        prompt_text=stage_b_prompt,
        max_tokens=stage_b_max_tokens,
        temperature=temperature,
        timeout_seconds=max(300, timeout_seconds // 3),
        max_retries=max_retries,
    )
    result["stage_b"] = stage_b
    if stage_b.get("status") == "ok" and not stage_b.get("parse_success"):
        salvaged = salvage_stage_b_response(stage_b.get("raw_content"))
        if salvaged is not None:
            stage_b["parse_success"] = True
            stage_b["parsed_response"] = salvaged
            stage_b["parse_error"] = "salvaged_partial_json"
    if stage_b.get("status") != "ok" or not stage_b.get("parse_success"):
        result["status"] = "stage_b_failed"
        return result

    parsed_b = stage_b.get("parsed_response") or {}
    reasoning = str(parsed_b.get("reasoning", "")).strip()
    if not reasoning:
        result["status"] = "stage_b_invalid"
        return result

    combined = {
        "reasoning": reasoning,
        "answer": answer,
        "evidence": evidence,
        "confidence": confidence,
    }
    result.update(
        {
            "status": "ok",
            "parse_success": True,
            "parsed_response": combined,
            "confidence_value": confidence,
            "usage_prompt_tokens": (stage_a.get("usage_prompt_tokens") or 0) + (stage_b.get("usage_prompt_tokens") or 0),
            "usage_completion_tokens": (stage_a.get("usage_completion_tokens") or 0)
            + (stage_b.get("usage_completion_tokens") or 0),
            "usage_total_tokens": (stage_a.get("usage_total_tokens") or 0) + (stage_b.get("usage_total_tokens") or 0),
            "latency_seconds": round((stage_a.get("latency_seconds") or 0.0) + (stage_b.get("latency_seconds") or 0.0), 3),
        }
    )
    return result


def iter_pending_rows(manifest_path: Path, output_path: Path) -> Iterator[dict]:
    done = {
        prompt_id
        for prompt_id, row in load_latest_rows(output_path).items()
        if row.get("parse_success") is True
    }
    for row in jsonl_iter(manifest_path):
        if row["prompt_id"] not in done:
            yield row


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pending_rows = list(iter_pending_rows(manifest_path, output_path))
    if not pending_rows:
        print(json.dumps({"status": "nothing_to_do", "manifest_path": str(manifest_path), "output_path": str(output_path)}))
        return

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {}
        row_iter = iter(pending_rows)
        while True:
            while len(futures) < args.max_workers:
                try:
                    row = next(row_iter)
                except StopIteration:
                    break
                fut = pool.submit(
                    run_two_stage_row,
                    row=row,
                    provider=args.provider,
                    model_name=args.model_name,
                    temperature=args.temperature,
                    stage_a_max_tokens=args.stage_a_max_tokens,
                    stage_b_max_tokens=args.stage_b_max_tokens,
                    timeout_seconds=args.timeout_seconds,
                    max_retries=args.max_retries,
                )
                futures[fut] = row["prompt_id"]

            if not futures:
                break

            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut, None)
                append_jsonl(output_path, fut.result())


if __name__ == "__main__":
    main()
