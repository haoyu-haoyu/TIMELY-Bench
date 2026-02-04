"""
Opt-in LLM annotation runner (DeepSeek-compatible, no baseline impact).

Key properties:
- API key only from environment variables (never logged/stored).
- Prompt fixed by prompt template + recorded sha256.
- Supports concurrency, RPS throttling, retry/backoff, resume, and sharded output.
- Writes audit-ready metadata with input/output hashes.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib import request, error

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR, TEMPORAL_ALIGNMENT_DIR


OUT_DIR = ROOT_DIR / "results" / "llm_annotations"
PROMPT_TEMPLATE_PATH = ROOT_DIR / "code" / "data_processing" / "prompt_templates" / "llm_annotation_prompt.txt"
FAILED_PATH = OUT_DIR / "failed_requests.jsonl"

ALLOWED_LABELS = {"SUPPORTIVE", "CONTRADICTORY", "AMBIGUOUS", "UNRELATED"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text()


def build_prompt(template: str, rec: Dict[str, object]) -> str:
    return template.format(
        pattern_name=rec["pattern_name"],
        pattern_severity=rec.get("pattern_severity", "unknown"),
        pattern_hour=rec["pattern_hour"],
        note_type=rec["note_type"],
        note_hour=rec["note_hour"],
        note_text_relevant=rec.get("note_text_relevant", ""),
    )


def rec_key(rec: Dict[str, object]) -> Tuple[object, object, object, object]:
    return (
        rec.get("stay_id"),
        rec.get("pattern_hour"),
        rec.get("pattern_name"),
        rec.get("note_id"),
    )


def normalize_label(label: str) -> str:
    if not label:
        return "AMBIGUOUS"
    label = label.strip().upper()
    return label if label in ALLOWED_LABELS else "AMBIGUOUS"


def fallback_evidence_span(note_text: str, pattern_name: str, max_len: int = 240) -> str:
    text = (note_text or "").strip()
    if not text:
        return "NO_EVIDENCE_AVAILABLE"

    # simple sentence split
    sentences = re.split(r"(?<=[\.!?])\s+", text)
    pat = (pattern_name or "").lower()
    chosen = None
    for sent in sentences:
        if pat and pat in sent.lower():
            chosen = sent
            break
    if chosen is None:
        chosen = sentences[0]
    chosen = chosen.strip()
    if len(chosen) > max_len:
        chosen = chosen[: max_len - 3] + "..."
    return chosen or text[:max_len]


def strip_markdown_code_block(text: str) -> str:
    """Strip markdown code block wrapper (```json...```) if present."""
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def parse_response(content: str) -> Tuple[str, str, str]:
    """Return (label, evidence_span, parse_status)."""
    parse_status = "json"
    label = ""
    evidence_span = ""

    # First strip markdown code block if present
    content_clean = strip_markdown_code_block(content)

    try:
        payload = json.loads(content_clean)
        if isinstance(payload, dict):
            label = str(payload.get("label", ""))
            evidence_span = str(payload.get("evidence_span", ""))
        else:
            parse_status = "json_non_dict"
    except Exception:
        parse_status = "regex"
        m = re.search(r"(SUPPORTIVE|CONTRADICTORY|AMBIGUOUS|UNRELATED)", content_clean.upper())
        if m:
            label = m.group(1)
        m_span = re.search(r'evidence_span\s*[:=]\s*"([^"]+)"', content_clean, re.IGNORECASE)
        if m_span:
            evidence_span = m_span.group(1)

    label = normalize_label(label)
    return label, evidence_span.strip(), parse_status


class RateLimiter:
    def __init__(self, rps: float):
        self.rps = max(rps, 0.1)
        self.interval = 1.0 / self.rps
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.next_allowed:
                sleep_for = self.next_allowed - now
                time.sleep(sleep_for)
                now = time.time()
            self.next_allowed = max(self.next_allowed, now) + self.interval


class ShardedWriter:
    def __init__(self, prefix: Path, shard_size: int):
        self.prefix = prefix
        self.shard_size = max(1, shard_size)
        self.part_idx = 1
        self.count_in_part = 0
        self.total_written = 0
        self.paths: List[Path] = []
        self.handle = None
        self._open_new()

    def _part_path(self) -> Path:
        return self.prefix.with_name(f"{self.prefix.name}_part{self.part_idx:04d}.jsonl")

    def _open_new(self):
        if self.handle:
            self.handle.close()
        path = self._part_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", buffering=1024 * 1024)
        if path not in self.paths:
            self.paths.append(path)
        self.count_in_part = 0

    def write(self, rec: Dict[str, object]):
        if self.count_in_part >= self.shard_size:
            self.part_idx += 1
            self._open_new()
        self.handle.write(json.dumps(rec, ensure_ascii=True) + "\n")
        self.count_in_part += 1
        self.total_written += 1

    def close(self):
        if self.handle:
            self.handle.close()
            self.handle = None


def load_done_keys(glob_pattern: str) -> Set[Tuple[object, object, object, object]]:
    done: Set[Tuple[object, object, object, object]] = set()
    for path in OUT_DIR.glob(glob_pattern):
        try:
            with path.open() as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    key = (
                        rec.get("stay_id"),
                        rec.get("pattern_hour"),
                        rec.get("pattern_name"),
                        rec.get("note_id"),
                    )
                    done.add(key)
        except Exception:
            continue
    return done


def deepseek_request(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
) -> Dict[str, object]:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a careful clinical reasoning assistant. Return compact JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    data = json.dumps(body).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8")), resp.status


def annotate_one(
    rec: Dict[str, object],
    prompt_template: str,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    rate_limiter: RateLimiter,
    inflight_sem: threading.Semaphore,
    dry_run: bool,
) -> Tuple[Optional[Dict[str, object]], Optional[Dict[str, object]], float]:
    """Return (result_record, failure_record, latency_sec)."""
    prompt = build_prompt(prompt_template, rec)
    attempts = 0
    start = time.time()

    while attempts <= max_retries:
        attempts += 1
        try:
            with inflight_sem:
                rate_limiter.wait()
                if dry_run:
                    # deterministic dry-run response
                    content = json.dumps({"label": "AMBIGUOUS", "evidence_span": "DRY_RUN"})
                    raw = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {}}
                    status = 200
                else:
                    raw, status = deepseek_request(
                        base_url=base_url,
                        api_key=api_key,
                        model=model,
                        prompt=prompt,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        timeout=timeout,
                    )

            content = ""
            finish_reason = ""
            usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
            if isinstance(raw, dict) and raw.get("choices"):
                choice0 = raw["choices"][0]
                finish_reason = choice0.get("finish_reason", "")
                content = (
                    choice0.get("message", {}).get("content", "")
                    if isinstance(choice0, dict)
                    else ""
                )

            label, evidence_span, parse_status = parse_response(content)
            evidence_source = "model"
            evidence_note = ""
            if not evidence_span:
                evidence_span = fallback_evidence_span(
                    str(rec.get("note_text_relevant", "")),
                    str(rec.get("pattern_name", "")),
                )
                evidence_source = "fallback"
                evidence_note = "empty_model_span"

            latency = time.time() - start
            result = {
                "stay_id": rec.get("stay_id"),
                "pattern_name": rec.get("pattern_name"),
                "pattern_hour": rec.get("pattern_hour"),
                "note_id": rec.get("note_id"),
                "note_hour": rec.get("note_hour"),
                "note_type": rec.get("note_type"),
                "time_delta_hours": rec.get("time_delta_hours"),
                "label": label,
                "evidence_span": evidence_span,
                "evidence_span_source": evidence_source,
                "evidence_note": evidence_note,
                "method": "llm",
                "provider": "deepseek" if not dry_run else "dry_run",
                "model_name": model,
                "raw_response": {
                    "status": status,
                    "finish_reason": finish_reason,
                    "usage": usage,
                    "parse_status": parse_status,
                    "content_preview": (content[:200] + "...") if len(content) > 200 else content,
                },
            }
            return result, None, latency

        except error.HTTPError as e:
            status = getattr(e, "code", "")
            retriable = status == 429 or (500 <= int(status) < 600 if isinstance(status, int) else False)
            if retriable and attempts <= max_retries:
                sleep_for = backoff_base * (2 ** (attempts - 1))
                time.sleep(sleep_for)
                continue
            failure = {
                "stay_id": rec.get("stay_id"),
                "pattern_name": rec.get("pattern_name"),
                "pattern_hour": rec.get("pattern_hour"),
                "note_id": rec.get("note_id"),
                "error": "http_error",
                "status": status,
                "attempts": attempts,
            }
            return None, failure, time.time() - start
        except Exception as e:  # network or parse errors
            if attempts <= max_retries:
                sleep_for = backoff_base * (2 ** (attempts - 1))
                time.sleep(sleep_for)
                continue
            failure = {
                "stay_id": rec.get("stay_id"),
                "pattern_name": rec.get("pattern_name"),
                "pattern_hour": rec.get("pattern_hour"),
                "note_id": rec.get("note_id"),
                "error": "exception",
                "message": str(e)[:200],
                "attempts": attempts,
            }
            return None, failure, time.time() - start


def load_records(input_path: Path, limit: int = 0) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with input_path.open() as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            if limit and i > limit:
                break
            try:
                note_type = str(row.get("note_type", "")).strip().lower()
                note_hour = float(row.get("note_hour", ""))
            except Exception:
                continue
            if note_type == "discharge":
                continue
            if note_hour < 0 or note_hour >= 24:
                continue
            rows.append(
                {
                    "stay_id": int(row.get("stay_id")),
                    "pattern_hour": float(row.get("pattern_hour")),
                    "pattern_name": row.get("pattern_name"),
                    "pattern_severity": row.get("pattern_severity"),
                    "note_id": str(row.get("note_id")),
                    "note_hour": note_hour,
                    "note_type": row.get("note_type"),
                    "note_text_relevant": row.get("note_text_relevant", ""),
                    "time_delta_hours": float(row.get("time_delta_hours")),
                }
            )
    return rows


def resolve_alignment_info() -> Tuple[str, str]:
    """Return (alignment_path, alignment_sha256) without rehashing huge files when possible."""
    meta_path = OUT_DIR / "ANNOTATION_METADATA.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        alignment_path = meta.get("alignment_source")
        alignment_sha = meta.get("alignment_sha256")
        if alignment_path and alignment_sha:
            return alignment_path, alignment_sha
    alignment_path = str(TEMPORAL_ALIGNMENT_DIR / "temporal_textual_alignment.csv")
    alignment_sha = ""
    return alignment_path, alignment_sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["llm"], default="llm")
    parser.add_argument("--provider", choices=["deepseek"], default="deepseek")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--input-path", type=str, default=str(OUT_DIR / "llm_annotation_set.csv"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--rps", type=float, default=5.0)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--backoff-base", type=float, default=1.0)
    parser.add_argument("--resume-glob", type=str, default="annotations_deepseek_*.jsonl")
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0, help="Local testing limit.")
    parser.add_argument("--dry-run", action="store_true", help="No network calls; outputs deterministic placeholders.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    if not args.dry_run and not api_key:
        raise EnvironmentError("DEEPSEEK_API_KEY is not set in environment")

    input_path = Path(args.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input annotation set: {input_path}")

    prompt_template = load_prompt_template(PROMPT_TEMPLATE_PATH)
    prompt_template_sha = sha256_file(PROMPT_TEMPLATE_PATH)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = OUT_DIR / f"annotations_deepseek_{run_id}"
    writer = ShardedWriter(prefix=prefix, shard_size=args.shard_size)

    done_keys = load_done_keys(args.resume_glob)

    records = load_records(input_path, limit=args.limit)
    total_input = len(records)

    # Dedup input by key and drop already-completed keys
    dedup: Dict[Tuple[object, object, object, object], Dict[str, object]] = {}
    for rec in records:
        dedup[rec_key(rec)] = rec

    to_run: List[Dict[str, object]] = []
    skipped = 0
    for key, rec in dedup.items():
        if key in done_keys:
            skipped += 1
            continue
        to_run.append(rec)

    rate_limiter = RateLimiter(args.rps)
    inflight_sem = threading.Semaphore(max(1, args.max_inflight))

    latencies: List[float] = []
    failures = 0
    completed = 0

    FAILED_PATH.parent.mkdir(parents=True, exist_ok=True)
    failed_handle = FAILED_PATH.open("a", buffering=1024 * 1024)

    print(
        f"DeepSeek annotation start: input={total_input}, dedup={len(dedup)}, "
        f"skipped(resume)={skipped}, to_run={len(to_run)}"
    )

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = [
            ex.submit(
                annotate_one,
                rec,
                prompt_template,
                base_url,
                api_key,
                args.model,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.timeout,
                args.max_retries,
                args.backoff_base,
                rate_limiter,
                inflight_sem,
                args.dry_run,
            )
            for rec in to_run
        ]

        for i, fut in enumerate(as_completed(futures), start=1):
            result, failure, latency = fut.result()
            latencies.append(latency)
            if result is not None:
                writer.write(result)
                completed += 1
            elif failure is not None:
                failed_handle.write(json.dumps(failure, ensure_ascii=True) + "\n")
                failures += 1

            if i % 500 == 0:
                elapsed = time.time() - start_time
                rate = i / max(elapsed, 1e-6)
                print(f"Processed {i:,}/{len(to_run):,} ({rate:,.2f} req/s); completed={completed:,}, failed={failures:,}")

    writer.close()
    failed_handle.close()

    output_files = []
    for path in writer.paths:
        if path.exists():
            # count rows per shard
            n_rows = 0
            with path.open() as f:
                for line in f:
                    if line.strip():
                        n_rows += 1
            output_files.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "rows": n_rows,
                }
            )

    failed_count = 0
    if FAILED_PATH.exists():
        with FAILED_PATH.open() as f:
            for line in f:
                if line.strip():
                    failed_count += 1

    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    duration = time.time() - start_time

    alignment_path, alignment_sha = resolve_alignment_info()

    metadata = {
        "method": "llm",
        "provider": args.provider,
        "model_name": args.model,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "run_id": run_id,
        "prompt_template_path": str(PROMPT_TEMPLATE_PATH),
        "prompt_template_sha256": prompt_template_sha,
        "input": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "n_rows": total_input,
            "n_dedup": len(dedup),
        },
        "alignment": {
            "path": alignment_path,
            "sha256": alignment_sha,
        },
        "generation_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "concurrency": {
            "workers": args.workers,
            "max_inflight": args.max_inflight,
            "rps": args.rps,
        },
        "retry_policy": {
            "max_retries": args.max_retries,
            "backoff_base": args.backoff_base,
        },
        "resume": {
            "resume_glob": args.resume_glob,
            "skipped": skipped,
        },
        "outputs": output_files,
        "failures": {
            "path": str(FAILED_PATH),
            "sha256": sha256_file(FAILED_PATH) if FAILED_PATH.exists() else "",
            "count": failed_count,
        },
        "stats": {
            "to_run": len(to_run),
            "completed": completed,
            "failed": failures,
            "avg_latency_sec": avg_latency,
            "duration_sec": duration,
        },
        "notes": [
            "API key is read from DEEPSEEK_API_KEY and never stored in outputs.",
            "Prompt contains only pattern metadata and note excerpts; no outcome labels.",
            "This artefact is opt-in and not used by baseline training scripts.",
        ],
    }

    meta_path = OUT_DIR / "ANNOTATION_METADATA_deepseek.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True))
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
