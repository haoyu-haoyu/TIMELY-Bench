"""
Run model inference on CRES tasks and export predictions.jsonl.

This is the canonical entrypoint for model-based CRES evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR


OUT_DIR = ROOT_DIR / "results" / "cres"
TASK_TO_FILE = {
    "trend_threshold": OUT_DIR / "trend_threshold.jsonl",
    "temporal_grounding": OUT_DIR / "temporal_grounding.jsonl",
    "diagnostic_consistency": OUT_DIR / "diagnostic_consistency.jsonl",
    "contrastive_inference": OUT_DIR / "contrastive_inference.jsonl",
}

TASK_LABELS = {
    "trend_threshold": ["yes", "no"],
    "temporal_grounding": ["high", "medium", "low"],
    "diagnostic_consistency": ["consistent", "inconsistent", "ambiguous"],
    "contrastive_inference": ["A_more_plausible", "B_more_plausible", "tie"],
}

PROMPT_TEMPLATES = {
    "trend_threshold": (
        "You are evaluating clinical threshold reasoning.\n"
        "Question: {question}\n"
        "Feature value: {value}\n"
        "Threshold: {threshold}\n"
        "Return exactly one label: yes or no."
    ),
    "temporal_grounding": (
        "You are evaluating temporal grounding quality between a detected pattern and a clinical note.\n"
        "Pattern: {pattern_name} at hour {pattern_hour}\n"
        "Note type: {note_type} at hour {note_hour}\n"
        "Time delta (note - pattern, hours): {time_delta_hours}\n"
        "Note snippet: {note_text_relevant}\n"
        "Return exactly one label: high, medium, or low."
    ),
    "diagnostic_consistency": (
        "You are comparing two diagnostic hypotheses.\n"
        "Primary hypothesis: {primary_hypothesis}\n"
        "Alternative hypothesis: {alternative_hypothesis}\n"
        "Supporting patterns: {supporting_patterns}\n"
        "Conflicting patterns: {conflicting_patterns}\n"
        "Multimorbidity context: {multimorbidity_context}\n"
        "Return exactly one label: consistent, inconsistent, or ambiguous."
    ),
    "contrastive_inference": (
        "You are doing contrastive evidence attribution.\n"
        "Pattern: {pattern_name}\n"
        "Option A: {option_a}\n"
        "Option B: {option_b}\n"
        "Return exactly one label: A_more_plausible, B_more_plausible, or tie."
    ),
}


def load_jsonl(path: Path):
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_label(task: str, text: str) -> str:
    t = str(text or "").strip()
    allowed = TASK_LABELS[task]
    low = t.lower()

    def _has_token(s: str, token: str) -> bool:
        return re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", s, flags=re.IGNORECASE) is not None

    # Exact match first.
    for lb in allowed:
        if t == lb:
            return lb

    if task == "trend_threshold":
        if _has_token(low, "yes"):
            return "yes"
        if _has_token(low, "no"):
            return "no"

    if task == "temporal_grounding":
        if _has_token(low, "high"):
            return "high"
        if _has_token(low, "medium"):
            return "medium"
        if _has_token(low, "low"):
            return "low"

    if task == "diagnostic_consistency":
        if _has_token(low, "inconsistent"):
            return "inconsistent"
        if _has_token(low, "consistent"):
            return "consistent"
        if _has_token(low, "ambiguous"):
            return "ambiguous"

    if task == "contrastive_inference":
        t_clean = t.strip()
        if re.fullmatch(r"[Aa]", t_clean):
            return "A_more_plausible"
        if re.fullmatch(r"[Bb]", t_clean):
            return "B_more_plausible"
        if re.fullmatch(r"(?i)tie", t_clean):
            return "tie"
        if "a_more_plausible" in t:
            return "A_more_plausible"
        if "b_more_plausible" in t:
            return "B_more_plausible"
        if _has_token(low, "tie"):
            return "tie"
        if _has_token(low, "option a"):
            return "A_more_plausible"
        if _has_token(low, "option b"):
            return "B_more_plausible"

    # Fallback to first class for deterministic output.
    return allowed[0]


def _build_prompt(task: str, row: dict) -> str:
    template = PROMPT_TEMPLATES[task]
    if task == "trend_threshold":
        return template.format(
            question=row.get("question", ""),
            value=row.get("value", ""),
            threshold=row.get("threshold", ""),
        )
    if task == "temporal_grounding":
        return template.format(
            pattern_name=row.get("pattern_name", ""),
            pattern_hour=row.get("pattern_hour", ""),
            note_type=row.get("note_type", ""),
            note_hour=row.get("note_hour", ""),
            time_delta_hours=row.get("time_delta_hours", ""),
            note_text_relevant=row.get("note_text_relevant", "")[:800],
        )
    if task == "diagnostic_consistency":
        return template.format(
            primary_hypothesis=row.get("primary_hypothesis", ""),
            alternative_hypothesis=row.get("alternative_hypothesis", ""),
            supporting_patterns=row.get("supporting_patterns", []),
            conflicting_patterns=row.get("conflicting_patterns", []),
            multimorbidity_context=row.get("multimorbidity_context", []),
        )
    if task == "contrastive_inference":
        return template.format(
            pattern_name=row.get("pattern_name", ""),
            option_a=row.get("option_a", {}),
            option_b=row.get("option_b", {}),
        )
    raise ValueError(task)


_OPENAI_KEY_CURSOR = 0
_GEMINI_KEY_CURSOR = 0


def _call_openai(model_name: str, prompt: str, temperature: float, max_tokens: int, request_timeout: float) -> str:
    global _OPENAI_KEY_CURSOR

    def _norm_base(url: str) -> str:
        url = (url or "").strip().rstrip("/")
        if not url:
            return "https://api.openai.com/v1"
        if url.endswith("/v1"):
            return url
        return url + "/v1"

    def _collect_keys() -> list[str]:
        keys = []
        pool_raw = os.environ.get("OPENAI_API_KEYS", "")
        if pool_raw:
            for chunk in pool_raw.replace("\n", ",").split(","):
                k = chunk.strip()
                if k and k not in keys:
                    keys.append(k)
        single = os.environ.get("OPENAI_API_KEY", "").strip()
        if single and single not in keys:
            keys.insert(0, single)
        return keys

    def _post_json(url: str, payload: dict, api_key: str):
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "curl/8.5.0",
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.getcode(), json.loads(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(body)
            except Exception:
                parsed = {"error": {"message": body}}
            return e.code, parsed

    def _extract_text_from_responses(resp_obj: dict) -> str:
        text = str(resp_obj.get("output_text", "") or "").strip()
        if text:
            return text
        output = resp_obj.get("output") or []
        if output and isinstance(output[0], dict):
            content = output[0].get("content") or []
            if content and isinstance(content[0], dict):
                return str(content[0].get("text", "") or "").strip()
        return ""

    def _extract_text_from_chat(resp_obj: dict) -> str:
        choices = resp_obj.get("choices") or []
        if choices and isinstance(choices[0], dict):
            msg = choices[0].get("message") or {}
            return str(msg.get("content", "") or "").strip()
        return ""

    keys = _collect_keys()
    if not keys:
        raise RuntimeError("OPENAI_API_KEY or OPENAI_API_KEYS is not set")

    base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
    norm_base = _norm_base(base_url)
    response_endpoint = f"{norm_base}/responses"
    chat_endpoint = f"{norm_base}/chat/completions"

    response_payloads = [
        {"model": model_name, "input": prompt, "max_output_tokens": max_tokens},
        {"model": model_name, "input": prompt},
        {"model": model_name, "input": prompt, "temperature": temperature, "max_output_tokens": max_tokens},
        {"model": model_name, "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]},
    ]
    chat_payloads = [
        {"model": model_name, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
        {"model": model_name, "messages": [{"role": "user", "content": prompt}]},
        {"model": model_name, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens},
    ]

    start = _OPENAI_KEY_CURSOR % len(keys)
    ordered = keys[start:] + keys[:start]
    error_msgs = []

    for rel_idx, api_key in enumerate(ordered):
        key_idx = (start + rel_idx) % len(keys)

        for payload in response_payloads:
            code, obj = _post_json(response_endpoint, payload, api_key)
            if 200 <= code < 300:
                text = _extract_text_from_responses(obj)
                if text:
                    _OPENAI_KEY_CURSOR = key_idx
                    return text
            else:
                msg = (obj.get("error") or {}).get("message", str(obj))
                error_msgs.append(f"responses_http_{code}: {msg}")

        for payload in chat_payloads:
            code, obj = _post_json(chat_endpoint, payload, api_key)
            if 200 <= code < 300:
                text = _extract_text_from_chat(obj)
                if text:
                    _OPENAI_KEY_CURSOR = key_idx
                    return text
            else:
                msg = (obj.get("error") or {}).get("message", str(obj))
                error_msgs.append(f"chat_http_{code}: {msg}")

    if error_msgs:
        raise RuntimeError(error_msgs[-1])
    raise RuntimeError("OpenAI call failed with unknown error")


def _call_gemini_native(model_name: str, prompt: str, temperature: float, max_tokens: int, request_timeout: float) -> str:
    global _GEMINI_KEY_CURSOR

    def _norm_base(url: str) -> str:
        url = (url or "").strip().rstrip("/")
        if not url:
            return "https://code.newcli.com/gemini"
        return url

    def _collect_keys() -> list[str]:
        keys = []
        key_envs = [
            os.environ.get("GEMINI_API_KEYS", ""),
            os.environ.get("OPENAI_API_KEYS", ""),
        ]
        for pool_raw in key_envs:
            if not pool_raw:
                continue
            for chunk in pool_raw.replace("\n", ",").split(","):
                k = chunk.strip()
                if k and k not in keys:
                    keys.append(k)
        single_envs = [
            os.environ.get("GEMINI_API_KEY", "").strip(),
            os.environ.get("OPENAI_API_KEY", "").strip(),
        ]
        for single in single_envs:
            if single and single not in keys:
                keys.insert(0, single)
        return keys

    def _parse_json(raw: str):
        try:
            return json.loads(raw)
        except Exception:
            return {"error": {"message": raw}}

    def _post_json_with_curl(url: str, payload: dict, api_key: str):
        body_payload = json.dumps(payload, ensure_ascii=True)
        cmd = [
            "curl",
            "-sS",
            "-m",
            str(int(max(1.0, request_timeout))),
            "-w",
            "\n__HTTP__%{http_code}",
            url,
            "-H",
            "Content-Type: application/json",
            "-H",
            f"Authorization: Bearer {api_key}",
            "-H",
            f"User-Agent: {os.environ.get('GEMINI_USER_AGENT', 'GeminiBridge/1.0')}",
            "-H",
            "Accept: application/json",
            "-d",
            body_payload,
        ]
        resolve_ip = os.environ.get("GEMINI_RESOLVE_IP", "").strip()
        if resolve_ip and "://" in url:
            host = url.split("://", 1)[1].split("/", 1)[0]
            if host:
                cmd[1:1] = ["--resolve", f"{host}:443:{resolve_ip}"]
        try:
            out = subprocess.check_output(cmd, text=True)
        except subprocess.CalledProcessError as e:
            msg = str(e.output or e)
            return 599, {"error": {"message": msg}}
        marker = "\n__HTTP__"
        if marker not in out:
            return 599, {"error": {"message": out}}
        raw_body, raw_code = out.rsplit(marker, 1)
        code = int(raw_code.strip() or "599")
        return code, _parse_json(raw_body)

    def _post_json(url: str, payload: dict, api_key: str):
        if os.environ.get("GEMINI_USE_CURL", "0").strip() == "1":
            return _post_json_with_curl(url, payload, api_key)
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        req = urllib.request.Request(
            url=url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": os.environ.get("GEMINI_USER_AGENT", "GeminiBridge/1.0"),
                "Accept": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=request_timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return resp.getcode(), _parse_json(body)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return e.code, _parse_json(body)
        except urllib.error.URLError as e:
            # CREATE's resolver may fail for some proxy domains; curl + --resolve is the fallback.
            if os.environ.get("GEMINI_RESOLVE_IP", "").strip():
                return _post_json_with_curl(url, payload, api_key)
            raise RuntimeError(str(e))

    def _extract_text(resp_obj: dict) -> str:
        candidates = resp_obj.get("candidates") or []
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            content = cand.get("content") or {}
            parts = content.get("parts") or []
            # Ignore "thought" parts when present; keep the final answer text.
            for part in parts:
                if not isinstance(part, dict):
                    continue
                text = str(part.get("text", "") or "").strip()
                if not text:
                    continue
                if part.get("thought") is True:
                    continue
                return text
        return ""

    keys = _collect_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY/GEMINI_API_KEYS (or OPENAI_API_KEY/OPENAI_API_KEYS) is not set")

    base_url = _norm_base(os.environ.get("GEMINI_BASE_URL", ""))
    endpoint = f"{base_url}/v1beta/models/{model_name}:generateContent"
    disable_thinking = os.environ.get("GEMINI_DISABLE_THINKING", "0").strip() == "1"

    def _mk_payload(include_temp: bool, include_generation: bool, with_role: bool) -> dict:
        payload = {
            "contents": (
                [{"role": "user", "parts": [{"text": prompt}]}]
                if with_role
                else [{"parts": [{"text": prompt}]}]
            )
        }
        if include_generation:
            generation_cfg = {"maxOutputTokens": max_tokens}
            if include_temp:
                generation_cfg["temperature"] = temperature
            payload["generationConfig"] = generation_cfg
        if disable_thinking:
            payload["thinkingConfig"] = {"thinkingBudget": 0}
        return payload

    payloads = [
        _mk_payload(include_temp=True, include_generation=True, with_role=True),
        _mk_payload(include_temp=False, include_generation=True, with_role=True),
        _mk_payload(include_temp=False, include_generation=False, with_role=False),
    ]

    start = _GEMINI_KEY_CURSOR % len(keys)
    ordered = keys[start:] + keys[:start]
    error_msgs = []

    for rel_idx, api_key in enumerate(ordered):
        key_idx = (start + rel_idx) % len(keys)
        for payload in payloads:
            code, obj = _post_json(endpoint, payload, api_key)
            if 200 <= code < 300:
                text = _extract_text(obj)
                if text:
                    _GEMINI_KEY_CURSOR = key_idx
                    return text
                error_msgs.append("gemini_http_2xx: empty_text")
            else:
                msg = (obj.get("error") or {}).get("message", str(obj))
                error_msgs.append(f"gemini_http_{code}: {msg}")

    if error_msgs:
        raise RuntimeError(error_msgs[-1])
    raise RuntimeError("Gemini native call failed with unknown error")


def _call_deepseek(model_name: str, prompt: str, temperature: float, max_tokens: int, request_timeout: float) -> str:
    """Call DeepSeek via OpenAI-compatible Chat Completions API."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required for backend=deepseek") from e

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=request_timeout)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def _heuristic_predict(task: str, row: dict) -> str:
    # Debug-only backend; not for final benchmark reporting.
    # Keep it label-free (never read gold answer/label fields).
    if task == "trend_threshold":
        try:
            value = float(row.get("value", float("-inf")))
            threshold = float(row.get("threshold", float("inf")))
            return "yes" if value >= threshold else "no"
        except Exception:
            return "no"
    if task == "temporal_grounding":
        dt = abs(float(row.get("time_delta_hours", 99)))
        if dt <= 1:
            return "high"
        if dt <= 3:
            return "medium"
        return "low"
    if task == "diagnostic_consistency":
        delta = float(row.get("score_delta", 0.0))
        if delta >= 2:
            return "consistent"
        if delta <= -2:
            return "inconsistent"
        return "ambiguous"
    if task == "contrastive_inference":
        try:
            dt_a = abs(float((row.get("option_a") or {}).get("time_delta_hours", 99)))
            dt_b = abs(float((row.get("option_b") or {}).get("time_delta_hours", 99)))
        except Exception:
            return "tie"
        if abs(dt_a - dt_b) <= 0.5:
            return "tie"
        return "A_more_plausible" if dt_a < dt_b else "B_more_plausible"
    return TASK_LABELS[task][0]


def main():
    parser = argparse.ArgumentParser(description="Run CRES model inference")
    parser.add_argument("--backend", choices=["openai", "gemini_native", "deepseek", "heuristic"], default="openai")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", default="")
    parser.add_argument("--tasks", default="trend_threshold,temporal_grounding,diagnostic_consistency,contrastive_inference")
    parser.add_argument("--max-per-task", type=int, default=0, help="0 means all rows")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=1.0)
    parser.add_argument("--retry-max-sleep", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--resume", action="store_true", help="Resume from existing predictions.jsonl in run_dir")
    parser.add_argument("--write-canonical", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in TASK_TO_FILE:
            raise ValueError(f"Unknown task: {t}")

    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_DIR / "model_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    template_payload = {
        "templates": PROMPT_TEMPLATES,
        "tasks": tasks,
        "backend": args.backend,
        "model_name": args.model_name,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    prompt_sha = hashlib.sha256(json.dumps(template_payload, sort_keys=True).encode("utf-8")).hexdigest()

    pred_path = run_dir / "predictions.jsonl"
    done_pairs: set[tuple[str, str]] = set()
    n_existing = 0
    if args.resume and pred_path.exists():
        with pred_path.open() as rf:
            for line in rf:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                task = str(obj.get("task", ""))
                rid = str(obj.get("id", ""))
                if task and rid:
                    done_pairs.add((task, rid))
                    n_existing += 1
        print(f"[INFO] Resume enabled: loaded {n_existing} existing predictions from {pred_path}")

    n_new = 0
    open_mode = "a" if args.resume and pred_path.exists() else "w"
    with pred_path.open(open_mode) as f:
        for task in tasks:
            rows = load_jsonl(TASK_TO_FILE[task])
            if args.max_per_task > 0:
                rows = rows[: args.max_per_task]

            for row in rows:
                row_id = str(row.get("id", ""))
                if (task, row_id) in done_pairs:
                    continue
                prompt = _build_prompt(task, row)
                raw = None
                last_err = None
                for attempt in range(args.max_retries + 1):
                    try:
                        if args.backend == "openai":
                            raw = _call_openai(
                                model_name=args.model_name,
                                prompt=prompt,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                            )
                        elif args.backend == "gemini_native":
                            raw = _call_gemini_native(
                                model_name=args.model_name,
                                prompt=prompt,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                            )
                        elif args.backend == "deepseek":
                            raw = _call_deepseek(
                                model_name=args.model_name,
                                prompt=prompt,
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                request_timeout=args.request_timeout,
                            )
                        else:
                            raw = _heuristic_predict(task, row)
                        break
                    except Exception as exc:  # network/rate limits/transient API issues
                        last_err = exc
                        if attempt >= args.max_retries:
                            raise
                        sleep_s = min(args.retry_sleep * (2 ** attempt), args.retry_max_sleep)
                        print(
                            f"[WARN] {task}:{row.get('id')} attempt {attempt + 1}/{args.max_retries} failed: {exc}; "
                            f"retrying in {sleep_s:.1f}s"
                        )
                        time.sleep(sleep_s)

                if raw is None and last_err is not None:
                    raise RuntimeError(f"Failed to get prediction for {task}:{row.get('id')}: {last_err}")

                pred_label = _normalize_label(task, raw)
                out = {
                    "id": row.get("id"),
                    "task": task,
                    "pred_label": pred_label,
                    "model_name": args.model_name,
                    "model_version": args.model_version,
                    "prompt_sha": prompt_sha,
                    "inference_config": {
                        "backend": args.backend,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "seed": args.seed,
                    },
                    "raw_response": str(raw)[:500],
                }
                f.write(json.dumps(out, ensure_ascii=True) + "\n")
                f.flush()
                n_new += 1
                n_total = n_existing + n_new
                if n_total % 50 == 0:
                    print(f"[INFO] {n_total} predictions written (task={task})", flush=True)

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "backend": args.backend,
        "model_name": args.model_name,
        "model_version": args.model_version,
        "tasks": tasks,
        "max_per_task": args.max_per_task,
        "predictions_path": str(pred_path),
        "resume": args.resume,
        "n_existing_predictions": n_existing,
        "n_new_predictions": n_new,
        "n_predictions": n_existing + n_new,
        "prompt_sha": prompt_sha,
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True))

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "evaluate_cres.py"),
        "--predictions-jsonl",
        str(pred_path),
        "--run-id",
        run_id,
    ]
    if args.write_canonical:
        cmd.append("--write-canonical")
    subprocess.run(cmd, check=True)

    print(f"Wrote predictions: {pred_path}")
    print(f"Run manifest: {run_dir / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
