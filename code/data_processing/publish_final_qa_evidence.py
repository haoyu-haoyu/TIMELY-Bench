#!/usr/bin/env python3
"""Publish final_qa_<job_id>.json/.md from Slurm logs and metadata."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR


def _parse_maxrss_to_kb(value: str) -> int | None:
    if not value:
        return None
    value = value.strip()
    m = re.match(r"^([0-9]*\.?[0-9]+)([KMGTP]?)$", value, re.IGNORECASE)
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).upper()
    factor = {
        "": 1 / 1024.0,
        "K": 1.0,
        "M": 1024.0,
        "G": 1024.0 * 1024.0,
        "T": 1024.0 * 1024.0 * 1024.0,
        "P": 1024.0 * 1024.0 * 1024.0 * 1024.0,
    }[unit]
    return int(num * factor)


def _run_sacct(job_id: str):
    cmd = [
        "sacct",
        "-j",
        str(job_id),
        "--format=JobID,JobName,State,ExitCode,Elapsed,Start,End,MaxRSS",
        "-P",
        "-n",
    ]
    try:
        out = subprocess.check_output(cmd, text=True)
    except Exception:
        return None

    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 8:
            continue
        rows.append(
            {
                "JobID": parts[0],
                "JobName": parts[1],
                "State": parts[2],
                "ExitCode": parts[3],
                "Elapsed": parts[4],
                "Start": parts[5],
                "End": parts[6],
                "MaxRSS": parts[7],
            }
        )
    if not rows:
        return None

    job_row = next((r for r in rows if r["JobID"] == str(job_id)), rows[0])
    batch_row = next((r for r in rows if r["JobID"] == f"{job_id}.batch"), None)
    return job_row, batch_row


def _normalize_path(path: Path, project_root: Path) -> str:
    try:
        rel = path.resolve().relative_to(project_root.resolve())
        return f"${{PROJECT_ROOT}}/{rel.as_posix()}"
    except Exception:
        return str(path)


def _extract_checks(stdout_log: Path):
    checks = []
    passed = False
    failed = False

    if not stdout_log.exists():
        return checks, passed, True

    for raw in stdout_log.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("[PASS] "):
            checks.append(line.replace("[PASS] ", "", 1))
        elif line.startswith("[INFO] Scanning "):
            checks.append(line.replace("[INFO] Scanning ", "scanned ", 1).replace(" for QA", ""))
        elif line.startswith("[INFO] Sampling "):
            checks.append(line.replace("[INFO] Sampling ", "sampled ", 1).replace(" for QA", ""))
        elif line.startswith("[FAIL] "):
            failed = True
        elif line == "[OK] Final QA checks passed":
            passed = True
    return checks, passed, failed


def main():
    parser = argparse.ArgumentParser(description="Publish final QA evidence into final_release/evidence")
    parser.add_argument("--job-id", required=True, help="Slurm job id")
    parser.add_argument("--job-name", default="timely_finalqa_noskip")
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--output-dir", default=str(ROOT_DIR / "final_release" / "evidence"))
    parser.add_argument("--project-root", default=str(ROOT_DIR))
    parser.add_argument("--state", default="")
    parser.add_argument("--exit-code", default="")
    parser.add_argument("--elapsed", default="")
    parser.add_argument("--start-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--batch-max-rss-kb", type=int, default=-1)
    parser.add_argument("--qa-name", default="final_qa_noskip")
    args = parser.parse_args()

    job_id = str(args.job_id)
    project_root = Path(args.project_root)
    stdout_log = Path(args.stdout_log)
    stderr_log = Path(args.stderr_log)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    state = args.state
    exit_code = args.exit_code
    elapsed = args.elapsed
    start_time = args.start_time
    end_time = args.end_time
    batch_max_rss_kb = args.batch_max_rss_kb if args.batch_max_rss_kb >= 0 else None

    sacct_rows = _run_sacct(job_id)
    if sacct_rows:
        job_row, batch_row = sacct_rows
        state = state or job_row.get("State", "")
        exit_code = exit_code or job_row.get("ExitCode", "")
        elapsed = elapsed or job_row.get("Elapsed", "")
        start_time = start_time or job_row.get("Start", "")
        end_time = end_time or job_row.get("End", "")
        if batch_max_rss_kb is None and batch_row:
            batch_max_rss_kb = _parse_maxrss_to_kb(batch_row.get("MaxRSS", ""))

    checks, passed_flag, failed_flag = _extract_checks(stdout_log)
    state_ok = str(state).upper().startswith("COMPLETED")
    exit_ok = str(exit_code) == "0:0"
    result = "PASS" if passed_flag and not failed_flag and state_ok and exit_ok else "FAIL"

    stdout_ref = _normalize_path(stdout_log, project_root)
    stderr_ref = _normalize_path(stderr_log, project_root)

    payload = {
        "qa_name": args.qa_name,
        "job_id": int(job_id) if job_id.isdigit() else job_id,
        "job_name": args.job_name,
        "state": state,
        "exit_code": exit_code,
        "elapsed": elapsed,
        "start_time": start_time,
        "end_time": end_time,
        "batch_max_rss_kb": batch_max_rss_kb,
        "checks": checks,
        "result": result,
        "log_files": {
            "stdout": stdout_ref,
            "stderr": stderr_ref,
        },
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }

    out_json = output_dir / f"final_qa_{job_id}.json"
    out_md = output_dir / f"final_qa_{job_id}.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    lines = [
        f"# Final QA Evidence (Job {job_id})",
        "",
        f"- Job: `{args.job_name}` (`{job_id}`)",
        f"- State: `{state}`",
        f"- ExitCode: `{exit_code}`",
        f"- Elapsed: `{elapsed}`",
        f"- Window: `{start_time}` -> `{end_time}`",
        f"- Batch MaxRSS: `{batch_max_rss_kb}K`" if batch_max_rss_kb is not None else "- Batch MaxRSS: `unknown`",
        "",
        f"Checks ({'all passed' if result == 'PASS' else 'failed'}):",
    ]
    for item in checks:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "Logs:",
            f"- `{stdout_ref}`",
            f"- `{stderr_ref}`",
        ]
    )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
