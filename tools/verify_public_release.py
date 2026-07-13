#!/usr/bin/env python3
"""Fail closed on common TIMELY-Bench public-release mistakes.

This audit is intentionally standard-library-only so it can run immediately
after cloning the repository.  It is a public-artifact safety and integrity
check, not a substitute for institutional disclosure review.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator


IGNORED_DIRECTORY_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "node_modules",
    "venv",
}

# Formats likely to contain row-level data, embeddings, weights, or serialized
# Python objects.  Aggregate JSON/CSV files remain permitted and are parsed below.
PROHIBITED_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".dta",
    ".feather",
    ".h5",
    ".hdf5",
    ".joblib",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pickle",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}

PROHIBITED_FILENAMES = {
    ".env",
    "credentials.json",
    "id_ed25519",
    "id_rsa",
    "service-account.json",
}

PROHIBITED_TOP_LEVEL_PATHS = {
    ("checkpoints",),
    ("data", "derived"),
    ("data", "processed"),
    ("data", "raw"),
    ("logs",),
    ("model_weights",),
    ("outputs",),
}

PROHIBITED_NAME_FRAGMENTS = {
    "canonical_responses",
    "judge_outputs",
    "patient_contexts",
    "prompt_manifest",
    "scored_prompts",
}

ROW_LEVEL_IDENTIFIER_COLUMNS = {
    "hadm_id",
    "note_id",
    "prompt_id",
    "stay_id",
    "subject_id",
}

REQUIRED_README_NUMBERS = {
    "V3 ICU stay count": "94,458",
    "hourly horizon": "168",
    "prompts per provider": "53,070",
    "frozen provider count": "9",
    "total frozen responses": "477,630",
    "judge prompt count": "500",
    "judge row count": "2,000",
}

REQUIRED_AGGREGATE_FILES = (
    "results/v3/cohort_v3_meta.json",
    "results/v3/structured_backbone_hourly_v3_meta.json",
    "results/v3/hourly_state_grid_168h_meta.json",
    "results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv",
    "results/cres_v3/phase65f_frozen_eval/phase65f_per_task_dimension_metrics.csv",
    "results/cres_v3/phase65f_frozen_eval/phase65f_condition_heatmap_data.csv",
    "results/cres_v3/phase65f_frozen_eval/phase65f_stratified_metrics.csv",
    "results/cres_v3/phase65f_frozen_eval/phase65f_temporal_degradation.csv",
    "results/cres_v3/phase65f_frozen_eval/phase65f_formal_summary.md",
    "results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_provider_summary.csv",
    "results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_condition_summary.csv",
    "results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_pairwise_agreement.csv",
    "results/cres_v3/phase65f_frozen_eval_local_final_sync/phase65f_judge_formal_summary.md",
)

BINARY_SUFFIXES = {
    ".bst",
    ".cls",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".pyc",
    ".svgz",
    ".woff",
    ".woff2",
}

# This is deliberately narrower than a file-level exemption.  The one approved
# literal is code that converts legacy CREATE paths to ${PROJECT_ROOT}; retaining
# the matcher is useful and does not disclose a project or user path.
ABSOLUTE_PATH_ALLOWLIST = {
    ("code/utils/standardize_results.py", "/" + "cephfs/"): (
        "legacy-prefix sanitizer; no account or project name is present"
    ),
    ("code/utils/standardize_results.py", "/" + "cephfs/.../TIMELY-Bench_Final/."): (
        "documentation for the legacy-prefix sanitizer; ellipsis contains no real path"
    ),
}


@dataclass(frozen=True, order=True)
class Finding:
    check: str
    path: str
    message: str


@dataclass
class VerificationReport:
    root: str
    checked_files: int = 0
    json_files: int = 0
    csv_files: int = 0
    allowlisted_matches: list[dict[str, str]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def to_dict(self) -> dict[str, object]:
        return {
            "root": self.root,
            "ok": self.ok,
            "checked_files": self.checked_files,
            "json_files": self.json_files,
            "csv_files": self.csv_files,
            "allowlisted_matches": self.allowlisted_matches,
            "findings": [asdict(item) for item in sorted(self.findings)],
        }


def _iter_files(root: Path) -> Iterator[Path]:
    for path in sorted(root.rglob("*")):
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in IGNORED_DIRECTORY_NAMES for part in relative_parts):
            continue
        if path.is_symlink() or path.is_file():
            yield path


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _add(report: VerificationReport, check: str, path: str, message: str) -> None:
    report.findings.append(Finding(check, path, message))


def _check_path_policy(path: Path, root: Path, report: VerificationReport) -> None:
    relative = _relative(path, root)
    parts = tuple(Path(relative).parts)
    lower_name = path.name.lower()
    if path.is_symlink():
        _add(report, "path-policy", relative, "symbolic links are not permitted in the public export")
    if path.suffix.lower() in PROHIBITED_SUFFIXES:
        _add(report, "path-policy", relative, f"prohibited artifact extension: {path.suffix.lower()}")
    if lower_name in PROHIBITED_FILENAMES or lower_name.startswith(".env."):
        _add(report, "path-policy", relative, "credential or environment filename is prohibited")
    for prefix in PROHIBITED_TOP_LEVEL_PATHS:
        if parts[: len(prefix)] == prefix:
            _add(report, "path-policy", relative, f"prohibited public path prefix: {'/'.join(prefix)}")
    normalized_name = lower_name.replace("-", "_")
    for fragment in PROHIBITED_NAME_FRAGMENTS:
        if fragment in normalized_name:
            _add(report, "path-policy", relative, f"filename resembles restricted row-level artifact: {fragment}")


def _strict_json_load(path: Path) -> object:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=reject_constant)


def _check_json(path: Path, root: Path, report: VerificationReport) -> None:
    report.json_files += 1
    try:
        _strict_json_load(path)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        _add(report, "json-parse", _relative(path, root), str(exc))


def _check_csv(path: Path, root: Path, report: VerificationReport) -> None:
    report.csv_files += 1
    relative = _relative(path, root)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if not header:
                _add(report, "csv-parse", relative, "CSV has no header")
                return
            normalized_header = {column.strip().lower() for column in header}
            restricted = sorted(normalized_header & ROW_LEVEL_IDENTIFIER_COLUMNS)
            if restricted:
                _add(
                    report,
                    "row-level-data",
                    relative,
                    "CSV exposes restricted row-level identifier columns: " + ", ".join(restricted),
                )
            width = len(header)
            for row_number, row in enumerate(reader, start=2):
                if len(row) != width:
                    _add(
                        report,
                        "csv-parse",
                        relative,
                        f"row {row_number} has {len(row)} columns; expected {width}",
                    )
                    break
    except (OSError, UnicodeError, csv.Error) as exc:
        _add(report, "csv-parse", relative, str(exc))


def _compile_absolute_path_patterns() -> tuple[re.Pattern[str], ...]:
    users = "/" + "Users/"
    scratch = "/" + "scratch/"
    ceph = "/" + "cephfs/"
    return tuple(
        re.compile(pattern)
        for pattern in (
            re.escape(users) + r"[^/\s\"'<>]+(?:/[^\s\"'<>]*)?",
            re.escape(scratch) + r"(?:prj|home|users?)/[^\s\"'<>]+",
            re.escape(ceph) + r"[^\s\"'<>]*",
        )
    )


def _compile_secret_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    return (
        ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
        ("private key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
        ("OpenAI-style key", re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b")),
        ("Anthropic-style key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
        ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_-]{30,}\b")),
        ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
        ("Hugging Face token", re.compile(r"\bhf_[A-Za-z0-9]{30,}\b")),
        (
            "assigned secret",
            re.compile(
                r"(?i)\b(?:api[_-]?key|client[_-]?secret|password|token)\b\s*[:=]\s*"
                r"[\"'][A-Za-z0-9_./+=-]{24,}[\"']"
            ),
        ),
    )


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _check_text(path: Path, root: Path, report: VerificationReport) -> None:
    if path.suffix.lower() in BINARY_SUFFIXES or path.is_symlink():
        return
    relative = _relative(path, root)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        _add(report, "content-scan", relative, str(exc))
        return
    if b"\x00" in raw:
        return
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        _add(report, "content-scan", relative, f"non-UTF-8 text file: {exc}")
        return

    for pattern in _compile_absolute_path_patterns():
        for match in pattern.finditer(text):
            allowed = ABSOLUTE_PATH_ALLOWLIST.get((relative, match.group(0)))
            if allowed:
                report.allowlisted_matches.append(
                    {"path": relative, "match": match.group(0), "reason": allowed}
                )
            else:
                _add(
                    report,
                    "absolute-path",
                    relative,
                    f"line {_line_number(text, match.start())}: {match.group(0)!r}",
                )
    for label, pattern in _compile_secret_patterns():
        for match in pattern.finditer(text):
            _add(
                report,
                "secret-scan",
                relative,
                f"line {_line_number(text, match.start())}: possible {label}",
            )


def _check_readme(root: Path, report: VerificationReport) -> None:
    path = root / "README.md"
    if not path.is_file():
        _add(report, "readme", "README.md", "required repository README is missing")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _add(report, "readme", "README.md", str(exc))
        return
    for label, value in REQUIRED_README_NUMBERS.items():
        if value not in text:
            _add(report, "readme", "README.md", f"missing key number for {label}: {value}")
    lowered = text.lower()
    for condition in ("aki", "delirium", "sepsis", "stroke"):
        if condition not in lowered:
            _add(report, "readme", "README.md", f"missing V3 condition name: {condition}")
    for relative in REQUIRED_AGGREGATE_FILES:
        if relative not in text:
            _add(report, "readme", "README.md", f"aggregate artifact is not referenced: {relative}")


def _check_required_artifacts(root: Path, report: VerificationReport) -> None:
    for relative in REQUIRED_AGGREGATE_FILES:
        path = root / relative
        if not path.is_file():
            _add(report, "aggregate-artifact", relative, "required aggregate artifact is missing")


def verify_repository(root: Path) -> VerificationReport:
    root = root.resolve()
    report = VerificationReport(root=str(root))
    if not root.is_dir():
        _add(report, "root", ".", f"not a directory: {root}")
        return report
    files = list(_iter_files(root))
    report.checked_files = len(files)
    for path in files:
        _check_path_policy(path, root, report)
        if path.is_symlink():
            continue
        suffix = path.suffix.lower()
        if suffix == ".json":
            _check_json(path, root, report)
        elif suffix == ".csv":
            _check_csv(path, root, report)
        _check_text(path, root, report)
    _check_readme(root, report)
    _check_required_artifacts(root, report)
    report.findings = sorted(set(report.findings))
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of tools/)",
    )
    parser.add_argument("--json", action="store_true", help="emit a JSON report")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = verify_repository(args.root)
    if args.json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    elif report.ok:
        print("PASS: public release verification completed with 0 findings.")
        print(
            f"Checked {report.checked_files} files "
            f"({report.json_files} JSON, {report.csv_files} CSV); "
            f"{len(report.allowlisted_matches)} narrow path match(es) allowlisted."
        )
    else:
        print(f"FAIL: public release verification found {len(report.findings)} issue(s).")
        for finding in report.findings:
            print(f"- [{finding.check}] {finding.path}: {finding.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
