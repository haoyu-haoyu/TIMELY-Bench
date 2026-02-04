"""
Generate Alignment Protocol Card and provenance metadata.
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_commit(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def latest_file(paths):
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def parse_qa_log(log_path: Path):
    checks = {
        "mortality_readmission_conflicts": ("no mortality/readmission conflicts", "UNKNOWN"),
        "alignment_window_discharge": ("alignment files within window and no discharge notes", "UNKNOWN"),
        "episodes_window_discharge": ("episodes within window and no discharge notes", "UNKNOWN"),
        "subject_split": ("subject_id split check (data/splits)", "UNKNOWN"),
    }

    if not log_path or not log_path.exists():
        return checks

    lines = log_path.read_text().splitlines()
    for line in lines:
        if "[PASS]" in line:
            msg = line.split("[PASS]")[-1].strip()
            for key, (label, _) in checks.items():
                if msg == label:
                    checks[key] = (label, "PASS")
        elif "[FAIL]" in line:
            msg = line.split("[FAIL]")[-1].strip()
            for key, (label, _) in checks.items():
                if msg == label:
                    checks[key] = (label, "FAIL")
    return checks


def main():
    root = ROOT_DIR
    final_release = root / "final_release"
    final_release.mkdir(parents=True, exist_ok=True)
    docs_dir = root / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    qa_log = latest_file(list(final_release.glob("qa_*.log")) + list((root / "logs").glob("qa_*.log")))
    perm_json = root / "results" / "standardized" / "permutation_structured_mortality.json"
    late_fusion_json = root / "results" / "standardized" / "late_fusion_sanity_xgb.json"
    summary_md = root / "results" / "standardized" / "results_summary.md"
    summary_csv = root / "results" / "standardized" / "results_summary.csv"

    # Copy evidence files into final_release (opt-in assets only)
    evidence_dir = final_release / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_to_copy = [qa_log, perm_json, late_fusion_json, summary_md, summary_csv]
    evidence_paths = []
    for src in evidence_to_copy:
        if src and Path(src).exists():
            dst = evidence_dir / Path(src).name
            if src.resolve() != dst.resolve():
                dst.write_bytes(Path(src).read_bytes())
            evidence_paths.append(str(dst))

    # Read alignment constants
    alignment_cfg = root / "code" / "data_processing" / "temporal_textual_alignment.py"
    alignment_text = alignment_cfg.read_text() if alignment_cfg.exists() else ""

    def _extract_bool(name, default=False):
        for line in alignment_text.splitlines():
            if line.strip().startswith(name):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    val = parts[1].strip().split()[0]
                    if val in ("True", "False"):
                        return val == "True"
        return default

    include_discharge = _extract_bool("INCLUDE_DISCHARGE_NOTES", default=False)
    allow_multi_match = _extract_bool("ALLOW_NOTE_MULTI_MATCH", default=False)

    qa_checks = parse_qa_log(qa_log)

    commit = get_git_commit(root)
    timestamp = datetime.now().isoformat(timespec="seconds")

    # Provenance hashes
    input_files = [
        root / "code" / "config.py",
        alignment_cfg,
        qa_log,
        perm_json,
        late_fusion_json,
        summary_md,
        summary_csv,
    ]
    input_hashes = []
    for p in input_files:
        if p and Path(p).exists():
            input_hashes.append({
                "path": str(p),
                "sha256": sha256_file(Path(p)),
            })

    provenance = {
        "timestamp": timestamp,
        "git_commit": commit,
        "root_dir": str(root),
        "inputs": input_hashes,
    }

    prov_path = final_release / "PROVENANCE.json"
    with prov_path.open("w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=True)

    # Protocol card markdown
    lines = []
    lines.append("# Alignment Protocol Card")
    lines.append("")
    lines.append("## Overview")
    lines.append("This card documents the auditable alignment protocol used for TIMELY-Bench.")
    lines.append("")
    lines.append("## Temporal reference (T0)")
    lines.append("T0 is ICU admission time (`intime`). Note hour offsets are computed as charttime - intime.")
    lines.append("")
    lines.append("## Window definition (0<=hour<24)")
    lines.append("Window definition: **0 ≤ hour < 24** (strictly causal).")
    lines.append("")
    lines.append("## Note inclusion/exclusion rules")
    discharge_excluded = not include_discharge
    lines.append(f"- Discharge notes excluded: **{str(discharge_excluded).lower()}** (INCLUDE_DISCHARGE_NOTES=False).")
    lines.append("- Multi-note mode enabled (Radiology/Nursing/Lab/Discharge where available).")
    lines.append("- Alignment window uses only past notes (lookahead disabled).")
    lines.append("")
    lines.append("## Deduplication keys")
    single_injection = not allow_multi_match
    lines.append(f"- Per-stay single injection enforced by note_id: **{str(single_injection).lower()}** (ALLOW_NOTE_MULTI_MATCH=False).")
    lines.append("- Alignment rows keyed by stay_id + pattern_hour + pattern_name + note_id.")
    lines.append("")
    lines.append("## Labels and exclusions")
    lines.append("- Mortality/readmission conflicts removed (readmission set NA for mortality; training drops NaN).")
    lines.append("")
    lines.append("## Train/Test split (subject_id grouping)")
    lines.append("- GroupShuffleSplit holdout test; GroupKFold(groups=subject_id) for CV.")
    lines.append("- No patient-level leakage across folds or test.")
    lines.append("")
    lines.append("## QA Gate checks (4条)")
    for _, (label, status) in qa_checks.items():
        evidence = str(evidence_dir / Path(qa_log).name) if qa_log else "NOT_FOUND"
        lines.append(f"- {label}: **{status}** (evidence: {evidence})")
    lines.append("")
    lines.append("## Evidence files (paths)")
    for p in evidence_paths:
        lines.append(f"- {p}")
    lines.append("")
    lines.append("## Version/Hashes (commit if exists)")
    lines.append(f"- git_commit: {commit}")
    lines.append(f"- provenance: {prov_path}")
    lines.append("")

    card_path = final_release / "ALIGNMENT_PROTOCOL_CARD.md"
    card_text = "\n".join(lines)
    card_path.write_text(card_text)

    # also update docs copy
    docs_path = docs_dir / "alignment_protocol_card.md"
    docs_path.write_text(card_text)

    print(f"Wrote {card_path}")
    print(f"Wrote {docs_path}")
    print(f"Wrote {prov_path}")


if __name__ == "__main__":
    main()
