"""
Run release audit checks and emit a report for the extension bundle.
"""

import argparse
import json
import hashlib
import re
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


def check_protocol_card(card_path: Path, root: Path):
    missing = []
    evidence_paths = []
    if not card_path.exists():
        return False, ["ALIGNMENT_PROTOCOL_CARD.md missing"], evidence_paths

    lines = card_path.read_text().splitlines()
    in_evidence = False
    for line in lines:
        if line.strip().startswith("## Evidence files"):
            in_evidence = True
            continue
        if in_evidence and line.startswith("## "):
            in_evidence = False
        if in_evidence and line.strip().startswith("-"):
            path_str = line.strip().lstrip("-").strip()
            evidence_paths.append(path_str)
        if "evidence:" in line:
            m = re.search(r"evidence:\s*([^\)]+)\)", line)
            if m:
                evidence_paths.append(m.group(1).strip())

    for p in evidence_paths:
        p_path = Path(p)
        if not p_path.is_absolute():
            p_path = (root / p_path).resolve()
        if not p_path.exists():
            missing.append(str(p_path))

    return len(missing) == 0, missing, evidence_paths


def check_manifest(manifest_path: Path, final_release: Path):
    if not manifest_path.exists():
        return False, ["manifest.json missing"]
    manifest = json.loads(manifest_path.read_text())
    required_keys = [
        "run_id", "timestamp", "git_commit", "python_version", "dependencies",
        "inputs", "outputs", "generation_commands", "file_tree_hash"
    ]
    missing = [k for k in required_keys if k not in manifest]
    if missing:
        return False, [f"manifest missing keys: {missing}"]

    mismatches = []
    for out in manifest.get("outputs", []):
        rel = out.get("path")
        if not rel:
            continue
        fpath = final_release / rel
        if not fpath.exists():
            mismatches.append(f"missing output file: {rel}")
            continue
        sha = sha256_file(fpath)
        if sha != out.get("sha256"):
            mismatches.append(f"hash mismatch: {rel}")

    if mismatches:
        return False, mismatches
    return True, []


def check_full_alignment_qc(root: Path, final_release: Path):
    qc_path = root / "results" / "qc" / "full_alignment_qc.json"
    final_qc_path = final_release / "qc" / "full_alignment_qc.json"
    if not qc_path.exists():
        return False, ["missing results/qc/full_alignment_qc.json"], None

    qc = json.loads(qc_path.read_text())
    alignment = qc.get("alignment", {})
    params = qc.get("parameters", {})

    discharge = int(alignment.get("discharge_rows", -1))
    out_of_range = int(alignment.get("note_hour_out_of_range_rows", -1))
    dup_rows = int(alignment.get("duplicate_rows", -1))
    max_rows = int(params.get("max_rows", 0))

    # Ensure QC evidence is also present in final_release
    final_qc_path.parent.mkdir(parents=True, exist_ok=True)
    final_qc_path.write_bytes(qc_path.read_bytes())

    issues = []
    if max_rows != 0:
        issues.append(f"QC is not full-scan (max_rows={max_rows})")
    if discharge != 0:
        issues.append(f"discharge_rows={discharge}")
    if out_of_range != 0:
        issues.append(f"note_hour_out_of_range_rows={out_of_range}")
    if dup_rows != 0:
        issues.append(f"duplicate_rows={dup_rows}")

    summary = {
        "path": str(final_qc_path if final_qc_path.exists() else qc_path),
        "discharge_rows": discharge,
        "note_hour_out_of_range_rows": out_of_range,
        "duplicate_rows": dup_rows,
        "max_rows": max_rows,
    }
    return len(issues) == 0, issues, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-full-scan-check", action="store_true")
    parser.add_argument("--verify-deepseek", action="store_true")
    args = parser.parse_args()

    root = ROOT_DIR
    final_release = root / "final_release"
    report_lines = []
    status_ok = True

    # 1) Protocol card paths
    card_path = final_release / "ALIGNMENT_PROTOCOL_CARD.md"
    ok, missing_paths, evidence_paths = check_protocol_card(card_path, root)
    if ok:
        report_lines.append("- Protocol card references: PASS")
    else:
        status_ok = False
        report_lines.append("- Protocol card references: FAIL")
        for p in missing_paths:
            report_lines.append(f"  - missing: {p}")

    # 2) Evidence location
    evidence_dir = final_release / "evidence"
    if not evidence_dir.exists():
        status_ok = False
        report_lines.append("- Evidence directory: FAIL (missing final_release/evidence)")
    else:
        bad_paths = []
        for p in evidence_paths:
            if "final_release/evidence" not in p:
                bad_paths.append(p)
        if bad_paths:
            status_ok = False
            report_lines.append("- Evidence directory: FAIL (paths outside evidence dir)")
            for p in bad_paths:
                report_lines.append(f"  - {p}")
        else:
            report_lines.append("- Evidence directory: PASS")

    # 2b) Full-scan QC evidence (no sampling)
    if args.skip_full_scan_check:
        report_lines.append("- Full-scan alignment QC: SKIP (--skip-full-scan-check)")
    else:
        qc_ok, qc_issues, qc_summary = check_full_alignment_qc(root, final_release)
        if qc_ok:
            report_lines.append("- Full-scan alignment QC: PASS")
            report_lines.append(
                "  - evidence: "
                f"{qc_summary['path']} (discharge={qc_summary['discharge_rows']}, "
                f"out_of_range={qc_summary['note_hour_out_of_range_rows']}, "
                f"dup={qc_summary['duplicate_rows']})"
            )
        else:
            status_ok = False
            report_lines.append("- Full-scan alignment QC: FAIL")
            if qc_summary:
                report_lines.append(
                    "  - evidence: "
                    f"{qc_summary['path']} (discharge={qc_summary['discharge_rows']}, "
                    f"out_of_range={qc_summary['note_hour_out_of_range_rows']}, "
                    f"dup={qc_summary['duplicate_rows']}, max_rows={qc_summary['max_rows']})"
                )
            for issue in qc_issues:
                report_lines.append(f"  - {issue}")

    # 3) Condition graph schema + mapping
    try:
        subprocess.check_call([
            sys.executable,
            str(root / "code" / "condition_graphs" / "validate_condition_graph.py"),
            "--check-mapping",
        ])
        report_lines.append("- Condition graphs: PASS")
    except subprocess.CalledProcessError:
        status_ok = False
        report_lines.append("- Condition graphs: FAIL")

    mapping_report = root / "code" / "condition_graphs" / "mapping_report.json"
    if mapping_report.exists():
        (final_release / "condition_graphs").mkdir(parents=True, exist_ok=True)
        (final_release / "condition_graphs" / mapping_report.name).write_bytes(mapping_report.read_bytes())

    # 4) CRES report
    cres_report = root / "results" / "cres" / "cres_evaluation_report.json"
    cres_manifest = root / "results" / "cres" / "cres_dataset_manifest.json"
    cres_ok = True
    missing = []
    if not cres_report.exists():
        cres_ok = False
        missing.append("cres_evaluation_report.json missing")
    else:
        r = json.loads(cres_report.read_text())
        for key in ["evidence_validity_rate", "label_distribution", "pattern_coverage", "note_type_coverage", "failure_case"]:
            if key not in r:
                cres_ok = False
                missing.append(f"cres_evaluation_report missing key: {key}")
    if not cres_manifest.exists():
        cres_ok = False
        missing.append("cres_dataset_manifest.json missing")

    if cres_ok:
        report_lines.append("- CRES report: PASS")
    else:
        status_ok = False
        report_lines.append("- CRES report: FAIL")
        for m in missing:
            report_lines.append(f"  - {m}")

    # 5) LLM annotation metadata + QA
    llm_meta = root / "results" / "llm_annotations" / "ANNOTATION_METADATA.json"
    try:
        subprocess.check_call([sys.executable, str(root / "code" / "data_processing" / "verify_llm_annotation_set.py")])
        report_lines.append("- LLM annotation QA: PASS")
    except subprocess.CalledProcessError:
        status_ok = False
        report_lines.append("- LLM annotation QA: FAIL")
    if not llm_meta.exists():
        status_ok = False
        report_lines.append("- LLM annotation metadata: FAIL (missing ANNOTATION_METADATA.json)")
    else:
        report_lines.append("- LLM annotation metadata: PASS")

    # 5b) DeepSeek branch QA (opt-in)
    deep_meta = root / "results" / "llm_annotations" / "ANNOTATION_METADATA_deepseek.json"
    if deep_meta.exists():
        if args.verify_deepseek:
            try:
                subprocess.check_call(
                    [
                        sys.executable,
                        str(root / "code" / "data_processing" / "verify_llm_annotation_set.py"),
                        "--metadata-path",
                        str(deep_meta),
                        "--summary-suffix",
                        "deepseek",
                    ]
                )
                report_lines.append("- DeepSeek annotation QA: PASS")
            except subprocess.CalledProcessError:
                status_ok = False
                report_lines.append("- DeepSeek annotation QA: FAIL")
        else:
            report_lines.append("- DeepSeek annotation QA: SKIP (use --verify-deepseek)")
        report_lines.append("- DeepSeek annotation metadata: PASS")
    else:
        report_lines.append("- DeepSeek annotation metadata: SKIP (not present)")

    # Refresh bundle + manifest after generating any new audit artefacts
    try:
        subprocess.check_call([sys.executable, str(root / "code" / "data_processing" / "build_final_release_bundle.py")])
        report_lines.append("- Bundle refresh: PASS")
    except subprocess.CalledProcessError:
        status_ok = False
        report_lines.append("- Bundle refresh: FAIL")

    # 6) Manifest integrity
    manifest_path = final_release / "manifest.json"
    ok, issues = check_manifest(manifest_path, final_release)
    if ok:
        report_lines.append("- Manifest integrity: PASS")
    else:
        status_ok = False
        report_lines.append("- Manifest integrity: FAIL")
        for i in issues:
            report_lines.append(f"  - {i}")

    # Emit report
    status = "PASS" if status_ok else "FAIL"
    report_text = "# Release Audit Report\n\n"
    report_text += f"Status: **{status}**\n\n"
    report_text += "\n".join(report_lines) + "\n"

    report_path = final_release / "RELEASE_AUDIT_REPORT.md"
    report_path.write_text(report_text)
    print(f"Wrote {report_path}")

    if not status_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
