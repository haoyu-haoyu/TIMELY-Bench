"""
Build final release bundle for extension assets (opt-in).
Copies protocol card, condition graphs, CRES outputs, LLM samples, and evidence files.
Generates manifest with size, sha256, and reproducibility metadata.
"""

import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path

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
        out = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return "unknown"


def tree_hash_from_outputs(outputs):
    payload = "\n".join(f"{o['path']}:{o['sha256']}" for o in sorted(outputs, key=lambda x: x["path"]))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def copy_entries(entries, final_release: Path):
    inputs = []
    outputs = []
    copied = []
    for src, dst in entries:
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
        inputs.append({
            "path": str(src),
            "size_bytes": src.stat().st_size,
            "sha256": sha256_file(src),
        })
        outputs.append({
            "path": str(dst.relative_to(final_release)),
            "size_bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
        })
        copied.append(dst)
    return inputs, outputs, copied


def latest_file(paths):
    existing = [p for p in paths if p.exists()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def main():
    root = ROOT_DIR
    final_release = root / "final_release"
    final_release.mkdir(parents=True, exist_ok=True)

    evidence_dir = final_release / "evidence"
    graphs_dir = final_release / "condition_graphs"
    cres_dir = final_release / "cres"
    llm_dir = final_release / "llm_annotations"
    qc_dir = final_release / "qc"

    # Protocol card & provenance
    protocol_src = latest_file([
        final_release / "ALIGNMENT_PROTOCOL_CARD.md",
        root / "docs" / "alignment_protocol_card.md",
    ])
    provenance_src = final_release / "PROVENANCE.json"

    # Evidence files
    qa_log = latest_file(list((root / "logs").glob("qa_*.log")) + list(final_release.glob("qa_*.log")))
    qa_err = None
    if qa_log is not None:
        qa_err = qa_log.with_suffix(".err")

    evidence_files = [
        qa_log,
        qa_err,
        root / "results" / "standardized" / "permutation_structured_mortality.json",
        root / "results" / "standardized" / "permutation_structured_mortality.csv",
        root / "results" / "standardized" / "late_fusion_sanity_xgb.json",
        root / "results" / "standardized" / "results_summary.csv",
        root / "results" / "standardized" / "results_summary.md",
    ]

    # Condition graphs
    graphs_src = [
        root / "code" / "condition_graphs" / "condition_graph_schema.json",
        root / "code" / "condition_graphs" / "graphs" / "aki_kdigo_graph.json",
        root / "code" / "condition_graphs" / "graphs" / "sepsis_sirs_graph.json",
        root / "code" / "condition_graphs" / "mapping_report.json",
    ]

    # QC evidence
    qc_src = [
        root / "results" / "qc" / "full_alignment_qc.json",
    ]

    # CRES outputs
    cres_src = [
        root / "results" / "cres" / "trend_threshold.jsonl",
        root / "results" / "cres" / "temporal_grounding.jsonl",
        root / "results" / "cres" / "temporal_grounding_index.jsonl",
        root / "results" / "cres" / "cres_build_meta.json",
        root / "results" / "cres" / "cres_eval_summary.json",
        root / "results" / "cres" / "cres_dataset_manifest.json",
        root / "results" / "cres" / "cres_evaluation_report.json",
    ]

    # LLM annotation outputs
    llm_src = [
        root / "results" / "llm_annotations" / "llm_annotation_set.csv",
        root / "results" / "llm_annotations" / "llm_annotation_prompts.jsonl",
        root / "results" / "llm_annotations" / "llm_annotation_summary.json",
        root / "results" / "llm_annotations" / "llm_annotation_meta.json",
        root / "results" / "llm_annotations" / "ANNOTATION_METADATA.json",
        root / "results" / "llm_annotations" / "summary_strata.json",
    ]

    # If metadata specifies a single annotations file, prefer it
    meta_path = root / "results" / "llm_annotations" / "ANNOTATION_METADATA.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        ann_path = Path(meta.get("annotations_path", ""))
        if ann_path.exists():
            llm_src.append(ann_path)
    else:
        llm_src.extend((root / "results" / "llm_annotations").glob("annotations_*.jsonl"))

    # DeepSeek branch (opt-in, never overwrites rule-based)
    deep_meta_path = root / "results" / "llm_annotations" / "ANNOTATION_METADATA_deepseek.json"
    if deep_meta_path.exists():
        llm_src.append(deep_meta_path)
        llm_src.append(root / "results" / "llm_annotations" / "summary_strata_deepseek.json")
        llm_src.append(root / "results" / "llm_annotations" / "failed_requests.jsonl")
        # New evidence validity and coverage files
        llm_src.append(root / "results" / "llm_annotations" / "evidence_validity_deepseek.json")
        llm_src.append(root / "results" / "llm_annotations" / "deepseek_coverage_summary.json")
        llm_src.append(root / "results" / "llm_annotations" / "ANNOTATION_AUDIT_PATCH.json")
        deep_meta = json.loads(deep_meta_path.read_text())
        outputs = deep_meta.get("outputs", [])
        if isinstance(outputs, list):
            for item in outputs:
                if isinstance(item, dict) and item.get("path"):
                    llm_src.append(Path(item["path"]))
        # Also include audited JSONL files
        for p in (root / "results" / "llm_annotations").glob("*_audited.jsonl"):
            llm_src.append(p)

    entries = []
    for src in [protocol_src, provenance_src]:
        if src:
            entries.append((src, final_release / Path(src).name))

    for src in evidence_files:
        if src:
            entries.append((Path(src), evidence_dir / Path(src).name))

    for src in graphs_src:
        entries.append((src, graphs_dir / src.name))

    for src in qc_src:
        entries.append((src, qc_dir / src.name))

    for src in cres_src:
        entries.append((src, cres_dir / src.name))

    for src in llm_src:
        entries.append((src, llm_dir / src.name))

    inputs, outputs, copied = copy_entries(entries, final_release)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    commit = get_git_commit(root)
    requirements = root / "requirements.txt"
    deps = {
        "requirements_path": str(requirements) if requirements.exists() else "",
        "requirements_sha256": sha256_file(requirements) if requirements.exists() else "",
    }

    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "root_dir": str(root),
        "git_commit": commit,
        "python_version": sys.version.split()[0],
        "dependencies": deps,
        "inputs": inputs,
        "outputs": outputs,
        "generation_commands": [
            "python3 code/data_processing/generate_protocol_card.py",
            "python3 code/condition_graphs/build_condition_graphs.py",
            "python3 code/condition_graphs/validate_condition_graph.py --check-mapping",
            "python3 code/data_processing/full_scan_alignment_qc.py --n-buckets 4096 --max-open-files 128 --cleanup",
            "python3 code/cres/build_cres_tasks.py --n-trend 2000 --n-grounding 2000 --seed 42",
            "python3 code/cres/evaluate_cres.py",
            "python3 code/data_processing/build_llm_annotation_set.py --n-per-stratum 50 --max-chunks 50 --seed 42",
            "python3 code/data_processing/run_llm_annotation.py --provider deepseek --model deepseek-chat --workers 32 --max-inflight 64 --rps 10",
            "python3 code/data_processing/summarize_llm_annotation_set.py",
            "python3 code/data_processing/verify_llm_annotation_set.py",
            "python3 code/data_processing/verify_llm_annotation_set.py --metadata-path results/llm_annotations/ANNOTATION_METADATA_deepseek.json --summary-suffix deepseek",
            "python3 code/data_processing/verify_deepseek_evidence_validity.py",
            "python3 code/data_processing/enhance_deepseek_strata_coverage.py",
            "python3 code/data_processing/postprocess_deepseek_annotations.py",
            "python3 code/data_processing/build_final_release_bundle.py",
        ],
    }
    manifest["file_tree_hash"] = tree_hash_from_outputs(outputs)

    manifest_path = final_release / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
