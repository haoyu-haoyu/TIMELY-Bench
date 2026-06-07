"""
Build final release bundle for extension assets (opt-in).
Copies protocol card, condition graphs, physiology templates, CRES outputs,
LLM samples, and evaluation evidence (including calibration/aligner/ablation).
Generates manifest with size, sha256, and reproducibility metadata.
"""

import json
import hashlib
import shutil
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


def latest_dir(paths):
    existing = [p for p in paths if p.exists() and p.is_dir()]
    if not existing:
        return None
    return max(existing, key=lambda p: p.stat().st_mtime)


def _extract_numeric_suffix(path: Path, prefix: str):
    stem = path.stem
    if not stem.startswith(prefix):
        return None
    suffix = stem[len(prefix):].strip("_")
    try:
        return int(suffix)
    except Exception:
        return None


def select_latest_final_qa(evidence_dir: Path):
    candidates = list(evidence_dir.glob("final_qa_*.json"))
    if not candidates:
        return None

    numeric = [(p, _extract_numeric_suffix(p, "final_qa")) for p in candidates]
    numeric = [(p, n) for p, n in numeric if n is not None]
    if numeric:
        return max(numeric, key=lambda x: x[1])[0]
    return max(candidates, key=lambda p: p.stat().st_mtime)


def prune_old_final_qa(evidence_dir: Path):
    latest_json = select_latest_final_qa(evidence_dir)
    if latest_json is None:
        return None, None

    latest_tag = latest_json.stem.replace("final_qa_", "")
    latest_md = evidence_dir / f"final_qa_{latest_tag}.md"
    for p in evidence_dir.glob("final_qa_*.*"):
        if p.name not in {latest_json.name, latest_md.name}:
            _remove_path(p)

    return latest_json, (latest_md if latest_md.exists() else None)


def _read_json_safe(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _is_valid_cres_run(run_dir: Path) -> bool:
    run_manifest = run_dir / "run_manifest.json"
    pred_file = run_dir / "predictions.jsonl"
    pred_manifest = run_dir / "predictions_manifest.json"
    eval_summary = run_dir / "cres_eval_summary.json"
    eval_report = run_dir / "cres_evaluation_report.json"

    for p in [run_manifest, pred_file, pred_manifest, eval_summary, eval_report]:
        if not p.exists():
            return False

    manifest = _read_json_safe(run_manifest) or {}
    backend = str(manifest.get("backend", "")).lower()
    if backend == "heuristic":
        return False

    n_predictions = int(manifest.get("n_predictions") or 0)
    if n_predictions <= 0:
        return False

    if pred_file.stat().st_size <= 0:
        return False

    return True


def collect_cres_runs(model_runs_dir: Path):
    if not model_runs_dir.exists():
        return []
    runs = [d for d in model_runs_dir.glob("*") if d.is_dir() and _is_valid_cres_run(d)]
    return sorted(runs, key=lambda p: p.stat().st_mtime)


def _normalize_path_text(value: str) -> str:
    prefixes = [
        str(ROOT_DIR),
        "${PROJECT_ROOT}",
    ]
    normalized = value
    for prefix in prefixes:
        clean = prefix.rstrip("/")
        if normalized == clean:
            normalized = "${PROJECT_ROOT}"
            continue
        normalized = normalized.replace(clean + "/", "${PROJECT_ROOT}/")
    return normalized


def _sanitize_json_object(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_json_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_json_object(v) for v in obj]
    if isinstance(obj, str):
        return _normalize_path_text(obj)
    return obj


def sanitize_json_paths(path: Path):
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    data = _sanitize_json_object(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True))


def _remove_path(path: Path):
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def rewrite_cres_self_contained_paths(cres_dir: Path, run_ids, canonical_run_id: str | None):
    """Rewrite CRES prediction path fields to final_release-internal paths."""
    run_ids = [rid for rid in run_ids if rid]
    if not run_ids and not canonical_run_id:
        return

    def _rewrite_json(path: Path, edits):
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        changed = False
        for edit in edits:
            changed = edit(data) or changed
        if changed:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")

    if canonical_run_id:
        internal_pred = f"${{PROJECT_ROOT}}/final_release/cres/model_runs/{canonical_run_id}/predictions.jsonl"

        _rewrite_json(
            cres_dir / "run_manifest.json",
            [lambda d: d.__setitem__("predictions_path", internal_pred) is None],
        )
        _rewrite_json(
            cres_dir / "cres_eval_summary.json",
            [lambda d: d.__setitem__("predictions_file", internal_pred) is None],
        )

        def _rewrite_report_summary(d):
            if not isinstance(d, dict):
                return False
            summary = d.get("summary")
            if not isinstance(summary, dict):
                return False
            summary["predictions_file"] = internal_pred
            return True

        _rewrite_json(
            cres_dir / "cres_evaluation_report.json",
            [_rewrite_report_summary],
        )

    for run_id in run_ids:
        internal_pred = f"${{PROJECT_ROOT}}/final_release/cres/model_runs/{run_id}/predictions.jsonl"
        run_dir = cres_dir / "model_runs" / run_id
        _rewrite_json(
            run_dir / "run_manifest.json",
            [lambda d, _pred=internal_pred: d.__setitem__("predictions_path", _pred) is None],
        )
        _rewrite_json(
            run_dir / "cres_eval_summary.json",
            [lambda d, _pred=internal_pred: d.__setitem__("predictions_file", _pred) is None],
        )

        def _rewrite_report_summary_run(d, _pred=internal_pred):
            if not isinstance(d, dict):
                return False
            summary = d.get("summary")
            if not isinstance(summary, dict):
                return False
            summary["predictions_file"] = _pred
            return True

        _rewrite_json(
            run_dir / "cres_evaluation_report.json",
            [_rewrite_report_summary_run],
        )


def harmonize_cres_prompt_metadata(cres_dir: Path, run_ids, canonical_run_id: str | None):
    """Keep run-level prompt metadata consistent with prediction manifests."""
    run_ids = [rid for rid in run_ids if rid]

    def _read(path: Path):
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _write(path: Path, payload):
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

    def _merge_prompt_shas(run_manifest_path: Path, pred_manifest_path: Path):
        run_manifest = _read(run_manifest_path)
        pred_manifest = _read(pred_manifest_path)
        if not isinstance(run_manifest, dict) or not isinstance(pred_manifest, dict):
            return

        merged = []
        for source in [pred_manifest.get("prompt_shas", []), run_manifest.get("prompt_shas", [])]:
            if isinstance(source, list):
                for item in source:
                    s = str(item).strip()
                    if s and s not in merged:
                        merged.append(s)

        single = str(run_manifest.get("prompt_sha", "")).strip()
        if single and single not in merged:
            merged.append(single)

        if not merged:
            return

        changed = False
        if run_manifest.get("prompt_shas") != merged:
            run_manifest["prompt_shas"] = merged
            changed = True

        canonical = single if single in merged else merged[-1]
        if run_manifest.get("prompt_sha") != canonical:
            run_manifest["prompt_sha"] = canonical
            changed = True

        if len(merged) > 1 and run_manifest.get("resume_prompt_migration") is not True:
            run_manifest["resume_prompt_migration"] = True
            changed = True

        if len(merged) <= 1 and "resume_prompt_migration" in run_manifest:
            run_manifest.pop("resume_prompt_migration", None)
            changed = True

        if changed:
            _write(run_manifest_path, run_manifest)

    if canonical_run_id:
        _merge_prompt_shas(
            cres_dir / "run_manifest.json",
            cres_dir / "predictions_manifest.json",
        )

    for run_id in run_ids:
        run_dir = cres_dir / "model_runs" / run_id
        _merge_prompt_shas(
            run_dir / "run_manifest.json",
            run_dir / "predictions_manifest.json",
        )


def collect_release_outputs(final_release: Path):
    """
    Collect a full output inventory of release files.
    Excludes self-referential files that would create hash recursion.
    """
    excluded_names = {".DS_Store", "CHECKSUMS.sha256", "manifest.json"}
    outputs = []
    for path in sorted(final_release.rglob("*")):
        if not path.is_file():
            continue
        if path.name in excluded_names:
            continue
        outputs.append(
            {
                "path": str(path.relative_to(final_release)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return outputs


def write_checksums_file(final_release: Path):
    """Write CHECKSUMS.sha256 for all files under final_release (except itself)."""
    lines = []
    for path in sorted(final_release.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {".DS_Store", "CHECKSUMS.sha256"}:
            continue
        rel = path.relative_to(final_release).as_posix()
        lines.append(f"{sha256_file(path)}  {rel}")

    checksums_path = final_release / "CHECKSUMS.sha256"
    checksums_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {checksums_path}")


def main():
    root = ROOT_DIR
    final_release = root / "final_release"
    final_release.mkdir(parents=True, exist_ok=True)

    # Keep PROVENANCE.json fresh for the current artifact state.
    subprocess.run(
        [sys.executable, str(root / "code" / "data_processing" / "generate_provenance.py")],
        check=True,
    )

    evidence_dir = final_release / "evidence"
    graphs_dir = final_release / "condition_graphs"
    physio_dir = final_release / "physiology_templates"
    cres_dir = final_release / "cres"
    llm_dir = final_release / "llm_annotations"
    qc_dir = final_release / "qc"
    aligner_dir = final_release / "aligner_comparison"
    ablation_dir = final_release / "note_ablation"
    calibration_dir = final_release / "calibration"
    state_space_dir = final_release / "state_space"

    # Remove legacy QA artefacts that cause QA-gate ambiguity in delivery packages.
    for pattern in ["qa_*.log", "qa_*.err"]:
        for p in final_release.glob(pattern):
            _remove_path(p)
        for p in evidence_dir.glob(pattern):
            _remove_path(p)

    # Keep only the latest canonical final_qa evidence pair.
    qa_json, qa_md = prune_old_final_qa(evidence_dir)

    # Keep only canonical run under final_release/cres/model_runs (rebuilt below).
    _remove_path(cres_dir / "model_runs")

    # Protocol card & provenance
    # Prefer the docs/ card (single source of truth). Do not pick based on mtime,
    # otherwise a stale final_release copy can overwrite newer docs content.
    protocol_src = root / "docs" / "ALIGNMENT_PROTOCOL_CARD.md"
    if not protocol_src.exists():
        protocol_src = final_release / "ALIGNMENT_PROTOCOL_CARD.md"
    provenance_src = final_release / "PROVENANCE.json"

    # Evidence files
    # Canonical QA evidence is final_qa_*.json/.md under final_release/evidence.
    if qa_json is None:
        qa_json = latest_file(list((root / "final_release" / "evidence").glob("final_qa_*.json")))
    if qa_md is None:
        qa_md = latest_file(list((root / "final_release" / "evidence").glob("final_qa_*.md")))

    evidence_files = [
        qa_json,
        qa_md,
        root / "results" / "standardized" / "permutation_structured_mortality.json",
        root / "results" / "standardized" / "permutation_structured_mortality.csv",
        root / "results" / "standardized" / "late_fusion_sanity_xgb.json",
        root / "results" / "standardized" / "results_summary.csv",
        root / "results" / "standardized" / "results_summary.md",
    ]
    # Also place canonical result summaries at the final_release root for discoverability
    # (some audit scripts treat these as top-level anchors).
    root_results_summary = [
        root / "results" / "standardized" / "results_summary.csv",
        root / "results" / "standardized" / "results_summary.md",
    ]

    # Condition graphs (dynamic collection to avoid stale hard-coded graph list)
    graph_files = [
        p for p in sorted((root / "code" / "condition_graphs" / "graphs").glob("*.json"))
        if not p.name.startswith("._")
    ]
    graphs_src = [root / "code" / "condition_graphs" / "condition_graph_schema.json"]
    graphs_src.extend(graph_files)
    graphs_src.append(root / "code" / "condition_graphs" / "mapping_report.json")

    # Physiology templates (canonical trajectories)
    physio_src = [
        root / "documentation" / "canonical_trajectories.json",
        root / "documentation" / "TRAJECTORIES_README.md",
    ]

    # QC evidence
    qc_src = [
        root / "results" / "qc" / "full_alignment_qc.json",
    ]

    # Additional evaluation artefacts required by assignment scope
    calibration_src = [
        root / "results" / "calibration" / "calibration_summary.csv",
        root / "results" / "calibration" / "calibration_fusion_summary.csv",
        root / "results" / "calibration" / "calibration_dl_summary.json",
    ]
    aligner_src = [
        root / "results" / "aligner_comparison" / "aligner_results.csv",
        root / "results" / "aligner_comparison" / "aligner_results.json",
    ]
    ablation_src = [
        root / "results" / "note_ablation" / "note_ablation_results.csv",
        root / "results" / "note_ablation" / "note_ablation_results.json",
    ]

    # Predefined splits (release requirement)
    canonical_splits_csv = root / "data" / "splits" / "predefined_splits.csv"
    fallback_splits_csv = root / "data" / "processed" / "predefined_splits.csv"
    splits_src = [
        canonical_splits_csv if canonical_splits_csv.exists() else fallback_splits_csv,
        root / "data" / "splits" / "split_summary.json",
    ]

    # CRES outputs
    # Prefer evaluated model_runs outputs (external predictions); fall back to legacy root files.
    valid_cres_runs = collect_cres_runs(root / "results" / "cres" / "model_runs")
    latest_cres_run = valid_cres_runs[-1] if valid_cres_runs else None
    cres_eval_summary = (
        latest_cres_run / "cres_eval_summary.json"
        if latest_cres_run and (latest_cres_run / "cres_eval_summary.json").exists()
        else root / "results" / "cres" / "cres_eval_summary.json"
    )
    cres_eval_report = (
        latest_cres_run / "cres_evaluation_report.json"
        if latest_cres_run and (latest_cres_run / "cres_evaluation_report.json").exists()
        else root / "results" / "cres" / "cres_evaluation_report.json"
    )
    cres_pred_manifest = (
        latest_cres_run / "predictions_manifest.json"
        if latest_cres_run and (latest_cres_run / "predictions_manifest.json").exists()
        else root / "results" / "cres" / "predictions_manifest.json"
    )
    cres_run_manifest = (
        latest_cres_run / "run_manifest.json"
        if latest_cres_run and (latest_cres_run / "run_manifest.json").exists()
        else root / "results" / "cres" / "run_manifest.json"
    )
    cres_src = [
        root / "results" / "cres" / "trend_threshold.jsonl",
        root / "results" / "cres" / "temporal_grounding.jsonl",
        root / "results" / "cres" / "temporal_grounding_index.jsonl",
        root / "results" / "cres" / "diagnostic_consistency.jsonl",
        root / "results" / "cres" / "contrastive_inference.jsonl",
        root / "results" / "cres" / "cres_build_meta.json",
        root / "results" / "cres" / "cres_dataset_manifest.json",
        cres_eval_summary,
        cres_eval_report,
        cres_pred_manifest,
        cres_run_manifest,
    ]

    # Explicit state-space reconstruction artifacts
    state_space_src = [
        root / "results" / "state_space" / "state_space_schema.json",
        root / "results" / "state_space" / "episode_state_trajectory.jsonl",
        root / "results" / "state_space" / "state_transition_summary.json",
        root / "documentation" / "STATE_SPACE_README.md",
    ]

    # LLM annotation outputs
    llm_src = [
        root / "results" / "llm_annotations" / "llm_annotation_set.csv",
        root / "results" / "llm_annotations" / "llm_annotation_prompts.jsonl",
        root / "results" / "llm_annotations" / "llm_annotation_summary.json",
        root / "results" / "llm_annotations" / "llm_annotation_meta.json",
        root / "results" / "llm_annotations" / "ANNOTATION_METADATA.json",
        root / "results" / "llm_annotations" / "summary_strata.json",
        root / "documentation" / "LLM_ANNOTATION_README.md",
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

    for src in root_results_summary:
        if src and Path(src).exists():
            entries.append((Path(src), final_release / Path(src).name))

    for src in graphs_src:
        entries.append((src, graphs_dir / src.name))

    for src in physio_src:
        entries.append((src, physio_dir / src.name))

    for src in qc_src:
        entries.append((src, qc_dir / src.name))

    for src in calibration_src:
        entries.append((src, calibration_dir / src.name))

    for src in aligner_src:
        entries.append((src, aligner_dir / src.name))

    for src in ablation_src:
        entries.append((src, ablation_dir / src.name))

    # Canonical splits go at the final_release root; metadata goes to evidence/.
    for src in splits_src:
        if src and Path(src).exists():
            if src.name == "predefined_splits.csv":
                entries.append((Path(src), final_release / src.name))
            else:
                entries.append((Path(src), evidence_dir / src.name))

    for src in cres_src:
        entries.append((src, cres_dir / src.name))

    # Copy all valid non-heuristic CRES runs; top-level summary points to canonical (latest).
    canonical_cres_run_id = latest_cres_run.name if latest_cres_run else None
    copied_run_ids = []
    for run_dir in valid_cres_runs:
        copied_run_ids.append(run_dir.name)
        for name in [
            "predictions.jsonl",
            "run_manifest.json",
            "predictions_manifest.json",
            "cres_eval_summary.json",
            "cres_evaluation_report.json",
        ]:
            src = run_dir / name
            if src.exists():
                entries.append((src, cres_dir / "model_runs" / run_dir.name / name))

    for src in state_space_src:
        entries.append((src, state_space_dir / src.name))

    for src in llm_src:
        entries.append((src, llm_dir / src.name))

    inputs, outputs, copied = copy_entries(entries, final_release)

    # Normalize machine-specific absolute paths across all release JSON metadata.
    for json_path in final_release.rglob("*.json"):
        sanitize_json_paths(json_path)

    # Ensure CRES points to final_release-internal predictions path.
    rewrite_cres_self_contained_paths(cres_dir, copied_run_ids, canonical_cres_run_id)
    harmonize_cres_prompt_metadata(cres_dir, copied_run_ids, canonical_cres_run_id)

    # Build a full release inventory for stronger traceability.
    outputs = collect_release_outputs(final_release)

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
        "output_inventory_scope": {
            "mode": "full_release_tree",
            "excluded": ["manifest.json", "CHECKSUMS.sha256", ".DS_Store"],
            "notes": "manifest and checksums are excluded to avoid self-referential hash recursion",
        },
        "generation_commands": [
            "python3 code/data_processing/generate_predefined_splits.py",
            "python3 code/data_processing/generate_protocol_card.py",
            "python3 code/condition_graphs/build_condition_graphs.py",
            "python3 code/condition_graphs/validate_condition_graph.py --check-mapping",
            "python3 code/state_space/reconstruct_state_space.py",
            "python3 code/data_processing/full_scan_alignment_qc.py --n-buckets 4096 --max-open-files 128 --cleanup",
            "python3 code/cres/build_cres_tasks.py --n-trend 900 --n-grounding 900 --n-diagnostic 900 --n-contrastive 900 --seed 42 --min-multimorbidity-ratio 0.3",
            "python3 code/cres/run_cres_model_eval.py --backend <openai|deepseek|heuristic> --model-name <model_id> --run-id <run_id> --write-canonical",
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
    sanitize_json_paths(manifest_path)
    print(f"Wrote {manifest_path}")

    # Rebuild checksums after all metadata is finalized.
    write_checksums_file(final_release)


if __name__ == "__main__":
    main()
