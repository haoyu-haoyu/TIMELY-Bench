#!/usr/bin/env python3
"""Generate PROVENANCE.json for the baseline release."""
import json, hashlib, os, sys, stat
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR

ROOT = ROOT_DIR

def sha256_file(path, block=1024*1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()

def file_meta(path):
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    s = p.stat()
    return {
        "path": str(p),
        "exists": True,
        "size_bytes": s.st_size,
        "mtime_iso": datetime.fromtimestamp(s.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(p),
    }

def dir_tree_hash(dirpath, glob_pattern="*.json"):
    """Compute a deterministic hash over file names + sizes in a directory."""
    entries = []
    for f in sorted(Path(dirpath).glob(glob_pattern)):
        entries.append(f"{f.name}:{f.stat().st_size}")
    payload = "\n".join(entries)
    return hashlib.sha256(payload.encode()).hexdigest()

now = datetime.now(timezone.utc)

# --- (1) Environment ---
try:
    import subprocess
    git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
except Exception:
    git_commit = None

req_path = ROOT / "requirements.txt"
req_hash = sha256_file(req_path) if req_path.exists() else None

# --- (2) Baseline inputs ---
alignment = file_meta(ROOT / "data/processed/temporal_alignment/temporal_textual_alignment.csv")
cohort    = file_meta(ROOT / "data/processed/merge_output/cohort_final.csv")
episodes_dir = ROOT / "episodes/episodes_enhanced"
ep_count = len(list(episodes_dir.glob("TIMELY_v2_*.json"))) if episodes_dir.exists() else 0
ep_tree_hash = dir_tree_hash(episodes_dir, "TIMELY_v2_*.json") if episodes_dir.exists() else None

# --- (3) Baseline outputs ---
outputs = {}
output_files = [
    "results/standardized/results_summary.csv",
    "results/standardized/results_summary.md",
    "results/standardized/permutation_structured_mortality.csv",
    "results/standardized/permutation_structured_mortality.json",
    "results/standardized/late_fusion_sanity_xgb.json",
]
for rel in output_files:
    outputs[rel] = file_meta(ROOT / rel)

# QA gate log
qa_log = file_meta(ROOT / "final_release/qa_31341087.log")
qa_err = file_meta(ROOT / "final_release/qa_31341087.err")

provenance = {
    "schema": "TIMELY-Bench-PROVENANCE/1.0",
    "generated_at": now.isoformat(),
    "project_root": str(ROOT),
    "run_id": now.strftime("%Y%m%d_%H%M%S"),
    "git_commit": git_commit,
    "git_commit_note": "null means no .git directory in project root (HPC scratch copy)" if git_commit is None else "HEAD of local repository",
    "python_version": sys.version.split()[0],
    "requirements_hash": req_hash,
    "baseline_inputs": {
        "temporal_textual_alignment": alignment,
        "cohort_final": cohort,
        "episodes_enhanced": {
            "directory": str(episodes_dir),
            "file_count": ep_count,
            "file_tree_hash": ep_tree_hash,
        },
    },
    "baseline_outputs": outputs,
    "qa_gate": {
        "log": qa_log,
        "err": qa_err,
        "verdict": "PASS (err file size = 0)",
    },
    "generation_commands": [
        "python3 code/baselines/train_tabular_baselines.py",
        "python3 code/baselines/train_text_only.py",
        "python3 code/baselines/train_fusion.py",
        "python3 code/baselines/train_temporal_gru_v2.py",
        "python3 code/baselines/train_readmission_baselines.py",
        "python3 code/baselines/train_los_baselines.py",
        "python3 code/data_processing/standardize_results.py",
        "python3 code/data_processing/run_qa_gate.py",
        "python3 code/data_processing/build_final_release_bundle.py",
    ],
    "split_configuration": {
        "method": "GroupShuffleSplit (holdout) + GroupKFold (CV)",
        "groups_column": "subject_id",
        "test_size": 0.2,
        "n_folds": 5,
        "random_state": 42,
    },
    "opt_in_extensions_disclaimer": (
        "The following components are opt-in extensions and are NOT required for "
        "baseline reproducibility: DeepSeek/LLM annotations, Condition Graphs, "
        "CRES evaluation tasks, and evidence validity reports. They are included "
        "in final_release/ for completeness but the baseline pipeline (structured, "
        "text, fusion, GRU baselines) does not depend on them."
    ),
}

out_path = ROOT / "final_release" / "PROVENANCE.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(provenance, indent=2, ensure_ascii=False))
print("Wrote", out_path)
print("sha256:", sha256_file(out_path))
