#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


TASK_DIMENSIONS: Dict[str, Dict[str, object]] = {
    "AKI-T1": {"mode": "binary + reasoning", "dims": ["D1", "D2", "D3", "D4", "D5", "D6"], "metric": "AUROC/AUPRC"},
    "AKI-S1": {"mode": "binary + reasoning", "dims": ["D1", "D2", "D3", "D4", "D6"], "metric": "AUROC/AUPRC"},
    "DEL-T1": {"mode": "binary + reasoning", "dims": ["D1", "D2", "D3", "D4", "D5", "D6"], "metric": "AUROC/AUPRC"},
    "DEL-S1": {"mode": "binary + reasoning", "dims": ["D1", "D3", "D4", "D6"], "metric": "AUROC/AUPRC"},
    "SEP-T1": {"mode": "binary + reasoning", "dims": ["D1", "D2", "D3", "D4", "D5", "D6"], "metric": "AUROC/AUPRC"},
    "SEP-S1": {"mode": "binary + reasoning", "dims": ["D1", "D2", "D3", "D6"], "metric": "AUROC/AUPRC"},
    "S-T1": {"mode": "binary + reasoning", "dims": ["D1", "D3", "D4", "D6"], "metric": "AUROC/AUPRC"},
    "S-T2": {"mode": "binary (stay-level)", "dims": ["D4"], "metric": "AUROC/AUPRC"},
    "S-T3": {"mode": "binary (stay-level)", "dims": ["D4"], "metric": "AUROC/AUPRC"},
    "S-T4": {"mode": "binary (stay-level)", "dims": ["D3", "D6"], "metric": "AUROC/AUPRC"},
    "S-R1": {"mode": "reasoning (retrospective)", "dims": ["D4", "D6"], "metric": "Rubric 1-5"},
    "S-R2": {"mode": "reasoning (retrospective)", "dims": ["D4", "D6"], "metric": "Rubric 1-5"},
    "S-R3": {"mode": "extraction (retrospective)", "dims": ["D2", "D6"], "metric": "Exact match"},
    "S-R4": {"mode": "reasoning (retrospective)", "dims": ["D4", "D6"], "metric": "Rubric 1-5"},
}

TASK_SAMPLES = {
    "AKI-T1": 2000,
    "AKI-S1": 1000,
    "DEL-T1": 2000,
    "DEL-S1": 1000,
    "SEP-T1": 2000,
    "SEP-S1": 500,
    "S-T1": 1000,
    "S-T2": 500,
    "S-T3": 500,
    "S-T4": 500,
    "S-R1": 500,
    "S-R2": 500,
    "S-R3": 277,
    "S-R4": 500,
}
FIXED_FULLSET_TASKS = {"S-R3"}

VARIANTS = ("full_multimodal", "structured_only", "text_only", "no_temporal_markers", "shuffled_timeline")

STRUCTURED_KEYS = {
    "aki": ["heart_rate", "map_merged", "temperature_c", "spo2", "creatinine", "bun", "potassium", "bicarbonate", "urineoutput_hourly", "rrt_active", "sofa_total"],
    "delirium": ["heart_rate", "map_merged", "gcs_total", "gcs_motor", "rass", "delirium_positive", "delirium_negative", "restraint_active", "sofa_total"],
    "sepsis": ["heart_rate", "map_merged", "temperature_c", "spo2", "lactate", "creatinine", "bilirubin_total", "sofa_total", "vasopressors_active", "vasopressor_dose_norepi_equiv", "fluid_balance"],
    "stroke": ["heart_rate", "map_merged", "temperature_c", "spo2", "gcs_total", "gcs_motor", "gcs_eye", "gcs_verbal", "sofa_total"],
}
NEURO_KEYS = ["gcs_motor", "left_arm_strength", "right_arm_strength", "left_leg_strength", "right_leg_strength", "speech", "commands", "orientation"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 6.5B prompt/sample build for CRES v3")
    p.add_argument("--root", default=".")
    p.add_argument("--output-dir", default="data/processed/v3/cres")
    p.add_argument("--results-dir", default="results/cres_v3")
    p.add_argument("--sample-size-total", type=int, default=12000)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    return p.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stable_rng(seed_text: str) -> np.random.Generator:
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16]
    return np.random.default_rng(int(h, 16))


def json_safe(value):
    if value is None:
        return None
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def resolve_manifest_path(root: Path, raw_path: str | Path) -> Path:
    raw = Path(str(raw_path))
    candidates: List[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        if "TIMELY-Bench_Final" in raw.parts:
            idx = raw.parts.index("TIMELY-Bench_Final")
            candidates.append(root.joinpath(*raw.parts[idx + 1 :]))
    else:
        candidates.append(root / raw)

    expanded: List[Path] = []
    for candidate in candidates:
        expanded.append(candidate)
        if str(candidate).endswith(".parquet"):
            expanded.append(Path(str(candidate) + ".parts"))
        if str(candidate).endswith(".parquet.parts"):
            expanded.append(Path(str(candidate)[: -len(".parts")]))

    seen = set()
    unique_candidates = []
    for candidate in expanded:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate
    tried = "\n".join(str(x) for x in unique_candidates)
    raise FileNotFoundError(f"Could not resolve manifest path from {raw_path!r}. Tried:\n{tried}")


def load_master(root: Path) -> pd.DataFrame:
    cols = [
        "instance_id",
        "condition",
        "task_id",
        "task_family",
        "task_mode",
        "layer",
        "stay_id",
        "subject_id",
        "hadm_id",
        "anchor_hour",
        "horizon_hours",
        "primary_label_key",
        "primary_label_type",
        "primary_label_binary",
        "primary_label_numeric",
        "primary_label_text",
        "trajectory_tier",
        "left_censored",
        "onset_confidence",
        "stroke_layer",
        "stroke_tier",
        "stroke_subtype_priority",
        "representation_profile",
        "available_representations",
    ]
    return pd.read_parquet(root / "data" / "processed" / "v3" / "cres" / "master_instance_manifest.parquet", columns=cols)


def choose_capped_temporal_rows(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["anchor_hour"] = pd.to_numeric(df["anchor_hour"], errors="coerce")
    early = df[df["anchor_hour"] <= 12].copy()
    late = df[df["anchor_hour"] >= 24].copy()
    if not early.empty:
        early["target_dist"] = (early["anchor_hour"] - 6).abs()
        early = early.sort_values(["stay_id", "target_dist", "anchor_hour", "instance_id"], kind="mergesort")
        early = early.drop_duplicates(subset=["stay_id"], keep="first")
        early["anchor_bucket"] = "early"
    if not late.empty:
        late["target_dist"] = (late["anchor_hour"] - 48).abs()
        late = late.sort_values(["stay_id", "target_dist", "anchor_hour", "instance_id"], kind="mergesort")
        late = late.drop_duplicates(subset=["stay_id"], keep="first")
        late["anchor_bucket"] = "late"
    kept = pd.concat([x for x in [early, late] if not x.empty], ignore_index=True)
    return kept


def build_sampling_pool(master: pd.DataFrame) -> pd.DataFrame:
    pools = []
    stay_level_tasks = {"S-T2", "S-T3", "S-T4", "S-R1", "S-R2", "S-R3", "S-R4"}
    for task_id, df in master.groupby("task_id", sort=False):
        if task_id in stay_level_tasks:
            part = df.copy()
            part["anchor_bucket"] = "stay_level"
        else:
            part = choose_capped_temporal_rows(df)
        pools.append(part)
    out = pd.concat(pools, ignore_index=True)
    out["primary_label_binary"] = out["primary_label_binary"].fillna(-1)
    out["left_censored"] = out["left_censored"].astype("string").fillna("<NA>")
    out["onset_confidence"] = out["onset_confidence"].astype("string").fillna("<NA>")
    out["trajectory_tier"] = out["trajectory_tier"].astype("string").fillna("<NA>")
    out["stroke_tier"] = out["stroke_tier"].astype("string").fillna("<NA>")
    return out


def sampling_columns(task_id: str) -> List[str]:
    if task_id == "AKI-T1":
        return ["trajectory_tier", "primary_label_binary", "anchor_bucket"]
    if task_id == "AKI-S1":
        return ["primary_label_binary", "anchor_bucket"]
    if task_id == "DEL-T1":
        return ["left_censored", "primary_label_binary", "anchor_bucket"]
    if task_id == "DEL-S1":
        return ["primary_label_binary", "anchor_bucket"]
    if task_id == "SEP-T1":
        return ["onset_confidence", "primary_label_binary", "anchor_bucket"]
    if task_id == "SEP-S1":
        return ["primary_label_binary", "anchor_bucket"]
    if task_id == "S-T1":
        return ["stroke_tier", "anchor_bucket"]
    if task_id in {"S-T2", "S-T3", "S-T4", "S-R1", "S-R2", "S-R3", "S-R4"}:
        return []
    return []


def allocate_counts(counts: pd.Series, n_target: int) -> Dict[str, int]:
    if n_target >= int(counts.sum()):
        return {str(k): int(v) for k, v in counts.items()}
    weights = counts / counts.sum()
    raw = weights * n_target
    base = np.floor(raw).astype(int)
    remainder = raw - base
    shortfall = int(n_target - base.sum())
    order = np.argsort(-remainder.to_numpy())
    base_arr = base.to_numpy().copy()
    for idx in order[:shortfall]:
        base_arr[idx] += 1
    return {str(k): int(v) for k, v in zip(counts.index.astype(str), base_arr)}


def normalized_task_samples(sample_size_total: int) -> Dict[str, int]:
    quotas = TASK_SAMPLES.copy()
    base_total = sum(quotas.values())
    if sample_size_total >= base_total:
        return quotas

    fixed_total = sum(quotas[t] for t in FIXED_FULLSET_TASKS)
    variable = {k: v for k, v in quotas.items() if k not in FIXED_FULLSET_TASKS}
    remaining_target = max(0, sample_size_total - fixed_total)
    alloc = allocate_counts(pd.Series(variable, dtype=np.int64), remaining_target)

    out = {}
    for task_id, value in quotas.items():
        out[task_id] = value if task_id in FIXED_FULLSET_TASKS else int(alloc[str(task_id)])
    return out


def stratified_sample(df: pd.DataFrame, task_id: str, n_target: int, random_state: int) -> pd.DataFrame:
    if task_id == "S-R3" or n_target >= len(df):
        return df.copy()
    cols = sampling_columns(task_id)
    if not cols:
        rng = np.random.default_rng(random_state)
        take = rng.choice(df.index.to_numpy(), size=min(n_target, len(df)), replace=False)
        return df.loc[np.sort(take)].copy()
    work = df.copy()
    work["stratum"] = work[cols].astype(str).agg("|".join, axis=1)
    counts = work["stratum"].value_counts(dropna=False).sort_index()
    alloc = allocate_counts(counts, min(n_target, len(work)))
    chunks = []
    for stratum, want in alloc.items():
        part = work[work["stratum"] == stratum]
        if want >= len(part):
            chunks.append(part)
        else:
            rng = stable_rng(f"{task_id}|{stratum}|{random_state}")
            take = rng.choice(part.index.to_numpy(), size=want, replace=False)
            chunks.append(part.loc[np.sort(take)])
    return pd.concat(chunks, ignore_index=True).drop(columns=["stratum"])


def build_sample(master: pd.DataFrame, random_state: int, sample_size_total: int) -> tuple[pd.DataFrame, Dict[str, int]]:
    pool = build_sampling_pool(master)
    quotas = normalized_task_samples(sample_size_total)
    sampled = []
    for task_id, task_df in pool.groupby("task_id", sort=False):
        sampled.append(stratified_sample(task_df, task_id, quotas[task_id], random_state))
    sample = pd.concat(sampled, ignore_index=True)
    sample["sample_id"] = np.arange(1, len(sample) + 1)
    return sample, quotas


def load_b2_manifest(root: Path, instance_ids: Iterable[str]) -> pd.DataFrame:
    cols = ["instance_id", "condition", "task_id", "task_mode", "stay_id", "anchor_hour", "b2_index_path"]
    df = pd.read_parquet(root / "data" / "processed" / "v3" / "cres" / "cres_B2_original_manifest.parquet", columns=cols)
    return df[df["instance_id"].isin(set(instance_ids))].copy()


def load_index_rows(root: Path, b2_manifest: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for index_rel, part in b2_manifest.groupby("b2_index_path"):
        idx = pd.read_parquet(resolve_manifest_path(root, index_rel))
        if "context_line_number" not in idx.columns:
            continue
        idx["stay_id"] = idx["stay_id"].astype(np.int64)
        if "anchor_hour_requested" in idx.columns:
            idx["anchor_hour"] = pd.to_numeric(idx["anchor_hour_requested"], errors="coerce")
        else:
            idx["anchor_hour"] = np.nan
        if part["task_mode"].iloc[0] == "retrospective":
            merged = part.merge(idx, on=["stay_id"], how="left")
        else:
            merged = part.merge(idx, on=["stay_id", "anchor_hour"], how="left")
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def load_context_objects(root: Path, needed_lines: Iterable[int]) -> Dict[int, dict]:
    needed = set(int(x) for x in needed_lines if pd.notna(x))
    out: Dict[int, dict] = {}
    path = root / "data" / "processed" / "v3" / "contexts" / "time_aware_patient_contexts_168h.jsonl"
    with path.open() as fh:
        for lineno, line in enumerate(fh, start=1):
            if lineno in needed:
                out[lineno] = json.loads(line)
                if len(out) == len(needed):
                    break
    return out


def trim_context(obj: dict, row: pd.Series) -> dict:
    anchor = row["anchor_hour"]
    task_mode = row["task_mode"]
    notes_policy = row.get("notes_policy")
    ctx = {
        "stay_id": obj["stay_id"],
        "static_context": obj.get("static_context", {}),
        "structured_timeline": [],
        "medication_timeline": [],
        "procedure_timeline": [],
        "diagnosis_pathway_events": [],
        "notes": [],
        "context_variant": obj.get("context_variant", "original"),
    }
    if task_mode == "retrospective":
        ctx["structured_timeline"] = obj.get("structured_timeline", [])
        ctx["medication_timeline"] = obj.get("medication_timeline", [])
        ctx["procedure_timeline"] = obj.get("procedure_timeline", [])
        ctx["diagnosis_pathway_events"] = obj.get("diagnosis_pathway_events", [])
        ctx["notes"] = obj.get("notes", [])
        return ctx
    ctx["structured_timeline"] = [x for x in obj.get("structured_timeline", []) if x.get("hour") is not None and x.get("hour") <= anchor]
    ctx["medication_timeline"] = [x for x in obj.get("medication_timeline", []) if x.get("event_start_hour", 10**9) <= anchor]
    ctx["procedure_timeline"] = [x for x in obj.get("procedure_timeline", []) if x.get("event_start_hour", 10**9) <= anchor]
    ctx["diagnosis_pathway_events"] = [x for x in obj.get("diagnosis_pathway_events", []) if x.get("event_time_hour", 10**9) <= anchor]
    notes = [x for x in obj.get("notes", []) if x.get("hour") is not None and x.get("hour") <= anchor]
    if notes_policy == "exclude_discharge":
        notes = [x for x in notes if str(x.get("note_type")) != "discharge"]
    ctx["notes"] = notes
    return ctx


def format_hour(hour, variant: str) -> str:
    if variant == "no_temporal_markers":
        return ""
    if hour is None or pd.isna(hour):
        return ""
    return f"[Hour {int(hour)}] "


def maybe_shuffle(lines: List[str], instance_id: str, variant: str) -> List[str]:
    if variant != "shuffled_timeline":
        return lines
    rng = stable_rng(f"{instance_id}|{variant}")
    idx = np.arange(len(lines))
    rng.shuffle(idx)
    return [lines[i] for i in idx]


def section1(static_context: dict) -> str:
    age = static_context.get("anchor_age", "?")
    sex = static_context.get("gender", "?")
    comorbidities = []
    for key in ["ckd", "diabetes", "copd", "cad"]:
        if static_context.get(key) not in [None, 0, False, "0"]:
            comorbidities.append(key)
    comorb = ", ".join(comorbidities) if comorbidities else "no explicit comorbidity flags"
    return f"=== SECTION 1: DEMOGRAPHICS & COMORBIDITIES ===\n{age}-year-old {sex}. {comorb}."


def section2(ctx: dict, condition: str, variant: str) -> str:
    keys = STRUCTURED_KEYS[condition]
    header = "=== SECTION 2: STRUCTURED TIMELINE ==="
    lines = []
    for row in ctx.get("structured_timeline", []):
        vals = row.get("values", {})
        parts = []
        for key in keys:
            val = vals.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            parts.append(f"{key}={val}")
        if parts:
            prefix = format_hour(row.get("hour"), variant)
            lines.append(prefix + "; ".join(parts))
    lines = maybe_shuffle(lines, str(ctx["stay_id"]), variant)
    if not lines:
        lines = ["No structured observations in window."]
    return header + "\n" + "\n".join(lines)


def section2b_stroke(ctx: dict, variant: str) -> str:
    header = "=== SECTION 2B: NEUROLOGICAL ASSESSMENTS ==="
    lines = []
    for row in ctx.get("structured_timeline", []):
        vals = row.get("values", {})
        parts = []
        for key in NEURO_KEYS:
            val = vals.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            parts.append(f"{key}={val}")
        if parts:
            prefix = format_hour(row.get("hour"), variant)
            lines.append(prefix + "; ".join(parts))
    lines = maybe_shuffle(lines, str(ctx["stay_id"]) + ":neuro", variant)
    if not lines:
        return header + "\nNo neurological assessments in window."
    return header + "\n" + "\n".join(lines)


def event_section(title: str, rows: List[dict], variant: str, time_key: str = "event_start_hour", name_key: str = "event_name") -> str:
    lines = []
    for row in rows:
        prefix = format_hour(row.get(time_key), variant)
        name = row.get(name_key) or row.get("event_type") or "event"
        lines.append(prefix + str(name))
    lines = maybe_shuffle(lines, title, variant)
    return f"=== {title} ===\n" + ("\n".join(lines) if lines else "No events in window.")


def note_section(ctx: dict, variant: str) -> str:
    lines = []
    for note in ctx.get("notes", []):
        prefix = format_hour(note.get("hour"), variant)
        note_type = note.get("note_type", "note")
        text = str(note.get("text", "")).strip().replace("\n", " ")
        lines.append(f"{prefix}[{note_type}] {text}")
    lines = maybe_shuffle(lines, str(ctx["stay_id"]) + ":notes", variant)
    return "=== SECTION 5: CLINICAL NOTES ===\n" + ("\n".join(lines) if lines else "No notes in window.")


def serialize_context(ctx: dict, row: pd.Series, variant: str) -> str:
    sections = [section1(ctx.get("static_context", {}))]
    if variant != "text_only":
        sections.append(section2(ctx, row["condition"], variant))
        if row["condition"] == "stroke" and row["task_mode"] == "temporal":
            sections.append(section2b_stroke(ctx, variant))
    if variant not in {"structured_only", "text_only"}:
        sections.append(event_section("SECTION 3: MEDICATIONS", ctx.get("medication_timeline", []), variant))
        sections.append(event_section("SECTION 4: PROCEDURES", ctx.get("procedure_timeline", []), variant))
    if variant != "structured_only":
        sections.append(note_section(ctx, variant))
    return "\n\n".join(sections)


def question_for(task_id: str, dim: str) -> str:
    q = {
        ("AKI-T1", "D1"): "At what hour did the earliest sign of AKI Stage 2 progression appear?",
        ("AKI-T1", "D2"): "Has serum creatinine crossed the KDIGO Stage 2 threshold? If yes, when?",
        ("AKI-T1", "D3"): "Is the creatinine trajectory improving, stable, or worsening?",
        ("AKI-T1", "D4"): "Is the observed renal injury more consistent with isolated AKI or AKI with concurrent sepsis physiology?",
        ("AKI-T1", "D5"): "Does this trajectory better match a typical AKI progression or an atypical pattern?",
        ("AKI-T1", "D6"): "List the three most important pieces of evidence supporting your AKI progression assessment.",
        ("AKI-S1", "D2"): "Has the patient crossed the threshold for RRT-related support escalation? If yes, when?",
        ("DEL-T1", "D2"): "Has the patient crossed the delirium positivity threshold based on available assessments? If yes, when?",
        ("DEL-T1", "D5"): "Which fits better: sedation-related altered consciousness or new-onset delirium?",
        ("SEP-T1", "D2"): "Have septic shock thresholds been crossed, including MAP and lactate criteria? If yes, when?",
        ("SEP-T1", "D5"): "Does this episode fit a high-confidence sepsis onset trajectory or a lower-confidence atypical trajectory?",
        ("S-T1", "D3"): "Is left-sided limb strength improving, stable, or worsening?",
        ("S-T2", "D4"): "Which side is more affected, and is the neurological pattern internally consistent?",
        ("S-T3", "D4"): "Are the neurological deficits and early imaging findings diagnostically consistent?",
        ("S-T4", "D3"): "Does the neurological sequence suggest improvement, stability, or progression?",
        ("S-R1", "D4"): "What is the most likely stroke mechanism based on the retrospective record?",
        ("S-R2", "D4"): "Which treatment strategy best matches the full retrospective record?",
        ("S-R3", "D2"): "What is the peak NIHSS value documented in the stay?",
        ("S-R4", "D4"): "Does the retrospective record support any stroke complication, and if so which one?",
    }
    generic = {
        "D1": "At what hour did the earliest sign of the target event appear?",
        "D2": "Has the relevant clinical threshold been crossed? If yes, when?",
        "D3": "Is the target clinical parameter improving, stable, or worsening?",
        "D4": "Which clinical explanation best fits the observed evidence?",
        "D5": "Which scenario better matches the observed data: typical or atypical?",
        "D6": "List the three most important pieces of supporting evidence.",
    }
    return q.get((task_id, dim), generic[dim])


def build_prompt(row: pd.Series, serialized_context: str, dim: str, variant: str) -> str:
    return (
        "[SYSTEM]\n"
        "You are a clinical reasoning assistant evaluating ICU patient data.\n"
        "You must reason step by step, citing specific measurements and timestamps.\n"
        "Answer based ONLY on the data provided.\n\n"
        f"[PATIENT CONTEXT]\n=== PATIENT CONTEXT (Condition: {row['condition']}, Task: {row['task_id']}, Anchor: Hour {row['anchor_hour']}) ===\n\n"
        f"{serialized_context}\n\n"
        f"[QUESTION]\n{question_for(row['task_id'], dim)}\n\n"
        "[FORMAT]\n"
        "{\n"
        '  "reasoning": "...",\n'
        '  "answer": "...",\n'
        '  "evidence": [{"timestamp": "...", "measurement": "...", "value": "..."}],\n'
        '  "confidence": "high/medium/low"\n'
        "}\n"
    )


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_dir = (root / args.output_dir).resolve()
    results_dir = (root / args.results_dir).resolve()
    ensure_dir(output_dir)
    ensure_dir(results_dir)

    master = load_master(root)
    sample, quotas = build_sample(master, args.random_state, args.sample_size_total)
    sample_path = output_dir / "cres_eval_sample_12k.parquet"
    sample.to_parquet(sample_path, index=False)

    b2_manifest = load_b2_manifest(root, sample["instance_id"].tolist())
    indexed = load_index_rows(root, b2_manifest)
    sample = sample.merge(indexed[["instance_id", "context_line_number", "notes_policy"]], on="instance_id", how="left")
    contexts = load_context_objects(root, sample["context_line_number"].dropna().astype(int).tolist())

    prompt_path = output_dir / "cres_eval_prompts_12k.jsonl"
    prompt_rows = 0
    variant_counts: Dict[str, int] = {}
    dim_counts: Dict[str, int] = {}
    with prompt_path.open("w") as fh:
        for _, row in sample.iterrows():
            obj = contexts.get(int(row["context_line_number"]))
            if obj is None:
                continue
            trimmed = trim_context(obj, row)
            dims = TASK_DIMENSIONS[row["task_id"]]["dims"]
            for variant in args.variants:
                serialized = serialize_context(trimmed, row, variant)
                for dim in dims:
                    payload = {
                        "prompt_id": f"{row['instance_id']}::{dim}::{variant}",
                        "instance_id": json_safe(row["instance_id"]),
                        "task_id": json_safe(row["task_id"]),
                        "condition": json_safe(row["condition"]),
                        "task_mode": json_safe(row["task_mode"]),
                        "variant_id": variant,
                        "dimension_id": dim,
                        "metric_family": TASK_DIMENSIONS[row["task_id"]]["metric"],
                        "anchor_hour": json_safe(row["anchor_hour"]),
                        "prompt_text": build_prompt(row, serialized, dim, variant),
                    }
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    prompt_rows += 1
                    variant_counts[variant] = variant_counts.get(variant, 0) + 1
                    dim_counts[dim] = dim_counts.get(dim, 0) + 1

    summary = {
        "sample_rows": int(len(sample)),
        "unique_stays": int(sample["stay_id"].nunique()),
        "prompt_rows": int(prompt_rows),
        "variants": variant_counts,
        "dimensions": dim_counts,
        "tasks": sample.groupby("task_id").size().to_dict(),
        "task_sample_targets": quotas,
        "outputs": {
            "sample_path": str(sample_path.relative_to(root)),
            "prompt_path": str(prompt_path.relative_to(root)),
        },
        "rules": {
            "stay_level_cap": "max 2 anchors per stay per task: one early <=12h and one late >=24h",
            "retrospective_rule": "stroke retrospective tasks serialize full-stay B2 context only",
        },
    }
    (results_dir / "phase65b_prompt_build_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
