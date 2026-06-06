#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build final stroke/sepsis feasibility report.")
    p.add_argument("--bq-json", required=True)
    p.add_argument("--project-plan-dir", required=True)
    p.add_argument("--discharge-csv", required=True)
    p.add_argument("--radiology-csv", required=True)
    p.add_argument("--nursing-csv", required=True)
    p.add_argument("--sepsis3-summary-json", required=True)
    p.add_argument("--sepsis-shock-summary-json", required=True)
    p.add_argument("--septic-shock-staging-summary-json", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    return p.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_excerpt(text: str, limit: int = 500) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _kw_patterns(categories: Dict[str, List[str]]) -> Dict[str, re.Pattern]:
    out = {}
    for cat, kws in categories.items():
        parts = [re.escape(k.lower()) for k in kws]
        out[cat] = re.compile("|".join(parts), re.IGNORECASE)
    return out


STROKE_DS_CATEGORIES = {
    "nihss": ["nihss", "nih stroke", "stroke scale"],
    "motor_deficit": ["hemiparesis", "hemiplegia", "hemipare", "weakness"],
    "speech_language": ["aphasia", "dysarthria", "speech"],
    "thrombolysis": ["tpa", "alteplase", "tenecteplase", "thrombolysis"],
    "thrombectomy": ["thrombectomy", "mechanical thrombectomy", "endovascular"],
    "infarction": ["infarct", "infarction", "ischemic stroke", "ischaemic stroke"],
    "haemorrhagic": ["hemorrhagic transformation", "hemorrhagic conversion", "haemorrhagic"],
    "oedema": ["midline shift", "cerebral edema", "cerebral oedema", "mass effect"],
    "vascular_territory": ["mca", "middle cerebral", "basilar", "pca", "aca", "anterior cerebral", "posterior cerebral"],
    "facial_neglect": ["facial droop", "neglect", "visual field"],
    "af_cardiac": ["atrial fibrillation", "afib", "a-fib", "cardioembolic"],
    "dysphagia": ["dysphagia", "swallow", "aspiration"],
}

STROKE_RAD_CATEGORIES = {
    "infarction": ["infarct", "infarction", "ischemia", "ischaemia"],
    "vessel": ["occlusion", "stenosis", "thrombus", "thrombosis"],
    "haemorrhage": ["hemorrhage", "hemorrhagic", "haemorrhage", "bleeding"],
    "oedema_shift": ["edema", "oedema", "midline shift", "mass effect", "herniation"],
    "territory": ["mca", "aca", "pca", "basilar", "vertebral", "carotid"],
    "perfusion": ["penumbra", "perfusion deficit", "mismatch"],
    "scoring": ["aspects"],
}

STROKE_BRAIN_FILTER = re.compile(
    r"(head|brain|ct head|mri brain|cta|mra|perfusion|cranial)",
    re.IGNORECASE,
)

STROKE_NURSING_NEURO = re.compile(
    r"(neuro check|neurological|gcs|pupils|motor|sensation|alert|oriented|confused|lethargic|obtunded|weakness|strength|speech|swallow)",
    re.IGNORECASE,
)

SEPSIS_DS_CATEGORIES = {
    "sepsis": ["sepsis", "septic", "sirs"],
    "infection_source": ["bacteremia", "bacteraemia", "blood culture", "source of infection", "pneumonia", "uti", "urinary tract", "cellulitis", "abscess", "peritonitis"],
    "treatment": ["antibiotic", "antimicrobial", "vasopressor", "norepinephrine", "levophed", "fluid resuscitation", "bolus"],
    "organ_dysfunction": ["organ dysfunction", "organ failure", "multi-organ", "sofa", "lactate", "acidosis"],
    "shock": ["septic shock", "shock", "hemodynamic", "haemodynamic"],
}


def _load_cohort_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _compute_overlap_matrix(cohorts: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, dict]]:
    sets = {name: set(pd.to_numeric(df["stay_id"], errors="coerce").dropna().astype(int).tolist()) for name, df in cohorts.items()}
    out: Dict[str, Dict[str, dict]] = {}
    for a, aset in sets.items():
        out[a] = {}
        for b, bset in sets.items():
            inter = aset & bset
            out[a][b] = {
                "n": len(inter),
                "pct_of_a": (len(inter) / len(aset)) if aset else 0.0,
                "pct_of_b": (len(inter) / len(bset)) if bset else 0.0,
            }
    return out


def _scan_discharge_notes(path: Path, cohort_hadm: Dict[str, set]) -> Tuple[dict, dict]:
    stroke_patterns = _kw_patterns(STROKE_DS_CATEGORIES)
    sepsis_patterns = _kw_patterns(SEPSIS_DS_CATEGORIES)
    coverage = {
        name: {"hadm_with_discharge": set(), "note_rows": 0, "richness": []}
        for name in cohort_hadm
    }
    stroke_by_hadm: Dict[int, dict] = {}
    sepsis_by_hadm: Dict[int, dict] = {}

    usecols = ["stay_id", "hadm_id", "note_id", "discharge_text"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=2000):
        chunk["hadm_id"] = pd.to_numeric(chunk["hadm_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["hadm_id"]).copy()
        chunk["hadm_id"] = chunk["hadm_id"].astype(int)
        for _, row in chunk.iterrows():
            hadm_id = int(row["hadm_id"])
            text = str(row.get("discharge_text") or "")
            text_lower = text.lower()
            for cond, hadm_set in cohort_hadm.items():
                if hadm_id in hadm_set:
                    coverage[cond]["hadm_with_discharge"].add(hadm_id)
                    coverage[cond]["note_rows"] += 1
            if hadm_id in cohort_hadm.get("stroke", set()):
                cats = [cat for cat, pat in stroke_patterns.items() if pat.search(text_lower)]
                richness = len(cats)
                stroke_by_hadm[hadm_id] = {
                    "categories": cats,
                    "richness": richness,
                    "excerpt": _clean_excerpt(text),
                }
                coverage["stroke"]["richness"].append(richness)
            if hadm_id in cohort_hadm.get("sepsis", set()):
                cats = [cat for cat, pat in sepsis_patterns.items() if pat.search(text_lower)]
                richness = len(cats)
                sepsis_by_hadm[hadm_id] = {
                    "categories": cats,
                    "richness": richness,
                    "excerpt": _clean_excerpt(text),
                }
                coverage["sepsis"]["richness"].append(richness)

    stroke_examples = sorted(stroke_by_hadm.values(), key=lambda x: x["richness"], reverse=True)[:5]
    sepsis_examples = sorted(sepsis_by_hadm.values(), key=lambda x: x["richness"], reverse=True)[:5]

    return {
        "stroke": stroke_by_hadm,
        "sepsis": sepsis_by_hadm,
    }, {
        "stroke_examples": stroke_examples,
        "sepsis_examples": sepsis_examples,
        "coverage": coverage,
    }


def _scan_radiology(path: Path, cohort_stay: Dict[str, set]) -> Tuple[dict, dict]:
    patterns = _kw_patterns(STROKE_RAD_CATEGORIES)
    stroke_rows = []
    coverage_any = {
        name: {"stay_with_radiology": set(), "note_rows": 0}
        for name in cohort_stay
    }
    reports_per_stay = Counter()
    for chunk in pd.read_csv(path, usecols=["stay_id", "note_id", "radiology_text"], chunksize=5000):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        for _, row in chunk.iterrows():
            stay_id = int(row["stay_id"])
            text = str(row.get("radiology_text") or "")
            for cond, stay_set in cohort_stay.items():
                if stay_id in stay_set:
                    coverage_any[cond]["stay_with_radiology"].add(stay_id)
                    coverage_any[cond]["note_rows"] += 1
            if stay_id in cohort_stay.get("stroke", set()):
                reports_per_stay[stay_id] += 1
                if STROKE_BRAIN_FILTER.search(text):
                    cats = [cat for cat, pat in patterns.items() if pat.search(text)]
                    stroke_rows.append(
                        {
                            "stay_id": stay_id,
                            "brain_related": True,
                            "categories": cats,
                            "richness": len(cats),
                            "excerpt": _clean_excerpt(text),
                        }
                    )
    return {
        "stroke_rows": stroke_rows,
        "reports_per_stay": reports_per_stay,
        "comparison_any": coverage_any,
    }, {}


def _scan_nursing(path: Path, cohort_stay: Dict[str, set]) -> dict:
    neuro_hits = Counter()
    any_hits = {
        name: {"stay_with_nursing": set(), "note_rows": 0}
        for name in cohort_stay
    }
    for chunk in pd.read_csv(path, usecols=["stay_id", "chart_text"], chunksize=10000):
        chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce").astype("Int64")
        chunk = chunk.dropna(subset=["stay_id"]).copy()
        chunk["stay_id"] = chunk["stay_id"].astype(int)
        for _, row in chunk.iterrows():
            stay_id = int(row["stay_id"])
            text = str(row.get("chart_text") or "")
            for cond, stay_set in cohort_stay.items():
                if stay_id in stay_set:
                    any_hits[cond]["stay_with_nursing"].add(stay_id)
                    any_hits[cond]["note_rows"] += 1
            if stay_id in cohort_stay.get("stroke", set()) and STROKE_NURSING_NEURO.search(text):
                neuro_hits[stay_id] += 1
    return {
        "stroke_neuro_hits": neuro_hits,
        "comparison_any": any_hits,
    }


def _describe_richness(values: List[int]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0, "max": 0, "histogram": {}}
    ser = pd.Series(values)
    return {
        "count": int(ser.count()),
        "mean": float(ser.mean()),
        "median": float(ser.median()),
        "min": int(ser.min()),
        "max": int(ser.max()),
        "histogram": {str(int(k)): int(v) for k, v in ser.value_counts().sort_index().items()},
    }


def _verdict_stroke(total_stays: int, discharge_cov: float, nihss_pct: float, brain_cov: float, avg_informative: float) -> Tuple[str, str]:
    if total_stays >= 2000 and discharge_cov >= 0.80 and nihss_pct >= 0.30 and brain_cov >= 0.50 and avg_informative >= 2:
        return "GO", "All predefined stroke feasibility thresholds are met."
    if total_stays < 500 or discharge_cov < 0.60:
        return "NO_GO", "Stroke cohort or discharge-summary coverage is critically insufficient."
    return "CONDITIONAL_GO", "Cohort and/or text coverage are usable but at least one key threshold is not met."


def _verdict_sepsis(cohort_size: int, lactate_cov: float, vaso_pct: float, sofa_available: bool, positive_rate: float, core_above_60: int) -> Tuple[str, str]:
    if cohort_size >= 10000 and lactate_cov >= 0.70 and vaso_pct >= 0.30 and sofa_available and 0.02 <= positive_rate <= 0.15 and core_above_60 >= 8:
        return "GO", "Sepsis cohort size, core physiologic coverage, and progression label rate all support execution."
    if cohort_size < 5000 or lactate_cov < 0.40:
        return "NO_GO", "Core sepsis cohort size or lactate coverage is too weak for reliable progression benchmarking."
    return "CONDITIONAL_GO", "Sepsis is feasible, but one or more supporting coverage thresholds remain below the ideal target."


def main() -> None:
    args = parse_args()

    bq = _read_json(Path(args.bq_json))
    project_plan_dir = Path(args.project_plan_dir)
    project_plan_dir.mkdir(parents=True, exist_ok=True)

    bq_dir = Path(bq["exports"]["stroke_cohort_csv"]).resolve().parent
    stroke_df = _load_cohort_csv(Path(bq["exports"]["stroke_cohort_csv"]))
    aki_df = _load_cohort_csv(Path(bq["exports"]["aki_cohort_csv"]))
    delirium_df = _load_cohort_csv(Path(bq["exports"]["delirium_cohort_csv"]))
    sepsis_df = _load_cohort_csv(Path(bq["exports"]["sepsis_cohort_csv"]))

    cohorts = {
        "AKI": aki_df,
        "Delirium": delirium_df,
        "Stroke": stroke_df,
        "Sepsis": sepsis_df,
    }
    cohort_hadm = {name.lower(): set(pd.to_numeric(df["hadm_id"], errors="coerce").dropna().astype(int).tolist()) for name, df in cohorts.items()}
    cohort_stay = {name.lower(): set(pd.to_numeric(df["stay_id"], errors="coerce").dropna().astype(int).tolist()) for name, df in cohorts.items()}

    discharge_rows, discharge_meta = _scan_discharge_notes(Path(args.discharge_csv), cohort_hadm)
    radiology_rows, _ = _scan_radiology(Path(args.radiology_csv), cohort_stay)
    nursing_rows = _scan_nursing(Path(args.nursing_csv), cohort_stay)

    stroke_discharge_by_hadm = discharge_rows["stroke"]
    sepsis_discharge_by_hadm = discharge_rows["sepsis"]
    stroke_rad = radiology_rows["stroke_rows"]
    stroke_reports_per_stay = radiology_rows["reports_per_stay"]

    stroke_hadm_total = int(stroke_df["hadm_id"].nunique())
    stroke_stay_total = int(stroke_df["stay_id"].nunique())
    stroke_with_ds = len(set(stroke_discharge_by_hadm.keys()))
    stroke_nihss = sum(1 for v in stroke_discharge_by_hadm.values() if "nihss" in v["categories"])
    stroke_any_kw = sum(1 for v in stroke_discharge_by_hadm.values() if v["richness"] > 0)
    stroke_brain_rad_stays = {r["stay_id"] for r in stroke_rad}
    stroke_infarct = len({r["stay_id"] for r in stroke_rad if "infarction" in r["categories"]})
    stroke_hem = len({r["stay_id"] for r in stroke_rad if "haemorrhage" in r["categories"]})
    stroke_vessel = len({r["stay_id"] for r in stroke_rad if "territory" in r["categories"] or "vessel" in r["categories"]})
    stroke_nursing_neuro_stays = len(nursing_rows["stroke_neuro_hits"])

    sepsis_hadm_total = int(sepsis_df["hadm_id"].nunique())
    sepsis_with_ds = len(set(sepsis_discharge_by_hadm.keys()))

    stroke_richness = [v["richness"] for v in stroke_discharge_by_hadm.values()]
    stroke_rad_richness = [r["richness"] for r in stroke_rad]
    sepsis_richness = [v["richness"] for v in sepsis_discharge_by_hadm.values()]

    sepsis_onset_summary = _read_json(Path(args.sepsis3_summary_json))
    sepsis_shock_summary = _read_json(Path(args.sepsis_shock_summary_json))
    septic_shock_staging_summary = _read_json(Path(args.septic_shock_staging_summary_json))

    stroke_structured = bq["stroke_structured_coverage"]
    sepsis_structured = bq["sepsis_structured_coverage"]
    stroke_cov_map = {row["variable"]: row for row in stroke_structured}
    sepsis_cov_map = {row["variable"]: row for row in sepsis_structured}

    overlap = _compute_overlap_matrix({
        "AKI": aki_df,
        "Delirium": delirium_df,
        "Stroke": stroke_df,
        "Sepsis": sepsis_df,
    })

    comparison_rows = {}
    for cond, df in cohorts.items():
        ck = cond.lower()
        n_stays = int(df["stay_id"].nunique())
        n_hadm = int(pd.to_numeric(df["hadm_id"], errors="coerce").dropna().astype(int).nunique())
        discharge_hadm_hits = len(discharge_meta["coverage"][ck]["hadm_with_discharge"])
        radiology_stay_hits = len(radiology_rows["comparison_any"][ck]["stay_with_radiology"])
        nursing_stay_hits = len(nursing_rows["comparison_any"][ck]["stay_with_nursing"])
        total_note_rows = (
            discharge_meta["coverage"][ck]["note_rows"]
            + radiology_rows["comparison_any"][ck]["note_rows"]
            + nursing_rows["comparison_any"][ck]["note_rows"]
        )
        comparison_rows[cond] = {
            "n_stays": n_stays,
            "pct_with_discharge_summary": (discharge_hadm_hits / n_hadm) if n_hadm else 0.0,
            "pct_with_radiology_reports": (radiology_stay_hits / n_stays) if n_stays else 0.0,
            "pct_with_other_note_types": (nursing_stay_hits / n_stays) if n_stays else 0.0,
            "avg_total_notes_per_stay": (total_note_rows / n_stays) if n_stays else 0.0,
        }

    stroke_avg_informative = ((stroke_with_ds + len(stroke_brain_rad_stays)) / stroke_stay_total) if stroke_stay_total else 0.0
    stroke_verdict, stroke_reason = _verdict_stroke(
        stroke_stay_total,
        (stroke_with_ds / stroke_hadm_total) if stroke_hadm_total else 0.0,
        (stroke_nihss / stroke_with_ds) if stroke_with_ds else 0.0,
        (len(stroke_brain_rad_stays) / stroke_stay_total) if stroke_stay_total else 0.0,
        stroke_avg_informative,
    )
    core_above_60 = sum(1 for row in sepsis_structured if row.get("status") == "ok" and row.get("pct_coverage", 0.0) >= 0.60)
    sepsis_vaso_pct = (septic_shock_staging_summary.get("stays_with_any_septic_shock", 0) / sepsis_onset_summary.get("cohort_size", max(1, sepsis_df["stay_id"].nunique())))
    sepsis_verdict, sepsis_reason = _verdict_sepsis(
        int(sepsis_df["stay_id"].nunique()),
        float(sepsis_cov_map.get("lactate", {}).get("pct_coverage", 0.0)),
        sepsis_vaso_pct,
        any(r["variable"] == "sofa_total" and r["status"] == "ok" for r in sepsis_structured),
        float(sepsis_shock_summary["positive_rate"]),
        core_above_60,
    )

    final_json = {
        "metadata": {
            "generated_from": "stroke_spesis数据检查.md",
            "note_source_preference": "local_fallback_if_needed",
            "bq_intermediate_source": str(Path(args.bq_json).resolve()),
        },
        "note_module_probe": bq["note_module_probe"],
        "stroke_feasibility": {
            "cohort_size": stroke_stay_total,
            "cohort_basic": bq["stroke_basic"],
            "discharge_summary_coverage_pct": (stroke_with_ds / stroke_hadm_total) if stroke_hadm_total else 0.0,
            "nihss_in_text_coverage_pct": (stroke_nihss / stroke_with_ds) if stroke_with_ds else 0.0,
            "any_stroke_keyword_pct": (stroke_any_kw / stroke_with_ds) if stroke_with_ds else 0.0,
            "discharge_keyword_richness": _describe_richness(stroke_richness),
            "discharge_examples": discharge_meta["stroke_examples"],
            "brain_radiology_coverage_pct": (len(stroke_brain_rad_stays) / stroke_stay_total) if stroke_stay_total else 0.0,
            "radiology_infarct_pct": (stroke_infarct / len(stroke_brain_rad_stays)) if stroke_brain_rad_stays else 0.0,
            "radiology_hemorrhage_pct": (stroke_hem / len(stroke_brain_rad_stays)) if stroke_brain_rad_stays else 0.0,
            "radiology_territory_or_vessel_pct": (stroke_vessel / len(stroke_brain_rad_stays)) if stroke_brain_rad_stays else 0.0,
            "radiology_richness": _describe_richness(stroke_rad_richness),
            "radiology_examples": sorted(stroke_rad, key=lambda x: x["richness"], reverse=True)[:5],
            "avg_brain_radiology_reports_per_stay": (sum(stroke_reports_per_stay.values()) / len(stroke_reports_per_stay)) if stroke_reports_per_stay else 0.0,
            "nursing_note_coverage_pct": (stroke_nursing_neuro_stays / stroke_stay_total) if stroke_stay_total else 0.0,
            "structured_data_baseline": "adequate" if sum(1 for r in stroke_structured if r["pct_coverage"] >= 0.7) >= 8 else "limited",
            "structured_coverage": stroke_structured,
            "overall_verdict": stroke_verdict,
            "reasoning": stroke_reason,
        },
        "sepsis_feasibility": {
            "cohort_size": int(sepsis_df["stay_id"].nunique()),
            "cohort_basic": bq["sepsis_basic"],
            "sepsis3_existing_summary": sepsis_onset_summary,
            "shock_existing_summary": sepsis_shock_summary,
            "shock_staging_summary": septic_shock_staging_summary,
            "shock_subset_size": int(sepsis_shock_summary["stays_with_any_shock_after_onset"]),
            "shock_positive_rate_pct": float(sepsis_shock_summary["positive_rate"]),
            "lactate_coverage_pct": float(sepsis_cov_map.get("lactate", {}).get("pct_coverage", 0.0)),
            "vasopressor_coverage_pct": sepsis_vaso_pct,
            "sofa_available": any(r["variable"] == "sofa_total" and r["status"] == "ok" for r in sepsis_structured),
            "map_coverage_pct": float(sepsis_cov_map.get("map", {}).get("pct_coverage", 0.0)),
            "discharge_summary_coverage_pct": (sepsis_with_ds / sepsis_hadm_total) if sepsis_hadm_total else 0.0,
            "discharge_keyword_richness": _describe_richness(sepsis_richness),
            "discharge_examples": discharge_meta["sepsis_examples"],
            "all_core_labs_above_70pct": all(
                sepsis_cov_map.get(var, {}).get("pct_coverage", 0.0) >= 0.70
                for var in ["lactate", "wbc", "creatinine", "bilirubin_total", "platelets", "bicarbonate", "bun", "albumin"]
            ),
            "structured_coverage": sepsis_structured,
            "overall_verdict": sepsis_verdict,
            "reasoning": sepsis_reason,
        },
        "cross_condition_overlap": overlap,
        "text_coverage_comparison": comparison_rows,
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(final_json, ensure_ascii=False, indent=2), encoding="utf-8")

    md = []
    md.append("# Stroke / Sepsis Feasibility Audit")
    md.append("")
    md.append("## Executive Summary")
    md.append(
        f"Stroke verdict: **{stroke_verdict}**. Sepsis verdict: **{sepsis_verdict}**. "
        "Stroke appears text-rich but still dependent on note quality, while sepsis remains structurally feasible with strong cohort size and progression labels."
    )
    md.append("")
    md.append("## Stroke Key Numbers")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Stroke ICU stays | {stroke_stay_total} |")
    md.append(f"| Discharge summary coverage | {((stroke_with_ds / stroke_hadm_total) if stroke_hadm_total else 0.0):.1%} |")
    md.append(f"| NIHSS mention rate in discharge summaries | {((stroke_nihss / stroke_with_ds) if stroke_with_ds else 0.0):.1%} |")
    md.append(f"| Brain-radiology coverage | {((len(stroke_brain_rad_stays) / stroke_stay_total) if stroke_stay_total else 0.0):.1%} |")
    md.append(f"| Avg informative notes per stay | {stroke_avg_informative:.2f} |")
    md.append("")
    md.append("## Sepsis Key Numbers")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    md.append(f"| Sepsis ICU stays | {int(sepsis_df['stay_id'].nunique())} |")
    md.append(f"| Shock subset size | {int(sepsis_shock_summary['stays_with_any_shock_after_onset'])} |")
    md.append(f"| Shock positive rate | {float(sepsis_shock_summary['positive_rate']):.2%} |")
    md.append(f"| Lactate coverage | {float(sepsis_cov_map.get('lactate', {}).get('pct_coverage', 0.0)):.1%} |")
    md.append(f"| Vasopressor coverage proxy | {sepsis_vaso_pct:.1%} |")
    md.append(f"| MAP coverage | {float(sepsis_cov_map.get('map', {}).get('pct_coverage', 0.0)):.1%} |")
    md.append("")
    md.append("## Cross-Condition Overlap (absolute counts)")
    md.append("")
    md.append("|  | AKI | Delirium | Stroke | Sepsis |")
    md.append("|---|---:|---:|---:|---:|")
    for row in ["AKI", "Delirium", "Stroke", "Sepsis"]:
        md.append(
            f"| {row} | {overlap[row]['AKI']['n']} | {overlap[row]['Delirium']['n']} | {overlap[row]['Stroke']['n']} | {overlap[row]['Sepsis']['n']} |"
        )
    md.append("")
    md.append("## Text Coverage Comparison")
    md.append("")
    md.append("| Metric | AKI | Delirium | Stroke | Sepsis |")
    md.append("|---|---:|---:|---:|---:|")
    for metric in ["n_stays", "pct_with_discharge_summary", "pct_with_radiology_reports", "pct_with_other_note_types", "avg_total_notes_per_stay"]:
        vals = []
        for cond in ["AKI", "Delirium", "Stroke", "Sepsis"]:
            value = comparison_rows[cond][metric]
            vals.append(f"{value:.1%}" if isinstance(value, float) and value <= 1 else f"{value:.2f}" if isinstance(value, float) else str(value))
        md.append(f"| {metric} | {' | '.join(vals)} |")
    md.append("")
    md.append("## Go / No-Go Verdicts")
    md.append("")
    md.append(f"- Stroke: **{stroke_verdict}**. {stroke_reason}")
    md.append(f"- Sepsis: **{sepsis_verdict}**. {sepsis_reason}")
    md.append("")
    md.append("## Recommended Next Steps")
    md.append("")
    md.append("- Keep stroke as a text-heavy condition and explicitly rely on discharge summary + radiology coverage.")
    md.append("- Keep sepsis as a structured-data-heavy condition using lactate, MAP, vasopressors, SOFA, and existing shock labels.")
    md.append("- If stroke NIHSS coverage remains low, define reasoning tasks around stroke narrative richness rather than structured severity scoring.")
    md.append("- If needed, tighten the stroke cohort to primary-diagnosis admissions for cleaner note narratives.")
    output_md.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
