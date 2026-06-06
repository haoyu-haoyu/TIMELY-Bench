#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    DEFAULT_DIAGNOSIS_COMORBIDITIES,
    DEFAULT_DIAGNOSIS_PATHWAY_EVENTS,
    DEFAULT_DISEASE_TIMELINES_FILE,
    DEFAULT_TEXT_FEATURES_FILE,
    DEFAULT_V3_COHORT_FILE,
    ensure_v3_directories,
)
from v3.io_utils import write_table  # type: ignore
from v3.mappings import match_prefixes, parse_icd_codes  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 diagnosis comorbidity and pathway event tables.")
    p.add_argument("--cohort-csv", default=str(DEFAULT_V3_COHORT_FILE))
    p.add_argument("--disease-timelines-json", default=str(DEFAULT_DISEASE_TIMELINES_FILE))
    p.add_argument("--text-features-csv", default=str(DEFAULT_TEXT_FEATURES_FILE))
    p.add_argument("--events-out", default=str(DEFAULT_DIAGNOSIS_PATHWAY_EVENTS))
    p.add_argument("--comorbidities-out", default=str(DEFAULT_DIAGNOSIS_COMORBIDITIES))
    p.add_argument("--stay-limit", type=int, default=None)
    return p.parse_args()


def _load_cohort(path: Path, stay_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")
    if stay_limit is not None:
        df = df.head(int(stay_limit)).copy()
    return df


def _build_comorbidities(cohort: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in cohort.iterrows():
        codes = parse_icd_codes(row.get("icd_codes"))
        flags = match_prefixes(codes)
        if "ckd" in row and pd.notna(row["ckd"]):
            flags["ckd"] = int(row["ckd"])
        rows.append(
            {
                "stay_id": int(row["stay_id"]),
                "subject_id": int(row["subject_id"]) if "subject_id" in row and pd.notna(row["subject_id"]) else None,
                "hadm_id": int(row["hadm_id"]) if "hadm_id" in row and pd.notna(row["hadm_id"]) else None,
                "raw_icd_codes": ",".join(codes),
                "diagnoses_text": row.get("diagnoses_text", ""),
                "num_conditions": int(row["num_conditions"]) if "num_conditions" in row and pd.notna(row["num_conditions"]) else 0,
                **flags,
            }
        )
    return pd.DataFrame(rows)


def _append_event(rows: list[dict], stay_id: int, event_time_hour: float | int, event_name: str, event_type: str, event_source: str, confidence: str, is_proxy: bool, details: dict | None = None) -> None:
    details = details or {}
    event_rule = str(details.pop("event_rule", event_name))
    rows.append(
        {
            "stay_id": int(stay_id),
            "event_time_hour": float(event_time_hour),
            "event_name": event_name,
            "event_type": event_type,
            "event_source": event_source,
            "event_rule": event_rule,
            "confidence": confidence,
            "is_proxy": int(is_proxy),
            "details_json": json.dumps(details, ensure_ascii=False),
        }
    )


def _build_events_from_cohort(cohort: pd.DataFrame, comorb: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    comorb_map = comorb.set_index("stay_id").to_dict(orient="index")
    for _, row in cohort.iterrows():
        sid = int(row["stay_id"])
        _append_event(rows, sid, 0, "admission", "anchor", "cohort", "high", False)

        if int(row.get("has_aki_final", 0) or 0) == 1 or float(row.get("aki_stage_max", 0) or 0) >= 1:
            _append_event(rows, sid, 0, "aki_present_proxy", "diagnosis_pathway", "cohort_final", "medium", True, {"aki_stage_max": row.get("aki_stage_max")})
        if int(row.get("has_sepsis_final", 0) or 0) == 1 or bool(row.get("sepsis3", False)):
            _append_event(rows, sid, 0, "sepsis_present_proxy", "diagnosis_pathway", "cohort_final", "medium", True, {"sepsis_sofa": row.get("sepsis_sofa")})
        if int(row.get("has_shock", 0) or 0) == 1:
            _append_event(rows, sid, 0, "shock_present_proxy", "diagnosis_pathway", "cohort_final", "medium", True)
        if int(row.get("has_respiratory_failure", 0) or 0) == 1 or int(row.get("has_ards", 0) or 0) == 1:
            _append_event(rows, sid, 0, "respiratory_failure_proxy", "diagnosis_pathway", "cohort_final", "medium", True)

        flags = comorb_map.get(sid, {})
        if int(flags.get("stroke_family", 0)) == 1:
            _append_event(rows, sid, 0, "stroke_cohort_membership", "diagnosis_pathway", "diagnoses_icd", "high", True)
        if int(flags.get("dementia", 0)) == 1:
            _append_event(rows, sid, 0, "cognitive_impairment_context", "diagnosis_pathway", "diagnoses_icd", "high", True)
    return rows


def _build_events_from_disease_timelines(path: Path, allowed_stays: set[int]) -> list[dict]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    rows: list[dict] = []
    for item in payload:
        try:
            sid = int(item["stay_id"])
        except Exception:
            continue
        if sid not in allowed_stays:
            continue
        timeline = item.get("disease_timeline") or {}
        primary = timeline.get("primary_disease") or "none"
        onset = timeline.get("onset_hour")
        if onset is not None:
            try:
                onset_f = float(onset)
                _append_event(rows, sid, onset_f, f"{primary}_onset_proxy", "onset", "disease_timelines", "low", True)
            except Exception:
                pass
        for phase in timeline.get("phases") or []:
            try:
                hour = float(phase.get("hour"))
            except Exception:
                continue
            _append_event(
                rows,
                sid,
                hour,
                f"{primary}_phase::{phase.get('phase', 'unknown')}",
                "phase",
                "disease_timelines",
                "low",
                True,
                {"severity": phase.get("severity")},
            )
    return rows


def _build_events_from_text_features(path: Path, allowed_stays: set[int]) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path)
    if "stay_id" not in df.columns:
        return []
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["stay_id"]).copy()
    df["stay_id"] = df["stay_id"].astype("int64")
    df = df[df["stay_id"].isin(allowed_stays)].copy()
    rows: list[dict] = []
    binary_cols = [col for col in df.columns if col != "stay_id"]
    for _, row in df.iterrows():
        sid = int(row["stay_id"])
        for col in binary_cols:
            try:
                value = int(row[col])
            except Exception:
                continue
            if value == 1:
                _append_event(
                    rows,
                    sid,
                    0,
                    f"text_documented::{col}",
                    "supporting_signal",
                    "patient_text_features",
                    "low",
                    True,
                )
    return rows


def main() -> None:
    args = parse_args()
    ensure_v3_directories()

    cohort = _load_cohort(Path(args.cohort_csv), args.stay_limit)
    comorb = _build_comorbidities(cohort)
    allowed_stays = set(int(v) for v in cohort["stay_id"].tolist())

    events = []
    events.extend(_build_events_from_cohort(cohort, comorb))
    events.extend(_build_events_from_disease_timelines(Path(args.disease_timelines_json), allowed_stays))
    events.extend(_build_events_from_text_features(Path(args.text_features_csv), allowed_stays))
    events_df = pd.DataFrame(events)
    if not events_df.empty:
        events_df = events_df.sort_values(["stay_id", "event_time_hour", "event_name"], kind="mergesort").drop_duplicates()

    comorb_out = Path(args.comorbidities_out)
    events_out = Path(args.events_out)
    comorb_out.parent.mkdir(parents=True, exist_ok=True)
    events_out.parent.mkdir(parents=True, exist_ok=True)
    comorb_written = write_table(comorb, comorb_out, index=False)
    events_written = write_table(events_df, events_out, index=False)
    print(f"Wrote {comorb_written}")
    print(f"Wrote {events_written}")


if __name__ == "__main__":
    main()
