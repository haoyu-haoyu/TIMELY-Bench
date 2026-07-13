#!/usr/bin/env python3
"""Generate the deterministic, wholly synthetic TIMELY-Bench fixture.

The records in this module were written specifically for software testing.  They
are not sampled, paraphrased, or statistically derived from MIMIC-IV records.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


FIXED_SEED = 20260713
CONDITIONS = ("aki", "delirium", "sepsis", "stroke")
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "fixtures"


def build_fixture(seed: int = FIXED_SEED) -> dict[str, Any]:
    """Return four manually authored synthetic cases.

    ``seed`` is recorded so downstream demonstrations can pin their own random
    operations to the fixture release.  The clinical examples themselves are
    intentionally fixed rather than randomly generated.
    """

    return {
        "schema_version": "1.0.0",
        "fixture_id": "timely-bench-public-synthetic-v1",
        "provenance": {
            "source": "Manually authored for TIMELY-Bench public software tests",
            "contains_real_patient_data": False,
            "contains_mimic_derived_data": False,
            "id_namespace": "SYNTHETIC-ONLY",
            "seed": seed,
            "time_semantics": (
                "All relative_hour values are measured from a fictional anchor at "
                "hour 0; negative values occur before the anchor."
            ),
        },
        "cases": [
            {
                "synthetic_id": "SYN-AKI-001",
                "condition": "aki",
                "anchor": {"name": "fictional_assessment", "relative_hour": 0},
                "events": [
                    {
                        "event_id": "SYN-AKI-E1",
                        "relative_hour": -12,
                        "feature": "creatinine",
                        "value": 0.8,
                        "unit": "mg/dL",
                    },
                    {
                        "event_id": "SYN-AKI-E2",
                        "relative_hour": -2,
                        "feature": "creatinine",
                        "value": 1.5,
                        "unit": "mg/dL",
                    },
                    {
                        "event_id": "SYN-AKI-E3",
                        "relative_hour": 0,
                        "feature": "urine_output_rate",
                        "value": 0.4,
                        "unit": "mL/kg/h",
                    },
                ],
                "notes": [
                    {
                        "note_id": "SYN-AKI-N1",
                        "relative_hour": -1,
                        "note_type": "fictional_progress_note",
                        "text": (
                            "Entirely fictional note: urine output has fallen during "
                            "this invented shift; repeat renal markers are planned."
                        ),
                    }
                ],
                "task_targets": [
                    {
                        "task_id": "aki_change_by_anchor",
                        "value": True,
                        "evidence_event_ids": ["SYN-AKI-E1", "SYN-AKI-E2"],
                    }
                ],
            },
            {
                "synthetic_id": "SYN-DELIRIUM-001",
                "condition": "delirium",
                "anchor": {"name": "fictional_assessment", "relative_hour": 0},
                "events": [
                    {
                        "event_id": "SYN-DEL-E1",
                        "relative_hour": -8,
                        "feature": "attention_screen",
                        "value": "attentive",
                        "unit": "category",
                    },
                    {
                        "event_id": "SYN-DEL-E2",
                        "relative_hour": -1,
                        "feature": "attention_screen",
                        "value": "inattentive",
                        "unit": "category",
                    },
                ],
                "notes": [
                    {
                        "note_id": "SYN-DEL-N1",
                        "relative_hour": -1,
                        "note_type": "fictional_nursing_note",
                        "text": (
                            "Entirely fictional note: the invented patient loses track "
                            "of a simple attention task and is intermittently disoriented."
                        ),
                    }
                ],
                "task_targets": [
                    {
                        "task_id": "delirium_state_at_anchor",
                        "value": "present",
                        "evidence_event_ids": ["SYN-DEL-E2"],
                    }
                ],
            },
            {
                "synthetic_id": "SYN-SEPSIS-001",
                "condition": "sepsis",
                "anchor": {"name": "fictional_assessment", "relative_hour": 0},
                "events": [
                    {
                        "event_id": "SYN-SEP-E1",
                        "relative_hour": -6,
                        "feature": "temperature",
                        "value": 39.1,
                        "unit": "degC",
                    },
                    {
                        "event_id": "SYN-SEP-E2",
                        "relative_hour": -3,
                        "feature": "lactate",
                        "value": 3.2,
                        "unit": "mmol/L",
                    },
                    {
                        "event_id": "SYN-SEP-E3",
                        "relative_hour": 0,
                        "feature": "mean_arterial_pressure",
                        "value": 62,
                        "unit": "mmHg",
                    },
                ],
                "notes": [
                    {
                        "note_id": "SYN-SEP-N1",
                        "relative_hour": -2,
                        "note_type": "fictional_progress_note",
                        "text": (
                            "Entirely fictional note: fever and low blood pressure in "
                            "this toy scenario prompted cultures and an antimicrobial plan."
                        ),
                    }
                ],
                "task_targets": [
                    {
                        "task_id": "sepsis_pattern_by_anchor",
                        "value": "compatible",
                        "evidence_event_ids": ["SYN-SEP-E1", "SYN-SEP-E2", "SYN-SEP-E3"],
                    }
                ],
            },
            {
                "synthetic_id": "SYN-STROKE-001",
                "condition": "stroke",
                "anchor": {"name": "fictional_assessment", "relative_hour": 0},
                "events": [
                    {
                        "event_id": "SYN-STR-E1",
                        "relative_hour": -4,
                        "feature": "left_arm_strength",
                        "value": 5,
                        "unit": "ordinal_0_to_5",
                    },
                    {
                        "event_id": "SYN-STR-E2",
                        "relative_hour": -1,
                        "feature": "left_arm_strength",
                        "value": 2,
                        "unit": "ordinal_0_to_5",
                    },
                    {
                        "event_id": "SYN-STR-E3",
                        "relative_hour": 0,
                        "feature": "speech_clarity",
                        "value": "slurred",
                        "unit": "category",
                    },
                ],
                "notes": [
                    {
                        "note_id": "SYN-STR-N1",
                        "relative_hour": -1,
                        "note_type": "fictional_neurology_note",
                        "text": (
                            "Entirely fictional note: new left arm weakness and slurred "
                            "speech are documented for this fabricated example."
                        ),
                    }
                ],
                "task_targets": [
                    {
                        "task_id": "stroke_change_by_anchor",
                        "value": "new_focal_change",
                        "evidence_event_ids": ["SYN-STR-E1", "SYN-STR-E2", "SYN-STR-E3"],
                    }
                ],
            },
        ],
    }


def validate_fixture(payload: dict[str, Any]) -> list[str]:
    """Validate invariants that matter for safe, anchor-bounded examples."""

    errors: list[str] = []
    provenance = payload.get("provenance", {})
    if provenance.get("contains_real_patient_data") is not False:
        errors.append("provenance must declare contains_real_patient_data=false")
    if provenance.get("contains_mimic_derived_data") is not False:
        errors.append("provenance must declare contains_mimic_derived_data=false")

    cases = payload.get("cases")
    if not isinstance(cases, list):
        return errors + ["cases must be a list"]
    condition_counts = Counter(case.get("condition") for case in cases)
    if condition_counts != Counter(CONDITIONS):
        errors.append(f"expected exactly one case per condition; found {dict(condition_counts)}")

    case_ids: set[str] = set()
    for case in cases:
        case_id = case.get("synthetic_id")
        if not isinstance(case_id, str) or not case_id.startswith("SYN-"):
            errors.append(f"invalid synthetic_id: {case_id!r}")
            continue
        if case_id in case_ids:
            errors.append(f"duplicate synthetic_id: {case_id}")
        case_ids.add(case_id)

        anchor_hour = case.get("anchor", {}).get("relative_hour")
        if anchor_hour != 0:
            errors.append(f"{case_id}: anchor relative_hour must equal 0")
        event_ids: set[str] = set()
        for event in case.get("events", []):
            event_id = event.get("event_id")
            if not isinstance(event_id, str) or not event_id.startswith("SYN-"):
                errors.append(f"{case_id}: invalid event_id {event_id!r}")
            elif event_id in event_ids:
                errors.append(f"{case_id}: duplicate event_id {event_id}")
            else:
                event_ids.add(event_id)
            if not _is_anchor_bounded(event.get("relative_hour")):
                errors.append(f"{case_id}: event occurs after the anchor")

        for note in case.get("notes", []):
            if not str(note.get("note_id", "")).startswith("SYN-"):
                errors.append(f"{case_id}: note_id is not synthetic")
            if not _is_anchor_bounded(note.get("relative_hour")):
                errors.append(f"{case_id}: note occurs after the anchor")
            if not str(note.get("text", "")).startswith("Entirely fictional note:"):
                errors.append(f"{case_id}: note lacks the fictional-content marker")

        for target in case.get("task_targets", []):
            unknown = set(target.get("evidence_event_ids", [])) - event_ids
            if unknown:
                errors.append(f"{case_id}: target references unknown events {sorted(unknown)}")

    return errors


def _is_anchor_bounded(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value <= 0


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_artifacts(seed: int = FIXED_SEED) -> dict[str, bytes]:
    fixture = build_fixture(seed)
    errors = validate_fixture(fixture)
    if errors:
        raise ValueError("invalid built-in fixture: " + "; ".join(errors))
    fixture_bytes = _json_bytes(fixture)
    cases = fixture["cases"]
    all_hours = [
        item["relative_hour"]
        for case in cases
        for group in ("events", "notes")
        for item in case[group]
    ]
    summary = {
        "schema_version": fixture["schema_version"],
        "fixture_id": fixture["fixture_id"],
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "case_count": len(cases),
        "condition_counts": dict(sorted(Counter(case["condition"] for case in cases).items())),
        "event_count": sum(len(case["events"]) for case in cases),
        "note_count": sum(len(case["notes"]) for case in cases),
        "relative_hour_min": min(all_hours),
        "relative_hour_max": max(all_hours),
        "all_observations_anchor_bounded": all(hour <= 0 for hour in all_hours),
        "contains_real_patient_data": False,
        "contains_mimic_derived_data": False,
    }
    return {
        "synthetic_cases.json": fixture_bytes,
        "golden_summary.json": _json_bytes(summary),
    }


def write_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in artifacts.items():
        (output_dir / name).write_bytes(content)


def check_artifacts(output_dir: Path, artifacts: dict[str, bytes]) -> list[str]:
    mismatches: list[str] = []
    for name, expected in artifacts.items():
        path = output_dir / name
        if not path.is_file():
            mismatches.append(f"missing {path}")
        elif path.read_bytes() != expected:
            mismatches.append(f"content differs from deterministic generator: {path}")
    return mismatches


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=FIXED_SEED)
    parser.add_argument(
        "--check",
        action="store_true",
        help="compare existing artifacts with the deterministic generator without writing",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifacts = build_artifacts(args.seed)
    if args.check:
        mismatches = check_artifacts(args.output_dir, artifacts)
        if mismatches:
            for mismatch in mismatches:
                print(f"ERROR: {mismatch}", file=sys.stderr)
            return 1
        print(f"OK: {len(artifacts)} synthetic artifacts match the deterministic generator.")
        return 0
    write_artifacts(args.output_dir, artifacts)
    print(f"WROTE: {len(artifacts)} deterministic synthetic artifacts to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
