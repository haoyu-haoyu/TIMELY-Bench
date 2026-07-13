#!/usr/bin/env python3
"""Print the public TIMELY-Bench V3 aggregate snapshot.

The script intentionally reads only tracked aggregate JSON/CSV files. It does
not require, infer, or reconstruct patient-level data.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(value) for value in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]
    template = "  ".join(f"{{:<{width}}}" for width in widths)
    print(template.format(*headers))
    print(template.format(*("-" * width for width in widths)))
    for row in rows:
        print(template.format(*row))


def main() -> None:
    cohort = read_json("results/v3/cohort_v3_meta.json")
    backbone = read_json("results/v3/structured_backbone_hourly_v3_meta.json")
    grid = read_json("results/v3/hourly_state_grid_168h_meta.json")
    cres = read_json("results/cres_v3/cres_master_manifest_summary.json")
    prompt = read_json("results/cres_v3/phase65b_prompt_build_summary.json")
    providers = read_csv(
        "results/cres_v3/phase65f_frozen_eval/phase65f_provider_metrics.csv"
    )
    judge = read_json(
        "results/cres_v3/phase65f_frozen_eval_local_final_sync/"
        "phase65f_judge_formal_summary.json"
    )

    print("TIMELY-Bench V3 public aggregate snapshot")
    print("=" * 42)
    print(f"Cohort: {cohort['n_stays']:,} ICU stays; {cohort['n_subjects']:,} subjects; "
          f"{cohort['n_hadm']:,} admissions")
    print(f"Structured backbone: {backbone['n_rows']:,} rows across "
          f"{backbone['n_parts']} parts")
    print(f"Hourly state grid: {grid['n_rows']:,} rows across {grid['n_parts']} parts; "
          f"horizon={grid['hours']}h")
    print(f"CRES manifest: {cres['manifest_rows']:,} rows; "
          f"{cres['unique_stays']:,} unique stays; {len(cres['tasks'])} task definitions")
    print(f"Evaluation sample: {prompt['sample_rows']:,} rows; "
          f"{prompt['prompt_rows']:,} prompts across {len(prompt['variants'])} variants")

    print("\nFrozen full-multimodal provider results")
    ranked = sorted(
        providers,
        key=lambda row: float(row["overall_macro_primary_score"]),
        reverse=True,
    )
    print_table(
        ["Rank", "Provider", "Tier", "Macro score", "Rows", "Parse success"],
        [
            [
                str(rank),
                row["provider"],
                row["tier"],
                f"{float(row['overall_macro_primary_score']):.6f}",
                f"{int(row['rows_actual']):,}",
                f"{int(row['parse_success_actual']):,}",
            ]
            for rank, row in enumerate(ranked, start=1)
        ],
    )

    print("\nFinal judge coverage")
    coverage_rows = [
        [
            row["judge_label"],
            row["judge_role"],
            f"{row['completed_ok_rows']:,}/{row['total_expected_rows']:,}",
            f"{row['avg_latency_seconds']:.4f}",
        ]
        for row in judge["judge_roster"]
    ]
    print_table(["Judge", "Role", "Coverage", "Mean latency (s)"], coverage_rows)

    print("\nBoundary: aggregate files above are public; patient-level inputs, filled "
          "prompts, canonical responses, per-instance scores, and judge rationales "
          "require controlled access.")


if __name__ == "__main__":
    main()
