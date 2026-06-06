#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    DEFAULT_FEATURE_DICTIONARY_CSV,
    DEFAULT_FEATURE_DICTIONARY_JSON,
    DEFAULT_FEATURE_DICTIONARY_MD,
    ensure_v3_directories,
)
from v3.feature_spec import FEATURE_SPECS, grouped_feature_records  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the TIMELY-Bench v3 feature dictionary.")
    p.add_argument("--json-out", default=str(DEFAULT_FEATURE_DICTIONARY_JSON))
    p.add_argument("--csv-out", default=str(DEFAULT_FEATURE_DICTIONARY_CSV))
    p.add_argument("--md-out", default=str(DEFAULT_FEATURE_DICTIONARY_MD))
    return p.parse_args()


def _write_csv(path: Path) -> None:
    rows = []
    for spec in FEATURE_SPECS:
        row = spec.__dict__.copy()
        row["itemids"] = ",".join(str(v) for v in spec.itemids)
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path) -> None:
    data = {
        "summary": {
            "n_features": len(FEATURE_SPECS),
            "by_validation_status": dict(Counter(spec.validation_status for spec in FEATURE_SPECS)),
            "backbone_features": [spec.name for spec in FEATURE_SPECS if spec.backbone],
        },
        "domains": grouped_feature_records(),
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(path: Path) -> None:
    lines = [
        "# TIMELY-Bench v3 Feature Dictionary",
        "",
        f"- Total features: **{len(FEATURE_SPECS)}**",
        "",
        "## Validation Summary",
        "",
    ]
    counts = Counter(spec.validation_status for spec in FEATURE_SPECS)
    for status, count in sorted(counts.items()):
        lines.append(f"- `{status}`: {count}")
    lines.append("")
    for domain, rows in grouped_feature_records().items():
        lines.append(f"## {domain}")
        lines.append("")
        lines.append("| Feature | Source | Acquisition mode | Validation status | Use |")
        lines.append("|---|---|---|---|---|")
        for row in rows:
            lines.append(
                f"| {row['name']} | {row['source']} | {row['acquisition_mode']} | {row['validation_status']} | {row['use']} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Field Definitions",
            "",
            "- `unit`: canonical output unit where applicable.",
            "- `normalization_rule`: explicit merge, conversion, or derivation rule used in v3.",
            "- `missingness_expectation`: expected sparsity profile before modeling or imputation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    json_out = Path(args.json_out)
    csv_out = Path(args.csv_out)
    md_out = Path(args.md_out)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(json_out)
    _write_csv(csv_out)
    _write_markdown(md_out)
    print(f"Wrote {json_out}")
    print(f"Wrote {csv_out}")
    print(f"Wrote {md_out}")


if __name__ == "__main__":
    main()
