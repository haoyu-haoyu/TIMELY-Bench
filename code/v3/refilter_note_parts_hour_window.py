#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import ROOT_DIR, V3_RAW_DATA_DIR, ensure_v3_directories  # type: ignore
from v3.io_utils import chunk_dir_path, relativize_value, write_table  # type: ignore


NOTE_OUTPUTS = {
    "discharge": "discharge_notes_v3.parquet",
    "nursing": "nursing_notes_168h.parquet",
    "lab_comment": "lab_comments_168h.parquet",
    "radiology": "radiology_notes_168h.parquet",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Refilter existing v3 note parts to the strict hour window.")
    p.add_argument(
        "--kinds",
        nargs="+",
        default=["nursing", "lab_comment", "radiology"],
        choices=sorted(NOTE_OUTPUTS),
        help="Note source kinds to refilter in place.",
    )
    p.add_argument("--hours", type=int, default=168)
    p.add_argument("--out-dir", default=str(V3_RAW_DATA_DIR))
    p.add_argument("--meta-json", default=str(V3_RAW_DATA_DIR / "extract_notes_bq_meta.json"))
    return p.parse_args()


def _filter_part(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    out = df.copy()
    if "hour_offset" not in out.columns:
        return out
    out["hour_offset"] = pd.to_numeric(out["hour_offset"], errors="coerce")
    out = out[out["hour_offset"].between(0, int(hours) - 1, inclusive="both")].copy()
    sort_cols = [col for col in ["stay_id", "charttime"] if col in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    out_dir = Path(args.out_dir)
    meta_path = Path(args.meta_json)
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    refilter_summary: dict[str, dict] = {}
    for kind in args.kinds:
        if kind == "discharge":
            continue
        out_path = out_dir / NOTE_OUTPUTS[kind]
        parts_dir = chunk_dir_path(out_path)
        if not parts_dir.exists():
            raise FileNotFoundError(parts_dir)

        n_rows_before = 0
        n_rows_after = 0
        n_rows_removed = 0
        n_parts = 0

        for part in sorted(parts_dir.glob("part_*.*")):
            if part.suffix not in {".parquet", ".csv"}:
                continue
            n_parts += 1
            if part.suffix == ".parquet":
                df = pd.read_parquet(part)
            else:
                df = pd.read_csv(part, low_memory=False)
            before = int(len(df))
            filtered = _filter_part(df, hours=int(args.hours))
            after = int(len(filtered))
            n_rows_before += before
            n_rows_after += after
            n_rows_removed += before - after
            write_table(filtered, part, index=False)

        refilter_summary[kind] = {
            "path": str(out_path),
            "n_parts": n_parts,
            "n_rows_before": n_rows_before,
            "n_rows_after": n_rows_after,
            "n_rows_removed": n_rows_removed,
            "hours": int(args.hours),
        }
        if meta.get("outputs", {}).get(kind):
            meta["outputs"][kind]["path"] = str(out_path)
            meta["outputs"][kind]["n_rows"] = n_rows_after
            meta["outputs"][kind]["n_parts"] = n_parts

    if meta:
        meta["refilter_summary"] = refilter_summary
        meta_path.write_text(
            json.dumps(relativize_value(meta, root=ROOT_DIR), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(json.dumps(relativize_value(refilter_summary, root=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
