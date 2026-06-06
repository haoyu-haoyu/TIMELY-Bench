#!/usr/bin/env python3
"""
Validation checks for Phase 2 note-centered alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import ROOT_DIR  # type: ignore


WINDOWS = ["W6", "W12", "W24", "D0", "leaked"]
MISSING_NOTE_STAYS = [30635125, 39438562, 39443966]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate note-centered alignment outputs.")
    p.add_argument(
        "--note-centered-dir",
        default=str(ROOT_DIR / "data" / "processed" / "note_centered"),
    )
    p.add_argument(
        "--stay-level-dir",
        default=str(ROOT_DIR / "data" / "processed" / "note_centered" / "stay_level"),
    )
    p.add_argument(
        "--metadata-csv",
        default=str(ROOT_DIR / "data" / "processed" / "text_embeddings" / "note_level_metadata_48h.csv"),
    )
    p.add_argument(
        "--timeseries-csv",
        default=str(ROOT_DIR / "data" / "processed" / "timeseries_sorted_72h.csv"),
    )
    p.add_argument(
        "--cohort-csv",
        default=str(ROOT_DIR / "data" / "raw" / "cohort_with_conditions.csv"),
        help="Used with --max-stays to validate smoke subsets consistently.",
    )
    p.add_argument(
        "--out-json",
        default=str(ROOT_DIR / "results" / "audit" / "phase2_note_centered_validation_summary.json"),
    )
    p.add_argument("--sample-size", type=int, default=100)
    p.add_argument("--max-stays", type=int, default=None, help="Optional debug subset size.")
    return p.parse_args()


def _iter_row_groups(path: Path, columns: List[str]):
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        yield pf.read_row_group(rg, columns=columns).to_pandas()


def _parquet_rows(path: Path) -> int:
    return int(pq.ParquetFile(path).metadata.num_rows)


def _sample_rows(path: Path, columns: List[str], max_rows: int) -> pd.DataFrame:
    out = []
    got = 0
    for df in _iter_row_groups(path, columns=columns):
        if df.empty:
            continue
        remain = max_rows - got
        if remain <= 0:
            break
        part = df.head(remain)
        out.append(part)
        got += len(part)
        if got >= max_rows:
            break
    if not out:
        return pd.DataFrame(columns=columns)
    return pd.concat(out, ignore_index=True)


def _load_selected_stays(cohort_csv: Path, max_stays: Optional[int]) -> Optional[Set[int]]:
    if max_stays is None or max_stays <= 0:
        return None
    cohort = pd.read_csv(cohort_csv, usecols=["stay_id"])
    stays = cohort["stay_id"].dropna().astype(np.int64).drop_duplicates().head(max_stays).tolist()
    return set(int(s) for s in stays)


def _contains_any_stay(path: Path, target_stays: Set[int]) -> bool:
    if not target_stays:
        return False
    for df in _iter_row_groups(path, columns=["stay_id"]):
        if df.empty:
            continue
        s = pd.to_numeric(df["stay_id"], errors="coerce").dropna().astype(np.int64)
        if any(int(x) in target_stays for x in s.tolist()):
            return True
    return False


def main() -> int:
    args = parse_args()
    note_centered_dir = Path(args.note_centered_dir)
    stay_level_dir = Path(args.stay_level_dir)
    metadata_csv = Path(args.metadata_csv)
    timeseries_csv = Path(args.timeseries_csv)
    cohort_csv = Path(args.cohort_csv)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    selected_stays = _load_selected_stays(cohort_csv=cohort_csv, max_stays=args.max_stays)

    result: Dict[str, object] = {
        "inputs": {
            "note_centered_dir": str(note_centered_dir),
            "stay_level_dir": str(stay_level_dir),
            "metadata_csv": str(metadata_csv),
            "timeseries_csv": str(timeseries_csv),
            "cohort_csv": str(cohort_csv),
            "max_stays": args.max_stays,
        },
        "tests": {},
    }

    # Files present
    files_ok = True
    structured_files = {}
    for w in WINDOWS:
        p = note_centered_dir / f"note_window_structured_{w}.parquet"
        structured_files[w] = str(p)
        if not p.exists():
            files_ok = False
    result["tests"]["files_exist"] = {"pass": files_ok, "details": structured_files}
    if not files_ok:
        out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1

    # Load note metadata count (same filters used by feature builders)
    meta = pd.read_csv(metadata_csv, usecols=["stay_id", "note_idx", "note_type", "chart_hour"])
    meta["stay_id"] = pd.to_numeric(meta["stay_id"], errors="coerce").fillna(-1).astype(np.int64)
    meta = meta[(meta["stay_id"] >= 0) & (meta["chart_hour"].notna())].copy()
    if selected_stays is not None:
        meta = meta[meta["stay_id"].isin(selected_stays)].copy()
    note_count = int(len(meta))
    rows_per_window = {w: _parquet_rows(Path(structured_files[w])) for w in WINDOWS}
    all_notes_all_windows = all(rows_per_window[w] == note_count for w in WINDOWS)
    result["tests"]["all_notes_all_windows"] = {
        "pass": all_notes_all_windows,
        "details": {"note_count": note_count, "rows_per_window": rows_per_window},
    }

    # Boundary checks (scan row-groups, metadata columns only)
    no_future_viol = 0
    d0_end_viol = 0
    d0_start_viol = 0
    leaked_future_rows = 0
    flag_mismatch = 0
    for w in WINDOWS:
        p = Path(structured_files[w])
        cols = ["chart_hour", "window_start", "window_end", "contains_future_data", "window_hours_actual"]
        for df in _iter_row_groups(p, columns=cols):
            if df.empty:
                continue
            chart = pd.to_numeric(df["chart_hour"], errors="coerce")
            ws = pd.to_numeric(df["window_start"], errors="coerce")
            we = pd.to_numeric(df["window_end"], errors="coerce")
            if w != "leaked":
                no_future_viol += int((we > chart).sum())
            else:
                leaked_future_rows += int((we > chart).sum())
            if w == "D0":
                d0_end_viol += int((we != chart).sum())
                d0_expect = (np.floor(chart / 24.0) * 24.0)
                d0_start_viol += int((ws != d0_expect).sum())

            flag = df["contains_future_data"].astype(bool)
            if w == "leaked":
                flag_mismatch += int((~flag).sum())
            else:
                flag_mismatch += int(flag.sum())

    result["tests"]["no_future_data_in_lookback"] = {
        "pass": no_future_viol == 0,
        "details": {"violations": no_future_viol},
    }
    result["tests"]["d0_no_future_data"] = {
        "pass": d0_end_viol == 0,
        "details": {"violations": d0_end_viol},
    }
    result["tests"]["d0_calendar_day_start"] = {
        "pass": d0_start_viol == 0,
        "details": {"violations": d0_start_viol},
    }
    result["tests"]["leaked_contains_future"] = {
        "pass": leaked_future_rows > 0,
        "details": {"rows_with_future": leaked_future_rows},
    }
    result["tests"]["contains_future_data_flag"] = {
        "pass": flag_mismatch == 0,
        "details": {"mismatch_rows": flag_mismatch},
    }

    # leaked vs W24 difference (sample)
    w24_p = Path(structured_files["W24"])
    lk_p = Path(structured_files["leaked"])
    # include a few feature columns
    w24_head = pd.read_parquet(w24_p, columns=None).head(1)
    feat_cols = [c for c in w24_head.columns if c.endswith("_mean")][:5]
    mean_to_base = {c: c[: -len("_mean")] for c in feat_cols}
    join_cols = ["stay_id", "note_idx"] + feat_cols
    w24_sample = _sample_rows(w24_p, columns=join_cols, max_rows=10000)
    lk_sample = _sample_rows(lk_p, columns=join_cols, max_rows=10000)
    merged = w24_sample.merge(lk_sample, on=["stay_id", "note_idx"], suffixes=("_w24", "_leaked"))
    diff_any = False
    if not merged.empty and feat_cols:
        for c in feat_cols:
            a = pd.to_numeric(merged[f"{c}_w24"], errors="coerce")
            b = pd.to_numeric(merged[f"{c}_leaked"], errors="coerce")
            if ((a - b).abs() > 1e-6).any():
                diff_any = True
                break
    result["tests"]["leaked_vs_clean_difference"] = {
        "pass": diff_any,
        "details": {"merged_sample_rows": int(len(merged)), "feature_sample": feat_cols},
    }

    # Feature math correctness on sample (W24)
    base_vars = [mean_to_base[c] for c in feat_cols]
    ts = pd.read_csv(timeseries_csv, usecols=["stay_id", "hour"] + base_vars)
    ts["stay_id"] = pd.to_numeric(ts["stay_id"], errors="coerce").fillna(-1).astype(np.int64)
    ts["hour"] = pd.to_numeric(ts["hour"], errors="coerce").fillna(-1).astype(np.int16)
    if selected_stays is not None:
        ts = ts[ts["stay_id"].isin(selected_stays)].copy()
    w24_math = _sample_rows(
        w24_p,
        columns=["stay_id", "note_idx", "window_start", "window_end"] + feat_cols,
        max_rows=max(1000, args.sample_size * 20),
    )
    if len(w24_math) > args.sample_size:
        w24_math = w24_math.sample(args.sample_size, random_state=42)
    math_fail = 0
    for row in w24_math.itertuples(index=False):
        sid = int(row.stay_id)
        ws = int(row.window_start)
        we = int(row.window_end)
        seg = ts[(ts["stay_id"] == sid) & (ts["hour"] >= ws) & (ts["hour"] <= we)]
        if seg.empty:
            continue
        for c in feat_cols:
            base_col = mean_to_base[c]
            target = getattr(row, c)
            actual = pd.to_numeric(seg[base_col], errors="coerce").mean()
            if pd.notna(target) and pd.notna(actual):
                if abs(float(target) - float(actual)) > 1e-4:
                    math_fail += 1
                    break
    result["tests"]["feature_math_correctness"] = {
        "pass": math_fail == 0,
        "details": {"sample_rows": int(len(w24_math)), "fail_rows": int(math_fail), "checked_feature": "mean"},
    }

    # Missing stays handling
    if selected_stays is None:
        target_missing = set(MISSING_NOTE_STAYS)
    else:
        target_missing = set(int(s) for s in MISSING_NOTE_STAYS if int(s) in selected_stays)

    if not target_missing:
        result["tests"]["missing_stays_handled"] = {
            "pass": True,
            "details": {"not_applicable_in_subset": True, "target_missing_stays": []},
        }
    else:
        missing_in_note_level = True
        for w in WINDOWS:
            if _contains_any_stay(Path(structured_files[w]), target_missing):
                missing_in_note_level = False
                break

        text_w24 = stay_level_dir / "text_W24_original.parquet"
        structured_w24 = stay_level_dir / "structured_W24.parquet"
        missing_present = False
        missing_flags_ok = False
        if text_w24.exists() and structured_w24.exists():
            tdf = pd.read_parquet(
                text_w24,
                columns=["stay_id", "text_has_notes"] + [f"emb_{i:04d}" for i in range(8)],
            )
            sdf = pd.read_parquet(structured_w24, columns=["stay_id", "has_notes"])
            chk = sdf.merge(tdf, on="stay_id", how="inner")
            chk = chk[chk["stay_id"].isin(list(target_missing))].copy()
            if len(chk) == len(target_missing):
                missing_present = True
                zero_ok = True
                for _, r in chk.iterrows():
                    if bool(r["has_notes"]) is True:
                        zero_ok = False
                    if bool(r["text_has_notes"]) is True:
                        zero_ok = False
                missing_flags_ok = zero_ok

        result["tests"]["missing_stays_handled"] = {
            "pass": bool(missing_in_note_level and missing_present and missing_flags_ok),
            "details": {
                "target_missing_stays": sorted(int(x) for x in target_missing),
                "not_in_note_level": missing_in_note_level,
                "present_in_stay_level": missing_present,
                "flags_ok": missing_flags_ok,
            },
        }

    # D0 boundary truncation distribution (informational)
    d0_total = 0
    d0_lt2 = 0
    for df in _iter_row_groups(Path(structured_files["D0"]), columns=["window_hours_actual"]):
        vals = pd.to_numeric(df["window_hours_actual"], errors="coerce")
        d0_total += int(vals.notna().sum())
        d0_lt2 += int((vals < 2.0).sum())
    frac_lt2 = (d0_lt2 / d0_total) if d0_total else None
    result["tests"]["d0_boundary_truncation_distribution"] = {
        "pass": True,
        "details": {"total_rows": d0_total, "rows_lt_2h": d0_lt2, "fraction_lt_2h": frac_lt2},
    }

    critical = [
        "files_exist",
        "all_notes_all_windows",
        "no_future_data_in_lookback",
        "d0_no_future_data",
        "d0_calendar_day_start",
        "leaked_contains_future",
        "contains_future_data_flag",
        "feature_math_correctness",
        "missing_stays_handled",
    ]
    result["all_pass"] = all(bool(result["tests"][k]["pass"]) for k in critical)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Wrote summary: {out_json}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
