#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v3.constants import (  # type: ignore
    ROOT_DIR,
    DEFAULT_HOURLY_STATE_GRID,
    DEFAULT_PROCEDURE_EVENTS,
    V3_PROCESSED_DIR,
    ensure_v3_directories,
)
from v3.io_utils import chunk_dir_path, iter_table_chunks, read_table, relativize_value, write_table  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build TIMELY-Bench v3 ARF cohort and labels.")
    p.add_argument("--hourly-grid", default=str(DEFAULT_HOURLY_STATE_GRID))
    p.add_argument("--procedure-events", default=str(DEFAULT_PROCEDURE_EVENTS))
    p.add_argument("--pf-threshold", type=float, default=300.0)
    p.add_argument("--prediction-interval", type=int, default=4)
    p.add_argument("--escalation-lookahead", type=int, default=24)
    p.add_argument("--extubation-lookahead", type=int, default=72)
    p.add_argument("--prolonged-hour", type=int, default=168)
    p.add_argument("--stay-limit", type=int, default=None)
    p.add_argument("--cohort-out", default=str(V3_PROCESSED_DIR / "arf" / "arf_cohort_v3.parquet"))
    p.add_argument("--labels-out", default=str(V3_PROCESSED_DIR / "arf" / "arf_labels_v3.parquet"))
    p.add_argument("--summary-json", default=str(V3_PROCESSED_DIR / "arf" / "arf_label_summary_v3.json"))
    p.add_argument("--resume", action="store_true", help="Resume from existing part files instead of rebuilding from scratch.")
    return p.parse_args()


def _load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["stay_id", "event_time_hour", "event_name", "event_type"])
    df = read_table(path)
    if "event_time_hour" not in df.columns and "event_start_hour" in df.columns:
        df = df.rename(columns={"event_start_hour": "event_time_hour"})
    return df


def main() -> None:
    args = parse_args()
    ensure_v3_directories()
    events = _load_events(Path(args.procedure_events))
    cohort_path = Path(args.cohort_out)
    labels_path = Path(args.labels_out)
    summary_path = Path(args.summary_json)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cohort_parts_dir = chunk_dir_path(cohort_path)
    labels_parts_dir = chunk_dir_path(labels_path)
    if cohort_parts_dir.exists() and not args.resume:
        shutil.rmtree(cohort_parts_dir)
    if labels_parts_dir.exists() and not args.resume:
        shutil.rmtree(labels_parts_dir)
    cohort_parts_dir.mkdir(parents=True, exist_ok=True)
    labels_parts_dir.mkdir(parents=True, exist_ok=True)

    intub_df = events[events.get("event_name", pd.Series(dtype=str)).astype(str).str.contains("intubation", case=False, na=False)].copy() if not events.empty and "event_name" in events.columns else pd.DataFrame(columns=["stay_id", "event_time_hour"])
    extub_df = events[events.get("event_name", pd.Series(dtype=str)).astype(str).str.contains("extubation", case=False, na=False)].copy() if not events.empty and "event_name" in events.columns else pd.DataFrame(columns=["stay_id", "event_time_hour"])
    for chunk_idx, grid in enumerate(iter_table_chunks(args.hourly_grid), start=1):
        cohort_part = cohort_parts_dir / f"part_{chunk_idx:05d}.parquet"
        labels_part = labels_parts_dir / f"part_{chunk_idx:05d}.parquet"
        if args.resume and cohort_part.exists() and labels_part.exists():
            print(f"Skipping existing {cohort_part.name} and {labels_part.name}")
            continue
        if args.stay_limit is not None:
            keep = grid["stay_id"].drop_duplicates().head(int(args.stay_limit)).tolist()
            grid = grid[grid["stay_id"].isin(keep)].copy()
        for col in ["pao2_fio2_ratio", "fio2", "peep", "ventilation_status"]:
            if col not in grid.columns:
                grid[col] = pd.NA
        grid["pao2_fio2_ratio"] = pd.to_numeric(grid["pao2_fio2_ratio"], errors="coerce")
        grid["fio2"] = pd.to_numeric(grid["fio2"], errors="coerce")
        grid["peep"] = pd.to_numeric(grid["peep"], errors="coerce")
        grid["ventilation_status"] = grid["ventilation_status"].fillna("none").astype(str)

        cohort_rows = []
        label_rows = []
        for stay_id, g in grid.sort_values(["stay_id", "hour"], kind="mergesort").groupby("stay_id", sort=False):
            g = g.copy()
            pf_mask = g["pao2_fio2_ratio"].notna() & (g["pao2_fio2_ratio"] < float(args.pf_threshold))
            pf_hours = g.loc[pf_mask, "hour"]
            invasive_hours = g.loc[g["ventilation_status"].str.lower().str.contains("invasive", na=False), "hour"]
            event_hours = pd.to_numeric(intub_df.loc[intub_df["stay_id"] == stay_id, "event_time_hour"], errors="coerce").dropna()
            onset_candidates = []
            if not pf_hours.empty:
                onset_candidates.append(int(pf_hours.min()))
            if not invasive_hours.empty:
                onset_candidates.append(int(invasive_hours.min()))
            if not event_hours.empty:
                onset_candidates.append(int(event_hours.min()))
            if not onset_candidates:
                continue
            onset_hour = min(onset_candidates)
            cohort_rows.append(
                {
                    "stay_id": int(stay_id),
                    "arf_onset_hour": int(onset_hour),
                    "onset_from_pf_ratio": int(not pf_hours.empty and int(pf_hours.min()) == onset_hour),
                    "onset_from_invasive_vent": int((not invasive_hours.empty and int(invasive_hours.min()) == onset_hour) or (not event_hours.empty and int(event_hours.min()) == onset_hour)),
                }
            )
            g2 = g.set_index("hour").reindex(range(int(g["hour"].max()) + 1))
            fio2 = pd.to_numeric(g2["fio2"], errors="coerce")
            peep = pd.to_numeric(g2["peep"], errors="coerce")
            vent = g2["ventilation_status"].fillna("none").astype(str)
            upper_t = int(g["hour"].max())
            t = onset_hour + int(args.prediction_interval)
            while t <= upper_t:
                t_fio2 = fio2.iloc[t] if t < len(fio2) else np.nan
                t_peep = peep.iloc[t] if t < len(peep) else np.nan
                esc_upper = min(upper_t, t + int(args.escalation_lookahead))
                future_fio2 = fio2.iloc[t + 1 : esc_upper + 1]
                future_peep = peep.iloc[t + 1 : esc_upper + 1]
                future_vent = vent.iloc[t + 1 : esc_upper + 1]
                escalation = int(
                    ((future_fio2 - t_fio2).fillna(-999) >= 0.10).any()
                    or ((future_peep - t_peep).fillna(-999) >= 3.0).any()
                    or future_vent.str.lower().str.contains("invasive", na=False).any()
                )
                ext_upper = min(upper_t, t + int(args.extubation_lookahead))
                ext_times = pd.to_numeric(extub_df.loc[extub_df["stay_id"] == stay_id, "event_time_hour"], errors="coerce").dropna()
                successful_ext = int(((ext_times > t) & (ext_times <= ext_upper)).any()) if not ext_times.empty else 0
                prolonged = int((t <= int(args.prolonged_hour)) and (vent.iloc[min(int(args.prolonged_hour), upper_t)].lower() != "none"))
                label_rows.append(
                    {
                        "stay_id": int(stay_id),
                        "prediction_hour": int(t),
                        "arf_onset_hour": int(onset_hour),
                        "label_escalation_24h": escalation,
                        "label_successful_extubation_72h": successful_ext,
                        "label_prolonged_ventilation_168h": prolonged,
                    }
                )
                t += int(args.prediction_interval)

        cohort_df = pd.DataFrame(cohort_rows)
        labels_df = pd.DataFrame(label_rows)
        cohort_written = write_table(cohort_df, cohort_part, index=False)
        labels_written = write_table(labels_df, labels_part, index=False)
        print(f"Wrote {cohort_written}")
        print(f"Wrote {labels_written}")
    cohort_all = read_table(cohort_path)
    labels_all = read_table(labels_path)
    cohort_total = int(len(cohort_all))
    label_rows_total = int(len(labels_all))
    escalation_sum = int(labels_all["label_escalation_24h"].sum()) if label_rows_total else 0
    successful_ext_sum = int(labels_all["label_successful_extubation_72h"].sum()) if label_rows_total else 0
    prolonged_sum = int(labels_all["label_prolonged_ventilation_168h"].sum()) if label_rows_total else 0
    part_count = len(list(cohort_parts_dir.glob("part_*.parquet")))
    summary = {
        "arf_cohort_stays": cohort_total,
        "label_rows": label_rows_total,
        "escalation_positive_rate": (escalation_sum / label_rows_total) if label_rows_total else None,
        "successful_extubation_positive_rate": (successful_ext_sum / label_rows_total) if label_rows_total else None,
        "prolonged_ventilation_positive_rate": (prolonged_sum / label_rows_total) if label_rows_total else None,
        "parts": part_count,
        "outputs": relativize_value({"cohort_parts": str(cohort_parts_dir), "labels_parts": str(labels_parts_dir)}, root=ROOT_DIR),
    }
    summary_path.write_text(
        json.dumps(relativize_value(summary, root=ROOT_DIR), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
