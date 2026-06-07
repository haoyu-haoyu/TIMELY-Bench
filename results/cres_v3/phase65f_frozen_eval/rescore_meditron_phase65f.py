#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path("${PROJECT_ROOT}")
OUTPUT_DIR = Path("${PROJECT_ROOT}/results/cres_v3/phase65f_frozen_eval")

sys.path.insert(0, str(PROJECT_ROOT / "code" / "v3"))

from run_phase65c_tier1a_scoring_v3 import build_joined_frame, derive_truth_and_prediction, normalize_text
from run_phase65f_frozen_eval_v1 import (
    iter_jsonl,
    load_registry,
    normalize_confidence_value,
    normalize_phase65f_answer,
    row_correctness,
)


def main() -> None:
    registry = load_registry(OUTPUT_DIR / "phase65f_frozen_provider_registry.json")
    entry = next(item for item in registry if item["provider"] == "meditron3_8b")
    joined = build_joined_frame(
        argparse.Namespace(
            sample_path=str(PROJECT_ROOT / "data/processed/v3/cres/cres_eval_sample_12k.parquet"),
            prompts_path=str(PROJECT_ROOT / "data/processed/v3/cres/cres_eval_prompts_12k.jsonl"),
            variant_id="full_multimodal",
        )
    )
    joined_prompt_ids = set(joined["prompt_id"].dropna().astype(str).tolist())
    joined_columns = list(joined.columns)
    joined_key_map = {}
    for values in joined.itertuples(index=False, name=None):
        row = dict(zip(joined_columns, values))
        key = (
            row.get("prompt_id"),
            row.get("instance_id"),
            row.get("task_id"),
            row.get("dimension_id"),
            row.get("variant_id"),
        )
        joined_key_map[key] = row

    rows = []
    for response_row in iter_jsonl(Path(entry["canonical_response_path"])):
        prompt_id = response_row.get("prompt_id")
        if prompt_id not in joined_prompt_ids:
            continue
        key = (
            prompt_id,
            response_row.get("instance_id"),
            response_row.get("task_id"),
            response_row.get("dimension_id"),
            response_row.get("variant_id"),
        )
        joined_row = joined_key_map.get(key)
        if joined_row is None:
            continue
        parsed_response = response_row.get("parsed_response")
        if isinstance(parsed_response, dict):
            minimal_parsed = {"answer": parsed_response.get("answer", "")}
        else:
            minimal_parsed = {"answer": ""}
        confidence = normalize_confidence_value(response_row)
        if confidence:
            minimal_parsed["confidence"] = confidence
        minimal_parsed = normalize_phase65f_answer(
            response_row.get("task_id"),
            response_row.get("dimension_id"),
            minimal_parsed,
        )
        merged_row = dict(joined_row)
        merged_row.update(
            {
                "provider": entry["provider"],
                "model_name": response_row.get("model_name"),
                "parse_success": response_row.get("parse_success"),
                "parsed_response": minimal_parsed,
                "confidence_value": confidence,
                "status": response_row.get("status"),
            }
        )
        derived = derive_truth_and_prediction(merged_row)
        if derived is None:
            continue
        derived["tier"] = entry["tier"]
        derived["primary_model_name"] = entry.get("primary_model_name")
        derived["row_correct"] = row_correctness(derived)
        derived["confidence_value"] = normalize_text(derived.get("confidence_value"))
        rows.append(derived)

    df = pd.DataFrame(rows)
    drop_cols = [col for col in ["model_name", "instance_id", "parse_success"] if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    out_path = OUTPUT_DIR / "_score_tmp" / "meditron3_8b_scored.parquet"
    df.to_parquet(out_path, index=False)
    subset = df[(df["task_id"] == "S-T3") & (df["dimension_id"] == "D4")]
    print(
        {
            "rows_total": len(df),
            "s_t3_d4_rows": len(subset),
            "s_t3_d4_pred_labels": subset["pred_label"].value_counts(dropna=False).to_dict() if len(subset) else {},
            "out_path": str(out_path),
        }
    )


if __name__ == "__main__":
    main()
