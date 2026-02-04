"""
Generate 5 LLM-style text features from radiology notes (0-24h).
Outputs a CSV compatible with llm_features_deepseek.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from config import NOTE_TIME_FILE, LLM_FEATURES_FILE


FEATURES = {
    "pneumonia": {
        "keywords": ["pneumonia", "pna", "consolidation", "infiltrate", "infiltration"],
        "negations": ["no pneumonia", "without pneumonia", "clear lungs"],
    },
    "edema": {
        "keywords": ["edema", "pulmonary edema", "fluid overload", "volume overload"],
        "negations": ["no edema", "no pulmonary edema", "euvolemic"],
    },
    "pleural_effusion": {
        "keywords": ["pleural effusion", "effusion", "pleural fluid"],
        "negations": ["no effusion", "no pleural effusion"],
    },
    "pneumothorax": {
        "keywords": ["pneumothorax", "ptx"],
        "negations": ["no pneumothorax", "without pneumothorax"],
    },
    "tubes_lines": {
        "keywords": ["tube", "tubes", "line", "lines", "et tube", "ett", "picc", "central line"],
        "negations": [],
    },
}


def extract_features(text: str) -> dict:
    text_lower = (text or "").lower()
    feats = {}
    for name, cfg in FEATURES.items():
        neg_hit = any(neg in text_lower for neg in cfg["negations"])
        pos_hit = any(kw in text_lower for kw in cfg["keywords"])
        if neg_hit:
            feats[name] = 0
        elif pos_hit:
            feats[name] = 1
        else:
            feats[name] = 0
    return feats


def main():
    parser = argparse.ArgumentParser(description="Generate LLM-style features from radiology notes")
    parser.add_argument("--input", default=str(NOTE_TIME_FILE), help="note_time.csv path")
    parser.add_argument("--output", default=str(LLM_FEATURES_FILE), help="output CSV path")
    parser.add_argument("--max-hours", type=int, default=24, help="max hour_offset window")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing notes file: {input_path}")

    print(f"Loading notes: {input_path}")
    notes = pd.read_csv(input_path)
    notes["stay_id"] = pd.to_numeric(notes["stay_id"], errors="coerce").fillna(-1).astype(int)
    notes["hour_offset"] = pd.to_numeric(notes["hour_offset"], errors="coerce")
    notes = notes[(notes["hour_offset"] >= 0) & (notes["hour_offset"] < args.max_hours)]

    if "radiology_text" not in notes.columns:
        raise ValueError("note_time.csv missing radiology_text column")

    print(f"Notes in window: {len(notes)}")

    rows = []
    for stay_id, group in notes.groupby("stay_id"):
        agg = {name: 0 for name in FEATURES}
        for text in group["radiology_text"].fillna("").tolist():
            feats = extract_features(text)
            for name, val in feats.items():
                agg[name] = max(agg[name], val)
        row = {"uid": str(stay_id), "stay_id": int(stay_id)}
        row.update(agg)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} rows -> {output_path}")


if __name__ == "__main__":
    main()
