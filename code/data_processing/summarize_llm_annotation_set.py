"""
Summarize LLM annotation sample coverage and stats.
"""

import json
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import ROOT_DIR


OUT_DIR = ROOT_DIR / "results" / "llm_annotations"


def main():
    sample_path = OUT_DIR / "llm_annotation_set.csv"
    if not sample_path.exists():
        raise FileNotFoundError("llm_annotation_set.csv not found; run build_llm_annotation_set.py first.")

    df = pd.read_csv(sample_path)

    stats = {
        "n_samples": int(len(df)),
        "note_type_counts": df["note_type"].value_counts().to_dict(),
        "time_delta_bucket_counts": pd.cut(
            df["time_delta_hours"],
            bins=[-1e9, -12, -6, 0],
            labels=["<=-12h", "-12h~-6h", "-6h~0h"]
        ).value_counts().to_dict(),
        "pattern_counts": df["pattern_name"].value_counts().head(50).to_dict(),
        "text_length_summary": {
            "min": int(df["note_text_relevant"].astype(str).str.len().min()),
            "max": int(df["note_text_relevant"].astype(str).str.len().max()),
            "mean": float(df["note_text_relevant"].astype(str).str.len().mean()),
        },
    }

    out_path = OUT_DIR / "llm_annotation_summary.json"
    out_path.write_text(json.dumps(stats, indent=2, ensure_ascii=True))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
