"""
Sort timeseries.csv by stay_id and hour.
Writes data/raw/timeseries_sorted.csv by default.
"""

import argparse
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Sort timeseries.csv by stay_id/hour")
    parser.add_argument(
        "--input",
        default=str(RAW_DATA_DIR / "timeseries.csv"),
        help="Input timeseries CSV path",
    )
    parser.add_argument(
        "--output",
        default=str(RAW_DATA_DIR / "timeseries_sorted.csv"),
        help="Output sorted CSV path",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    if "stay_id" not in df.columns or "hour" not in df.columns:
        raise ValueError("timeseries.csv missing stay_id/hour columns")

    print("Sorting by stay_id, hour...")
    df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce")
    df["hour"] = pd.to_numeric(df["hour"], errors="coerce")
    df = df.sort_values(["stay_id", "hour"], kind="mergesort")

    print(f"Writing: {output_path}")
    df.to_csv(output_path, index=False)
    print("Done.")


if __name__ == "__main__":
    main()
