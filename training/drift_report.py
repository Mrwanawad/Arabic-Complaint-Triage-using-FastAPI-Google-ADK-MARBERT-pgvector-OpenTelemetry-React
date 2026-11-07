"""Compare recent tickets with the training data using Evidently, and flag retraining.

Usage: python -m training.drift_report --reference data/synthetic.csv --current data/recent.csv
The current file needs normalized `text` and predicted `category` columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from training.common import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data/drift_report.html"))
    args = parser.parse_args()

    reference = load_dataset(args.reference)
    current = pd.read_csv(args.current, encoding="utf-8")

    def features(frame: pd.DataFrame, text_column: str) -> pd.DataFrame:
        text = frame[text_column].astype(str)
        return pd.DataFrame(
            {
                "category": frame["category"],
                "length": text.str.len(),
                "words": text.str.split().str.len(),
                "latin_ratio": text.str.count(r"[a-z]") / text.str.len().clip(lower=1),
            }
        )

    snapshot = Report([DataDriftPreset()]).run(
        features(current, "text"), features(reference, "normalized")
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.out))
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
