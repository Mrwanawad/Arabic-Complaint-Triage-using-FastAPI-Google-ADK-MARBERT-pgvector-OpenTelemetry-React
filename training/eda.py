"""Exploratory charts: class balance, length by script, and an Arabic word cloud.

Usage: python -m training.eda --data data/synthetic.csv --out reports/eda
"""

from __future__ import annotations

import argparse
from pathlib import Path

import arabic_reshaper
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from bidi.algorithm import get_display
from wordcloud import WordCloud

from training.common import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--out", type=Path, default=Path("reports/eda"))
    parser.add_argument("--font", default=r"C:\Windows\Fonts\tahoma.ttf")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    frame = load_dataset(args.data)
    frame["length"] = frame["normalized"].str.split().str.len()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, column in zip(axes, ["category", "urgency", "sentiment"], strict=True):
        sns.countplot(data=frame, x=column, ax=axis, color="#2a6fbb")
        axis.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(args.out / "class_balance.png", dpi=130)

    px.box(frame, x="script", y="length", color="category").write_html(args.out / "length.html")

    arabic_text = " ".join(frame.loc[frame["script"] == "arabic", "normalized"])
    cloud = WordCloud(font_path=args.font, width=1000, height=500, background_color="white")
    cloud.generate(get_display(arabic_reshaper.reshape(arabic_text)))
    cloud.to_file(args.out / "wordcloud.png")
    print(f"Saved charts to {args.out}")


if __name__ == "__main__":
    main()
