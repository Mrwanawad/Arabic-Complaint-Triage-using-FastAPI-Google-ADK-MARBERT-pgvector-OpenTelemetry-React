"""Find likely mislabeled rows with cleanlab, and export them to Label Studio for relabeling.

Usage: python -m training.label_quality --data data/synthetic.csv --head category
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cleanlab.filter import find_label_issues
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from training.common import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--head", default="category", choices=["category", "urgency", "sentiment"])
    parser.add_argument("--out", type=Path, default=Path("data/label_review.json"))
    args = parser.parse_args()

    frame = load_dataset(args.data)
    encoder = LabelEncoder()
    labels = encoder.fit_transform(frame[args.head])
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
        CalibratedClassifierCV(LinearSVC(class_weight="balanced"), cv=3),
    )
    probabilities = cross_val_predict(
        model, frame["normalized"], labels, cv=5, method="predict_proba"
    )
    issues = find_label_issues(labels, probabilities, return_indices_ranked_by="self_confidence")

    # Label Studio task format with the suggested label as a pre-annotation.
    tasks = []
    for index in issues:
        suggested = encoder.inverse_transform([probabilities[index].argmax()])[0]
        tasks.append(
            {
                "data": {
                    "text": frame["text"].iloc[index],
                    "current_label": frame[args.head].iloc[index],
                },
                "predictions": [
                    {
                        "score": float(probabilities[index].max()),
                        "result": [
                            {
                                "from_name": args.head,
                                "to_name": "text",
                                "type": "choices",
                                "value": {"choices": [suggested]},
                            }
                        ],
                    }
                ],
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(issues)} of {len(frame)} rows flagged for {args.head}. Tasks: {args.out}")


if __name__ == "__main__":
    main()
