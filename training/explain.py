"""Explain sample predictions with LIME against the baseline's category model.

Usage: python -m training.explain "النت واقع من امبارح"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
from lime.lime_text import LimeTextExplainer

from triage.domain.normalization import normalize
from triage.domain.pii import mask_pii


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text")
    parser.add_argument("--model", type=Path, default=Path("artifacts/baseline.joblib"))
    parser.add_argument("--head", default="category")
    args = parser.parse_args()

    model = joblib.load(args.model)["models"][args.head]
    normalized = normalize(mask_pii(args.text))
    explainer = LimeTextExplainer(class_names=[str(c) for c in model.classes_])
    probabilities = model.predict_proba([normalized])[0]
    top = int(probabilities.argmax())
    explanation = explainer.explain_instance(
        normalized, model.predict_proba, labels=[top], num_features=8, num_samples=500
    )
    print(f"Prediction: {model.classes_[top]} ({probabilities[top]:.2f})")
    for token, weight in explanation.as_list(label=top):
        print(f"{weight:+.3f}  {token}")


if __name__ == "__main__":
    main()
