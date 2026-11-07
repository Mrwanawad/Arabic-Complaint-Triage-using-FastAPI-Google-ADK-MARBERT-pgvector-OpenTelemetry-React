"""Train the TF-IDF + calibrated LinearSVC baseline and compare imbalance strategies.

Usage: python -m training.train_baseline --data data/synthetic.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from training.common import HEADS, head_metrics, high_urgency_recall, load_dataset, report, split
from training.tracking import log_run


def _features() -> FeatureUnion:
    return FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
        ]
    )


def _model(strategy: str) -> ImbPipeline:
    class_weight = "balanced" if strategy == "class_weight" else None
    calibrated = CalibratedClassifierCV(LinearSVC(class_weight=class_weight), cv=3)
    steps = [("features", _features())]
    if strategy == "oversample":
        steps.append(("sampler", RandomOverSampler(random_state=42)))
    steps.append(("classifier", calibrated))
    return ImbPipeline(steps)


def serving_pipeline(model: ImbPipeline) -> Pipeline:
    """Resampling only affects fit, so serving drops it, along with the imbalanced-learn dependency
    that the production image does not install."""
    return Pipeline([step for step in model.steps if step[0] != "sampler"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/baseline.joblib"))
    args = parser.parse_args()

    frame = load_dataset(args.data)
    train, test = split(frame)
    print(f"train={len(train)} test={len(test)}")

    models = {}
    results: dict[str, dict] = {}
    for head in HEADS:
        best_score = -1.0
        results[head] = {}
        for strategy in ("class_weight", "oversample"):
            model = _model(strategy).fit(train["normalized"], train[head])
            predictions = model.predict(test["normalized"])
            metrics = head_metrics(test[head], predictions)
            if head == "urgency":
                metrics["high_recall"] = high_urgency_recall(test[head], predictions)
            results[head][strategy] = metrics
            print(f"[{head}/{strategy}] {metrics}")
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                models[head] = serving_pipeline(model)
                best_predictions = predictions
                results[head]["selected"] = strategy
        print(report(test[head], best_predictions))

    slices = {}
    for script, group in test.groupby("script"):
        slices[script] = head_metrics(
            group["category"], models["category"].predict(group["normalized"])
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "version": "baseline-v1"}, args.out)
    summary = {"heads": results, "category_by_script": slices, "rows": len(frame)}
    args.out.with_suffix(".metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    log_run("baseline", {"data": str(args.data)}, summary, [args.out.with_suffix(".metrics.json")])
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
