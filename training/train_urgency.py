"""Train the LightGBM urgency model on out-of-sample encoder outputs plus ticket metadata.

The model must strictly improve on the encoder urgency head without losing recall or macro-F1, or it
is not saved and serving keeps the encoder head.

The synthetic dataset has no channel, hour, or repeat-count columns, so they are SIMULATED below
with an assumed correlation. Simulated runs are never kept unless --allow-simulated is passed.

Usage: python -m training.train_urgency --data data/synthetic.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

from training.common import high_urgency_recall, load_dataset, split
from training.tracking import log_run
from triage.adapters.outbound.classifiers import (
    CHANNELS,
    BaselineClassifier,
    OnnxClassifier,
    urgency_feature_vector,
)
from triage.application.ports import UrgencyFeatures


def _load_classifier(artifacts: Path):
    if (artifacts / "encoder" / "model.onnx").exists():
        return OnnxClassifier(artifacts / "encoder")
    return BaselineClassifier(artifacts / "baseline.joblib")


def _metadata(frame, rng: np.random.Generator) -> tuple[list[UrgencyFeatures], bool]:
    required = {"channel", "hour", "repeat_complaints_24h"}
    if required <= set(frame.columns):
        return [
            UrgencyFeatures(row.channel, int(row.hour), int(row.repeat_complaints_24h))
            for row in frame.itertuples()
        ], False
    features = [
        UrgencyFeatures(
            str(rng.choice(CHANNELS)),
            int(rng.integers(0, 24)),
            int(rng.poisson(1.2 if urgency == "high" else 0.4)),
        )
        for urgency in frame["urgency"]
    ]
    return features, True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--allow-simulated", action="store_true", help="for demos only")
    args = parser.parse_args()

    _, holdout = split(load_dataset(args.data))
    classifier = _load_classifier(args.artifacts)
    predictions = [classifier.predict(text) for text in holdout["normalized"]]
    features, simulated = _metadata(holdout, np.random.default_rng(42))
    x = np.array(
        [urgency_feature_vector(p, f) for p, f in zip(predictions, features, strict=True)],
        dtype=np.float64,
    )
    y = (holdout["urgency"] == "high").astype(int).to_numpy()

    train_index, test_index = next(
        GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=42).split(
            x, y, groups=holdout.get("template_id")
        )
    )
    booster = lgb.train(
        {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 15,
            "min_data_in_leaf": 5,
            "is_unbalance": True,
            "verbosity": -1,
            "seed": 42,
        },
        lgb.Dataset(x[train_index], y[train_index]),
        num_boost_round=200,
    )

    truth = ["high" if value else "normal" for value in y[test_index]]
    lightgbm_labels = [
        "high" if value >= 0.5 else "normal" for value in booster.predict(x[test_index])
    ]
    encoder_labels = [predictions[i].urgency.value for i in test_index]
    results = {
        "lightgbm_high_recall": high_urgency_recall(truth, lightgbm_labels),
        "encoder_high_recall": high_urgency_recall(truth, encoder_labels),
        "lightgbm_macro_f1": float(f1_score(truth, lightgbm_labels, average="macro")),
        "encoder_macro_f1": float(f1_score(truth, encoder_labels, average="macro")),
        "simulated_metadata": simulated,
        "feature_source": classifier.version,
        "test_rows": len(truth),
    }
    recall_delta = results["lightgbm_high_recall"] - results["encoder_high_recall"]
    f1_delta = results["lightgbm_macro_f1"] - results["encoder_macro_f1"]
    # A tie adds a model to the path for nothing, and simulated features cannot prove a real gain.
    wins = recall_delta >= 0 and f1_delta >= 0 and (recall_delta > 0 or f1_delta > 0)
    wins = wins and (not simulated or args.allow_simulated)
    results["kept_in_serving"] = wins

    args.artifacts.mkdir(parents=True, exist_ok=True)
    model_path = args.artifacts / "urgency_lgbm.txt"
    if wins:
        booster.save_model(str(model_path))
    else:
        model_path.unlink(missing_ok=True)
    metrics_path = args.artifacts / "urgency_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log_run("urgency-lightgbm", {"data": str(args.data)}, results, [metrics_path])
    print(json.dumps(results, indent=2))
    if simulated:
        print("WARNING: metadata was simulated, so this result says nothing about real data.")


if __name__ == "__main__":
    main()
