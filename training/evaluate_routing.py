"""Show the trade-off behind ENCODER_THRESHOLD: how much traffic auto-routes, and how often it is
right, on the held-out split. Lower-confidence traffic would go to the LLM.

Usage: python -m training.evaluate_routing --data data/synthetic.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common import load_dataset, split
from triage.adapters.outbound.classifiers import OnnxClassifier
from triage.domain.models import Urgency

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--model", type=Path, default=Path("artifacts/encoder"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/routing_eval.json"))
    args = parser.parse_args()

    _, test = split(load_dataset(args.data))
    classifier = OnnxClassifier(args.model)
    rows = [
        (classifier.predict(row.normalized), row.category, row.urgency) for row in test.itertuples()
    ]

    table = []
    for threshold in THRESHOLDS:
        kept = [r for r in rows if r[0].confidence >= threshold]
        missed = [r for r in kept if r[2] == "high" and r[0].urgency is not Urgency.HIGH]
        correct = sum(r[0].category.value == r[1] for r in kept)
        table.append(
            {
                "threshold": threshold,
                "auto_routed": round(len(kept) / len(rows), 3),
                "category_accuracy": round(correct / len(kept), 3) if kept else None,
                "urgent_missed_while_auto_routed": len(missed),
            }
        )
    results = {"rows": len(rows), "version": classifier.version, "table": table}
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
