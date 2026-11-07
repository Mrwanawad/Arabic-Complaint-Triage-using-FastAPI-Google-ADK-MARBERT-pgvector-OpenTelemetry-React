"""Measure few-shot retrieval quality with RAGAS.

Retrieved complaints count as relevant when they share the query's category and urgency. Non-LLM
RAGAS metrics compare them with references drawn from the same labels, so no API key is needed.

Usage: python -m evals.eval_retrieval_ragas --data data/synthetic.csv --k 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference, NonLLMContextRecall

from training.common import load_dataset, split
from triage.bootstrap import build_embedder
from triage.settings import Settings


async def evaluate(data: Path, k: int, max_queries: int, settings: Settings) -> dict:
    train, test = split(load_dataset(data))
    embedder = build_embedder(settings)
    print(f"Embedder: {type(embedder).__name__}")
    matrix = np.array([embedder.embed(text) for text in train["normalized"]], dtype=np.float32)

    # Numeric similarity on exact label strings makes the RAGAS metrics label-based.
    train_labels = (train["category"] + "|" + train["urgency"]).tolist()
    sample = test.sample(min(max_queries, len(test)), random_state=42)

    precision_metric = NonLLMContextPrecisionWithReference()
    recall_metric = NonLLMContextRecall()
    precision, recall, hit = [], [], []
    for row in sample.itertuples():
        scores = matrix @ np.array(embedder.embed(row.normalized), dtype=np.float32)
        top = np.argsort(-scores)[:k]
        retrieved = [train_labels[index] for index in top]
        label = f"{row.category}|{row.urgency}"
        turn = SingleTurnSample(
            retrieved_contexts=retrieved,
            reference_contexts=[label],
        )
        precision.append(await precision_metric.single_turn_ascore(turn))
        recall.append(await recall_metric.single_turn_ascore(turn))
        hit.append(sum(value == label for value in retrieved) / k)

    return {
        "embedder": type(embedder).__name__,
        "k": k,
        "queries": len(sample),
        "context_precision": float(np.mean(precision)),
        "context_recall": float(np.mean(recall)),
        "label_precision_at_k": float(np.mean(hit)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ragas_retrieval.json"))
    args = parser.parse_args()
    results = asyncio.run(evaluate(args.data, args.k, args.max_queries, Settings()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
