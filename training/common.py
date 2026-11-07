"""Shared training helpers: loading, group-aware splits, and metrics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit

from triage.domain.normalization import normalize
from triage.domain.pii import mask_pii

HEADS = ("category", "urgency", "sentiment")


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8")
    frame["normalized"] = [normalize(mask_pii(text)) for text in frame["text"]]
    return frame


def split(
    frame: pd.DataFrame, test_size: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group by template when present, so near-duplicates cannot leak into the test set."""
    groups = frame["template_id"] if "template_id" in frame.columns else frame.index
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(frame, groups=groups))
    return frame.iloc[train_index].copy(), frame.iloc[test_index].copy()


def head_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, zero_division=0)


def high_urgency_recall(y_true, y_pred) -> float:
    high = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t == "high"]
    return sum(t == p for t, p in high) / len(high) if high else 0.0
