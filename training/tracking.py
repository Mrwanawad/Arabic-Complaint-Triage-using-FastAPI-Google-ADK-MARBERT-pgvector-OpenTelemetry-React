"""Thin MLflow wrapper that flattens nested metrics. Skips quietly when MLflow is not installed."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path


def _flatten(values: Mapping, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in values.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, int | float) and not isinstance(value, bool):
            flat[name] = float(value)
    return flat


def log_run(name: str, params: Mapping, metrics: Mapping, artifacts: Iterable[Path] = ()) -> None:
    try:
        import mlflow
    except ImportError:
        print("MLflow not installed: run not tracked")
        return
    mlflow.set_experiment("arabic-complaint-triage")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics(_flatten(metrics))
        for artifact in artifacts:
            mlflow.log_artifact(str(artifact))
