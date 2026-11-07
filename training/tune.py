"""Optuna search over encoder hyperparameters. Trials use a validation split, never the test set.

Usage: python -m training.tune --data data/synthetic.csv --trials 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna

from training.common import load_dataset, split
from training.train_encoder import evaluate, predict, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train_full, _ = split(load_dataset(args.data))
    train_frame, validation = split(train_full, test_size=0.2, seed=7)

    def objective(trial: optuna.Trial) -> float:
        model, _, loader, device = train_model(
            train_frame,
            validation,
            epochs=args.epochs,
            batch_size=trial.suggest_categorical("batch_size", [8, 16]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            dropout=trial.suggest_float("dropout", 0.1, 0.3),
        )
        metrics = evaluate(predict(model, loader, device), validation)
        # Missing an outage costs most, so urgency recall shares the objective.
        return metrics["category"]["macro_f1"] + metrics["urgency"]["high_recall"]

    study = optuna.create_study(direction="maximize", study_name="marbert-triage")
    study.optimize(objective, n_trials=args.trials)
    print(f"best={study.best_value:.4f} params={study.best_params}")


if __name__ == "__main__":
    main()
