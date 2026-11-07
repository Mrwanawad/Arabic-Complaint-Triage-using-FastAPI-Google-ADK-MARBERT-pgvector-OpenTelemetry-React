"""Fine-tune MARBERT with one classification head each for category, urgency, and sentiment.

Usage: python -m training.train_encoder --data data/synthetic.csv --epochs 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from training.common import HEADS, head_metrics, high_urgency_recall, load_dataset, report, split
from training.tracking import log_run
from triage.domain.models import Category, Sentiment, Urgency

LABELS = {
    "category": [c.value for c in Category],
    "urgency": [u.value for u in Urgency],
    "sentiment": [s.value for s in Sentiment],
}
DEFAULT_BASE_MODEL = "UBC-NLP/MARBERT"


class MultiHeadEncoder(nn.Module):
    def __init__(self, base_model: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.base_model_name = base_model
        self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({h: nn.Linear(hidden, len(LABELS[h])) for h in HEADS})

    def forward(self, input_ids, attention_mask, token_type_ids=None) -> dict[str, torch.Tensor]:
        output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        cls = self.dropout(output.last_hidden_state[:, 0])
        return {head: layer(cls) for head, layer in self.heads.items()}


class ComplaintDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length: int) -> None:
        self.encodings = tokenizer(
            frame["normalized"].tolist(), truncation=True, max_length=max_length
        )
        self.labels = {head: [LABELS[head].index(value) for value in frame[head]] for head in HEADS}

    def __len__(self) -> int:
        return len(self.labels["category"])

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item.update({f"label_{head}": self.labels[head][index] for head in HEADS})
        return item


def build_loader(dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
        labels = {f"label_{h}": torch.tensor([b.pop(f"label_{h}") for b in batch]) for h in HEADS}
        padded = tokenizer.pad(batch, return_tensors="pt")
        return {**padded, **labels}

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def class_weights(frame, device) -> dict[str, torch.Tensor]:
    """Weights cover only labels seen in training. Absent labels keep a neutral weight of 1."""
    weights = {}
    for head in HEADS:
        values = np.ones(len(LABELS[head]), dtype=np.float32)
        present = sorted(set(frame[head]))
        balanced = compute_class_weight("balanced", classes=np.array(present), y=frame[head])
        for label, weight in zip(present, balanced, strict=True):
            values[LABELS[head].index(label)] = weight
        weights[head] = torch.tensor(values, device=device)
    return weights


@torch.no_grad()
def predict(model, loader, device) -> dict[str, list[str]]:
    model.eval()
    predictions: dict[str, list[str]] = {head: [] for head in HEADS}
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch.get("token_type_ids", None).to(device) if "token_type_ids" in batch else None,
        )
        for head in HEADS:
            indexes = logits[head].argmax(dim=-1).cpu().tolist()
            predictions[head].extend(LABELS[head][i] for i in indexes)
    return predictions


def evaluate(predictions: dict[str, list[str]], frame) -> dict:
    metrics = {head: head_metrics(frame[head], predictions[head]) for head in HEADS}
    metrics["urgency"]["high_recall"] = high_urgency_recall(
        frame["urgency"], predictions["urgency"]
    )
    return metrics


def train_model(
    train_frame,
    validation_frame,
    base_model: str = DEFAULT_BASE_MODEL,
    epochs: int = 4,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_length: int = 128,
    dropout: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = MultiHeadEncoder(base_model, dropout).to(device)

    train_loader = build_loader(
        ComplaintDataset(train_frame, tokenizer, max_length), tokenizer, batch_size, True
    )
    validation_loader = build_loader(
        ComplaintDataset(validation_frame, tokenizer, max_length), tokenizer, batch_size * 2, False
    )
    best_score, best_state = -1.0, None
    weights = class_weights(train_frame, device)
    losses = {head: nn.CrossEntropyLoss(weight=weights[head]) for head in HEADS}

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, total_steps // 10, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["token_type_ids"].to(device) if "token_type_ids" in batch else None,
                )
                loss = sum(
                    losses[head](logits[head], batch[f"label_{head}"].to(device)) for head in HEADS
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
        metrics = evaluate(predict(model, validation_loader, device), validation_frame)
        score = metrics["category"]["macro_f1"] + metrics["urgency"]["high_recall"]
        print(
            f"epoch {epoch + 1}/{epochs} loss={running / len(train_loader):.3f} "
            f"val_category_f1={metrics['category']['macro_f1']:.3f} "
            f"val_urgency_recall={metrics['urgency']['high_recall']:.3f}"
        )
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, tokenizer, validation_loader, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--out", type=Path, default=Path("checkpoints/encoder"))
    args = parser.parse_args()

    train_full, test_frame = split(load_dataset(args.data))
    # The best epoch is chosen on validation, so the test set stays unseen until the final report.
    train_frame, validation_frame = split(train_full, test_size=0.15, seed=7)
    model, tokenizer, _, device = train_model(
        train_frame,
        validation_frame,
        args.base_model,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.max_length,
    )
    test_loader = build_loader(
        ComplaintDataset(test_frame, tokenizer, args.max_length),
        tokenizer,
        args.batch_size * 2,
        False,
    )
    predictions = predict(model, test_loader, device)
    metrics = evaluate(predictions, test_frame)
    for head in HEADS:
        print(head, report(test_frame[head], predictions[head]), sep="\n")

    scripts = test_frame["script"].to_numpy()
    metrics["category_by_script"] = {
        script: head_metrics(
            test_frame.loc[scripts == script, "category"],
            [p for p, s in zip(predictions["category"], scripts, strict=True) if s == script],
        )
        for script in sorted(set(scripts))
    }

    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "model.pt")
    tokenizer.save_pretrained(args.out)
    (args.out / "labels.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "version": f"marbert-v1-{args.data.stem}",
                "base_model": args.base_model,
                "max_length": args.max_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path = args.out / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log_run(
        "encoder",
        {
            "base_model": args.base_model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "data": str(args.data),
        },
        metrics,
        [metrics_path],
    )
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
