"""Export the trained encoder to ONNX, quantize to int8, and measure the accuracy delta.

Usage: python -m training.export_onnx --checkpoint checkpoints/encoder --data data/synthetic.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn
from transformers import AutoTokenizer

from training.common import HEADS, load_dataset, split
from training.train_encoder import LABELS, MultiHeadEncoder
from triage.adapters.outbound.classifiers import OnnxClassifier


class ExportWrapper(nn.Module):
    """ONNX needs positional tensor outputs rather than a dict."""

    def __init__(self, model: MultiHeadEncoder) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.model(input_ids, attention_mask, token_type_ids)
        return tuple(logits[head] for head in HEADS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/encoder"))
    parser.add_argument("--data", type=Path, default=Path("data/synthetic.csv"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/encoder"))
    args = parser.parse_args()

    metadata = json.loads((args.checkpoint / "labels.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = MultiHeadEncoder(metadata["base_model"])
    model.load_state_dict(torch.load(args.checkpoint / "model.pt", map_location="cpu"))
    model.eval()

    args.out.mkdir(parents=True, exist_ok=True)
    float_path = args.out / "model.float.onnx"
    sample = tokenizer("النت واقع", return_tensors="pt", return_token_type_ids=True)
    torch.onnx.export(
        ExportWrapper(model),
        (sample["input_ids"], sample["attention_mask"], sample["token_type_ids"]),
        str(float_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=list(HEADS),
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "token_type_ids": {0: "batch", 1: "sequence"},
            **{head: {0: "batch"} for head in HEADS},
        },
        opset_version=17,
        dynamo=False,
    )
    quantized_path = args.out / "model.onnx"
    quantize_dynamic(float_path, quantized_path, weight_type=QuantType.QInt8)
    float_mb = float_path.stat().st_size / 1e6
    float_path.unlink()
    for leftover in args.out.glob("*.data"):
        leftover.unlink()

    tokenizer.backend_tokenizer.save(str(args.out / "tokenizer.json"))
    (args.out / "labels.json").write_text(
        json.dumps({"labels": LABELS, "version": f"{metadata['version']}-int8"}, indent=2),
        encoding="utf-8",
    )
    shutil.copy(args.checkpoint / "metrics.json", args.out / "metrics.json")

    _, test = split(load_dataset(args.data))
    classifier = OnnxClassifier(args.out, metadata["max_length"])
    started = time.perf_counter()
    latencies = []
    agreement = {head: 0 for head in HEADS}
    correct = dict.fromkeys(HEADS, 0)
    torch_predictions = {head: [] for head in HEADS}
    with torch.no_grad():
        for text in test["normalized"]:
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
            logits = model(**encoded)
            for head in HEADS:
                torch_predictions[head].append(LABELS[head][int(logits[head].argmax())])
    for index, text in enumerate(test["normalized"]):
        tick = time.perf_counter()
        prediction = classifier.predict(text)
        latencies.append((time.perf_counter() - tick) * 1000)
        onnx_values = {
            "category": prediction.category.value,
            "urgency": prediction.urgency.value,
            "sentiment": prediction.sentiment.value,
        }
        for head in HEADS:
            agreement[head] += onnx_values[head] == torch_predictions[head][index]
            correct[head] += onnx_values[head] == test[head].iloc[index]
    size = len(test)
    report = {
        "float_mb": round(float_mb, 1),
        "int8_mb": round(quantized_path.stat().st_size / 1e6, 1),
        "agreement_with_pytorch": {h: round(agreement[h] / size, 4) for h in HEADS},
        "onnx_accuracy": {h: round(correct[h] / size, 4) for h in HEADS},
        "latency_ms_p50": round(float(np.percentile(latencies, 50)), 1),
        "latency_ms_p95": round(float(np.percentile(latencies, 95)), 1),
        "evaluated": size,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    (args.out / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
