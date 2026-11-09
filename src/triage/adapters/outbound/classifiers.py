"""Local model adapters: ONNX encoder, scikit-learn baseline, LightGBM urgency, Whisper."""

from __future__ import annotations

import io
import json
import threading
from pathlib import Path

import joblib
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from triage.application.ports import (
    NullTelemetry,
    Telemetry,
    TranscriptionFailed,
    UrgencyFeatures,
)
from triage.domain.models import Category, Prediction, Sentiment, Urgency

CATEGORIES = [c.value for c in Category]
URGENCIES = [u.value for u in Urgency]
SENTIMENTS = [s.value for s in Sentiment]
CHANNELS = ["api", "tickets", "whatsapp", "social", "voice"]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=-1, keepdims=True)


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxClassifier:
    """Multi-head encoder. model.onnx returns one logits output per head, named in labels.json."""

    def __init__(
        self, model_dir: Path, max_length: int = 128, telemetry: Telemetry | None = None
    ) -> None:
        self._telemetry = telemetry or NullTelemetry()
        metadata = json.loads((model_dir / "labels.json").read_text(encoding="utf-8"))
        self._labels: dict[str, list[str]] = metadata["labels"]
        self.version: str = metadata["version"]
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._unknown_id = self._tokenizer.token_to_id("[UNK]")
        self._session = _session(model_dir / "model.onnx")
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._outputs = [o.name for o in self._session.get_outputs()]

    def _head(self, logits: dict[str, np.ndarray], head: str) -> tuple[str, float]:
        probabilities = _softmax(logits[head])[0]
        index = int(probabilities.argmax())
        return self._labels[head][index], float(probabilities[index])

    def predict(self, normalized_text: str) -> Prediction:
        encoding = self._tokenizer.encode(normalized_text)
        # A rising share of unknown tokens signals new slang the model has never seen.
        unknown = encoding.ids.count(self._unknown_id) / max(len(encoding.ids), 1)
        self._telemetry.record("triage.unknown_token_ratio", unknown)
        features = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
            "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
        }
        outputs = self._session.run(
            self._outputs, {k: v for k, v in features.items() if k in self._inputs}
        )
        logits = dict(zip(self._outputs, outputs, strict=True))
        category, category_confidence = self._head(logits, "category")
        urgency, urgency_confidence = self._head(logits, "urgency")
        sentiment, sentiment_confidence = self._head(logits, "sentiment")
        return Prediction(
            Category(category),
            category_confidence,
            Urgency(urgency),
            urgency_confidence,
            Sentiment(sentiment),
            sentiment_confidence,
            self.version,
        )


class BaselineClassifier:
    """TF-IDF with calibrated LinearSVC, one pipeline per head."""

    def __init__(self, path: Path) -> None:
        bundle = joblib.load(path)
        self._models = bundle["models"]
        self.version: str = bundle["version"]

    def _head(self, head: str, text: str) -> tuple[str, float]:
        model = self._models[head]
        probabilities = model.predict_proba([text])[0]
        index = int(probabilities.argmax())
        return str(model.classes_[index]), float(probabilities[index])

    def predict(self, normalized_text: str) -> Prediction:
        category, category_confidence = self._head("category", normalized_text)
        urgency, urgency_confidence = self._head("urgency", normalized_text)
        sentiment, sentiment_confidence = self._head("sentiment", normalized_text)
        return Prediction(
            Category(category),
            category_confidence,
            Urgency(urgency),
            urgency_confidence,
            Sentiment(sentiment),
            sentiment_confidence,
            self.version,
        )


def urgency_feature_vector(prediction: Prediction, features: UrgencyFeatures) -> list[float]:
    """Shared with training so serving can never drift from the trained feature order."""
    high_probability = (
        prediction.urgency_confidence
        if prediction.urgency is Urgency.HIGH
        else 1 - prediction.urgency_confidence
    )
    negative = (
        prediction.sentiment_confidence if prediction.sentiment is Sentiment.NEGATIVE else 0.0
    )
    categories = [float(prediction.category.value == c) for c in CATEGORIES]
    channels = [float(features.channel == c) for c in CHANNELS]
    return [
        high_probability,
        prediction.category_confidence,
        negative,
        float(features.hour),
        float(features.repeat_complaints_24h),
        *categories,
        *channels,
    ]


class LightGbmUrgency:
    def __init__(self, path: Path, decision_threshold: float = 0.5) -> None:
        import lightgbm as lgb

        self._booster = lgb.Booster(model_file=str(path))
        self._threshold = decision_threshold

    def predict(self, prediction: Prediction, features: UrgencyFeatures) -> tuple[Urgency, float]:
        row = np.array([urgency_feature_vector(prediction, features)], dtype=np.float64)
        high = float(self._booster.predict(row)[0])
        if high >= self._threshold:
            return Urgency.HIGH, high
        return Urgency.NORMAL, 1 - high


class WhisperTranscriber:
    """Loads lazily because the model is large and many deployments disable voice."""

    def __init__(self, model: str = "base") -> None:
        # A model size downloads from Hugging Face on first load; a directory loads offline.
        self._model_name = model
        self._model = None
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Loads the model ahead of the first request, which would otherwise time out."""
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                self._model = WhisperModel(self._model_name, device="cpu", compute_type="int8")

    def transcribe(self, audio: bytes) -> str:
        import av

        self.warm()
        try:
            segments, _ = self._model.transcribe(io.BytesIO(audio), language="ar", vad_filter=True)
            return " ".join(segment.text.strip() for segment in segments).strip()
        except (av.error.FFmpegError, ValueError, IndexError) as error:
            raise TranscriptionFailed("The audio could not be decoded") from error
