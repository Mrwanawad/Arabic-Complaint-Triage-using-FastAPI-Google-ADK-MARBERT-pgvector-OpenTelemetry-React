"""Embedding adapters. Both return 384-dimensional unit vectors to match schema.sql."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from sklearn.feature_extraction.text import HashingVectorizer
from tokenizers import Tokenizer

from triage.domain.normalization import embedding_text

EMBEDDING_DIM = 384


class OnnxEmbedder:
    """SentenceTransformers model exported to ONNX, so serving needs no torch."""

    def __init__(self, model_dir: Path, max_length: int = 128) -> None:
        self._tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self._tokenizer.enable_truncation(max_length=max_length)
        self._session = ort.InferenceSession(
            str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"]
        )
        self._inputs = {i.name for i in self._session.get_inputs()}

    def embed(self, text: str) -> list[float]:
        encoding = self._tokenizer.encode(embedding_text(text))
        mask = np.array([encoding.attention_mask], dtype=np.int64)
        features = {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "attention_mask": mask,
            "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
        }
        tokens = self._session.run(None, {k: v for k, v in features.items() if k in self._inputs})[
            0
        ]
        weights = mask[..., None].astype(np.float32)
        pooled = (tokens * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        vector = pooled[0] / max(float(np.linalg.norm(pooled[0])), 1e-9)
        return vector.tolist()


class HashingEmbedder:
    """Character n-gram hashing. A lexical stand-in for development when the ONNX model is absent:
    it has no semantic understanding, so production must not rely on it."""

    def __init__(self) -> None:
        self._vectorizer = HashingVectorizer(
            n_features=EMBEDDING_DIM,
            analyzer="char_wb",
            ngram_range=(2, 4),
            norm="l2",
            alternate_sign=False,
        )

    def embed(self, text: str) -> list[float]:
        vector = self._vectorizer.transform([embedding_text(text)]).toarray()[0]
        if not vector.any():
            vector[0] = 1.0
        return vector.tolist()
