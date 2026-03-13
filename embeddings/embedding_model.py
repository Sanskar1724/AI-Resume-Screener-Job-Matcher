"""Sentence embedding model wrapper."""

from __future__ import annotations

from typing import Iterable

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


class EmbeddingModel:
    """Encapsulates sentence-transformers embedding generation."""

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Iterable[str]) -> list[list[float]]:
        """Encode multiple texts into dense vectors."""
        vectors = self.model.encode(
            list(texts), convert_to_numpy=True, normalize_embeddings=True)
        return vectors.tolist()

    def encode_single(self, text: str) -> list[float]:
        """Encode a single text into one dense vector."""
        vector = self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True)[0]
        return vector.tolist()
