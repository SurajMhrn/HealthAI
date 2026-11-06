"""Cross-encoder re-ranking with bge-reranker-base."""

from __future__ import annotations

from typing import Protocol


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher means more relevant."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One score per passage."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder on CPU."""

    def __init__(self, model_name: str, revision: str, batch_size: int, max_length: int) -> None:
        from sentence_transformers import CrossEncoder  # heavy import, only when needed

        self.model = CrossEncoder(model_name, revision=revision, device="cpu", max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logits for each passage."""
        if not passages:
            return []
        scores = self.model.predict([(query, p) for p in passages], batch_size=self.batch_size)
        return [float(s) for s in scores]
