"""Dense retrieval: bge-small embeddings in a FAISS inner-product index (cosine on unit vectors)."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import faiss
import numpy as np


class Embedder(Protocol):
    """Anything that turns texts into L2-normalised float32 vectors."""

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return an array of shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    """bge embeddings via sentence-transformers on CPU.

    bge v1.5 expects an instruction prefix on queries only; passages are embedded as-is.
    """

    def __init__(
        self, model_name: str, revision: str, query_instruction: str, batch_size: int, max_seq_length: int
    ) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, only when needed

        self.model = SentenceTransformer(model_name, revision=revision, device="cpu")
        self.model.max_seq_length = max_seq_length
        self.query_instruction = query_instruction
        self.batch_size = batch_size

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Embed texts (queries get the instruction prefix) and L2-normalise."""
        if is_query:
            texts = [self.query_instruction + t for t in texts]
        vectors = self.model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=len(texts) > 500
        )
        return np.asarray(vectors, dtype=np.float32)


class DenseIndex:
    """Exact FAISS index (IndexFlatIP). Row ``i`` is chunk ``i`` in chunks.jsonl order."""

    def __init__(self, index: faiss.Index) -> None:
        self.index = index

    @classmethod
    def build(cls, vectors: np.ndarray) -> DenseIndex:
        """Create an index from unit vectors."""
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        return cls(index)

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        """Read an index written by :meth:`save`."""
        return cls(faiss.read_index(str(path)))

    def save(self, path: Path) -> None:
        """Write the index to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    @property
    def size(self) -> int:
        """Number of vectors in the index."""
        return int(self.index.ntotal)

    def search(self, query_vector: np.ndarray, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-``k`` (index, cosine) pairs; ``allowed`` restricts the search inside FAISS."""
        query = np.ascontiguousarray(query_vector.reshape(1, -1), dtype=np.float32)
        if allowed is not None:
            if allowed.size == 0:
                return []
            params = faiss.SearchParameters(sel=faiss.IDSelectorBatch(allowed.astype(np.int64)))
            scores, ids = self.index.search(query, min(k, allowed.size), params=params)
        else:
            scores, ids = self.index.search(query, min(k, self.size))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0], strict=True) if i >= 0]
