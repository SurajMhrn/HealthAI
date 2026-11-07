"""Sparse retrieval with BM25 (rank_bm25's Okapi BM25)."""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)*")

# A short English stop-word list; clinical words are never on it.
STOP_WORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were what when "
    "which who with does do did how why should".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stop words. Keeps codes like ``e11.9`` and ``covid-19`` whole."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOP_WORDS]


class Bm25Index:
    """BM25 over chunk texts. Built in memory; ~6k chunks take about a second."""

    def __init__(self, texts: list[str]) -> None:
        self._bm25 = BM25Okapi([tokenize(t) or ["_"] for t in texts])
        self.size = len(texts)

    def search(self, query: str, k: int, allowed: np.ndarray | None = None) -> list[tuple[int, float]]:
        """Top-``k`` (index, score) pairs, restricted to ``allowed`` indices when given."""
        scores = np.asarray(self._bm25.get_scores(tokenize(query)), dtype=np.float64)
        candidates = np.arange(self.size) if allowed is None else allowed
        if candidates.size == 0:
            return []
        chosen = candidates[np.argsort(-scores[candidates], kind="stable")[:k]]
        return [(int(i), float(scores[i])) for i in chosen if scores[i] > 0]
