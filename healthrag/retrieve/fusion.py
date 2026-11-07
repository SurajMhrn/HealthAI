"""Reciprocal rank fusion (Cormack et al., 2009)."""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(ranked_lists: list[list[int]], k: int) -> list[tuple[int, float]]:
    """Fuse ranked lists of ids: score(d) = sum over lists of 1 / (k + rank), rank starting at 1.

    Ties are broken by the id's best rank in any list, then by id, so the output is deterministic.
    """
    scores: dict[int, float] = defaultdict(float)
    best_rank: dict[int, int] = {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            scores[doc] += 1.0 / (k + rank)
            best_rank[doc] = min(best_rank.get(doc, rank), rank)
    return sorted(scores.items(), key=lambda item: (-item[1], best_rank[item[0]], item[0]))
