"""The retrieval pipeline and its four comparable modes.

``dense``          FAISS only
``bm25``           BM25 only
``hybrid``         BM25 + FAISS fused with RRF
``hybrid_rerank``  hybrid, then the cross-encoder re-ranks the top ``rerank_top_n``

Filters are applied first, inside each retriever, so fusion and re-ranking only
ever see chunks that pass them.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from healthrag.config import RetrievalSettings, Settings
from healthrag.retrieve.bm25 import Bm25Index
from healthrag.retrieve.dense import DenseIndex, Embedder, SentenceTransformerEmbedder
from healthrag.retrieve.filters import Filters, allowed_indices
from healthrag.retrieve.fusion import reciprocal_rank_fusion
from healthrag.retrieve.rerank import CrossEncoderReranker, Reranker

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

INDEX_FILE = "dense.faiss"
MANIFEST_FILE = "manifest.json"


@dataclass
class Hit:
    """One retrieved chunk with the score from the final stage and per-stage ranks."""

    record: dict
    score: float
    ranks: dict[str, int] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """The chunk's ID."""
        return str(self.record["chunk_id"])


@dataclass
class SearchResult:
    """Hits plus how long each stage took, in milliseconds."""

    hits: list[Hit]
    timings_ms: dict[str, float]


def load_chunks(path: Path) -> list[dict]:
    """Read chunks.jsonl into a list of records."""
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def chunks_fingerprint(records: list[dict]) -> str:
    """Hash of the chunk IDs in order; ties an index to the exact chunks file it was built from."""
    return hashlib.sha256("\n".join(r["chunk_id"] for r in records).encode("utf-8")).hexdigest()


class Retriever:
    """Hybrid retriever over redacted chunks."""

    def __init__(
        self,
        records: list[dict],
        dense: DenseIndex,
        embedder: Embedder,
        reranker: Reranker | None,
        config: RetrievalSettings,
    ) -> None:
        if dense.size != len(records):
            raise ValueError(f"Index has {dense.size} vectors but there are {len(records)} chunks")
        self.records = records
        self.dense = dense
        self.embedder = embedder
        self.reranker = reranker
        self.config = config
        self.bm25 = Bm25Index([r["text"] for r in records])

    @classmethod
    def from_settings(cls, settings: Settings, with_reranker: bool = True) -> Retriever:
        """Load chunks, the FAISS index and the models named in config."""
        r = settings.retrieval
        records = load_chunks(settings.paths.chunks_file)
        manifest = json.loads((r.index_dir / MANIFEST_FILE).read_text(encoding="utf-8"))
        if manifest["chunks_fingerprint"] != chunks_fingerprint(records):
            raise RuntimeError("The index is out of date with chunks.jsonl. Run `make index`.")
        if manifest["embedding_model"] != r.embedding_model or manifest["embedding_revision"] != r.embedding_revision:
            raise RuntimeError("The index was built with a different embedding model. Run `make index`.")
        embedder = SentenceTransformerEmbedder(
            r.embedding_model, r.embedding_revision, r.query_instruction, r.embed_batch_size, r.max_seq_length
        )
        reranker = (
            CrossEncoderReranker(r.rerank_model, r.rerank_revision, r.rerank_batch_size, r.max_seq_length)
            if with_reranker
            else None
        )
        return cls(records, DenseIndex.load(r.index_dir / INDEX_FILE), embedder, reranker, r)

    def search(self, query: str, filters: Filters | None = None, mode: str = "hybrid_rerank") -> SearchResult:
        """Retrieve the top ``final_k`` chunks for ``query`` using one of :data:`MODES`."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        c = self.config
        timings: dict[str, float] = {}
        start = time.perf_counter()
        allowed = allowed_indices(self.records, filters)
        timings["filter"] = (time.perf_counter() - start) * 1000

        dense_hits: list[tuple[int, float]] = []
        bm25_hits: list[tuple[int, float]] = []
        if mode != "bm25":
            start = time.perf_counter()
            vector = self.embedder.encode([query], is_query=True)[0]
            dense_hits = self.dense.search(vector, c.candidates_per_retriever, allowed)
            timings["dense"] = (time.perf_counter() - start) * 1000
        if mode != "dense":
            start = time.perf_counter()
            bm25_hits = self.bm25.search(query, c.candidates_per_retriever, allowed)
            timings["bm25"] = (time.perf_counter() - start) * 1000

        dense_rank = {i: rank for rank, (i, _) in enumerate(dense_hits, start=1)}
        bm25_rank = {i: rank for rank, (i, _) in enumerate(bm25_hits, start=1)}

        if mode == "dense":
            ranked = dense_hits
        elif mode == "bm25":
            ranked = bm25_hits
        else:
            start = time.perf_counter()
            ranked = reciprocal_rank_fusion([[i for i, _ in dense_hits], [i for i, _ in bm25_hits]], c.rrf_k)
            timings["fusion"] = (time.perf_counter() - start) * 1000

        if mode == "hybrid_rerank":
            if self.reranker is None:
                raise RuntimeError("This retriever was loaded without a re-ranker")
            start = time.perf_counter()
            pool = [i for i, _ in ranked[: c.rerank_top_n]]
            scores = self.reranker.score(query, [self.records[i]["text"] for i in pool])
            ranked = sorted(zip(pool, scores, strict=True), key=lambda item: (-item[1], item[0]))
            timings["rerank"] = (time.perf_counter() - start) * 1000

        hits = []
        for i, score in ranked[: c.final_k]:
            ranks = {}
            if i in dense_rank:
                ranks["dense"] = dense_rank[i]
            if i in bm25_rank:
                ranks["bm25"] = bm25_rank[i]
            hits.append(Hit(self.records[i], float(score), ranks))
        timings["total"] = sum(timings.values())
        return SearchResult(hits, timings)
