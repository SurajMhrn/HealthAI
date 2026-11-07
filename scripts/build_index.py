"""Embed every redacted chunk with bge-small and write the FAISS index.

Run with ``python -m scripts.build_index`` (or ``make index``) after ``make ingest``.
Writes ``indexes/dense.faiss`` and ``indexes/manifest.json``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone

from healthrag.config import get_settings
from healthrag.retrieve.dense import DenseIndex, SentenceTransformerEmbedder
from healthrag.retrieve.retriever import INDEX_FILE, MANIFEST_FILE, chunks_fingerprint, load_chunks

log = logging.getLogger("build_index")


def main() -> int:
    """Build the dense index from data/processed/chunks.jsonl."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    r = settings.retrieval
    records = load_chunks(settings.paths.chunks_file)
    log.info("Embedding %d chunks with %s@%s", len(records), r.embedding_model, r.embedding_revision[:8])
    embedder = SentenceTransformerEmbedder(
        r.embedding_model, r.embedding_revision, r.query_instruction, r.embed_batch_size, r.max_seq_length
    )
    started = time.perf_counter()
    vectors = embedder.encode([rec["text"] for rec in records])
    seconds = time.perf_counter() - started
    index = DenseIndex.build(vectors)
    index.save(r.index_dir / INDEX_FILE)
    manifest = {
        "embedding_model": r.embedding_model,
        "embedding_revision": r.embedding_revision,
        "dimension": int(vectors.shape[1]),
        "chunks": len(records),
        "chunks_fingerprint": chunks_fingerprint(records),
        "embedding_seconds": round(seconds, 1),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (r.index_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log.info("Wrote %s (%d x %d) in %.1f s", r.index_dir / INDEX_FILE, len(records), vectors.shape[1], seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
