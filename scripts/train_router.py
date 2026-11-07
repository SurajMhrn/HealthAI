"""Train the logistic-regression router on bge-small embeddings of the train split.

Run with ``python -m scripts.train_router`` (or ``make train-router``) after
``make router-labels``. Saves ``indexes/router.joblib``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

import joblib
from sklearn.linear_model import LogisticRegression

from healthrag.config import get_settings
from healthrag.retrieve.dense import SentenceTransformerEmbedder

log = logging.getLogger("train_router")


def load_split(path, split: str) -> list[dict]:  # noqa: ANN001
    """Items of one split from router_labels.jsonl."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [row for row in rows if row["split"] == split]


def main() -> int:
    """Fit the classifier and save it with the embedding model it expects."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    r, rt = settings.retrieval, settings.router
    train = load_split(rt.labels_file, "train")
    embedder = SentenceTransformerEmbedder(
        r.embedding_model, r.embedding_revision, r.query_instruction, r.embed_batch_size, r.max_seq_length
    )
    vectors = embedder.encode([row["query"] for row in train])
    labels = [row["label"] for row in train]
    classifier = LogisticRegression(C=rt.logistic_c, class_weight="balanced", max_iter=2000, random_state=rt.seed)
    classifier.fit(vectors, labels)
    train_accuracy = float(classifier.score(vectors, labels))
    rt.model_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "classifier": classifier,
            "embedding_model": r.embedding_model,
            "embedding_revision": r.embedding_revision,
            "train_items": len(train),
            "train_accuracy": train_accuracy,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        rt.model_file,
    )
    log.info("Trained on %d items (train accuracy %.3f); saved %s", len(train), train_accuracy, rt.model_file)
    return 0


if __name__ == "__main__":
    sys.exit(main())
