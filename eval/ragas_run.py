"""RAGAS faithfulness and answer relevancy on a fixed subset of answerable golden items.

    python -m eval.ragas_run                          # every config in config.yaml
    python -m eval.ragas_run --config rerank_tiered
    python -m eval.ragas_run --config rerank_tiered --repeat   # uncached second run: judge noise

Needs ``pip install -r requirements-eval.txt`` (RAGAS and its LangChain dependencies) and
``make eval-run`` first. Writes ``eval/results/<config>_ragas.csv`` (or ``_ragas_repeat.csv``).

Why a subset: the free judge tier allows roughly 200K tokens a day, so faithfulness and answer
relevancy run on a fixed, stratified 20-item subset (``eval/subsets.json``). Context
quality is measured without a judge instead (hit@k, MRR, support coverage in ``eval.run``).

Judge replies are cached on disk, so a run stopped by a rate limit resumes where it left off;
items already scored are skipped. ``--repeat`` disables the cache so the same answers are
judged again, which measures how much the judge's own scores vary.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import sys
import time

from healthrag.config import Settings, get_settings
from healthrag.evaluation.ratelimit import call_with_patience
from healthrag.evaluation.subsets import load_or_create
from healthrag.retrieve.dense import SentenceTransformerEmbedder

log = logging.getLogger("eval.ragas")

FIELDS = ["id", "type", "faithfulness", "answer_relevancy", "judge_model", "error"]
MIN_SECONDS_BETWEEN_ITEMS = 2.0  # stays well under the free tier's 30 requests per minute
MAX_WAIT_HOURS = 36


def make_subset(settings: Settings) -> list[str]:
    """The fixed RAGAS subset from eval/subsets.json (created on first use)."""
    golden = [json.loads(line) for line in settings.evaluation.golden_file.read_text(encoding="utf-8").splitlines()]
    ragas, _ = load_or_create(settings.evaluation.subsets_file, golden, settings.evaluation.seed)
    return ragas


class PinnedEmbeddings:
    """RAGAS embeddings backed by the project's pinned bge-small model."""

    def __init__(self, settings: Settings) -> None:
        r = settings.retrieval
        self._embedder = SentenceTransformerEmbedder(
            r.embedding_model, r.embedding_revision, r.query_instruction, r.embed_batch_size, r.max_seq_length
        )

    def embed_text(self, text: str, **kwargs) -> list[float]:  # noqa: ANN003
        """Embed one text."""
        return self._embedder.encode([text])[0].tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:  # noqa: ANN003
        """Embed one text (async interface)."""
        return self.embed_text(text)

    def embed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:  # noqa: ANN003
        """Embed several texts."""
        return self._embedder.encode(texts).tolist()

    async def aembed_texts(self, texts: list[str], **kwargs) -> list[list[float]]:  # noqa: ANN003
        """Embed several texts (async interface)."""
        return self.embed_texts(texts)


def build_metrics(settings: Settings, use_cache: bool):  # noqa: ANN201
    """Faithfulness and AnswerRelevancy wired to the judge endpoint."""
    from openai import AsyncOpenAI
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings.base import BaseRagasEmbedding
    from ragas.llms import llm_factory
    from ragas.metrics.collections import AnswerRelevancy, Faithfulness

    base_url, api_key = settings.judge_endpoint()
    if not (base_url and api_key):
        raise RuntimeError("Set JUDGE_BASE_URL/JUDGE_API_KEY (or the LARGE_* endpoint) in .env")
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key.get_secret_value(),
        max_retries=settings.llm.max_retries,
        timeout=settings.llm.timeout_seconds,
    )
    cache = DiskCacheBackend(cache_dir=str(settings.llm.cache_dir / "ragas")) if use_cache else None
    llm = llm_factory(settings.llm.judge_model, client=client, cache=cache, temperature=0.0, max_tokens=2048)
    BaseRagasEmbedding.register(PinnedEmbeddings)  # satisfies RAGAS's type check without subclassing
    embeddings = PinnedEmbeddings(settings)
    return Faithfulness(llm=llm), AnswerRelevancy(llm=llm, embeddings=embeddings)


def _score(value) -> float:  # noqa: ANN001
    number = float(getattr(value, "value", value))
    return number if math.isfinite(number) else float("nan")


def run_config(name: str, settings: Settings, subset: list[str], repeat: bool) -> bool:
    """Score one config's answers; returns False if it had to stop early (e.g. rate limit)."""
    results_dir = settings.evaluation.results_dir
    answers_file = results_dir / f"{name}_answers.jsonl"
    if not answers_file.exists():
        raise FileNotFoundError(f"{answers_file} missing; run `python -m eval.run --config {name}` first")
    answers = {a["id"]: a for a in map(json.loads, answers_file.read_text(encoding="utf-8").splitlines())}
    out = results_dir / f"{name}_ragas{'_repeat' if repeat else ''}.csv"
    done = {}
    if out.exists():
        with out.open(encoding="utf-8") as handle:
            done = {row["id"]: row for row in csv.DictReader(handle) if not row["error"]}
    faithfulness, relevancy = build_metrics(settings, use_cache=not repeat)
    rows = list(done.values())
    finished = True
    for item_id in subset:
        if item_id in done:
            continue
        a = answers[item_id]
        started = time.perf_counter()
        try:
            f = call_with_patience(
                lambda a=a: asyncio.run(faithfulness.ascore(a["question"], a["answer"], a["contexts"])),
                f"{name} {item_id} faithfulness",
                MAX_WAIT_HOURS,
            )
            r = call_with_patience(
                lambda a=a: asyncio.run(relevancy.ascore(a["question"], a["answer"])),
                f"{name} {item_id} relevancy",
                MAX_WAIT_HOURS,
            )
            rows.append({"id": item_id, "type": a["type"], "faithfulness": round(_score(f), 4),
                         "answer_relevancy": round(_score(r), 4), "judge_model": settings.llm.judge_model,
                         "error": ""})  # fmt: skip
        except Exception as error:  # noqa: BLE001 - record and stop; rate limits end the day's run
            log.warning("%s %s: %s", name, item_id, str(error)[:200])
            rows.append({"id": item_id, "type": a["type"], "faithfulness": "", "answer_relevancy": "",
                         "judge_model": settings.llm.judge_model, "error": type(error).__name__})  # fmt: skip
            if "RateLimit" in type(error).__name__ or "429" in str(error):
                finished = False
                break
        time.sleep(max(0.0, MIN_SECONDS_BETWEEN_ITEMS - (time.perf_counter() - started)))
    order = {item_id: n for n, item_id in enumerate(subset)}
    rows.sort(key=lambda row: order.get(row["id"], len(order)))
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    scored = sum(1 for row in rows if not row["error"])
    log.info("%s: %d / %d items scored -> %s", name, scored, len(subset), out)
    return finished


def main(argv: list[str] | None = None) -> int:
    """Score the requested configs (default: all); exit 2 if a rate limit stopped the run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append")
    parser.add_argument("--repeat", action="store_true", help="judge again without the cache (noise run)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers", "ragas", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    subset = make_subset(settings)
    for name in args.config or list(settings.evaluation.configs):
        if not run_config(name, settings, subset, args.repeat):
            log.warning("Stopped by the judge's rate limit; run the same command again later to resume.")
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
