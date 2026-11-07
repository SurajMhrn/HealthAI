"""Run the golden set through one or more pipeline configurations and record judge-free metrics.

    python -m eval.run                       # all five configs from config.yaml
    python -m eval.run --config rerank_tiered

For each config this writes ``eval/results/<config>.csv`` (one row per golden item) and
``eval/results/<config>_answers.jsonl`` (question, answer, contexts and reference, for RAGAS).

Judge-free metrics per item:

* ``hit_rank``: rank of the first retrieved chunk that carries a golden support (empty = miss)
* ``support_coverage``: share of an item's supports found in the top-k (multi-hop items have two)
* ``answer_match``: the answer contains the expected facts (dates accept common formats)
* ``refused``: the answer says the information is not in the documents

Guardrails are switched off here: they have their own evaluation (``make eval-guard``), and
leaving them on would spend one extra model call per question.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time

from healthrag.config import Settings, get_settings
from healthrag.evaluation.matching import answer_matches, supports
from healthrag.evaluation.ratelimit import call_with_patience
from healthrag.evaluation.subsets import load_or_create
from healthrag.generate.factory import query_redactor
from healthrag.generate.llm import LlmClient
from healthrag.generate.pipeline import Answerer
from healthrag.generate.router import COMPLEX, SIMPLE, FixedRouter, make_rules_router
from healthrag.retrieve.filters import Filters
from healthrag.retrieve.retriever import Retriever

log = logging.getLogger("eval.run")

FIELDS = ["id", "type", "answerable", "route", "model", "hit_rank", "support_coverage", "answer_match",
          "refused", "n_citations", "invalid_citations", "answer_phi_hits", "tokens_in", "tokens_out",
          "est_cost_usd", "latency_ms", "cached"]  # fmt: skip


def load_golden(settings: Settings) -> list[dict]:
    """Golden items from eval/golden.jsonl."""
    path = settings.evaluation.golden_file
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


MAX_WAIT_HOURS = 36  # a free-tier daily cap can take most of a day to clear


def answer_with_patience(answerer: Answerer, item: dict, name: str, number: int):  # noqa: ANN201
    """Answer one item, waiting out rate limits (answers are cached, so nothing is spent twice)."""
    return call_with_patience(
        lambda: answerer.answer(item["question"], Filters(**item.get("filters", {})), log=False),
        f"{name} item {number}",
        MAX_WAIT_HOURS,
    )


def run_config(name: str, settings: Settings, answerer: Answerer, items: list[dict]) -> None:
    """Answer every golden item with one configuration and write its CSV and answers file."""
    config = settings.evaluation.configs[name]
    if config.items == "comparison":
        _, comparison = load_or_create(settings.evaluation.subsets_file, items, settings.evaluation.seed)
        keep = set(comparison)
        items = [item for item in items if item["id"] in keep]
    log.info("%s: %d items", name, len(items))
    answerer.retrieval_mode = config.retrieval_mode
    answerer.router = make_rules_router(settings) if config.generator == "tiered" else FixedRouter(
        SIMPLE if config.generator == "small" else COMPLEX
    )
    out_dir = settings.evaluation.results_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, answers = [], []
    for number, item in enumerate(items, start=1):
        started = time.perf_counter()
        answer = answer_with_patience(answerer, item, name, number)
        elapsed = (time.perf_counter() - started) * 1000
        records = [h.record for h in answer.hits]
        hit_rank = next(
            (rank for rank, r in enumerate(records, 1) if any(supports(r, s) for s in item["support"])), None
        )
        covered = sum(1 for s in item["support"] if any(supports(r, s) for r in records))
        answerable = item["type"] != "unanswerable"
        rows.append(
            {
                "id": item["id"],
                "type": item["type"],
                "answerable": answerable,
                "route": answer.route.tier,
                "model": answer.model,
                "hit_rank": hit_rank or "",
                "support_coverage": round(covered / len(item["support"]), 3) if item["support"] else "",
                "answer_match": answer_matches(answer.text, item["answer_contains"]) if answerable else "",
                "refused": answer.refused,
                "n_citations": len(answer.citations),
                "invalid_citations": len(answer.invalid_citations),
                "answer_phi_hits": answer.answer_phi_hits,
                "tokens_in": answer.tokens_in,
                "tokens_out": answer.tokens_out,
                "est_cost_usd": round(answer.estimated_cost_usd, 8),
                "latency_ms": round(answer.timings_ms.get("total", elapsed), 1),
                "cached": answer.timings_ms.get("generate", 1.0) == 0.0,
            }
        )
        answers.append(
            {
                "id": item["id"],
                "type": item["type"],
                "question": answer.question,
                "answer": answer.text,
                "contexts": [r["text"] for r in records],
                "reference": item.get("reference", ""),
            }
        )
        if number % 10 == 0:
            log.info("%s: %d / %d", name, number, len(items))
    with (out_dir / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / f"{name}_answers.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for a in answers:
            handle.write(json.dumps(a, ensure_ascii=False) + "\n")
    log.info("%s: wrote %d rows", name, len(rows))


def main(argv: list[str] | None = None) -> int:
    """Run the requested configs (default: all)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", action="append", help="config name from config.yaml (repeatable)")
    parser.add_argument("--limit", type=int, help="only the first N golden items (for a quick check)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
    settings = get_settings()
    names = args.config or list(settings.evaluation.configs)
    items = load_golden(settings)[: args.limit] if args.limit else load_golden(settings)
    retriever = Retriever.from_settings(settings)
    client = LlmClient(settings)
    answerer = Answerer(settings, retriever, make_rules_router(settings), client, query_redactor(settings))
    for name in names:
        run_config(name, settings, answerer, items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
