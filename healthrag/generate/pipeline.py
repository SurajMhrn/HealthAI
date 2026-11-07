"""Question answering: redact the question, route, retrieve, generate, check the answer, log.

1. The question is redacted with Presidio, so nothing unredacted reaches retrieval, the
   LLM or the logs.
2. The router picks the small or large tier and gives a reason.
3. Hybrid retrieval (with filters) returns the top chunks.
4. The chosen model answers from those chunks only, citing ``[chunk_id]``.
5. Presidio checks the answer again; any hit is redacted, counted and logged.
6. A telemetry record goes to ``logs/queries.jsonl``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from healthrag.config import Settings
from healthrag.generate.llm import ChatClient, estimated_cost
from healthrag.generate.prompts import build_messages, extract_citations, is_refusal, normalize_citations
from healthrag.generate.router import RouteDecision, Router
from healthrag.generate.telemetry import log_query
from healthrag.redact.engine import Redactor
from healthrag.retrieve.filters import Filters
from healthrag.retrieve.retriever import Hit, Retriever


@dataclass
class Answer:
    """The answer and everything needed to show and audit it."""

    text: str
    question: str
    citations: list[str]
    invalid_citations: list[str]
    hits: list[Hit]
    route: RouteDecision
    refused: bool
    answer_phi_hits: int
    model: str
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    timings_ms: dict[str, float] = field(default_factory=dict)
    guardrail: str | None = None


class Answerer:
    """Ties redaction, routing, retrieval and generation together."""

    def __init__(
        self, settings: Settings, retriever: Retriever, router: Router, client: ChatClient, redactor: Redactor
    ) -> None:
        self.settings = settings
        self.retriever = retriever
        self.router = router
        self.client = client
        self.redactor = redactor

    def answer(self, question: str, filters: Filters | None = None, log: bool = True) -> Answer:
        """Answer one question within ``filters``."""
        g = self.settings.generation
        timings: dict[str, float] = {}
        overall = time.perf_counter()

        started = time.perf_counter()
        safe_question = self.redactor.redact(question).text
        timings["redact_query"] = (time.perf_counter() - started) * 1000

        route = self.router.route(safe_question)
        timings["route"] = route.latency_ms

        result = self.retriever.search(safe_question, filters, g.retrieval_mode)
        for stage, ms in result.timings_ms.items():
            timings[f"retrieve_{stage}"] = ms
        records = [h.record for h in result.hits]

        if records:
            messages = build_messages(safe_question, records, g.not_found_answer, g.max_context_chars_per_chunk)
            response = self.client.chat(route.tier, messages)  # type: ignore[arg-type]
            raw_answer, model = normalize_citations(response.text), response.model
            tokens_in, tokens_out = response.tokens_in, response.tokens_out
            timings["generate"] = response.latency_ms
        else:
            raw_answer, model, tokens_in, tokens_out = g.not_found_answer, "none", 0, 0
            timings["generate"] = 0.0

        started = time.perf_counter()
        checked = self.redactor.redact(raw_answer)
        timings["redact_answer"] = (time.perf_counter() - started) * 1000
        timings["total"] = (time.perf_counter() - overall) * 1000

        retrieved_ids = {h.chunk_id for h in result.hits}
        citations = extract_citations(checked.text)
        cost = estimated_cost(self.settings, model, tokens_in, tokens_out) + estimated_cost(
            self.settings, route.model or "", route.tokens_in, route.tokens_out
        )
        answer = Answer(
            text=checked.text,
            question=safe_question,
            citations=[c for c in citations if c in retrieved_ids],
            invalid_citations=[c for c in citations if c not in retrieved_ids],
            hits=result.hits,
            route=route,
            refused=is_refusal(checked.text, g.not_found_answer),
            answer_phi_hits=len(checked.detections),
            model=model,
            tokens_in=tokens_in + route.tokens_in,
            tokens_out=tokens_out + route.tokens_out,
            estimated_cost_usd=cost,
            timings_ms=timings,
        )
        if log:
            log_query(g.telemetry_file, telemetry_record(answer, filters))
        return answer


def telemetry_record(answer: Answer, filters: Filters | None) -> dict:
    """The JSON line written to logs/queries.jsonl. Contains only redacted text."""
    return {
        "query": answer.question,
        "filters": {k: v for k, v in vars(filters).items() if v} if filters else {},
        "router": answer.route.router,
        "route": answer.route.tier,
        "route_reason": answer.route.reason,
        "model": answer.model,
        "retrieved": [h.chunk_id for h in answer.hits],
        "citations": answer.citations,
        "invalid_citations": answer.invalid_citations,
        "refused": answer.refused,
        "answer_phi_hits": answer.answer_phi_hits,
        "guardrail": answer.guardrail,
        "tokens_in": answer.tokens_in,
        "tokens_out": answer.tokens_out,
        "estimated_cost_usd": round(answer.estimated_cost_usd, 8),
        "actual_cost_usd": 0.0,
        "latency_ms": {k: round(v, 1) for k, v in answer.timings_ms.items()},
    }
