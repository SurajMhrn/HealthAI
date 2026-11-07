"""FastAPI service.

    uvicorn healthrag.api:app --port 8000      (or: make api)

Endpoints:

* ``POST /ask``: answer a question (optionally filtered) with citations, route, latency,
  cost and guardrail status
* ``GET /chunks/{chunk_id}``: one redacted chunk, for citation pop-ups
* ``GET /patients``: pseudonymous patient IDs, for the UI's patient filter
* ``GET /health``: liveness

Everything returned is redacted: chunk text comes from chunks.jsonl and answers pass the
output checks.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from healthrag.config import get_settings
from healthrag.generate.factory import build_answerer
from healthrag.generate.pipeline import Answerer
from healthrag.retrieve.filters import Filters

app = FastAPI(title="Safe clinical RAG assistant", version="0.1.0")


class AskRequest(BaseModel):
    """A question and optional retrieval filters."""

    question: str = Field(min_length=1, max_length=2000)
    patient_id: str | None = None
    doc_type: Literal["patient_record", "guideline"] | None = None
    date_from: str | None = None
    date_to: str | None = None


class Citation(BaseModel):
    """A cited (or retrieved) chunk, redacted."""

    chunk_id: str
    source: str
    section: str
    page: int | None
    date: str | None
    text: str
    cited: bool


class AskResponse(BaseModel):
    """The answer and what it took to produce it."""

    answer: str
    refused: bool
    guardrail: str | None
    route: str
    route_reason: str
    router: str
    model: str
    sources: list[Citation]
    invalid_citations: list[str]
    answer_phi_hits: int
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    latency_ms: dict[str, float]


@lru_cache(maxsize=1)
def answerer() -> Answerer:
    """Load the models once per process."""
    return build_answerer(get_settings())


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check (doesn't load the models)."""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Answer a question from the documents."""
    filters = Filters(request.doc_type, request.patient_id, request.date_from, request.date_to)
    a = answerer().answer(request.question, filters)
    cited = set(a.citations)
    sources = [
        Citation(
            chunk_id=h.chunk_id,
            source=h.record["source"],
            section=h.record["section"],
            page=h.record.get("page"),
            date=h.record.get("date"),
            text=h.record["text"],
            cited=h.chunk_id in cited,
        )
        for h in a.hits
    ]
    return AskResponse(
        answer=a.text,
        refused=a.refused,
        guardrail=a.guardrail,
        route=a.route.tier,
        route_reason=a.route.reason,
        router=a.route.router,
        model=a.model,
        sources=sources,
        invalid_citations=a.invalid_citations,
        answer_phi_hits=a.answer_phi_hits,
        tokens_in=a.tokens_in,
        tokens_out=a.tokens_out,
        estimated_cost_usd=a.estimated_cost_usd,
        latency_ms={k: round(v, 1) for k, v in a.timings_ms.items()},
    )


@app.get("/chunks/{chunk_id}", response_model=Citation)
def chunk(chunk_id: str) -> Citation:
    """One redacted chunk by ID."""
    for record in answerer().retriever.records:
        if record["chunk_id"] == chunk_id:
            return Citation(
                chunk_id=chunk_id, source=record["source"], section=record["section"], page=record.get("page"),
                date=record.get("date"), text=record["text"], cited=False,
            )  # fmt: skip
    raise HTTPException(status_code=404, detail="unknown chunk_id")


@app.get("/patients")
def patients() -> list[dict[str, str | int]]:
    """Pseudonymous patient IDs with their chunk counts (no identifiers)."""
    counts = Counter(r["patient_id"] for r in answerer().retriever.records if r.get("patient_id"))
    return [{"patient_id": pid, "chunks": n} for pid, n in sorted(counts.items())]
