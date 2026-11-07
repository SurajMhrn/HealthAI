"""Routing between the small and the large model. Three interchangeable routers:

``rules``      keyword rules: multi-patient, summarize/compare/trend, or several clinical terms -> large
``rules_llm``  the rules first; if they say "simple", the small model classifies the question (JSON)
``trained``    logistic regression on bge-small embeddings, trained by ``scripts/train_router.py``

Every decision carries a ``reason`` that is logged with the query.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from healthrag.config import Settings
from healthrag.generate.llm import ChatClient
from healthrag.retrieve.dense import Embedder

SIMPLE, COMPLEX = "simple", "complex"
TIER_FOR_LABEL = {SIMPLE: "small", COMPLEX: "large"}

CLASSIFIER_PROMPT = """Classify a question for a clinical document assistant.

"simple": answerable from one fact in one patient record or one guideline passage
(e.g. one medication, one date, one recommendation).
"complex": needs several documents or reasoning across facts: summaries, comparisons, trends over
time, questions about several patients, or several conditions together.

Reply with JSON only: {"label": "simple" or "complex", "reason": "<at most 12 words>"}"""


@dataclass(frozen=True)
class RouteDecision:
    """Which tier to use and why."""

    label: str
    reason: str
    router: str
    latency_ms: float
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0

    @property
    def tier(self) -> str:
        """``small`` or ``large``."""
        return TIER_FOR_LABEL[self.label]


class Router(Protocol):
    """Anything that labels a question simple or complex."""

    name: str

    def route(self, question: str) -> RouteDecision:
        """Decide the tier for ``question``."""
        ...


def _contains(text: str, phrase: str) -> bool:
    if phrase != phrase.strip():  # phrases with spaces around them, like " vs ", match literally
        return phrase in f" {text} "
    return re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text) is not None


class RulesRouter:
    """Keyword rules. Fast, free, and easy to explain."""

    name = "rules"

    def __init__(self, complex_phrases: list[str], clinical_terms: list[str], multi_entity_threshold: int) -> None:
        self.complex_phrases = [p.lower() for p in complex_phrases]
        self.clinical_terms = [t.lower() for t in clinical_terms]
        self.threshold = multi_entity_threshold

    def route(self, question: str) -> RouteDecision:
        """Complex if a complex phrase appears or enough distinct clinical terms are mentioned."""
        started = time.perf_counter()
        text = question.lower()
        phrases = [p.strip() for p in self.complex_phrases if _contains(text, p)]
        terms = [t for t in self.clinical_terms if _contains(text, t)]
        if phrases:
            label, reason = COMPLEX, f"complex phrase: {', '.join(phrases)}"
        elif len(terms) >= self.threshold:
            label, reason = COMPLEX, f"{len(terms)} clinical terms: {', '.join(terms)}"
        else:
            label, reason = SIMPLE, "no complex phrase" + (f"; term: {terms[0]}" if terms else "")
        return RouteDecision(label, reason, self.name, (time.perf_counter() - started) * 1000)


class RulesLlmRouter:
    """Rules first; questions the rules call simple are double-checked by the small model."""

    name = "rules_llm"

    def __init__(self, rules: RulesRouter, client: ChatClient) -> None:
        self.rules = rules
        self.client = client

    def route(self, question: str) -> RouteDecision:
        """Use the rules' "complex", otherwise ask the small model.

        Latency is the rules' time plus the model call's measured latency (kept with cached
        replies), so re-runs report the real cost of an uncached call.
        """
        first = self.rules.route(question)
        if first.label == COMPLEX:
            return RouteDecision(COMPLEX, f"rules: {first.reason}", self.name, first.latency_ms)
        messages = [{"role": "system", "content": CLASSIFIER_PROMPT}, {"role": "user", "content": question}]
        response = self.client.chat("small", messages, json_mode=True)
        label, reason = parse_classifier_reply(response.text)
        return RouteDecision(
            label,
            f"llm: {reason}",
            self.name,
            first.latency_ms + response.latency_ms,
            model=response.model,
            tokens_in=response.tokens_in,
            tokens_out=response.tokens_out,
        )


def parse_classifier_reply(text: str) -> tuple[str, str]:
    """Read ``{"label": ..., "reason": ...}``; anything unreadable falls back to simple."""
    try:
        data = json.loads(text)
        label = str(data.get("label", "")).strip().lower()
        if label in (SIMPLE, COMPLEX):
            return label, str(data.get("reason", "")).strip()[:120]
    except (json.JSONDecodeError, AttributeError):
        pass
    return SIMPLE, "unparseable classifier reply, defaulted to simple"


class TrainedRouter:
    """Logistic regression on question embeddings."""

    name = "trained"

    def __init__(self, model, embedder: Embedder) -> None:  # noqa: ANN001 (a fitted scikit-learn classifier)
        self.model = model
        self.embedder = embedder

    @classmethod
    def load(cls, path: Path, embedder: Embedder) -> TrainedRouter:
        """Load a classifier saved by ``scripts/train_router.py``."""
        import joblib

        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `make train-router` first.")
        return cls(joblib.load(path)["classifier"], embedder)

    def route(self, question: str) -> RouteDecision:
        """Predict simple/complex and report the probability of "complex"."""
        started = time.perf_counter()
        vector = self.embedder.encode([question])
        p_complex = float(self.model.predict_proba(vector)[0][list(self.model.classes_).index(COMPLEX)])
        label = COMPLEX if p_complex >= 0.5 else SIMPLE
        return RouteDecision(
            label, f"p(complex) = {p_complex:.2f}", self.name, (time.perf_counter() - started) * 1000
        )

    def predict(self, vectors: np.ndarray) -> list[str]:
        """Labels for pre-computed embeddings (used in evaluation)."""
        return [str(label) for label in self.model.predict(vectors)]


def make_rules_router(settings: Settings) -> RulesRouter:
    """The rules router configured from settings."""
    r = settings.router
    return RulesRouter(r.complex_phrases, r.clinical_terms, r.multi_entity_threshold)


def make_router(name: str, settings: Settings, client: ChatClient | None, embedder: Embedder | None) -> Router:
    """Build the router called ``name`` (rules, rules_llm or trained)."""
    if name == "rules":
        return make_rules_router(settings)
    if name == "rules_llm":
        if client is None:
            raise ValueError("rules_llm needs an LLM client")
        return RulesLlmRouter(make_rules_router(settings), client)
    if name == "trained":
        if embedder is None:
            raise ValueError("trained router needs an embedder")
        return TrainedRouter.load(settings.router.model_file, embedder)
    raise ValueError(f"unknown router {name!r}")
