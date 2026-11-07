"""Build a ready-to-use :class:`Answerer` from settings (shared by the CLI, API and UI)."""

from __future__ import annotations

from healthrag.config import Settings
from healthrag.generate.llm import LlmClient
from healthrag.generate.pipeline import Answerer
from healthrag.generate.router import make_router
from healthrag.ingest.pipeline import deny_list_names, load_all_patients
from healthrag.ingest.synthea import clinical_vocabulary
from healthrag.redact.engine import Redactor
from healthrag.retrieve.retriever import Retriever


def query_redactor(settings: Settings) -> Redactor:
    """The same redactor used at ingestion: deny-list from patients.csv, clinical allow-list."""
    r = settings.redaction
    patients = [p for rows in load_all_patients(settings).values() for p in rows]
    return Redactor(
        r.spacy_model,
        r.language,
        r.score_threshold,
        deny_list=deny_list_names(patients),
        allow_list=clinical_vocabulary(list(settings.patient_csv_dirs().values())),
    )


def build_answerer(settings: Settings, router_name: str | None = None) -> Answerer:
    """Load the retriever, models, router and redactor."""
    retriever = Retriever.from_settings(settings, with_reranker=settings.generation.retrieval_mode == "hybrid_rerank")
    client = LlmClient(settings)
    router = make_router(router_name or settings.router.default, settings, client, retriever.embedder)
    return Answerer(settings, retriever, router, client, query_redactor(settings))
