"""Presidio redaction engine: spaCy NER plus the custom recognizers.

Only the entity types in :data:`healthrag.redact.recognizers.ENTITIES` are detected.
Presidio's generic ``DATE_TIME`` is deliberately not requested, so encounter and
condition dates survive; dates of birth are caught by a dedicated recognizer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from healthrag.redact.recognizers import (
    DENY_LIST_RECOGNIZER,
    ENTITIES,
    PLACEHOLDERS,
    custom_recognizers,
    name_deny_list_recognizer,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One span that was replaced: what it was, where, and which recognizer found it."""

    entity_type: str
    start: int
    end: int
    score: float
    recognizer: str
    original: str


@dataclass
class Redaction:
    """Redacted text plus the detections that were applied to it."""

    text: str
    detections: list[Detection] = field(default_factory=list)


def resolve_overlaps(results: list[RecognizerResult]) -> list[RecognizerResult]:
    """Keep non-overlapping results, preferring higher scores, then longer spans.

    Doing this ourselves (instead of inside the anonymizer) means we know exactly
    which detection produced each placeholder.
    """
    ranked = sorted(results, key=lambda r: (-r.score, -(r.end - r.start), r.start))
    kept: list[RecognizerResult] = []
    for result in ranked:
        if all(result.end <= k.start or result.start >= k.end for k in kept):
            kept.append(result)
    return sorted(kept, key=lambda r: r.start)


def _recognizer_name(result: RecognizerResult) -> str:
    metadata = result.recognition_metadata or {}
    return str(metadata.get(RecognizerResult.RECOGNIZER_NAME_KEY, "unknown"))


class Redactor:
    """Detect and replace PHI with typed placeholders such as ``<PERSON>``.

    Args:
        spacy_model: spaCy pipeline used for NER (e.g. ``en_core_web_lg``).
        language: language code passed to Presidio.
        score_threshold: detections scoring below this are ignored.
        deny_list: exact names to always redact (built from ``patients.csv``).
        allow_list: exact strings never to redact, e.g. clinical terms and codes
            that NER mistakes for names or phone numbers.
    """

    def __init__(
        self,
        spacy_model: str,
        language: str = "en",
        score_threshold: float = 0.3,
        deny_list: Iterable[str] = (),
        allow_list: Iterable[str] = (),
    ) -> None:
        self.language = language
        self.score_threshold = score_threshold
        nlp_engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": language, "model_name": spacy_model}],
            }
        ).create_engine()
        registry = RecognizerRegistry(supported_languages=[language])
        registry.load_predefined_recognizers(languages=[language], nlp_engine=nlp_engine)
        for recognizer in custom_recognizers():
            registry.add_recognizer(recognizer)
        self.analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry, supported_languages=[language])
        self.anonymizer = AnonymizerEngine()
        self.allow_list = sorted(set(allow_list))
        names = list(deny_list)
        self.deny_list: PatternRecognizer | None = name_deny_list_recognizer(names) if names else None
        self.operators = {
            entity: OperatorConfig("replace", {"new_value": placeholder}) for entity, placeholder in PLACEHOLDERS.items()
        }

    def analyze(
        self, text: str, nlp_artifacts: NlpArtifacts | None = None, use_deny_list: bool = True
    ) -> list[RecognizerResult]:
        """Run all recognizers on ``text`` and return non-overlapping results."""
        ad_hoc = [self.deny_list] if use_deny_list and self.deny_list is not None else None
        results = self.analyzer.analyze(
            text=text,
            language=self.language,
            entities=ENTITIES,
            score_threshold=self.score_threshold,
            nlp_artifacts=nlp_artifacts,
            ad_hoc_recognizers=ad_hoc,
            allow_list=self.allow_list or None,
        )
        return resolve_overlaps(results)

    def apply(self, text: str, results: list[RecognizerResult]) -> Redaction:
        """Replace the given (non-overlapping) results with their placeholders."""
        anonymized = self.anonymizer.anonymize(text=text, analyzer_results=results, operators=self.operators)
        detections = [
            Detection(r.entity_type, r.start, r.end, r.score, _recognizer_name(r), text[r.start : r.end])
            for r in results
        ]
        return Redaction(text=anonymized.text, detections=detections)

    def redact(self, text: str, use_deny_list: bool = True) -> Redaction:
        """Redact a single text."""
        return self.apply(text, self.analyze(text, use_deny_list=use_deny_list))

    def redact_both_ways(self, texts: list[str], batch_size: int = 32) -> Iterator[tuple[Redaction, Redaction]]:
        """Redact each text with and without the deny-list, running spaCy only once per text.

        Yields ``(with_deny_list, without_deny_list)`` pairs in input order.
        """
        batches = self.analyzer.nlp_engine.process_batch(texts, language=self.language, batch_size=batch_size)
        for text, artifacts in batches:
            with_list = self.apply(text, self.analyze(text, artifacts, use_deny_list=True))
            without_list = self.apply(text, self.analyze(text, artifacts, use_deny_list=False))
            yield with_list, without_list


def is_deny_list_detection(detection: Detection) -> bool:
    """True if the detection came from the patient-name deny-list."""
    return detection.recognizer == DENY_LIST_RECOGNIZER
