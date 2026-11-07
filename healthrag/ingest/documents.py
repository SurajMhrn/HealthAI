"""Plain data types shared by the ingestion steps."""

from __future__ import annotations

from dataclasses import dataclass, field

PATIENT_RECORD = "patient_record"
GUIDELINE = "guideline"


@dataclass(frozen=True)
class TextUnit:
    """One line of a document, with the section it belongs to.

    ``date`` is the ISO date of a dated row in a patient record (e.g. an encounter).
    ``page`` is the 1-based page number for PDF text.
    """

    text: str
    section: str
    date: str | None = None
    page: int | None = None


@dataclass
class Document:
    """A source document made of ordered text units."""

    source: str
    doc_type: str
    title: str
    units: list[TextUnit] = field(default_factory=list)
    patient_id: str | None = None
    date: str | None = None


@dataclass(frozen=True)
class Chunk:
    """A piece of a document plus the metadata stored with it.

    ``date`` is the latest date in the chunk and ``date_start`` the earliest. For
    guidelines both hold the publication date. Demographics chunks have no date, so
    a date of birth can never end up in metadata.
    """

    text: str
    source: str
    doc_type: str
    section: str
    ordinal: int
    patient_id: str | None = None
    date: str | None = None
    date_start: str | None = None
    page: int | None = None
