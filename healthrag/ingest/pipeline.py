"""Ingestion pipeline: documents, chunks, redaction, then chunks.jsonl and the redaction report.

Nothing is written to ``data/processed`` until every chunk has been redacted.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from healthrag.config import Settings
from healthrag.guard.patterns import injection_matches
from healthrag.ids import chunk_id, patient_id
from healthrag.ingest.chunking import chunk_document
from healthrag.ingest.documents import Chunk, Document
from healthrag.ingest.pdf import guideline_document
from healthrag.ingest.synthea import NAME_FIELDS, build_patient_documents, clinical_vocabulary, load_patients
from healthrag.redact.engine import Redaction, Redactor
from healthrag.redact.leak_scan import LeakScanResult, known_identifiers, scan_chunks

log = logging.getLogger(__name__)


@dataclass
class ProcessedChunk:
    """A chunk with its raw text and both redacted versions (with and without the deny-list)."""

    chunk: Chunk
    chunk_id: str
    raw_text: str
    redacted: Redaction
    redacted_without_deny_list: Redaction

    def record(self) -> dict:
        """The JSON record written to chunks.jsonl (redacted text only)."""
        c = self.chunk
        return {
            "chunk_id": self.chunk_id,
            "text": self.redacted.text,
            "source": c.source,
            "doc_type": c.doc_type,
            "patient_id": c.patient_id,
            "date": c.date,
            "date_start": c.date_start,
            "section": c.section,
            "page": c.page,
            "redactions": dict(sorted(Counter(d.entity_type for d in self.redacted.detections).items())),
            "injection_flags": injection_matches(self.redacted.text),
        }


@dataclass
class IngestResult:
    """Everything the redaction report needs."""

    patients: list[dict[str, str]]
    documents: list[Document]
    chunks: list[ProcessedChunk]
    leaks: LeakScanResult
    leaks_without_deny_list: LeakScanResult
    redaction_seconds: float
    allow_list_size: int


def deny_list_names(patients: list[dict[str, str]]) -> list[str]:
    """All values of the name fields in patients.csv."""
    return [p[field].strip() for p in patients for field in NAME_FIELDS if (p.get(field) or "").strip()]


def load_all_patients(settings: Settings) -> dict[str, list[dict[str, str]]]:
    """Patients of every enabled dataset, keyed by dataset name."""
    return {name: load_patients(folder) for name, folder in settings.patient_csv_dirs().items()}


def build_documents(settings: Settings, patients_by_dataset: dict[str, list[dict[str, str]]]) -> list[Document]:
    """Patient documents from each Synthea dataset plus one document per guideline PDF."""
    documents: list[Document] = []
    for name, folder in settings.patient_csv_dirs().items():
        documents += build_patient_documents(
            patients_by_dataset[name], folder, settings.require_salt(), settings.patient_id.length, f"synthea-{name}"
        )
    for source in settings.guidelines.sources:
        path = settings.paths.raw_guidelines / f"{source.name}.pdf"
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing. Run `make data` first.")
        documents.append(
            guideline_document(
                path,
                source=path.name,
                title=source.short_title,
                published=source.published,
                heading_max_chars=settings.pdf.heading_max_chars,
                boilerplate_min_share=settings.pdf.boilerplate_min_share,
                boilerplate_min_pages=settings.pdf.boilerplate_min_pages,
            )
        )
    return documents


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records as JSON lines (UTF-8), replacing the file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)


def run(settings: Settings) -> IngestResult:
    """Build, chunk and redact every document, write chunks.jsonl, and run the leak scans."""
    by_dataset = load_all_patients(settings)
    patients = [p for rows in by_dataset.values() for p in rows]
    if len({p["Id"] for p in patients}) != len(patients):
        raise ValueError("The same Synthea UUID appears in more than one patient dataset")
    documents = build_documents(settings, by_dataset)
    chunks = [c for doc in documents for c in chunk_document(doc, settings.chunking.size, settings.chunking.overlap)]
    log.info("%d documents -> %d chunks", len(documents), len(chunks))

    r = settings.redaction
    allow_list = clinical_vocabulary(list(settings.patient_csv_dirs().values()))
    redactor = Redactor(
        r.spacy_model, r.language, r.score_threshold, deny_list=deny_list_names(patients), allow_list=allow_list
    )
    started = time.perf_counter()
    processed: list[ProcessedChunk] = []
    texts = [c.text for c in chunks]
    for chunk, (with_list, without_list) in zip(chunks, redactor.redact_both_ways(texts, r.batch_size), strict=True):
        cid = chunk_id(chunk.source, chunk.ordinal, chunk.section, with_list.text)
        processed.append(ProcessedChunk(chunk, cid, chunk.text, with_list, without_list))
        if len(processed) % 500 == 0:
            log.info("redacted %d / %d chunks", len(processed), len(chunks))
    seconds = time.perf_counter() - started

    records = [p.record() for p in processed]
    write_jsonl(settings.paths.chunks_file, records)
    log.info("wrote %d chunks to %s", len(records), settings.paths.chunks_file)

    salt = settings.require_salt()
    ids = {p["Id"]: patient_id(p["Id"], salt, settings.patient_id.length) for p in patients}
    identifiers = known_identifiers(patients, ids)
    without = [{**rec, "text": p.redacted_without_deny_list.text} for rec, p in zip(records, processed, strict=True)]
    return IngestResult(
        patients=patients,
        documents=documents,
        chunks=processed,
        leaks=scan_chunks(identifiers, records),
        leaks_without_deny_list=scan_chunks(identifiers, without),
        redaction_seconds=seconds,
        allow_list_size=len(allow_list),
    )
