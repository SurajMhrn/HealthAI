"""Section-aware chunking.

A chunk never spans two sections. Inside a section, whole lines are packed into
chunks of at most ``size`` characters, and each chunk repeats up to ``overlap``
characters of trailing lines from the previous chunk. Every chunk starts with a
short header naming the document and section, so it still makes sense on its own.
"""

from __future__ import annotations

from collections.abc import Iterator

from healthrag.ingest.documents import Chunk, Document, TextUnit


def chunk_header(document: Document, section: str) -> str:
    """Header line put at the top of every chunk, e.g. ``[Patient record | Medications]``."""
    return f"[{document.title} | {section}]"


def group_by_section(units: list[TextUnit]) -> Iterator[tuple[str, list[TextUnit]]]:
    """Yield runs of consecutive units that share a section, in document order."""
    run: list[TextUnit] = []
    for unit in units:
        if run and unit.section != run[0].section:
            yield run[0].section, run
            run = []
        run.append(unit)
    if run:
        yield run[0].section, run


def split_long_unit(unit: TextUnit, limit: int) -> list[TextUnit]:
    """Split a unit longer than ``limit`` characters at word boundaries."""
    if len(unit.text) <= limit:
        return [unit]
    pieces: list[TextUnit] = []
    current = ""
    for word in unit.text.split(" "):
        while len(word) > limit:  # a single very long token: hard split
            if current:
                pieces.append(TextUnit(current, unit.section, unit.date, unit.page))
                current = ""
            pieces.append(TextUnit(word[:limit], unit.section, unit.date, unit.page))
            word = word[limit:]
        candidate = f"{current} {word}" if current else word
        if len(candidate) > limit:
            pieces.append(TextUnit(current, unit.section, unit.date, unit.page))
            current = word
        else:
            current = candidate
    if current:
        pieces.append(TextUnit(current, unit.section, unit.date, unit.page))
    return pieces


def _joined_length(units: list[TextUnit]) -> int:
    return sum(len(u.text) for u in units) + max(0, len(units) - 1)


def _overlap_tail(window: list[TextUnit], overlap: int) -> list[TextUnit]:
    tail: list[TextUnit] = []
    for unit in reversed(window):
        if _joined_length([unit, *tail]) > overlap:
            break
        tail.insert(0, unit)
    return tail


def pack_units(units: list[TextUnit], size: int, overlap: int) -> list[list[TextUnit]]:
    """Greedily pack units into windows of at most ``size`` characters (joined by newlines)."""
    pieces = [piece for unit in units for piece in split_long_unit(unit, size)]
    windows: list[list[TextUnit]] = []
    current: list[TextUnit] = []
    for piece in pieces:
        if current and _joined_length([*current, piece]) > size:
            windows.append(current)
            current = _overlap_tail(current, overlap)
            while current and _joined_length([*current, piece]) > size:
                current.pop(0)
        current.append(piece)
    if current:
        windows.append(current)
    return windows


def chunk_document(document: Document, size: int, overlap: int) -> list[Chunk]:
    """Split a document into section-aware chunks with metadata.

    ``size`` and ``overlap`` count characters of the chunk body; the header line is
    added on top.
    """
    chunks: list[Chunk] = []
    for section, units in group_by_section(document.units):
        header = chunk_header(document, section)
        for window in pack_units(units, size, overlap):
            dates = sorted(u.date for u in window if u.date)
            pages = [u.page for u in window if u.page is not None]
            chunks.append(
                Chunk(
                    text=header + "\n" + "\n".join(u.text for u in window),
                    source=document.source,
                    doc_type=document.doc_type,
                    section=section,
                    ordinal=len(chunks),
                    patient_id=document.patient_id,
                    date=dates[-1] if dates else document.date,
                    date_start=dates[0] if dates else document.date,
                    page=pages[0] if pages else None,
                )
            )
    return chunks
