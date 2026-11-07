"""Extract guideline PDFs page by page, with simple heading detection.

pypdf only reads the text layer, so scanned PDFs (images) produce no text.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

from healthrag.ingest.documents import GUIDELINE, Document, TextUnit

log = logging.getLogger(__name__)

FRONT_MATTER = "Front matter"

_NUMBERED_HEADING = re.compile(r"^(?:\d{1,2}(?:\.\d{1,2}){0,3}\.?|[IVX]{1,5}\.|[A-H]\.)\s+[A-Z]\S*")
_NAMED_HEADING = re.compile(r"^(?:Appendix|Chapter|Part|Section|Module)\s+[A-Z0-9]{1,4}\b", re.IGNORECASE)
_TOC_ENTRY = re.compile(r"(?:\.\s*){4,}\s*\d+\s*$|\s\d{1,3}$")
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d{1,4}(?:\s+of\s+\d{1,4})?$", re.IGNORECASE)
_SPACES = re.compile(r"[ \t ]+")


def extract_pages(path: Path) -> list[str]:
    """Text of each page, in order (an empty string for pages with no text layer)."""
    reader = PdfReader(path)
    return [page.extract_text() or "" for page in reader.pages]


def clean_lines(page_text: str) -> list[str]:
    """Split page text into trimmed, non-empty lines and drop bare page numbers."""
    lines = []
    for raw in page_text.splitlines():
        line = _SPACES.sub(" ", raw).strip()
        if line and not _PAGE_NUMBER.match(line):
            lines.append(line)
    return lines


def _boilerplate_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.lower())


def repeated_lines(pages: list[list[str]], min_share: float, min_pages: int) -> set[str]:
    """Normalised lines that appear on at least ``min_share`` of pages (running headers and footers).

    Only applied to documents with at least ``min_pages`` pages.
    """
    if len(pages) < min_pages:
        return set()
    counts = Counter(key for lines in pages for key in {_boilerplate_key(line) for line in lines})
    threshold = max(2, int(min_share * len(pages)))
    return {key for key, count in counts.items() if count >= threshold}


def is_heading(line: str, max_chars: int) -> bool:
    """Guess whether a line is a section heading.

    Accepts numbered headings (``3.2 Diagnosis``), named ones (``Appendix B``) and short
    all-caps lines. Rejects table-of-contents entries and lines that read like sentences.
    """
    if len(line) > max_chars or len(line) < 3 or line.endswith((".", ",", ";", ":")):
        return False
    if _TOC_ENTRY.search(line):
        return False
    if _NUMBERED_HEADING.match(line) or _NAMED_HEADING.match(line):
        return True
    letters = [ch for ch in line if ch.isalpha()]
    return len(letters) >= 4 and line.upper() == line and len(line.split()) <= 10


def guideline_document(
    path: Path,
    source: str,
    title: str,
    published: str,
    heading_max_chars: int,
    boilerplate_min_share: float,
    boilerplate_min_pages: int,
) -> Document:
    """Read a guideline PDF into a document whose units carry page numbers and sections."""
    pages = [clean_lines(text) for text in extract_pages(path)]
    boilerplate = repeated_lines(pages, boilerplate_min_share, boilerplate_min_pages)
    section = FRONT_MATTER
    units: list[TextUnit] = []
    for page_number, lines in enumerate(pages, start=1):
        for line in lines:
            if _boilerplate_key(line) in boilerplate:
                continue
            if is_heading(line, heading_max_chars):
                section = line
            units.append(TextUnit(text=line, section=section, page=page_number))
    empty_pages = sum(1 for lines in pages if not lines)
    log.info("%s: %d pages, %d lines, %d pages without text", source, len(pages), len(units), empty_pages)
    return Document(source=source, doc_type=GUIDELINE, title=title, units=units, date=published)
