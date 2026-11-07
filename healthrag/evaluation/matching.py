"""Matching golden-set expectations against retrieved chunks and answers."""

from __future__ import annotations

import re

_SPACES = re.compile(r"\s+")
# Models often write non-breaking or typographic dashes (e.g. "GLP‑1"); treat them all as "-".
_DASHES = str.maketrans({c: "-" for c in "‐‑‒–—−"})


def normalize(text: str) -> str:
    """Lower-case, drop soft hyphens, join PDF line-break hyphenation, collapse whitespace."""
    text = text.replace("\u00ad", "").translate(_DASHES).lower()
    text = re.sub(r"(\w)- \n?(\w)", r"\1-\2", text.replace("\n", " "))
    return _SPACES.sub(" ", text).strip()


def supports(record: dict, support: dict) -> bool:
    """Whether a chunk record carries one golden support (source prefix + a phrase in its text)."""
    if support.get("source") and support["source"] not in record["source"]:
        return False
    if support.get("section") and record.get("section") != support["section"]:
        return False
    return all(normalize(part) in normalize(record["text"]) for part in _as_list(support.get("contains", "")))


def _as_list(value: str | list[str]) -> list[str]:
    return value if isinstance(value, list) else [value]


def answer_matches(answer: str, required: list[str | list[str]]) -> bool:
    """Every requirement is met (case-insensitive, whitespace-normalised).

    A requirement is a phrase, or a list of alternative phrases of which any one is enough
    (e.g. the same date written in different formats).
    """
    text = normalize(answer)
    return all(any(normalize(alt) in text for alt in _as_list(req)) for req in required)
