"""Metadata filters, applied before any ranking or fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Filters:
    """Restrict retrieval to matching chunks. ``None`` means "no restriction".

    Dates are ISO strings compared as text, so ``"2019"`` and ``"2019-04"`` work as
    coarse bounds. A chunk matches a date range when its own range
    (``date_start`` .. ``date``) overlaps it; chunks without a date never match
    a date filter.
    """

    doc_type: str | None = None
    patient_id: str | None = None
    date_from: str | None = None
    date_to: str | None = None

    def is_empty(self) -> bool:
        """True when no filter is set."""
        return not (self.doc_type or self.patient_id or self.date_from or self.date_to)


def matches(record: dict, filters: Filters) -> bool:
    """Whether one chunk record passes the filters."""
    if filters.doc_type and record.get("doc_type") != filters.doc_type:
        return False
    if filters.patient_id and record.get("patient_id") != filters.patient_id:
        return False
    if filters.date_from or filters.date_to:
        end = record.get("date")
        start = record.get("date_start") or end
        if not end:
            return False
        # Compare on the common prefix length so "2019" bounds work against "2019-04-02".
        if filters.date_from and end[: len(filters.date_from)] < filters.date_from:
            return False
        if filters.date_to and start[: len(filters.date_to)] > filters.date_to:
            return False
    return True


def allowed_indices(records: list[dict], filters: Filters | None) -> np.ndarray | None:
    """Indices of records that pass ``filters``, or None when nothing is filtered."""
    if filters is None or filters.is_empty():
        return None
    return np.array([i for i, record in enumerate(records) if matches(record, filters)], dtype=np.int64)
