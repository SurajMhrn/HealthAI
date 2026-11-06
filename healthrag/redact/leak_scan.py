"""Leak scan: look for every known identifier from patients.csv in the redacted chunks."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from healthrag.ingest.synthea import DEMOGRAPHICS, NAME_FIELDS

# Identifier kinds and the patients.csv column(s) they come from.
IDENTIFIER_COLUMNS: dict[str, tuple[str, ...]] = {
    "name": NAME_FIELDS,
    "ssn": ("SSN",),
    "drivers_license": ("DRIVERS",),
    "passport": ("PASSPORT",),
    "address": ("ADDRESS",),
    "uuid": ("Id",),
    "birth_date": ("BIRTHDATE",),
}


@dataclass(frozen=True)
class KnownIdentifier:
    """An identifier value that must not appear in any redacted chunk."""

    kind: str
    value: str
    patient_id: str


@dataclass(frozen=True)
class Leak:
    """One occurrence of a known identifier in a redacted chunk."""

    kind: str
    value: str
    chunk_id: str


@dataclass
class LeakScanResult:
    """Leaks found, plus how many identifiers of each kind were checked."""

    checked: Counter[str] = field(default_factory=Counter)
    leaks: list[Leak] = field(default_factory=list)

    def leaks_by_kind(self) -> Counter[str]:
        """Number of leak occurrences per identifier kind."""
        return Counter(leak.kind for leak in self.leaks)

    def leaked_values_by_kind(self) -> Counter[str]:
        """Number of distinct leaked identifier values per kind."""
        return Counter(kind for kind, _ in {(leak.kind, leak.value) for leak in self.leaks})


def known_identifiers(patients: list[dict[str, str]], patient_ids: dict[str, str]) -> list[KnownIdentifier]:
    """Every identifier value in ``patients.csv``.

    ``patient_ids`` maps each Synthea UUID to its pseudonymous ``patient_id``.
    Birth dates are taken as their ``YYYY-MM-DD`` part.
    """
    found: list[KnownIdentifier] = []
    for patient in patients:
        pid = patient_ids[patient["Id"]]
        for kind, columns in IDENTIFIER_COLUMNS.items():
            for column in columns:
                value = (patient.get(column) or "").strip()
                if kind == "birth_date":
                    value = value[:10]
                if value:
                    found.append(KnownIdentifier(kind, value, pid))
    return found


def _token_pattern(values: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(re.escape(v) for v in sorted(set(values), key=len, reverse=True))
    return re.compile(rf"(?<![A-Za-z0-9])(?:{alternatives})(?![A-Za-z0-9])", re.IGNORECASE)


def scan_chunks(identifiers: list[KnownIdentifier], chunks: list[dict]) -> LeakScanResult:
    """Count occurrences of known identifiers in redacted chunk records.

    Each chunk record needs ``chunk_id``, ``text``, ``section`` and ``patient_id``.
    Matching is case-insensitive and only counts whole tokens, so ``Abe604`` does not
    match inside ``Abe6041``.

    Birth dates are only checked in the patient's own demographics chunks. Elsewhere the
    same calendar date can legitimately appear as an encounter date, which is kept.
    """
    result = LeakScanResult()
    global_ids = [i for i in identifiers if i.kind != "birth_date"]
    birth_dates = {i.patient_id: i.value for i in identifiers if i.kind == "birth_date"}
    result.checked.update(i.kind for i in identifiers)

    by_value: dict[str, list[KnownIdentifier]] = defaultdict(list)
    for ident in global_ids:
        by_value[ident.value.lower()].append(ident)
    pattern = _token_pattern([i.value for i in global_ids]) if global_ids else None

    for chunk in chunks:
        text = chunk["text"]
        if pattern is not None:
            for match in pattern.finditer(text):
                # Report the value as it appears in patients.csv, once per kind.
                found = {(ident.kind, ident.value) for ident in by_value[match.group(0).lower()]}
                for kind, value in sorted(found):
                    result.leaks.append(Leak(kind, value, chunk["chunk_id"]))
        birth_date = birth_dates.get(chunk.get("patient_id") or "")
        if birth_date and chunk.get("section") == DEMOGRAPHICS and birth_date in text:
            result.leaks.append(Leak("birth_date", birth_date, chunk["chunk_id"]))
    return result
