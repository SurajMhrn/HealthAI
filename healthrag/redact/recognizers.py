"""Custom Presidio recognizers for Synthea-style identifiers.

spaCy's NER often misses Synthea names because they carry numeric suffixes
(``Abe604 Frami345``), and Presidio's built-in SSN recognizer rejects the ``999-``
prefix Synthea uses. These recognizers cover those gaps.

Entity types and the placeholder each one is replaced with:

==================  =================
entity type         placeholder
==================  =================
PERSON              ``<PERSON>``
DATE_OF_BIRTH       ``<DATE_OF_BIRTH>``
ADDRESS             ``<ADDRESS>``
PHONE_NUMBER        ``<PHONE_NUMBER>``
SSN                 ``<RECORD_ID>``
DRIVER_LICENSE      ``<RECORD_ID>``
PASSPORT            ``<RECORD_ID>``
RECORD_UUID         ``<RECORD_ID>``
==================  =================
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

PLACEHOLDERS: dict[str, str] = {
    "PERSON": "<PERSON>",
    "DATE_OF_BIRTH": "<DATE_OF_BIRTH>",
    "ADDRESS": "<ADDRESS>",
    "PHONE_NUMBER": "<PHONE_NUMBER>",
    "SSN": "<RECORD_ID>",
    "DRIVER_LICENSE": "<RECORD_ID>",
    "PASSPORT": "<RECORD_ID>",
    "RECORD_UUID": "<RECORD_ID>",
}

ENTITIES: list[str] = list(PLACEHOLDERS)

DENY_LIST_RECOGNIZER = "PatientNameDenyListRecognizer"

# A capitalised word that ends in a lowercase letter followed by 1-4 digits:
# "Abe604", "O'Keefe54", "Ramón646". "COVID-19" and "B12" don't match.
_SUFFIXED_NAME = r"\b[A-ZÀ-Þ][A-Za-zÀ-ÿ'’]*(?:-[A-ZÀ-Þ][A-Za-zÀ-ÿ'’]*)*[a-zß-ÿ]\d{1,4}\b"

_UUID = r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"

_SSN = r"\b\d{3}-\d{2}-\d{4}\b"

# One letter followed by 6-9 digits: Massachusetts licences (Synthea uses "S9999...")
# and the format used by many other states.
_DRIVER_LICENSE = r"\b[A-Z]\d{6,9}\b"

# Synthea writes passports as a letter, 6-9 digits and a letter: "X12345678X", "X743270X".
_PASSPORT = r"\b[A-Z]\d{6,9}[A-Z]\b"

# "<number> <street words>, <city>, <state> [ZIP]" on one line.
_STREET = r"\d{1,6}(?:[ ][A-Za-z0-9.'’#-]+){1,8}?"
_CITY = r"[A-Z][A-Za-z.'’-]*(?:[ ][A-Z][A-Za-z.'’-]*){0,3}"
_STATE = r"(?:[A-Z]{2}|[A-Z][a-z]+(?:[ ][A-Z][a-z]+){0,2})"
_ADDRESS_LINE = rf"\b{_STREET},[ ]*{_CITY},[ ]*{_STATE}(?:[ ]+\d{{5}}(?:-\d{{4}})?)?\b"

_DATE = (
    r"(?:\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r"|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.? \d{1,2},? \d{4}"
    r"|\d{1,2} (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4})"
)


class LabelledValueRecognizer(EntityRecognizer):
    """Finds a value by the label in front of it and returns only the value's span.

    Each regex must have one capture group around the value. Used for things like
    ``Date of birth: 1985-03-12``, where the label tells us what an otherwise generic
    value means.
    """

    def __init__(self, entity: str, patterns: Iterable[str], name: str, score: float = 0.95) -> None:
        self._patterns = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in patterns]
        self._score = score
        super().__init__(supported_entities=[entity], name=name, supported_language="en")

    def load(self) -> None:
        """Nothing to load; patterns are compiled in the constructor."""

    def analyze(
        self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None
    ) -> list[RecognizerResult]:
        """Return one result per labelled value found in ``text``."""
        entity = self.supported_entities[0]
        if entities and entity not in entities:
            return []
        results = []
        for pattern in self._patterns:
            for match in pattern.finditer(text):
                start, end = match.span(1)
                if end > start:
                    results.append(RecognizerResult(entity, start, end, self._score))
        return sorted(results, key=lambda r: r.start)


def synthea_name_recognizer() -> PatternRecognizer:
    """Names with Synthea's numeric suffix, e.g. ``Abe604`` or ``Frami345``."""
    return PatternRecognizer(
        supported_entity="PERSON",
        name="SyntheaNameRecognizer",
        patterns=[Pattern("suffixed name", _SUFFIXED_NAME, 0.9)],
        global_regex_flags=re.UNICODE,  # case-sensitive: the capital letters matter
    )


def record_uuid_recognizer() -> PatternRecognizer:
    """Synthea record UUIDs (patients, encounters, providers)."""
    return PatternRecognizer(
        supported_entity="RECORD_UUID",
        name="RecordUuidRecognizer",
        patterns=[Pattern("uuid", _UUID, 0.95)],
    )


def ssn_recognizer() -> PatternRecognizer:
    """US social security numbers in ``123-45-6789`` form, including Synthea's ``999-`` range."""
    return PatternRecognizer(
        supported_entity="SSN",
        name="SsnFormatRecognizer",
        patterns=[Pattern("ssn dashed", _SSN, 0.85)],
        context=["ssn", "social security"],
    )


def driver_license_recognizer() -> PatternRecognizer:
    """Driver's licence numbers: one capital letter and 6-9 digits."""
    return PatternRecognizer(
        supported_entity="DRIVER_LICENSE",
        name="DriverLicenseFormatRecognizer",
        patterns=[Pattern("letter + 6-9 digits", _DRIVER_LICENSE, 0.6)],
        context=["licence", "license", "driver", "dl"],
        global_regex_flags=re.UNICODE,
    )


def passport_recognizer() -> PatternRecognizer:
    """Passport numbers in Synthea's ``X12345678X`` form (6-9 digits between two letters)."""
    return PatternRecognizer(
        supported_entity="PASSPORT",
        name="PassportFormatRecognizer",
        patterns=[Pattern("letter + 6-9 digits + letter", _PASSPORT, 0.85)],
        context=["passport"],
        global_regex_flags=re.UNICODE,
    )


def address_recognizer() -> LabelledValueRecognizer:
    """Addresses after an ``Address:`` label (whole line)."""
    return LabelledValueRecognizer(
        entity="ADDRESS",
        name="LabelledAddressRecognizer",
        patterns=[r"^[ \t]*(?:home |street |mailing )?address[ \t]*:[ \t]*(\S[^\n]*?)[ \t]*$"],
    )


def address_line_recognizer() -> PatternRecognizer:
    """US address lines in free text: ``12 Main St Unit 4, Springfield, MA 01103``."""
    return PatternRecognizer(
        supported_entity="ADDRESS",
        name="AddressLineRecognizer",
        patterns=[Pattern("street, city, state zip", _ADDRESS_LINE, 0.7)],
        global_regex_flags=re.UNICODE,
    )


def date_of_birth_recognizer() -> LabelledValueRecognizer:
    """Dates of birth only: dates after ``Date of birth:``, ``DOB:`` or ``born (on)``.

    Other dates (encounters, conditions) are left alone on purpose, because the
    retrieval date filter needs them.
    """
    return LabelledValueRecognizer(
        entity="DATE_OF_BIRTH",
        name="DateOfBirthRecognizer",
        patterns=[
            r"\b(?:date of birth|birth ?date|d\.?o\.?b\.?)[ \t]*[:\-]?[ \t]*(" + _DATE + r")",
            r"\bborn(?: on)?[ \t]+(" + _DATE + r")",
        ],
    )


def name_deny_list_recognizer(names: Iterable[str]) -> PatternRecognizer:
    """Exact names taken from ``patients.csv`` (a safety net for the pattern recognizers)."""
    unique = sorted({name.strip() for name in names if name and name.strip()}, key=lambda n: (-len(n), n))
    return PatternRecognizer(supported_entity="PERSON", name=DENY_LIST_RECOGNIZER, deny_list=unique)


def custom_recognizers() -> list[EntityRecognizer]:
    """All custom recognizers except the deny-list, which depends on the data."""
    return [
        synthea_name_recognizer(),
        record_uuid_recognizer(),
        ssn_recognizer(),
        driver_license_recognizer(),
        passport_recognizer(),
        address_recognizer(),
        address_line_recognizer(),
        date_of_birth_recognizer(),
    ]
