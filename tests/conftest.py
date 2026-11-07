"""Shared fixtures. Everything here is offline and uses the small CSVs in tests/fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from healthrag.config import get_settings
from healthrag.ingest.pipeline import deny_list_names
from healthrag.ingest.synthea import build_patient_documents, clinical_vocabulary, load_patients
from healthrag.redact.engine import Redactor

FIXTURES = Path(__file__).parent / "fixtures"
SYNTHEA_FIXTURES = FIXTURES / "synthea"
TEST_SALT = "test-salt-not-secret"


@pytest.fixture(scope="session")
def patients() -> list[dict[str, str]]:
    """The two fake patients in tests/fixtures/synthea/patients.csv."""
    return load_patients(SYNTHEA_FIXTURES)


@pytest.fixture(scope="session")
def patient_documents(patients: list[dict[str, str]]):  # noqa: ANN201
    """Patient documents built from the fixture CSVs."""
    return build_patient_documents(patients, SYNTHEA_FIXTURES, TEST_SALT, 16, "synthea-test")


@pytest.fixture(scope="session")
def redactor(patients: list[dict[str, str]]) -> Redactor:
    """Redactor configured like the real pipeline (spaCy is loaded once per test session)."""
    r = get_settings().redaction
    return Redactor(
        r.spacy_model,
        r.language,
        r.score_threshold,
        deny_list=deny_list_names(patients),
        allow_list=clinical_vocabulary([SYNTHEA_FIXTURES]),
    )
