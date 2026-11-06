"""Turn Synthea CSV exports into one readable text document per patient.

The demographics section deliberately includes the patient's name, date of birth,
address, SSN, driver's licence and passport numbers, so that redaction is exercised
on realistic identifiers. Everything here is synthetic.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from healthrag.ids import patient_id as make_patient_id
from healthrag.ingest.documents import PATIENT_RECORD, Document, TextUnit

DEMOGRAPHICS = "Demographics"
CONDITIONS = "Conditions"
MEDICATIONS = "Medications"
ENCOUNTERS = "Encounters"

NAME_FIELDS = ("FIRST", "MIDDLE", "LAST", "MAIDEN")

_GENDER = {"M": "male", "F": "female"}
_MARITAL = {"M": "married", "S": "single", "D": "divorced", "W": "widowed"}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts (empty list if the file is missing)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_by_patient(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group rows from conditions/medications/encounters by their ``PATIENT`` column."""
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("PATIENT", "")].append(row)
    return grouped


def iso_date(value: str | None) -> str | None:
    """Return the YYYY-MM-DD part of a Synthea date or timestamp, or None if empty."""
    value = (value or "").strip()
    return value[:10] if len(value) >= 10 else None


def full_name(patient: dict[str, str]) -> str:
    """Prefix, given, middle, family name and suffix joined with spaces."""
    parts = [patient.get(key, "").strip() for key in ("PREFIX", "FIRST", "MIDDLE", "LAST", "SUFFIX")]
    return " ".join(part for part in parts if part)


def _value(row: dict[str, str], key: str) -> str:
    return (row.get(key) or "").strip()


def address_line(patient: dict[str, str]) -> str:
    """Street, city, state and ZIP on one line."""
    street = _value(patient, "ADDRESS")
    city = _value(patient, "CITY")
    state_zip = " ".join(part for part in (_value(patient, "STATE"), _value(patient, "ZIP")) if part)
    return ", ".join(part for part in (street, city, state_zip) if part)


def demographics_lines(patient: dict[str, str]) -> list[str]:
    """Readable demographics, including the identifiers that redaction must remove."""
    death_date = iso_date(patient.get("DEATHDATE"))
    lines = [
        f"Name: {full_name(patient)}",
        f"Previous surname: {_value(patient, 'MAIDEN')}" if _value(patient, "MAIDEN") else "",
        f"Date of birth: {iso_date(patient.get('BIRTHDATE'))}",
        f"Vital status: deceased ({death_date[:4]})" if death_date else "Vital status: alive",
        f"Gender: {_GENDER.get(_value(patient, 'GENDER'), _value(patient, 'GENDER'))}",
        f"Race: {_value(patient, 'RACE')}; ethnicity: {_value(patient, 'ETHNICITY')}",
        f"Marital status: {_MARITAL.get(_value(patient, 'MARITAL'), _value(patient, 'MARITAL') or 'unknown')}",
        f"Address: {address_line(patient)}",
        f"SSN: {_value(patient, 'SSN')}" if _value(patient, "SSN") else "",
        f"Driver's licence: {_value(patient, 'DRIVERS')}" if _value(patient, "DRIVERS") else "",
        f"Passport: {_value(patient, 'PASSPORT')}" if _value(patient, "PASSPORT") else "",
        f"Synthea ID: {_value(patient, 'Id')}",
    ]
    return [line for line in lines if line]


def _period(start: str | None, stop: str | None) -> str:
    if start and stop:
        return f"{start} to {stop}" if stop != start else start
    return f"{start} (ongoing)" if start else "date unknown"


def _reason(row: dict[str, str]) -> str:
    reason = _value(row, "REASONDESCRIPTION")
    return f"; reason: {reason}" if reason else ""


def condition_line(row: dict[str, str]) -> str:
    """One condition row, e.g. ``2015-03-02 to 2015-03-16: Acute viral pharyngitis (disorder)``."""
    period = _period(iso_date(row.get("START")), iso_date(row.get("STOP")))
    return f"{period}: {_value(row, 'DESCRIPTION')} [code {_value(row, 'CODE')}]"


def medication_line(row: dict[str, str]) -> str:
    """One medication row with its period and the reason it was prescribed."""
    period = _period(iso_date(row.get("START")), iso_date(row.get("STOP")))
    return f"{period}: {_value(row, 'DESCRIPTION')} [code {_value(row, 'CODE')}]{_reason(row)}"


def encounter_line(row: dict[str, str]) -> str:
    """One encounter row with its class, description and reason."""
    start = iso_date(row.get("START")) or "date unknown"
    return f"{start}, {_value(row, 'ENCOUNTERCLASS')}: {_value(row, 'DESCRIPTION')}{_reason(row)}"


def _dated_units(
    rows: list[dict[str, str]], section: str, render: Callable[[dict[str, str]], str]
) -> list[TextUnit]:
    ordered = sorted(rows, key=lambda r: (iso_date(r.get("START")) or "", _value(r, "DESCRIPTION")))
    return [TextUnit(text=render(row), section=section, date=iso_date(row.get("START"))) for row in ordered]


def build_patient_document(
    patient: dict[str, str],
    conditions: list[dict[str, str]],
    medications: list[dict[str, str]],
    encounters: list[dict[str, str]],
    pseudonymous_id: str,
    dataset: str = "synthea",
) -> Document:
    """Build one patient's document: demographics, conditions, medications, encounters."""
    units = [TextUnit(text=line, section=DEMOGRAPHICS) for line in demographics_lines(patient)]
    units += _dated_units(conditions, CONDITIONS, condition_line)
    units += _dated_units(medications, MEDICATIONS, medication_line)
    units += _dated_units(encounters, ENCOUNTERS, encounter_line)
    return Document(
        source=f"{dataset}:{pseudonymous_id}",
        doc_type=PATIENT_RECORD,
        title="Patient record",
        units=units,
        patient_id=pseudonymous_id,
    )


VOCABULARY_FILES = ("conditions.csv", "medications.csv", "encounters.csv")
VOCABULARY_COLUMNS = ("DESCRIPTION", "REASONDESCRIPTION", "CODE", "REASONCODE")


def clinical_vocabulary(csv_dirs: list[Path], min_token_length: int = 3) -> set[str]:
    """Clinical terms and codes from Synthea's coded rows, for the redaction allow-list.

    Returns every description and code, plus each alphabetic word in the descriptions.
    These come from standard code systems (SNOMED CT, RxNorm), never from patient
    identifiers. Synthea names always carry digits, so no name can match an
    alphabetic word here.
    """
    vocabulary: set[str] = set()
    for folder in csv_dirs:
        for name in VOCABULARY_FILES:
            for row in read_csv(folder / name):
                for column in VOCABULARY_COLUMNS:
                    value = _value(row, column)
                    if not value:
                        continue
                    vocabulary.add(value)
                    words = re.findall(r"[A-Za-z][A-Za-z'-]*[A-Za-z]", value)
                    vocabulary.update(w for w in words if len(w) >= min_token_length)
    return vocabulary


def load_patients(csv_dir: Path) -> list[dict[str, str]]:
    """Rows of ``patients.csv``, sorted by Synthea UUID for a stable order."""
    patients = read_csv(csv_dir / "patients.csv")
    if not patients:
        raise FileNotFoundError(f"No patients found in {csv_dir / 'patients.csv'}. Run `make data` first.")
    return sorted(patients, key=lambda p: p["Id"])


def build_patient_documents(
    patients: list[dict[str, str]], csv_dir: Path, salt: str, id_length: int, dataset: str = "synthea"
) -> list[Document]:
    """One document per patient from the Synthea CSV export in ``csv_dir``."""
    conditions = group_by_patient(read_csv(csv_dir / "conditions.csv"))
    medications = group_by_patient(read_csv(csv_dir / "medications.csv"))
    encounters = group_by_patient(read_csv(csv_dir / "encounters.csv"))
    documents = []
    for patient in patients:
        uuid = patient["Id"]
        documents.append(
            build_patient_document(
                patient,
                conditions.get(uuid, []),
                medications.get(uuid, []),
                encounters.get(uuid, []),
                make_patient_id(uuid, salt, id_length),
                dataset,
            )
        )
    return documents
