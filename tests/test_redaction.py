"""End-to-end redaction with Presidio + spaCy on fixture documents."""

from healthrag.ids import patient_id
from healthrag.ingest.chunking import chunk_document
from healthrag.ingest.synthea import DEMOGRAPHICS
from healthrag.redact.engine import Redactor
from healthrag.redact.leak_scan import known_identifiers, scan_chunks
from tests.conftest import TEST_SALT



def _records(redactor: Redactor, documents, use_deny_list: bool = True) -> list[dict]:  # noqa: ANN001
    records = []
    for doc in documents:
        for chunk in chunk_document(doc, 1000, 150):
            redacted = redactor.redact(chunk.text, use_deny_list=use_deny_list)
            records.append(
                {
                    "chunk_id": f"{chunk.source}#{chunk.ordinal}",
                    "text": redacted.text,
                    "section": chunk.section,
                    "patient_id": chunk.patient_id,
                }
            )
    return records


def test_chunk_with_known_fake_phi_comes_out_clean(redactor: Redactor, patients, patient_documents) -> None:  # noqa: ANN001
    records = _records(redactor, patient_documents)
    ids = {p["Id"]: patient_id(p["Id"], TEST_SALT, 16) for p in patients}
    result = scan_chunks(known_identifiers(patients, ids), records)
    assert result.leaks == []

    demographics = next(r["text"] for r in records if r["section"] == DEMOGRAPHICS)
    for placeholder in ("<PERSON>", "<DATE_OF_BIRTH>", "<ADDRESS>", "<RECORD_ID>"):
        assert placeholder in demographics
    for raw in ("Rosalind817", "Okuneva707", "Hartmann412", "999-41-2234", "S99945871", "X41258763X", "Kautzer"):
        assert raw not in demographics


def test_free_text_phi_is_redacted(redactor: Redactor) -> None:
    text = (
        "Patient Abe604 Frami345 (SSN 999-12-3456, passport X743270X) was born on 1985-03-12. "
        "Call (555) 123-4567. Lives at 12 Elm Street, Springfield, MA 01103. "
        "Record 1d604da9-9a81-4ba9-80c2-de3375d59b40."
    )
    out = redactor.redact(text).text
    for raw in ("Abe604", "Frami345", "999-12-3456", "X743270X", "1985-03-12", "123-4567", "Elm Street", "1d604da9"):
        assert raw not in out
    for placeholder in ("<PERSON>", "<RECORD_ID>", "<DATE_OF_BIRTH>", "<PHONE_NUMBER>", "<ADDRESS>"):
        assert placeholder in out


def test_dob_is_redacted_but_encounter_dates_survive(redactor: Redactor, patient_documents) -> None:  # noqa: ANN001
    # Patient 2 was born on 2009-02-03 and has a well-child visit on that same date.
    doc = next(d for d in patient_documents if any("2009-02-03" in u.text for u in d.units))
    chunks = {c.section: redactor.redact(c.text).text for c in chunk_document(doc, 1000, 150)}
    assert "Date of birth: <DATE_OF_BIRTH>" in chunks[DEMOGRAPHICS]
    assert "2009-02-03" not in chunks[DEMOGRAPHICS]
    assert "2009-02-03, wellness: Well child visit (procedure)" in chunks["Encounters"]
    assert "2014-06-10" in chunks["Conditions"]


def test_clinical_terms_are_not_redacted(redactor: Redactor) -> None:
    text = "2012-05-04 (ongoing): Essential hypertension (disorder) [code 59621000]"
    assert redactor.redact(text).text == text


def test_suffixed_names_are_caught_without_the_deny_list(redactor: Redactor) -> None:
    text = "Name: Mr. Zebulon771 Quigley88"
    assert redactor.redact(text, use_deny_list=False).text == "Name: Mr. <PERSON>"
