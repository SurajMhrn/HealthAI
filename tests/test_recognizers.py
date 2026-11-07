"""One test (or more) per custom recognizer. These run without spaCy."""

from presidio_analyzer import RecognizerResult

from healthrag.redact import recognizers as rec


def spans(results: list[RecognizerResult], text: str) -> list[str]:
    return [text[r.start : r.end] for r in results]


def test_synthea_name_recognizer_finds_suffixed_names() -> None:
    text = "Seen by Abe604 Frami345 with Rosalind817 O'Keefe54 and José512."
    found = spans(rec.synthea_name_recognizer().analyze(text, ["PERSON"]), text)
    assert found == ["Abe604", "Frami345", "Rosalind817", "O'Keefe54", "José512"]


def test_synthea_name_recognizer_ignores_clinical_tokens() -> None:
    text = "COVID-19 vaccine, vitamin B12, HbA1c 7.2, Type 2 diabetes, SARS-CoV-2, 120 ACTUAT inhaler"
    assert rec.synthea_name_recognizer().analyze(text, ["PERSON"]) == []


def test_record_uuid_recognizer() -> None:
    text = "Synthea ID: 2d68ad16-268a-478c-1f84-d0f1976e1a46."
    assert spans(rec.record_uuid_recognizer().analyze(text, ["RECORD_UUID"]), text) == [
        "2d68ad16-268a-478c-1f84-d0f1976e1a46"
    ]


def test_ssn_recognizer_accepts_synthea_999_range() -> None:
    text = "SSN: 999-41-2234; phone 555-0100"
    assert spans(rec.ssn_recognizer().analyze(text, ["SSN"]), text) == ["999-41-2234"]


def test_driver_license_recognizer() -> None:
    text = "Driver's licence: S99945871 (ICD E11.9, code 44054006)"
    assert spans(rec.driver_license_recognizer().analyze(text, ["DRIVER_LICENSE"]), text) == ["S99945871"]


def test_passport_recognizer_handles_short_numbers() -> None:
    text = "Passport: X41258763X, old passport X743270X"
    found = spans(rec.passport_recognizer().analyze(text, ["PASSPORT"]), text)
    assert found == ["X41258763X", "X743270X"]
    # A passport must not also be read as a driver's licence.
    assert rec.driver_license_recognizer().analyze(text, ["DRIVER_LICENSE"]) == []


def test_labelled_address_recognizer_takes_the_whole_line() -> None:
    text = "Address: 418 Kautzer Overpass Suite 9, Worcester, Massachusetts 01608\nSSN: 999-41-2234"
    found = spans(rec.address_recognizer().analyze(text, ["ADDRESS"]), text)
    assert found == ["418 Kautzer Overpass Suite 9, Worcester, Massachusetts 01608"]


def test_address_line_recognizer_in_free_text() -> None:
    text = "Mail forms to 810 Vermont Avenue NW, Washington, DC 20420 by Friday."
    found = spans(rec.address_line_recognizer().analyze(text, ["ADDRESS"]), text)
    assert found == ["810 Vermont Avenue NW, Washington, DC 20420"]


def test_date_of_birth_recognizer_only_takes_birth_dates() -> None:
    text = (
        "Date of birth: 1961-07-14\n"
        "2019-11-02, ambulatory: Encounter for symptom (procedure)\n"
        "The patient was born on March 4, 1990. DOB 03/04/1990."
    )
    found = spans(rec.date_of_birth_recognizer().analyze(text, ["DATE_OF_BIRTH"]), text)
    assert found == ["1961-07-14", "March 4, 1990", "03/04/1990"]


def test_name_deny_list_recognizer_matches_exact_names() -> None:
    recognizer = rec.name_deny_list_recognizer(["Hartmann412", "Okuneva707", ""])
    text = "Née hartmann412; not Hartmann4120."
    assert spans(recognizer.analyze(text, ["PERSON"]), text) == ["hartmann412"]


def test_every_entity_has_a_typed_placeholder() -> None:
    assert set(rec.PLACEHOLDERS.values()) == {
        "<PERSON>",
        "<DATE_OF_BIRTH>",
        "<ADDRESS>",
        "<PHONE_NUMBER>",
        "<RECORD_ID>",
    }
