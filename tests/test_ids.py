import pytest

from healthrag.ids import chunk_id, patient_id

UUID = "3f1c2a9e-5b7d-4e8f-9a01-23456789abcd"


def test_patient_id_is_deterministic() -> None:
    assert patient_id(UUID, "salt-a", 16) == patient_id(UUID, "salt-a", 16)


def test_patient_id_never_equals_or_contains_raw_uuid() -> None:
    pid = patient_id(UUID, "salt-a", 16)
    assert pid != UUID
    assert UUID not in pid
    assert UUID.replace("-", "")[:16] != pid


def test_patient_id_depends_on_salt_and_uuid() -> None:
    assert patient_id(UUID, "salt-a", 16) != patient_id(UUID, "salt-b", 16)
    assert patient_id(UUID, "salt-a", 16) != patient_id("8a7b6c5d-4e3f-4a1b-8c9d-0e1f2a3b4c5d", "salt-a", 16)


def test_patient_id_is_truncated_hex() -> None:
    pid = patient_id(UUID, "salt-a", 16)
    assert len(pid) == 16
    int(pid, 16)


def test_patient_id_ignores_uuid_case_and_whitespace() -> None:
    assert patient_id(UUID.upper(), "s", 16) == patient_id(f" {UUID} ", "s", 16)


def test_patient_id_rejects_empty_salt() -> None:
    with pytest.raises(ValueError):
        patient_id(UUID, "", 16)


def test_chunk_id_is_deterministic_and_content_addressed() -> None:
    first = chunk_id("doc.pdf", 3, "Intro", "some <PERSON> text")
    assert first == chunk_id("doc.pdf", 3, "Intro", "some <PERSON> text")
    assert first != chunk_id("doc.pdf", 4, "Intro", "some <PERSON> text")
    assert first != chunk_id("doc.pdf", 3, "Intro", "other text")
