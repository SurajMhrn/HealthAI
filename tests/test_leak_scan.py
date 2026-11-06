from healthrag.redact.leak_scan import KnownIdentifier, known_identifiers, scan_chunks

IDS = [
    KnownIdentifier("name", "Abe604", "p1"),
    KnownIdentifier("ssn", "999-12-3456", "p1"),
    KnownIdentifier("birth_date", "1985-03-12", "p1"),
]


def _chunk(cid: str, text: str, section: str = "Encounters", pid: str = "p1") -> dict:
    return {"chunk_id": cid, "text": text, "section": section, "patient_id": pid}


def test_planted_leaks_are_found() -> None:
    chunks = [_chunk("c1", "Name: abe604"), _chunk("c2", "SSN 999-12-3456 and Abe604")]
    result = scan_chunks(IDS, chunks)
    assert result.leaks_by_kind() == {"name": 2, "ssn": 1}
    assert result.leaked_values_by_kind() == {"name": 1, "ssn": 1}


def test_only_whole_tokens_count() -> None:
    assert scan_chunks(IDS, [_chunk("c1", "Abe6041 and x999-12-34567")]).leaks == []


def test_birth_date_only_counts_in_own_demographics() -> None:
    chunks = [
        _chunk("enc", "1985-03-12, wellness: visit"),
        _chunk("demo-other", "Date of birth: 1985-03-12", section="Demographics", pid="p2"),
        _chunk("demo", "Date of birth: 1985-03-12", section="Demographics"),
    ]
    leaks = scan_chunks(IDS, chunks).leaks
    assert [(leak.kind, leak.chunk_id) for leak in leaks] == [("birth_date", "demo")]


def test_known_identifiers_cover_all_columns(patients) -> None:  # noqa: ANN001
    ids = known_identifiers(patients, {p["Id"]: f"pid{i}" for i, p in enumerate(patients)})
    kinds = {i.kind for i in ids}
    assert kinds == {"name", "ssn", "drivers_license", "passport", "address", "uuid", "birth_date"}
    assert KnownIdentifier("name", "Hartmann412", "pid0") in ids
    assert KnownIdentifier("birth_date", "1961-07-14", "pid0") in ids
