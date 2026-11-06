from healthrag.ingest.chunking import chunk_document, pack_units, split_long_unit
from healthrag.ingest.documents import GUIDELINE, PATIENT_RECORD, Document, TextUnit
from healthrag.ingest.pdf import clean_lines, is_heading, repeated_lines
from healthrag.ingest.synthea import CONDITIONS, DEMOGRAPHICS, ENCOUNTERS, MEDICATIONS, clinical_vocabulary
from tests.conftest import SYNTHEA_FIXTURES


def test_patient_document_has_sections_and_identifiers(patient_documents) -> None:  # noqa: ANN001
    doc = patient_documents[0]
    assert doc.doc_type == PATIENT_RECORD
    assert [s for s in (DEMOGRAPHICS, CONDITIONS, MEDICATIONS, ENCOUNTERS) if any(u.section == s for u in doc.units)] == [
        DEMOGRAPHICS,
        CONDITIONS,
        MEDICATIONS,
        ENCOUNTERS,
    ]
    demographics = "\n".join(u.text for u in doc.units if u.section == DEMOGRAPHICS)
    assert "Name: Mrs. Rosalind817 Ines45 Okuneva707" in demographics
    assert "Date of birth: 1961-07-14" in demographics
    assert "SSN: 999-41-2234" in demographics
    assert "Passport: X41258763X" in demographics
    assert "Address: 418 Kautzer Overpass Suite 9, Worcester, Massachusetts 01608" in demographics
    # The source and patient_id are pseudonymous, never the raw UUID.
    assert "3f1c2a9e" not in doc.source and doc.patient_id and "3f1c2a9e" not in doc.patient_id


def test_patient_chunks_carry_dates_but_demographics_do_not(patient_documents) -> None:  # noqa: ANN001
    chunks = chunk_document(patient_documents[0], 1000, 150)
    by_section = {c.section: c for c in chunks}
    assert by_section[DEMOGRAPHICS].date is None and by_section[DEMOGRAPHICS].date_start is None
    assert by_section[CONDITIONS].date_start == "2012-05-04"
    assert by_section[CONDITIONS].date == "2019-11-02"
    assert all(c.patient_id == patient_documents[0].patient_id for c in chunks)


def test_chunks_respect_size_overlap_and_sections() -> None:
    units = [TextUnit(f"line {i:03d} " + "x" * 40, "A") for i in range(30)]
    units += [TextUnit(f"other {i}", "B") for i in range(3)]
    doc = Document(source="d", doc_type=GUIDELINE, title="Doc", units=units, date="2025-03")
    chunks = chunk_document(doc, size=200, overlap=60)
    for chunk in chunks:
        header, body = chunk.text.split("\n", 1)
        assert header == f"[Doc | {chunk.section}]"
        assert len(body) <= 200
        assert chunk.date == "2025-03"
    a_chunks = [c for c in chunks if c.section == "A"]
    assert len(a_chunks) > 1
    # Consecutive chunks share their boundary line (overlap).
    first_last_line = a_chunks[0].text.splitlines()[-1]
    assert a_chunks[1].text.splitlines()[1] == first_last_line
    # Section B never mixes with A.
    assert all("line" not in c.text for c in chunks if c.section == "B")
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_long_lines_are_split_on_words() -> None:
    unit = TextUnit(" ".join(["word"] * 100), "S")
    pieces = split_long_unit(unit, 50)
    assert all(len(p.text) <= 50 for p in pieces)
    assert " ".join(p.text for p in pieces) == unit.text
    assert len(pack_units([unit], 50, 10)) == len(pieces)


def test_heading_detection() -> None:
    for line in ("3.2 Diagnosis", "IV. Algorithm", "Appendix B: Evidence Tables", "RECOMMENDATIONS", "A. Background"):
        assert is_heading(line, 90), line
    for line in (
        "Introduction ........................ 5",
        "The Work Group recommends treating adults with hypertension.",
        "1",
        "x" * 120,
    ):
        assert not is_heading(line, 90), line


def test_running_headers_and_page_numbers_are_removed() -> None:
    bodies = ["Blood pressure", "Statins", "Spirometry", "Inhalers", "Diet", "Exercise", "Sleep", "Smoking", "Alcohol"]
    pages = [clean_lines(f"VA/DoD Guideline March 2025 Page {i} of 9\n{body}\n{i}") for i, body in enumerate(bodies, 1)]
    assert all(len(lines) == 2 for lines in pages)  # bare page number dropped
    boilerplate = repeated_lines(pages, min_share=0.3, min_pages=5)
    assert "va/dod guideline march # page # of #" in boilerplate
    assert boilerplate == {"va/dod guideline march # page # of #"}


def test_clinical_vocabulary_has_terms_and_codes_but_no_names() -> None:
    vocab = clinical_vocabulary([SYNTHEA_FIXTURES])
    assert {"Hydrochlorothiazide", "hypertension", "59621000", "Asthma (disorder)"} <= vocab
    assert not {"Rosalind817", "Okuneva707", "José512"} & vocab
