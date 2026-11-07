"""Markdown redaction report built from a finished ingest run.

Every number in the report is computed here from the run's data; nothing is typed in.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from healthrag.config import Settings
from healthrag.ingest.documents import GUIDELINE, PATIENT_RECORD
from healthrag.ingest.pipeline import IngestResult, ProcessedChunk
from healthrag.ingest.synthea import DEMOGRAPHICS
from healthrag.redact.leak_scan import IDENTIFIER_COLUMNS, LeakScanResult
from healthrag.redact.recognizers import ENTITIES, PLACEHOLDERS

ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")

_KIND_LABELS = {
    "name": "Names (first, middle, last, maiden)",
    "ssn": "SSN",
    "drivers_license": "Driver's licence",
    "passport": "Passport",
    "address": "Street address",
    "uuid": "Synthea UUID",
    "birth_date": "Date of birth (own demographics chunk)",
}


def _bucket(p: ProcessedChunk) -> str:
    if p.chunk.doc_type == GUIDELINE:
        return "guideline"
    return "demographics" if p.chunk.section == DEMOGRAPHICS else "clinical"


def _table(header: list[str], rows: list[list[object]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return lines


def entity_count_rows(chunks: list[ProcessedChunk]) -> list[list[object]]:
    """Applied redactions per entity type, split by where they happened."""
    counts: dict[str, Counter[str]] = {bucket: Counter() for bucket in ("demographics", "clinical", "guideline")}
    for p in chunks:
        counts[_bucket(p)].update(d.entity_type for d in p.redacted.detections)
    rows: list[list[object]] = []
    for entity in ENTITIES:
        per_bucket = [counts[b][entity] for b in ("demographics", "clinical", "guideline")]
        rows.append([f"`{entity}`", f"`{PLACEHOLDERS[entity]}`", *per_bucket, sum(per_bucket)])
    totals = [sum(counts[b].values()) for b in ("demographics", "clinical", "guideline")]
    rows.append(["**Total**", "", *totals, sum(totals)])
    return rows


def recognizer_rows(chunks: list[ProcessedChunk]) -> list[list[object]]:
    """Applied redactions per recognizer (after overlap resolution), with and without the deny-list."""
    with_list = Counter(d.recognizer for p in chunks for d in p.redacted.detections)
    without = Counter(d.recognizer for p in chunks for d in p.redacted_without_deny_list.detections)
    names = sorted(set(with_list) | set(without), key=lambda n: (-with_list[n], n))
    return [[f"`{name}`", with_list[name], without[name]] for name in names]


def leak_rows(with_list: LeakScanResult, without: LeakScanResult) -> list[list[object]]:
    """Per identifier kind: identifiers checked, and leaks (occurrences / distinct values) for both runs."""
    rows: list[list[object]] = []
    w_occ, w_val = with_list.leaks_by_kind(), with_list.leaked_values_by_kind()
    o_occ, o_val = without.leaks_by_kind(), without.leaked_values_by_kind()
    for kind in IDENTIFIER_COLUMNS:
        rows.append([_KIND_LABELS[kind], with_list.checked[kind], w_occ[kind], w_val[kind], o_occ[kind], o_val[kind]])
    rows.append(
        [
            "**Total**",
            sum(with_list.checked.values()),
            len(with_list.leaks),
            sum(w_val.values()),
            len(without.leaks),
            sum(o_val.values()),
        ]
    )
    return rows


def leak_examples(result: LeakScanResult, limit: int) -> list[str]:
    """A few concrete leaks (synthetic values) so they can be inspected."""
    seen: list[str] = []
    for leak in result.leaks:
        line = f"- {_KIND_LABELS[leak.kind]}: `{leak.value}` in chunk `{leak.chunk_id}`"
        if line not in seen:
            seen.append(line)
        if len(seen) == limit:
            break
    return seen


def date_survival(chunks: list[ProcessedChunk]) -> tuple[int, int]:
    """ISO dates in patient clinical sections before and after redaction."""
    clinical = [p for p in chunks if _bucket(p) == "clinical"]
    before = sum(len(ISO_DATE.findall(p.raw_text)) for p in clinical)
    after = sum(len(ISO_DATE.findall(p.redacted.text)) for p in clinical)
    return before, after


def pick_samples(chunks: list[ProcessedChunk], limit: int) -> list[tuple[str, ProcessedChunk]]:
    """Deterministic before/after samples covering the main cases."""

    def has(entity: str) -> Callable[[ProcessedChunk], bool]:
        return lambda p: any(d.entity_type == entity for d in p.redacted.detections)

    wanted: list[tuple[str, Callable[[ProcessedChunk], bool]]] = [
        ("Patient demographics", lambda p: _bucket(p) == "demographics"),
        ("Patient demographics with a previous surname", lambda p: _bucket(p) == "demographics" and "Previous surname" in p.raw_text),
        ("Patient encounters: dates are kept", lambda p: p.chunk.section == "Encounters" and not p.redacted.detections),
        ("Patient clinical section with a redaction (false positive)", lambda p: _bucket(p) == "clinical" and bool(p.redacted.detections)),
        ("Guideline with a phone number", lambda p: _bucket(p) == "guideline" and has("PHONE_NUMBER")(p)),
        ("Guideline with an address", lambda p: _bucket(p) == "guideline" and has("ADDRESS")(p)),
        ("Guideline with a person name", lambda p: _bucket(p) == "guideline" and has("PERSON")(p)),
    ]  # fmt: skip
    ordered = sorted(chunks, key=lambda p: (p.chunk.doc_type != PATIENT_RECORD, p.chunk.source, p.chunk.ordinal))
    picked: list[tuple[str, ProcessedChunk]] = []
    for label, predicate in wanted:
        match = next((p for p in ordered if predicate(p) and all(p is not q for _, q in picked)), None)
        if match is not None:
            picked.append((label, match))
        if len(picked) == limit:
            break
    return picked


def false_positive_rows(chunks: list[ProcessedChunk], limit: int) -> list[list[object]]:
    """Most frequent spans redacted in patient clinical sections, which contain no identifiers by construction."""
    spans = Counter(
        (d.entity_type, d.original) for p in chunks if _bucket(p) == "clinical" for d in p.redacted.detections
    )
    return [[f"`{entity}`", f"`{text}`", count] for (entity, text), count in spans.most_common(limit)]


def build_report(result: IngestResult, settings: Settings) -> str:
    """Render the full Markdown report."""
    chunks = result.chunks
    s, r = settings.synthea, settings.redaction
    patient_chunks = [p for p in chunks if p.chunk.doc_type == PATIENT_RECORD]
    guideline_chunks = [p for p in chunks if p.chunk.doc_type == GUIDELINE]
    deceased = sum(1 for p in result.patients if (p.get("DEATHDATE") or "").strip())
    n_guidelines = sum(1 for d in result.documents if d.doc_type == GUIDELINE)
    dates_before, dates_after = date_survival(chunks)
    clinical_chunks = sum(1 for p in chunks if _bucket(p) == "clinical")
    clinical_touched = sum(1 for p in chunks if _bucket(p) == "clinical" and p.redacted.detections)
    guideline_touched = sum(1 for p in guideline_chunks if p.redacted.detections)
    leaks, leaks_without = result.leaks, result.leaks_without_deny_list
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    out: list[str] = [
        "# Redaction report",
        "",
        f"Generated by `python -m healthrag.ingest` on {generated}. All patient data is synthetic.",
        "",
        "## Run",
        "",
        *_table(
            ["Setting", "Value"],
            [
                ["Synthea", f"{s.version}, seed {s.seed}, clinician seed {s.clinician_seed}, "
                            f"reference date {s.reference_date}, `-p {s.population}`, {s.state}"],
                ["Patient datasets", ", ".join(settings.patient_datasets)
                                     + (f" (sample: {settings.synthea_sample.description})"
                                        if "sample" in settings.patient_datasets else "")],
                ["Patients in patients.csv", f"{len(result.patients)} ({len(result.patients) - deceased} alive, {deceased} deceased)"],
                ["Documents", f"{len(result.patients)} patient records, {n_guidelines} guideline PDFs"],
                ["Chunks", f"{len(chunks)} ({len(patient_chunks)} patient, {len(guideline_chunks)} guideline)"],
                ["Chunk size / overlap", f"{settings.chunking.size} / {settings.chunking.overlap} characters"],
                ["NER model", f"spaCy `{r.spacy_model}` {version(r.spacy_model)} (spaCy {version('spacy')})"],
                ["Presidio", f"analyzer {version('presidio-analyzer')}, anonymizer {version('presidio-anonymizer')}"],
                ["Score threshold", r.score_threshold],
                ["Allow-list", f"{result.allow_list_size} clinical terms and codes from the Synthea condition, "
                               "medication and encounter rows (never redacted)"],
                ["Redaction time", f"{result.redaction_seconds:.1f} s for both passes (spaCy runs once per chunk)"],
            ],
        ),  # fmt: skip
        "",
        "## Redactions per entity type",
        "",
        "Replacements applied in `data/processed/chunks.jsonl` (the run with the deny-list). Overlapping "
        "detections are resolved first, so each replaced span is counted once.",
        "",
        *_table(["Entity type", "Placeholder", "Patient demographics", "Patient clinical sections", "Guidelines", "Total"],
                entity_count_rows(chunks)),  # fmt: skip
        "",
        "Applied replacements by recognizer. When two recognizers find the same span, the higher score "
        "wins, so the deny-list (score 1.0) takes most names that the suffix-name pattern also finds.",
        "",
        *_table(["Recognizer", "With deny-list", "Without deny-list"], recognizer_rows(chunks)),
        "",
        "## Leak scan",
        "",
        "Every identifier value in `patients.csv` is searched for in the redacted text of every chunk "
        "(case-insensitive, whole tokens only). A leak is one occurrence of one identifier in one chunk. "
        "Dates of birth are only searched in that patient's own demographics chunks, because the same "
        "calendar date may legitimately appear as an encounter date elsewhere.",
        "",
        *_table(
            ["Identifier", "Values checked", "Leaks with deny-list", "Distinct values leaked",
             "Leaks without deny-list", "Distinct values leaked (no deny-list)"],
            leak_rows(leaks, leaks_without),
        ),  # fmt: skip
        "",
        f"**Leaks found: {len(leaks.leaks)} with the deny-list, {len(leaks_without.leaks)} without it.**",
        "",
    ]
    for label, scan in (("with the deny-list", leaks), ("without the deny-list", leaks_without)):
        examples = leak_examples(scan, 10)
        if examples:
            out += [f"Examples of leaks {label}:", "", *examples, ""]

    out += [
        "### Does recall depend on the deny-list?",
        "",
        dependence_summary(leaks, leaks_without),
        "",
        "## Dates and over-redaction",
        "",
        f"- ISO dates in patient clinical sections: {dates_before} before redaction, {dates_after} after. "
        "Only dates of birth are redacted; encounter, condition and medication dates are kept for the date filter.",
        f"- Patient clinical-section chunks with any redaction: {clinical_touched} of {clinical_chunks}. These "
        "sections hold only coded Synthea rows (no names, IDs or addresses), so every redaction there is a false positive.",
        f"- Guideline chunks with any redaction: {guideline_touched} of {len(guideline_chunks)}.",
        "",
    ]
    fp_rows = false_positive_rows(chunks, 15)
    if fp_rows:
        out += ["Most frequent spans redacted in patient clinical sections:", "",
                *_table(["Entity type", "Span", "Count"], fp_rows), ""]  # fmt: skip

    out += [
        "Most guideline `PERSON` redactions are author names in reference lists and work-group pages. "
        "They are public, not patient data, so removing them only costs a little retrieval text.",
        "",
        "## Before / after samples",
        "",
    ]
    for label, p in pick_samples(chunks, r.report_samples):
        c = p.chunk
        out += [
            f"### {label}",
            "",
            f"`{p.chunk_id}` · {c.doc_type} · section `{c.section}` · date `{c.date}`",
            "",
            "Before:",
            "",
            "```text",
            p.raw_text,
            "```",
            "",
            "After:",
            "",
            "```text",
            p.redacted.text,
            "```",
            "",
        ]
    return "\n".join(out).rstrip() + "\n"


def dependence_summary(with_list: LeakScanResult, without: LeakScanResult) -> str:
    """One paragraph comparing the two leak scans, using only their numbers."""
    w, o = len(with_list.leaks), len(without.leaks)
    names_w, names_o = with_list.leaks_by_kind()["name"], without.leaks_by_kind()["name"]
    if w == o:
        return (
            f"No difference in this run: {w} leaks with the deny-list and {o} without it "
            f"(name leaks: {names_w} vs {names_o}). The pattern recognizers alone catch every name "
            "the deny-list catches, so recall does not depend on the deny-list here."
        )
    return (
        f"Yes: removing the deny-list changes the leak count from {w} to {o} "
        f"(name leaks: {names_w} vs {names_o}). Without it, the pattern recognizers and spaCy miss the "
        "values listed above."
    )


def write_report(result: IngestResult, settings: Settings) -> Path:
    """Write the report to the configured path and return it."""
    path = settings.paths.redaction_report
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_report(result, settings), encoding="utf-8", newline="\n")
    return path
