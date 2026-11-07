"""Build eval/golden.jsonl: hand-written guideline items plus patient items derived from the CSVs.

Run with ``python -m scripts.make_golden`` (or ``make golden``) after ``make ingest``.

* Guideline lookups, guideline multi-hop and unanswerable guideline questions come from
  ``eval/golden_manual.jsonl`` (written by hand against the recommendation statements).
* Patient lookups (first-recorded date, medication for a reason, last encounter of a type),
  patient multi-hop (diagnosis date + medication) and unanswerable patient questions are
  generated from the Synthea CSVs, so every expected answer is checkable.

Patient items carry the pseudonymous ``patient_id``, which depends on ``PATIENT_ID_SALT``;
with a different salt, run this script again. Every item stays ``"status": "draft"`` until
it has been reviewed by hand.
"""

from __future__ import annotations

import json
import random
import re
import sys
from collections import defaultdict
from datetime import date

from healthrag.config import PROJECT_ROOT, get_settings
from healthrag.ids import patient_id
from healthrag.ingest.synthea import iso_date, read_csv

MANUAL = PROJECT_ROOT / "eval" / "golden_manual.jsonl"
GOLDEN = PROJECT_ROOT / "eval" / "golden.jsonl"
PER_TYPE = {"patient_lookup_condition": 15, "patient_lookup_medication": 15, "patient_lookup_encounter": 15,
            "patient_multihop": 10}  # fmt: skip
ENCOUNTER_WORDS = {"wellness": "wellness", "ambulatory": "ambulatory", "emergency": "emergency",
                   "urgentcare": "urgent care", "inpatient": "inpatient", "outpatient": "outpatient"}  # fmt: skip
UNANSWERABLE_PATIENT = [
    "What is the patient's blood type?",
    "Which allergies are documented for this patient?",
    "What was the patient's most recent blood pressure reading?",
    "What was the patient's latest HbA1c value?",
    "Which immunizations has the patient received?",
    "Which insurance plan covers this patient?",
    "What is the name of the patient's primary care physician?",
]


def date_variants(iso: str) -> list[str]:
    """The same date in formats a model might use."""
    d = date.fromisoformat(iso)
    month = d.strftime("%B")
    return [iso, f"{month} {d.day}, {d.year}", f"{d.day} {month} {d.year}", f"{month[:3]} {d.day}, {d.year}",
            f"{d.month}/{d.day}/{d.year}", f"{d.month:02d}/{d.day:02d}/{d.year}"]  # fmt: skip


def clean(description: str) -> str:
    """Drop SNOMED's trailing semantic tag, e.g. "Asthma (disorder)" -> "Asthma"."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", description).strip()


def first_word(description: str) -> str:
    """The leading drug name of a medication description, lower-case."""
    match = re.search(r"[A-Za-z][A-Za-z-]{2,}", description)
    return match.group(0).lower() if match else description.lower()


def patient_items(settings, rng: random.Random) -> list[dict]:  # noqa: ANN001
    """Lookup, multi-hop and unanswerable items for patients in every enabled dataset."""
    salt = settings.require_salt()
    candidates: dict[str, list[dict]] = defaultdict(list)
    pids: list[tuple[str, str]] = []
    for dataset, folder in settings.patient_csv_dirs().items():
        conditions = defaultdict(list)
        for row in read_csv(folder / "conditions.csv"):
            conditions[row["PATIENT"]].append(row)
        medications = defaultdict(list)
        for row in read_csv(folder / "medications.csv"):
            medications[row["PATIENT"]].append(row)
        encounters = defaultdict(list)
        for row in read_csv(folder / "encounters.csv"):
            encounters[row["PATIENT"]].append(row)
        for uuid in sorted(conditions.keys() | medications.keys() | encounters.keys()):
            pid = patient_id(uuid, salt, settings.patient_id.length)
            source = f"synthea-{dataset}:{pid}"
            filters = {"doc_type": "patient_record", "patient_id": pid}
            pids.append((pid, source))
            by_desc = defaultdict(list)
            for row in conditions[uuid]:
                by_desc[row["DESCRIPTION"]].append(row)
            meds_by_reason = defaultdict(set)
            for row in medications[uuid]:
                if row.get("REASONDESCRIPTION"):
                    meds_by_reason[row["REASONDESCRIPTION"]].add(row["DESCRIPTION"])
            for desc, rows in by_desc.items():
                if not desc.endswith("(disorder)") or len(rows) != 1:
                    continue
                start = iso_date(rows[0]["START"])
                candidates["patient_lookup_condition"].append({
                    "question": f"When was {clean(desc).lower()} first recorded for this patient?",
                    "reference": f"{clean(desc)} was first recorded on {start}.",
                    "answer_contains": [date_variants(start)],
                    "support": [{"source": source, "section": "Conditions", "contains": [start, desc]}],
                    "filters": filters})  # fmt: skip
                meds = meds_by_reason.get(desc, set())
                if len(meds) == 1:
                    med = next(iter(meds))
                    candidates["patient_multihop"].append({
                        "question": f"When was the patient's {clean(desc).lower()} diagnosed, and which medication "
                                    "was prescribed for it?",
                        "reference": f"{clean(desc)} was first recorded on {start}; it was treated with {med}.",
                        "answer_contains": [date_variants(start), first_word(med)],
                        "support": [{"source": source, "section": "Conditions", "contains": [start, desc]},
                                    {"source": source, "section": "Medications", "contains": [med, desc]}],
                        "filters": filters})  # fmt: skip
            for reason, meds in meds_by_reason.items():
                if len(meds) == 1:
                    med = next(iter(meds))
                    candidates["patient_lookup_medication"].append({
                        "question": f"Which medication was prescribed for the patient's {clean(reason).lower()}?",
                        "reference": f"{med}.",
                        "answer_contains": [first_word(med)],
                        "support": [{"source": source, "section": "Medications", "contains": [med, reason]}],
                        "filters": filters})  # fmt: skip
            last_by_class: dict[str, str] = {}
            for row in encounters[uuid]:
                cls, start = row["ENCOUNTERCLASS"], iso_date(row["START"])
                if cls in ENCOUNTER_WORDS and start and start > last_by_class.get(cls, ""):
                    last_by_class[cls] = start
            for cls, start in last_by_class.items():
                candidates["patient_lookup_encounter"].append({
                    "question": f"What was the date of the patient's most recent {ENCOUNTER_WORDS[cls]} encounter?",
                    "reference": f"{start}.",
                    "answer_contains": [date_variants(start)],
                    "support": [{"source": source, "section": "Encounters", "contains": [f"{start}, {cls}"]}],
                    "filters": filters})  # fmt: skip

    items: list[dict] = []
    for item_type, count in PER_TYPE.items():
        pool = candidates[item_type]
        chosen, used_patients = [], set()
        for item in rng.sample(pool, len(pool)):  # one item per patient keeps the set diverse
            if item["filters"]["patient_id"] not in used_patients:
                chosen.append(item)
                used_patients.add(item["filters"]["patient_id"])
            if len(chosen) == count:
                break
        items += [{"type": item_type, **item} for item in chosen]
    for question, (pid, _) in zip(UNANSWERABLE_PATIENT, rng.sample(pids, len(UNANSWERABLE_PATIENT)), strict=True):
        items.append({"type": "unanswerable", "question": question, "reference": "", "answer_contains": [],
                      "support": [], "filters": {"doc_type": "patient_record", "patient_id": pid}})  # fmt: skip
    return items


def main() -> int:
    """Write golden.jsonl and print counts per type."""
    settings = get_settings()
    rng = random.Random(settings.router.seed)
    manual = [json.loads(line) for line in MANUAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    generated = patient_items(settings, rng)
    for number, item in enumerate(generated, start=1):
        item["id"] = f"pt{number:02d}"
    items = [{**item, "status": "draft"} for item in manual + generated]
    GOLDEN.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8", newline="\n")
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item["type"]] += 1
    for item_type, n in sorted(counts.items()):
        print(f"{item_type}: {n}")
    print(f"wrote {len(items)} items to {GOLDEN}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
