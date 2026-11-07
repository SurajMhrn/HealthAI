"""Build eval/router_labels.jsonl: questions labelled simple/complex by how they were constructed.

Run with ``python -m scripts.make_router_labels`` (or ``make router-labels``).

* ``simple``: single-record or single-passage lookups (one medication, one date, one recommendation).
* ``complex``: summaries, comparisons, trends, several patients, or several conditions together.

The label comes from the template a question was built from, never from an LLM, so the
LLM router and the trained router can be compared fairly. Two templates per label are held
out entirely for the test split, and the hand-written paraphrases in
``eval/router_paraphrases.jsonl`` are test-only too, so a model can't score well by
memorising template wording. Every item stays ``"status": "draft"`` until reviewed by hand.
"""

from __future__ import annotations

import json
import random
import sys

from healthrag.config import PROJECT_ROOT, get_settings

PARAPHRASES = PROJECT_ROOT / "eval" / "router_paraphrases.jsonl"

CONDITIONS = [
    "hypertension", "prediabetes", "type 2 diabetes", "hyperlipidemia", "asthma", "COPD",
    "chronic kidney disease", "obesity", "anemia", "osteoarthritis", "acute bronchitis", "viral sinusitis",
]  # fmt: skip
MEDICATIONS = [
    "metformin", "lisinopril", "hydrochlorothiazide", "amlodipine", "atorvastatin", "simvastatin",
    "albuterol", "fluticasone", "insulin", "naproxen",
]  # fmt: skip
ENCOUNTERS = ["wellness", "emergency", "urgent care", "ambulatory", "inpatient"]
GUIDELINES = ["hypertension", "type 2 diabetes", "lipid", "COPD", "asthma"]
TOPICS = [
    "first-line drug therapy", "the treatment target", "screening", "home monitoring", "lifestyle changes",
    "when to refer to a specialist", "follow-up intervals", "inhaler technique", "statin intensity",
]  # fmt: skip
YEARS = ["2015", "2018", "2020", "2022"]

SIMPLE_TEMPLATES = {
    "s_med_for": "What medication is the patient taking for {cond}?",
    "s_first_recorded": "When was {cond} first recorded for this patient?",
    "s_active": "Is the patient's {cond} still active?",
    "s_last_visit": "What was the date of the patient's last {enc} visit?",
    "s_dose": "What dose of {med} was prescribed?",
    "s_guideline_rec": "What does the {gl} guideline recommend for {topic}?",
    "s_has_record": "Does the patient have a record of {cond}?",
    "s_med_start": "When did the patient start {med}?",
    # held out for test
    "s_reason_med": "Why was {med} prescribed?",
    "s_guideline_def": "How does the {gl} guideline define {topic}?",
}
COMPLEX_TEMPLATES = {
    "c_summarize": "Summarize the patient's history of {cond}.",
    "c_compare_conditions": "Compare how {cond} and {cond2} were treated for this patient.",
    "c_trend": "How has the patient's treatment for {cond} changed over time?",
    "c_multi_patient": "Which patients have both {cond} and {cond2}?",
    "c_overview_encounters": "Give an overview of all encounters related to {cond}.",
    "c_compare_guidelines": "Compare the {gl} and {gl2} guidelines on {topic}.",
    "c_adherence": "Does the patient's {cond} treatment follow the {gl} guideline recommendations?",
    "c_since_year": "What has happened in the patient's care since {year}?",
    # held out for test
    "c_count_patients": "How many patients were prescribed {med} for {cond}?",
    "c_timeline": "Build a timeline of the patient's {cond} and {cond2} diagnoses and medications.",
}
HELD_OUT = {"s_reason_med", "s_guideline_def", "c_count_patients", "c_timeline"}
PER_TEMPLATE = 16


def fill(template: str, rng: random.Random) -> str:
    """Fill a template's slots with random (seeded) values; two-slot fields get distinct values."""
    cond, cond2 = rng.sample(CONDITIONS, 2)
    gl, gl2 = rng.sample(GUIDELINES, 2)
    return template.format(
        cond=cond, cond2=cond2, med=rng.choice(MEDICATIONS), enc=rng.choice(ENCOUNTERS),
        gl=gl, gl2=gl2, topic=rng.choice(TOPICS), year=rng.choice(YEARS),
    )  # fmt: skip


def build(seed: int) -> list[dict]:
    """All templated items plus the paraphrases, de-duplicated, in a stable order."""
    rng = random.Random(seed)
    items: list[dict] = []
    seen: set[str] = set()
    for label, templates in (("simple", SIMPLE_TEMPLATES), ("complex", COMPLEX_TEMPLATES)):
        for template_id, template in templates.items():
            made = 0
            for _ in range(PER_TEMPLATE * 10):
                if made == PER_TEMPLATE:
                    break
                question = fill(template, rng)
                if question in seen:
                    continue
                seen.add(question)
                made += 1
                items.append(
                    {
                        "query": question,
                        "label": label,
                        "source": f"template:{template_id}",
                        "split": "test" if template_id in HELD_OUT else "train",
                        "status": "draft",
                    }
                )
    for line in PARAPHRASES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            items.append({**item, "source": "paraphrase", "split": "test", "status": "draft"})
    return items


def main() -> int:
    """Write the labels file and print the split sizes."""
    settings = get_settings()
    items = build(settings.router.seed)
    path = settings.router.labels_file
    path.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items), encoding="utf-8", newline="\n")
    for split in ("train", "test"):
        part = [i for i in items if i["split"] == split]
        complex_n = sum(1 for i in part if i["label"] == "complex")
        print(f"{split}: {len(part)} items ({complex_n} complex, {len(part) - complex_n} simple)")
    print(f"wrote {len(items)} items to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
