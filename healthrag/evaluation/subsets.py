"""Fixed, stratified subsets of the golden set, written once to eval/ and then reused.

* RAGAS subset (20 answerable items): judged with RAGAS; also the regression subset.
* Comparison subset (40 items): the RAGAS subset plus 14 more answerable and 6 unanswerable
  items. The four comparison configs run on it, so every comparison is paired on the same
  questions. Groq's free tier allows about 200K tokens per model per day, which isn't enough
  to run all five configs on the full golden set in one day.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

RAGAS_PER_TYPE = {"guideline_lookup": 6, "guideline_multihop": 2, "patient_lookup_condition": 3,
                  "patient_lookup_medication": 3, "patient_lookup_encounter": 3, "patient_multihop": 3}  # fmt: skip
EXTRA_PER_TYPE = {"guideline_lookup": 4, "patient_lookup_condition": 3, "patient_lookup_medication": 3,
                  "patient_lookup_encounter": 3, "patient_multihop": 1, "unanswerable": 6}  # fmt: skip


def _by_type(golden: list[dict]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for item in golden:
        groups[item["type"]].append(item["id"])
    return groups


def _pick(groups: dict[str, list[str]], per_type: dict[str, int], exclude: set[str], rng: random.Random) -> list[str]:
    ids: list[str] = []
    for item_type, n in per_type.items():
        pool = [i for i in groups[item_type] if i not in exclude]
        ids += sorted(rng.sample(pool, min(n, len(pool))))
    return ids


def load_or_create(path: Path, golden: list[dict], seed: int) -> tuple[list[str], list[str]]:
    """Return ``(ragas_ids, comparison_ids)``, creating the subset file on first use."""
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["ragas"], data["comparison"]
    rng = random.Random(seed)
    groups = _by_type(golden)
    ragas = _pick(groups, RAGAS_PER_TYPE, set(), rng)
    comparison = ragas + _pick(groups, EXTRA_PER_TYPE, set(ragas), rng)
    path.write_text(
        json.dumps({"seed": seed, "ragas": ragas, "comparison": comparison,
                    "ragas_per_type": RAGAS_PER_TYPE, "extra_per_type": EXTRA_PER_TYPE}, indent=2) + "\n",
        encoding="utf-8",
    )  # fmt: skip
    return ragas, comparison
