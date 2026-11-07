"""Evaluation helpers: matching, statistics, golden-set integrity and the regression rule. Offline."""

from __future__ import annotations

import json
import math

import pytest

from eval.regress import judge_noise
from healthrag.config import PROJECT_ROOT
from healthrag.evaluation.matching import answer_matches, normalize, supports
from healthrag.evaluation.stats import bootstrap_ci, mean, paired_difference, verdict
from scripts.make_golden import clean, date_variants, first_word

GOLDEN = PROJECT_ROOT / "eval" / "golden.jsonl"
MANUAL = PROJECT_ROOT / "eval" / "golden_manual.jsonl"


def test_normalize_handles_pdf_and_model_artifacts() -> None:
    assert normalize("self­management") == "selfmanagement"
    assert normalize("GLP‑1 receptor–agonist") == "glp-1 receptor-agonist"
    assert normalize("long-\nacting  beta") == "long-acting beta"


def test_answer_match_with_alternatives() -> None:
    required = [date_variants("2022-12-10"), "hydrochlorothiazide"]
    assert answer_matches("Started on December 10, 2022: Hydrochlorothiazide 25 MG.", required)
    assert answer_matches("Since 2022-12-10 the patient takes hydrochlorothiazide.", required)
    assert not answer_matches("Started in 2022: hydrochlorothiazide.", required)


def test_supports_checks_source_section_and_text() -> None:
    record = {"source": "synthea-generated:abc", "section": "Medications", "text": "[x]\n2022-12-10: Lisinopril 10 MG"}
    assert supports(record, {"source": "synthea-generated:abc", "section": "Medications", "contains": ["lisinopril"]})
    assert not supports(record, {"source": "synthea-generated:abc", "section": "Conditions", "contains": "lisinopril"})
    assert not supports(record, {"source": "va-dod", "contains": "lisinopril"})


def test_bootstrap_and_paired_difference() -> None:
    low, high = bootstrap_ci([1.0] * 10 + [0.0] * 10, 500, 1)
    assert low < 0.5 < high
    diff, low, high, n = paired_difference([1, 1, 1, 1], [0, 0, 0, 0], 500, 1)
    assert diff == 1.0 and low == high == 1.0 and n == 4
    diff, *_ = paired_difference([1.0, math.nan], [0.0, 0.0], 100, 1)
    assert diff == 1.0  # NaN pairs are dropped
    assert math.isnan(mean([math.nan]))


def test_verdict_wording() -> None:
    assert verdict(-0.1, 0.2, "a", "b") == "no clear difference"
    assert verdict(0.05, 0.2, "a", "b") == "a better"
    assert verdict(0.05, 0.2, "a", "b", higher_is_better=False) == "b better"


def test_judge_noise() -> None:
    assert judge_noise({"x": 0.8, "y": 1.0}, {"x": 0.6, "y": 1.0, "z": 0.1}) == pytest.approx(0.1)


def test_golden_helpers() -> None:
    assert clean("Essential hypertension (disorder)") == "Essential hypertension"
    assert first_word("24 HR Metformin hydrochloride 500 MG") == "metformin"
    assert "December 10, 2022" in date_variants("2022-12-10")


def test_golden_set_is_well_formed() -> None:
    items = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids))
    assert all(i["status"] == "draft" for i in items)
    for item in items:
        if item["type"] == "unanswerable":
            assert not item["support"] and not item["answer_contains"]
        else:
            assert item["support"] and item["answer_contains"] and item["reference"]
    manual_ids = {json.loads(line)["id"] for line in MANUAL.read_text(encoding="utf-8").splitlines()}
    assert manual_ids <= set(ids)
