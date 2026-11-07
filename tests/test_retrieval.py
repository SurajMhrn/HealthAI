"""Retrieval tests with a tiny fake embedder and re-ranker, so no models are downloaded."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from healthrag.config import load_settings
from healthrag.retrieve.bm25 import Bm25Index, tokenize
from healthrag.retrieve.dense import DenseIndex
from healthrag.retrieve.filters import Filters, allowed_indices, matches
from healthrag.retrieve.fusion import reciprocal_rank_fusion
from healthrag.retrieve.retriever import MODES, Retriever

RECORDS = [
    {"chunk_id": "g1", "text": "Statin therapy lowers LDL cholesterol in adults", "doc_type": "guideline",
     "patient_id": None, "section": "Lipids", "date": "2025-12", "date_start": "2025-12", "source": "lipids.pdf"},
    {"chunk_id": "g2", "text": "Metformin is first-line therapy for type 2 diabetes", "doc_type": "guideline",
     "patient_id": None, "section": "Diabetes", "date": "2023-05", "date_start": "2023-05", "source": "dm.pdf"},
    {"chunk_id": "p1", "text": "2019-04-02: Hydrochlorothiazide 25 MG for hypertension", "doc_type": "patient_record",
     "patient_id": "aaaa", "section": "Medications", "date": "2019-04-02", "date_start": "2015-01-10", "source": "s:aaaa"},
    {"chunk_id": "p2", "text": "2021-06-01, emergency: Emergency room admission", "doc_type": "patient_record",
     "patient_id": "aaaa", "section": "Encounters", "date": "2021-06-01", "date_start": "2021-06-01", "source": "s:aaaa"},
    {"chunk_id": "p3", "text": "2020-02-02: Metformin 500 MG for prediabetes", "doc_type": "patient_record",
     "patient_id": "bbbb", "section": "Medications", "date": "2020-02-02", "date_start": "2020-02-02", "source": "s:bbbb"},
    {"chunk_id": "p4", "text": "Name: <PERSON>", "doc_type": "patient_record",
     "patient_id": "bbbb", "section": "Demographics", "date": None, "date_start": None, "source": "s:bbbb"},
]  # fmt: skip


class HashEmbedder:
    """Bag-of-words vectors from hashed tokens: similar wording gives similar vectors."""

    dim = 64

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                out[row, int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class OverlapReranker:
    """Scores by shared tokens, with a bonus for the word 'metformin' to make its effect visible."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = set(tokenize(query))
        return [len(q & set(tokenize(p))) + (5.0 if "metformin" in p.lower() else 0.0) for p in passages]


@pytest.fixture()
def retriever() -> Retriever:
    embedder = HashEmbedder()
    dense = DenseIndex.build(embedder.encode([r["text"] for r in RECORDS]))
    return Retriever(RECORDS, dense, embedder, OverlapReranker(), load_settings().retrieval)


def test_rrf_matches_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60))
    assert fused[1] == pytest.approx(1 / 61 + 1 / 62)
    assert fused[3] == pytest.approx(1 / 63 + 1 / 61)
    assert fused[2] == pytest.approx(1 / 62)
    assert [doc for doc, _ in reciprocal_rank_fusion([[1, 2, 3], [3, 1]], k=60)] == [1, 3, 2]


def test_rrf_rewards_agreement_between_retrievers() -> None:
    order = [doc for doc, _ in reciprocal_rank_fusion([[7, 8], [9, 8]], k=60)]
    assert order[0] == 8


def test_tokenizer_keeps_clinical_codes() -> None:
    assert tokenize("COVID-19 and E11.9 in the HbA1c") == ["covid-19", "e11.9", "hba1c"]


def test_bm25_respects_allowed_indices() -> None:
    index = Bm25Index([r["text"] for r in RECORDS])
    assert index.search("metformin", 5)[0][0] in (1, 4)
    only_patients = np.array([2, 3, 4, 5])
    assert [i for i, _ in index.search("metformin", 5, only_patients)] == [4]


def test_filters_by_type_patient_and_date_overlap() -> None:
    assert matches(RECORDS[2], Filters(patient_id="aaaa", date_from="2016", date_to="2016"))  # inside 2015..2019
    assert not matches(RECORDS[2], Filters(date_from="2020"))
    assert not matches(RECORDS[5], Filters(date_from="2000"))  # no date, so never matches a date filter
    assert allowed_indices(RECORDS, Filters()) is None
    assert allowed_indices(RECORDS, Filters(doc_type="guideline")).tolist() == [0, 1]


def test_dense_search_with_faiss_selector() -> None:
    embedder = HashEmbedder()
    index = DenseIndex.build(embedder.encode([r["text"] for r in RECORDS]))
    query = embedder.encode(["metformin"], is_query=True)[0]
    assert {i for i, _ in index.search(query, 2)} == {1, 4}
    assert [i for i, _ in index.search(query, 5, np.array([4, 5]))][0] == 4
    assert index.search(query, 5, np.array([], dtype=np.int64)) == []


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_applies_filters_before_ranking(retriever: Retriever, mode: str) -> None:
    result = retriever.search("metformin therapy", Filters(patient_id="bbbb"), mode)
    assert result.hits and all(h.record["patient_id"] == "bbbb" for h in result.hits)
    assert result.hits[0].chunk_id == "p3"
    assert result.timings_ms["total"] >= 0


def test_rerank_reorders_the_fused_pool(retriever: Retriever) -> None:
    hybrid = [h.chunk_id for h in retriever.search("statin therapy for adults", mode="hybrid").hits]
    reranked = [h.chunk_id for h in retriever.search("statin therapy for adults", mode="hybrid_rerank").hits]
    assert hybrid[0] == "g1"
    assert reranked[0] in ("g2", "p3")  # the fake re-ranker boosts metformin chunks
    assert len(reranked) == len(set(reranked))


def test_hits_report_ranks_per_retriever(retriever: Retriever) -> None:
    hit = retriever.search("emergency room admission", mode="hybrid").hits[0]
    assert hit.chunk_id == "p2"
    assert hit.ranks == {"dense": 1, "bm25": 1}


def test_mismatched_index_is_rejected() -> None:
    embedder = HashEmbedder()
    dense = DenseIndex.build(embedder.encode(["only one"]))
    with pytest.raises(ValueError):
        Retriever(RECORDS, dense, embedder, None, load_settings().retrieval)
