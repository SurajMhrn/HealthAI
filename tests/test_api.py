"""API tests with a fake answerer: no models, no network."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from healthrag import api
from healthrag.config import load_settings
from healthrag.redact.engine import Redactor
from tests.test_generation import make_answerer


@pytest.fixture()
def client(redactor: Redactor, monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:  # noqa: ANN001
    settings = load_settings()
    settings.generation.telemetry_file = tmp_path / "q.jsonl"
    fake, _ = make_answerer(settings, "Metformin 500 MG for prediabetes [p3].", redactor)
    monkeypatch.setattr(api, "answerer", lambda: fake)
    return TestClient(api.app)


def test_health(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok"}


def test_ask_returns_answer_sources_and_telemetry_fields(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is the patient taking?", "patient_id": "bbbb"})
    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Metformin")
    assert body["sources"] and all(s["source"] == "s:bbbb" for s in body["sources"])
    assert {"route", "route_reason", "model", "latency_ms", "estimated_cost_usd", "guardrail"} <= set(body)


def test_ask_validates_input(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x", "doc_type": "emails"}).status_code == 422


def test_chunk_lookup_and_patients(client: TestClient) -> None:
    assert client.get("/chunks/g1").json()["section"] == "Lipids"
    assert client.get("/chunks/nope").status_code == 404
    patients = client.get("/patients").json()
    assert {p["patient_id"] for p in patients} == {"aaaa", "bbbb"}
