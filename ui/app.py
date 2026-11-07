"""Streamlit chat UI. Talks to the FastAPI service.

    streamlit run ui/app.py      (or: make ui; the API must be running: make api)

Set ``API_URL`` if the API isn't on http://localhost:8000.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 120.0

st.set_page_config(page_title="Clinical RAG assistant", layout="wide")


@st.cache_data(ttl=600)
def patient_ids() -> list[str]:
    """Pseudonymous patient IDs from the API."""
    response = httpx.get(f"{API_URL}/patients", timeout=TIMEOUT)
    response.raise_for_status()
    return [p["patient_id"] for p in response.json()]


def ask(question: str, filters: dict) -> dict:
    """Send a question to the API."""
    response = httpx.post(f"{API_URL}/ask", json={"question": question, **filters}, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json()


def show_answer(result: dict) -> None:
    """Answer text plus expandable sources (cited ones first)."""
    st.markdown(result["answer"])
    sources = sorted(result["sources"], key=lambda s: not s["cited"])
    for source in sources:
        label = f"{'Cited' if source['cited'] else 'Retrieved'} [{source['chunk_id']}] {source['source']} · {source['section']}"
        if source.get("page"):
            label += f" · p.{source['page']}"
        with st.expander(label):
            st.text(source["text"])
    if result["invalid_citations"]:
        st.warning(f"Citations that don't match a retrieved chunk: {', '.join(result['invalid_citations'])}")


def show_panel(result: dict) -> None:
    """Route, latency, cost and guardrail status for the last answer."""
    st.subheader("Last answer")
    st.markdown(f"**Route:** {result['route']}  \n`{result['model']}`")
    st.caption(f"{result['router']}: {result['route_reason']}")
    st.markdown(f"**Latency:** {result['latency_ms'].get('total', 0):.0f} ms")
    st.markdown(f"**Estimated cost:** ${result['estimated_cost_usd']:.6f}")
    st.caption("Reference prices; free-tier calls cost $0.")
    st.markdown(f"**Guardrail:** {result['guardrail'] or 'none'}")
    st.markdown(f"**PHI hits in answer:** {result['answer_phi_hits']}")
    with st.expander("Latency by stage"):
        st.json(result["latency_ms"])


st.title("Clinical RAG assistant")
st.caption("Answers come only from synthetic patient records and public VA/DoD guidelines. Identifiers are redacted.")

with st.sidebar:
    st.header("Filters")
    doc_type = st.selectbox("Documents", ["all", "patient_record", "guideline"])
    patient = None
    if doc_type != "guideline":
        try:
            patient = st.selectbox("Patient (pseudonymous ID)", ["any", *patient_ids()])
        except httpx.HTTPError:
            st.error(f"Can't reach the API at {API_URL}")
    date_from = st.text_input("From date (YYYY or YYYY-MM-DD)", "")
    date_to = st.text_input("To date", "")

filters = {
    "doc_type": None if doc_type == "all" else doc_type,
    "patient_id": None if patient in (None, "any") else patient,
    "date_from": date_from or None,
    "date_to": date_to or None,
}

if "history" not in st.session_state:
    st.session_state.history = []

chat, panel = st.columns([3, 1])
with chat:
    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            show_answer(turn["result"])
    question = st.chat_input("Ask about a patient record or a guideline")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Retrieving and answering..."):
            try:
                result = ask(question, filters)
            except httpx.HTTPStatusError as error:
                detail = error.response.json().get("detail") if "json" in error.response.headers.get("content-type", "") else None
                st.error(detail or f"Request failed: {error}")
            except httpx.HTTPError as error:
                st.error(f"Request failed: {error}")
            else:
                show_answer(result)
                st.session_state.history.append({"question": question, "result": result})
with panel:
    if st.session_state.history:
        show_panel(st.session_state.history[-1]["result"])
