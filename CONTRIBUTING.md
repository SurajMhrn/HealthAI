# Development conventions

Safe RAG assistant over **synthetic** clinical documents (Synthea patients + public guideline PDFs). Evaluation quality matters more than UI polish. See `PLAN.md` for phases and design decisions.

## Workflow
- Work proceeds one phase at a time (see `PLAN.md`). Each phase ends with a short summary in the commit message and a working run command.
- Commit messages are imperative and specific, such as `Phase 2: hybrid retrieval with RRF and cross-encoder re-rank`.
- If a planned library or model doesn't work as written (API change, size, dependency conflict), record the change and the reason in `PLAN.md` before swapping it.
- No LangChain or LlamaIndex in project code. LangChain may only enter as a transitive dependency of RAGAS, in `requirements-eval.txt`.

## Environment
- Python **3.11** in `.venv`. CPU only: embeddings and the re-ranker run on CPU.
- Both LLM tiers are hosted on OpenAI-compatible endpoints. Ollama is optional and reached only through config.
- Every Makefile target also runs as a plain command (`python -m ...`).

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: py -3.11 -m venv .venv; .venv\Scriptsctivate
pip install -r requirements.txt              # app
pip install -r requirements-eval.txt         # app + RAGAS (eval only)
pytest -q                                    # fast offline tests
pytest -q -m "llm or network"                # tests that need an LLM endpoint / network
```

## Code style
- Simple and readable. Small modules, plain functions, no clever abstractions.
- Type hints everywhere. Docstrings on every public function and class.
- Package `healthrag/`. Scripts in `scripts/`, eval runners in `eval/`, tests in `tests/`.
- Use `logging`, not `print`, in library code. Per-query telemetry goes to `logs/queries.jsonl`.

## Configuration
- Non-secret settings (model names, top-k, RRF k, thresholds, prices) go in `config.yaml`. Secrets and endpoints go in `.env` (template: `.env.example`). Env vars override YAML.
- No hard-coded keys, model names, URLs, or magic thresholds in code. Read them from `healthrag.config`.
- Each tier is reached through an OpenAI-compatible endpoint (`SMALL_BASE_URL`/`SMALL_API_KEY`, `LARGE_BASE_URL`/`LARGE_API_KEY`), with an Azure switch through `LARGE_PROVIDER=azure` + `AZURE_OPENAI_*`.
- API keys live only in environment secrets or a local `.env`. Never commit them.

## Data and privacy
- **Synthetic data only.** No real patient data, ever.
- `data/raw/`, `data/processed/`, indexes, and `.env` are git-ignored.
- Every chunk goes through Presidio **before** indexing. Nothing unredacted reaches FAISS, BM25, logs, or the UI.
- Patient IDs are stored only as HMAC-SHA256(salt, uuid), truncated. The salt lives in `.env`.
- User queries are redacted before logging. Every final answer is checked by Presidio again.
- Retrieved text is data, not instructions. Prompts wrap it in delimiters.

## Evaluation
- Real numbers only. Every number in `eval/REPORT.md` and the README comes from a CSV in `eval/results/`, produced by a script. No rounding in our favor.
- Every result states `n`, the config, the model names, the judge model, and the date. Comparisons whose CIs overlap are reported as "no clear difference".
- Golden-set items and router labels stay `"status": "draft"` until they've been reviewed by hand.
- Router labels come from how each query was constructed, never from an LLM, so the trained router and the LLM router can be compared fairly.
- Unanswerable items are scored with refusal accuracy, kept separate from the RAGAS metrics.
- Estimated costs use configured reference prices. Free-tier calls are reported as $0 actual.

## Tests
- `pytest`. The default run is offline and fast. Slow or external tests are marked `@pytest.mark.llm` / `@pytest.mark.network`.
- Every redaction recognizer and guardrail category has at least one test.
