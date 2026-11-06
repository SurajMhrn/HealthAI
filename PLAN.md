# Plan: Safe Clinical RAG Assistant

**Problem.** Clinicians spend a lot of time searching patient records and long guidelines. Generic chatbots don't fit healthcare: they invent answers, leak patient identifiers, give personal medical advice, and are rarely measured.

**This project** is a question-answering assistant over **synthetic** patient records (Synthea) and public clinical guidelines. It:
- answers only from sources, citing chunk IDs
- never shows patient identifiers
- refuses diagnosis or treatment advice for a specific person
- routes questions between a cheap model and a strong model
- backs every claim with real evaluation numbers

Evaluation quality is the priority; UI polish is secondary.

## Design decisions

| Topic | Decision | Rationale |
|---|---|---|
| Runtime | Cloud dev environment (Linux, CPU only). Nothing runs on the laptop | Keeps heavy dependencies, models and data off the local machine |
| Small model | Hosted 7-8B model on an OpenAI-compatible endpoint: Qwen2.5 7B, falling back to Llama 3.1 8B | No local GPU. Ollama stays a config option (`SMALL_BASE_URL`) and appears in docker-compose |
| Large model + RAGAS judge | Free OpenAI-compatible endpoint (Groq / OpenRouter / GitHub Models), with an Azure OpenAI switch through env vars | The judge comes from a different model family than the large model, to avoid self-preference bias |
| LangChain | Only as a transitive dependency of RAGAS, in `requirements-eval.txt`. Never imported by project code | RAGAS 0.4.3 requires `langchain*`. Keeps the app and Docker image free of it |
| LLM client | One `openai` SDK client for both tiers (different `base_url` / model) | Simple. Azure is just a different client class |
| Embedder and re-ranker | BAAI bge-small-en-v1.5 + bge-reranker-base on CPU | No GPU is needed at this corpus size |
| Router | Rules, rules + small-model classifier, and a **trained classifier** (logistic regression on embeddings), compared head to head | Adds a real ML training component with a proper train/test split |
| PDF parsing | `pypdf` (BSD license) | Small and permissive. No OCR, so scanned PDFs won't parse |
| Synthea | Pinned Synthea release jar, 100 patients, fixed seed, CSV export | Reproducible. Needs Java 17+ |
| Guideline PDFs | `scripts/get_guidelines.py` downloads a reviewed list of public guideline PDFs (URLs in `config.yaml`) | Reproducible in any environment |
| `make eval` | `Makefile`, and every target also runs as `python -m ...` | Works with or without `make` |

## Layout (target)
```
healthrag/            # the package
  config.py           # loads config.yaml + .env (pydantic)
  ingest/             # synthea → text docs, PDF parsing, chunking
  redact/             # Presidio engine, custom recognizers, report
  retrieve/           # bm25, dense (FAISS), RRF fusion, filters, rerank
  generate/           # llm client, prompts, router (rules, LLM, trained), pipeline, telemetry
  guard/              # NeMo Guardrails config (config.yml, *.co) + wrapper
  api.py              # FastAPI
ui/app.py             # Streamlit
scripts/              # get_synthea, get_guidelines, build_index, compare_retrieval, train_router
eval/                 # golden.jsonl, attacks.jsonl, router_labels.jsonl, runners, results/, REPORT.md
tests/
config.yaml  .env.example  requirements.txt  requirements-eval.txt  Makefile
```

## Phases

**Phase 1: Ingestion and redaction**
- Data scripts:
  - `scripts/get_synthea.py` generates about 100 patients.
  - `scripts/get_guidelines.py` downloads the reviewed PDFs.
- Patient CSVs → one readable document per patient (demographics, conditions, medications, encounters). Guideline PDFs → text by page, with simple heading detection.
- Chunk with metadata: `chunk_id` (deterministic hash), `source`, `doc_type`, `patient_id` (HMAC-SHA256 with a salt from `.env`, truncated), `date`, `section`.
- Presidio runs on every chunk before indexing.
  - Typed placeholders: `<PERSON>`, `<DATE_OF_BIRTH>`, `<ADDRESS>`, `<PHONE_NUMBER>`, `<RECORD_ID>`.
  - Synthea names carry numeric suffixes (e.g. `Abe604`), which spaCy NER tends to miss. Custom recognizers cover those, plus Synthea UUIDs and record numbers, plus a deny-list built from `patients.csv`.
- Redaction report:
  - counts per entity type
  - a few before/after samples
  - a leak scan that checks every known identifier from `patients.csv` against the redacted chunks and reports any hits
- Tests: a chunk containing known fake PHI comes out clean, plus a test for each recognizer.

**Phase 2: Retrieval**
- BM25 (`rank_bm25`) and FAISS (bge-small, cosine) fused with RRF (`k` set in config).
- Filters on `doc_type`, date range, and `patient_id`, applied before fusion.
- Cross-encoder (bge-reranker-base) re-ranks the top 20 down to the top 5.
- `scripts/compare_retrieval.py` runs the same queries through dense, BM25, hybrid, and hybrid + re-rank, side by side.

**Phase 3: Generation and tiering**
- Grounded prompt: answer only from the context, cite `[chunk_id]`, and say "not in the documents" otherwise. Retrieved text is wrapped in delimiters and treated as data.
- Router, three variants selectable in config, each logging its `route_reason`:
  1. **Rules.** Multi-patient, summarize/compare/trend, or multiple entities → large.
  2. **Rules + LLM classifier.** Rules first, then a JSON classifier on the small model.
  3. **Trained classifier.**
     - Data: about 300 queries labeled `simple` / `complex` in `eval/router_labels.jsonl`, marked `draft` until reviewed. Labels come from how each query was built (single-record lookup vs multi-document/summary), not from an LLM, so the comparison isn't circular.
     - Model: logistic regression on bge-small embeddings (scikit-learn), trained by `scripts/train_router.py`.
     - Split: the test split includes hand-written paraphrases, so the model can't score well just by memorizing templates.
  - The three routers are compared on accuracy, precision/recall/F1 for `complex`, latency, and cost per query.
- Telemetry to `logs/queries.jsonl`: route, latency per stage, tokens in/out, estimated cost, and whether a guardrail fired. The query is redacted before logging.
- Cost: free-tier calls cost $0, so estimated cost uses configurable reference prices per model. Reports state that.
- Presidio runs again on the final answer, and any hit is counted and logged.

**Phase 4: Guardrails**
- NeMo Guardrails input rails for individual diagnosis/treatment advice, off-topic questions, prompt injection, and identity reveal. Output rails for identity leaks.
- Injection hidden in documents: flag suspicious chunks at ingestion and check the retrieved context before generation. Tests use a deliberately poisoned fixture document.
- `eval/attacks.jsonl`: 40-50 prompts with expected `block` or `allow`, roughly a third of them harmless. The report gives block rate, false-positive rate, and a per-category breakdown.

**Phase 5: Evaluation**
- `eval/golden.jsonl`: about 100 items, all `"status": "draft"` until reviewed by hand, mixing easy lookups, multi-hop questions, and unanswerable questions. Lookup answers come from the structured CSVs, so they're checkable. Sources are stored as doc + section + supporting text, so they survive re-chunking.
- RAGAS (faithfulness, answer relevancy, context precision, context recall) with a configurable judge.
- Beyond RAGAS:
  - retrieval hit@5 and MRR against golden sources (no judge needed)
  - refusal accuracy on unanswerable items, scored separately
  - paired bootstrap 95% CIs on the differences between configs
- Configs: dense vs hybrid vs hybrid + re-rank (with the tiered generator), and small vs large vs tiered (with hybrid + re-rank). That's 5 runs.
- Output: CSVs in `eval/results/`, and a generated `eval/REPORT.md` with tables and plain-language conclusions. Numbers come only from the CSVs.
- `make eval` / `python -m eval.regress`: fails if faithfulness drops more than a threshold below `eval/results/baseline.json`. The threshold is set above the measured judge noise (two runs of the same config), and the check runs on a fixed regression subset to save quota.

**Phase 6: UI and packaging**
- Streamlit chat with clickable citations that expand to the redacted source chunk, and a side panel for route, latency, cost, and guardrail status.
- Dockerfile + docker-compose (app + optional Ollama).
- README with the problem, a Mermaid architecture diagram, run steps, the eval summary (real numbers from REPORT.md), and known limitations.

## Risks and open items
- **Free-tier limits:** a full RAGAS pass makes roughly 9 judge calls per question, so 5 configs × about 100 questions comes to several thousand calls. Plan: count the calls first, then spread runs over days, use a smaller subset, or use a cheap paid judge. LLM responses are cached on disk, keyed by the full prompt, so re-runs don't spend quota again.
- **NeMo Guardrails 0.24.1** no longer lists LangChain as a dependency. Its OpenAI-compatible model connector needs to be verified to work without it.
- **Docker in the cloud:** the dev environment may not run Docker. If not, the Dockerfile and compose file need a separate test.
- **Dates:** only dates of birth are redacted. Encounter dates stay, because the date filter needs them. HIPAA Safe Harbor would also require generalizing those dates, and the README states this.
- **RAGAS API churn:** the API changed between 0.2, 0.3 and 0.4. Pin the version and check the current API before writing the runner.
