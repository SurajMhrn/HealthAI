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
| Runtime | Local development on Windows 11, Python 3.11, CPU only. The venv, Synthea jar, spaCy model and data all stay inside the project folder | First planned for a cloud dev environment, which wasn't available. Nothing in the project needs a GPU or a specific OS, and every `make` target also runs as `python -m ...` |
| Small model | Hosted 7-8B model on an OpenAI-compatible endpoint: Qwen2.5 7B, falling back to Llama 3.1 8B | No local GPU. Ollama stays a config option (`SMALL_BASE_URL`) and appears in docker-compose |
| Large model + RAGAS judge | Free OpenAI-compatible endpoint (Groq / OpenRouter / GitHub Models), with an Azure OpenAI switch through env vars | The judge comes from a different model family than the large model, to avoid self-preference bias |
| LangChain | Only as a transitive dependency of RAGAS, in `requirements-eval.txt`. Never imported by project code | RAGAS 0.4.3 requires `langchain*`. Keeps the app and Docker image free of it |
| LLM client | One `openai` SDK client for both tiers (different `base_url` / model) | Simple. Azure is just a different client class |
| Embedder and re-ranker | BAAI bge-small-en-v1.5 + bge-reranker-base on CPU | No GPU is needed at this corpus size |
| Router | Rules, rules + small-model classifier, and a **trained classifier** (logistic regression on embeddings), compared head to head | Adds a real ML training component with a proper train/test split |
| PDF parsing | `pypdf` (BSD license) | Small and permissive. No OCR, so scanned PDFs won't parse |
| Synthea | Two populations, both CSV: (1) a run of the pinned v4.0.0 jar (sha256-checked), `-p 100`, seed 42, fixed reference date; (2) Synthea's pre-generated sample CSVs (111 patients, sha256-pinned archive). `patient_datasets` in config picks either or both | Generated data is reproducible and tunable; the sample set needs no Java and adds a second, independently generated population. Needs Java 17+ for (1) |
| Guideline PDFs | `scripts/get_guidelines.py` downloads five VA/DoD clinical practice guidelines (hypertension 2026, type 2 diabetes 2023, lipids 2025, COPD 2026, asthma 2025), sha256-pinned in `config.yaml` | U.S. federal works (public domain), text-based PDFs. CDC MMWR refused scripted downloads (403) and WHO IRIS links now serve a JavaScript app |
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

## Changes made during Phase 1
- **Allow-list for clinical vocabulary.** spaCy tagged condition names (e.g. "Hyperlipidemia") as `PERSON`, and Presidio's phone recognizer matched SNOMED/RxNorm codes. Redaction now skips exact strings taken from Synthea's coded condition, medication and encounter rows. These come from code systems, never from identifiers, and Synthea names always carry digits, so no name can be allow-listed. The leak scan confirms it.
- **Passport and licence formats.** Some Synthea passports have 6-7 digits (`X743270X`); the first pattern assumed 8 and the leak scan found 8 leaks. Both patterns now accept 6-9 digits.
- **Downloads fall back to `curl`.** On this network, Python's TLS connections to GitHub's release-asset host were reset while `curl` (Windows TLS) worked, so the download helper retries and then falls back to `curl`.

## Changes made during Phase 2
- **Model runtime.** bge-small-en-v1.5 and bge-reranker-base (both MIT) run through `sentence-transformers` on CPU-only PyTorch, pinned to exact Hugging Face revisions in `config.yaml`. The FAISS index is exact (`IndexFlatIP` on unit vectors = cosine); at about 6.5k chunks an approximate index isn't needed.
- **Filters inside each retriever.** BM25 scores are masked to the allowed chunks, and FAISS searches with an `IDSelectorBatch`, so fusion and re-ranking never see a filtered-out chunk. Date filters match when a chunk's own date range overlaps the requested one.
- **Measured cost of re-ranking.** Embedding all 6,485 chunks took 422.5 s on this laptop. In `reports/retrieval_comparison.md`, hybrid retrieval has a median latency of 24 ms per query and hybrid + re-rank 3,118 ms (the cross-encoder scores 20 pairs of up to 512 tokens on CPU). If that is too slow for the UI, the re-rank pool can be cut in config.
- **The 13-query smoke set doesn't separate the modes** (all four find a relevant chunk for 13/13). Choosing between them waits for the Phase 5 golden set.

## Changes made during Phase 3
- **Models.** Groq no longer serves Llama 3.1 8B or Qwen2.5 7B, and GitHub Models is unreachable from the development network (requests to `models.github.ai` get a bare `200 OK` from something in between, even without a key). All three roles therefore run on Groq: small `openai/gpt-oss-20b`, large `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`. The judge is still a different family from the large model. Any OpenAI-compatible endpoint can be swapped in through `.env` and `config.yaml`.
- **Reference prices** come from Groq's model docs (2026-11-07): gpt-oss-20b $0.075/$0.30, gpt-oss-120b $0.15/$0.60, qwen3.8-27b $0.80/$4.00 per million input/output tokens.
- **The query is redacted before anything else happens**, so the retriever, the LLM and the logs only ever see redacted text.
- **Citation formats.** gpt-oss sometimes cites with full-width brackets (`【id】`). Citations are normalised to `[id]` before they are checked against the retrieved chunks.
- **Router results** (`reports/router_comparison.md`, n = 98): rules 0.837 accuracy, rules + LLM 0.867, trained 0.643. The trained router reaches 0.984 on its own training data but 0.569 on held-out templates, so it memorises template wording; more varied training questions are needed before it is useful. Rules and rules + LLM have overlapping CIs, so `rules` (free, sub-millisecond) stays the default.

## Changes made during Phase 4
- **NeMo Guardrails without LangChain: verified.** 0.24.1 ships its own OpenAI-compatible client; the `openai` engine with a `base_url` reaches Groq directly, and `langchain` is not installed. The rails run through `LLMRails.check()` (input rails before retrieval, output rails after generation) around our own pipeline.
- **Rails.** Input: prompt-injection patterns, identity-reveal patterns, then an LLM self-check on the small model (individual advice, off-topic, injection, identity). Output: Presidio on the answer; any identifier blocks it. Retrieved chunks that match injection patterns are dropped before generation, and every chunk carries `injection_flags` from ingestion.
- **Results** (`reports/guardrails_report.md`, 30 harmful + 15 harmless prompts, 3 runs): block rate 0.867 / 0.900 / 0.867 (mean 0.878), false-positive rate 0.000 in every run, output rail 5/5 leaks blocked and 5/5 clean answers allowed, poisoned fixture 3/3 flagged with 0 false flags, 0 of 6,485 real chunks flagged.
- **Known weakness: off-topic.** Only 2-3 of 6 off-topic prompts are blocked. An earlier self-check prompt blocked a normal record lookup ("What medications is the patient taking for asthma?"); after it was rewritten to list allowed lookups, it became lenient on off-topic requests. It was not tuned further against the evaluation prompts; a separate development set is needed first.
- **Run-to-run variance.** The self-check isn't deterministic even at temperature 0 (one prompt flipped between runs), so the evaluation repeats the input rails three times and reports every run.

## Risks and open items
- **Free-tier limits:** a full RAGAS pass makes roughly 9 judge calls per question, so 5 configs × about 100 questions comes to several thousand calls. Plan: count the calls first, then spread runs over days, use a smaller subset, or use a cheap paid judge. LLM responses are cached on disk, keyed by the full prompt, so re-runs don't spend quota again.
- **NeMo Guardrails 0.24.1** no longer lists LangChain as a dependency. Its OpenAI-compatible model connector needs to be verified to work without it.
- **Docker:** development is local on Windows; the Dockerfile and compose file need Docker Desktop (or a separate Linux host) to be tested.
- **Dates:** only dates of birth are redacted. Encounter dates stay, because the date filter needs them. HIPAA Safe Harbor would also require generalizing those dates, and the README states this.
- **RAGAS API churn:** the API changed between 0.2, 0.3 and 0.4. Pin the version and check the current API before writing the runner.
