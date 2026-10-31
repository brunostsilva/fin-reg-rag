# FinRegRAG

A Retrieval-Augmented Generation (RAG) service for answering questions over financial and
payments regulation documents, built with Python, FastAPI, and a pluggable
embeddings/vector-store/LLM architecture.

The project is designed to be **runnable end-to-end with zero external dependencies and no
API keys** (it ships with offline, deterministic defaults for embeddings and generation), while
also exposing a clear path to a **production-style deployment** with OpenAI models and a
Postgres + pgvector vector store.

> **Note on the sample data:** the documents in `data/sample_docs/` are **synthetic** texts
> written for this project, loosely inspired by real-world regulatory topics (Brazil's instant
> payments scheme, Open Finance, AML/KYC). They do **not** represent official regulatory text
> from any central bank or regulator. A production deployment would ingest real source
> documents instead.

## Why this project exists

This is a demonstration project built to show how to combine solid backend engineering
practices (clean architecture, dependency injection, automated tests, CI, containerization)
with a Retrieval-Augmented Generation pipeline, instead of a bare "chat with your PDF" script.
The emphasis is on the **engineering around the LLM** — pluggable providers, testability
without hitting paid APIs, and a realistic path from a local prototype to a containerized,
Postgres-backed deployment.

## Features

- **FastAPI service** with `/ingest`, `/query`, and `/health` endpoints (interactive docs at
  `/docs`).
- **Pluggable embeddings providers**: a dependency-free, deterministic local hashing embedder
  (default, no download or API key needed) or OpenAI embeddings.
- **Pluggable vector stores**: FAISS (local, file-backed, zero setup) or Postgres + `pgvector`
  (via `docker-compose`, for a production-style deployment).
- **Pluggable LLM providers**: an offline extractive "fake" generator (default, returns the
  most relevant retrieved passage — no API key needed) or OpenAI chat completions.
- **Source-attributed answers**: every response includes the retrieved chunks, their source
  document, and similarity score.
- **Fully offline test suite** (20+ tests) that never hits a network call or requires an API
  key, so it runs the same way locally and in CI.
- **Dockerized**, with a `docker-compose.yml` that wires the API to a Postgres/pgvector
  instance.

## Architecture

```mermaid
flowchart TD
    Client([Client]) -->|POST /ingest, POST /query, GET /health| API[FastAPI app]
    API --> Pipeline[RagPipeline]

    Pipeline -->|embed text| Embeddings{{EmbeddingsProvider}}
    Pipeline -->|add / search| Store{{VectorStore}}
    Pipeline -->|question + retrieved chunks| LLM{{LLMProvider}}

    Embeddings --- E1[LocalHashEmbeddings<br/>offline, default]
    Embeddings --- E2[OpenAIEmbeddings]

    Store --- S1[FaissVectorStore<br/>local file, default]
    Store --- S2[PgVectorStore<br/>Postgres + pgvector]

    LLM --- L1[FakeExtractiveLLM<br/>offline, default]
    LLM --- L2[OpenAIChatLLM]
```

**Ingestion flow:** documents in `DOCS_DIR` → split into overlapping chunks → embedded →
stored in the vector store with their source metadata.

**Query flow:** question → embedded → top-k most similar chunks retrieved → passed as context
to the LLM provider → answer returned together with the source chunks and similarity scores.

Each provider is selected at runtime via configuration (see [Configuration](#configuration))
and implements a small `Protocol` interface (see `src/finregrag/core/*/base.py`), so adding a
new provider (e.g. a local Ollama model, or Qdrant as a vector store) means writing one small
class and registering it in the corresponding `factory.py`, without touching the pipeline or
the API layer.

## Project structure

```
fin-reg-rag/
├── src/finregrag/
│   ├── main.py                 # FastAPI app entrypoint
│   ├── config.py               # Settings (env-driven)
│   ├── dependencies.py         # Wires providers into a RagPipeline
│   ├── api/
│   │   ├── routes.py           # /ingest, /query, /health
│   │   └── schemas.py          # Request/response models
│   ├── core/
│   │   ├── chunking.py         # Text splitting
│   │   ├── embeddings.py       # LocalHashEmbeddings / OpenAIEmbeddings
│   │   ├── rag_pipeline.py     # Orchestrates retrieval + generation
│   │   ├── llm/                # FakeExtractiveLLM / OpenAIChatLLM
│   │   └── vectorstore/        # FaissVectorStore / PgVectorStore
│   └── ingestion/
│       └── loader.py           # Loads .md/.txt documents from disk
├── scripts/ingest.py           # CLI to (re)index documents without the API
├── data/sample_docs/           # Synthetic sample regulation documents
├── tests/                      # Fully offline test suite (pytest)
├── docker-compose.yml          # API + Postgres/pgvector
├── Dockerfile
└── example.env                 # Copy to .env to configure the service
```

## Quickstart (local, no API keys needed)

Requires Python 3.9+.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp example.env .env

# Index the sample documents into a local FAISS index
python scripts/ingest.py

# Start the API
uvicorn finregrag.main:app --reload
```

Then open `http://localhost:8000/docs` for the interactive API docs, or query it directly:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the nighttime transaction limit for the instant payment arrangement?"}'
```

By default, `/query` will:
1. Embed the question with the deterministic local embedder.
2. Retrieve the most similar chunks from the FAISS index.
3. Return the single most relevant passage as an "extractive" answer (no LLM call), together
   with all retrieved sources and their similarity scores.

This makes the whole pipeline runnable and testable without any external dependency — useful
both for CI and for anyone evaluating this project without wanting to configure API keys.

## Switching to real LLM generation (OpenAI)

Set the following in `.env` (copied from `example.env`):

```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini   # or any chat-completion-capable model

EMBEDDINGS_PROVIDER=openai
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

With `LLM_PROVIDER=openai`, `/query` will send the retrieved chunks as context to a real chat
model and return a grounded, synthesized answer instead of the raw passage. The system prompt
(see `core/llm/openai_llm.py`) instructs the model to answer strictly from the provided context
and to cite its source.

## Running with Docker Compose (Postgres + pgvector)

For a more production-representative setup, using Postgres with the `pgvector` extension
instead of a local FAISS file:

```bash
docker compose up --build
```

This starts:
- `postgres`: a `pgvector/pgvector` Postgres instance, with a `document_chunks` table created
  automatically on first connection.
- `api`: the FastAPI service, configured via environment variables to use `VECTOR_STORE=pgvector`.

You can still choose the LLM/embeddings provider via the `LLM_PROVIDER`, `EMBEDDINGS_PROVIDER`,
and `OPENAI_API_KEY` environment variables (see `docker-compose.yml`), independently of the
vector store backend.

## Configuration

All configuration is environment-variable driven (see `src/finregrag/config.py` and
`example.env`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `fake` | `fake` (offline, extractive) or `openai` |
| `OPENAI_API_KEY` | _(empty)_ | Required when any `*_PROVIDER` is `openai` |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model used for generation |
| `EMBEDDINGS_PROVIDER` | `local` | `local` (offline hashing embedder) or `openai` |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `VECTOR_STORE` | `faiss` | `faiss` (local file) or `pgvector` (Postgres) |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Where the FAISS index is persisted |
| `POSTGRES_DSN` | `postgresql://finregrag:finregrag@localhost:5432/finregrag` | Postgres connection string, used when `VECTOR_STORE=pgvector` |
| `DOCS_DIR` | `./data/sample_docs` | Directory scanned for `.md`/`.txt` documents to ingest |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | Text splitting parameters |
| `TOP_K` | `4` | Number of chunks retrieved per query |

## API reference

### `POST /ingest`
Loads and indexes every `.md`/`.txt` file in `DOCS_DIR`, then persists the index (FAISS).
Re-ingesting a document replaces its previously indexed chunks instead of duplicating them, so
the endpoint is safe to call repeatedly. Returns how many documents and chunks were indexed.

### `POST /query`
```json
{ "question": "...", "top_k": 4 }
```
Returns the generated answer plus the list of retrieved source chunks (text, source file,
chunk index, similarity score).

### `GET /health`
Returns service status and the number of chunks currently in the vector store.

## Testing

```bash
pytest -v
```

The test suite always uses the offline `fake` LLM provider, the `local` hashing embedder, and
a temporary FAISS index — no network access or API key is required, which is also how CI
(`.github/workflows/ci.yml`) runs it across Python 3.9 and 3.12.

## Known limitations

- The default `local` embeddings provider is a deterministic hashing scheme meant to keep the
  whole pipeline testable and runnable offline — it is **not** a substitute for a learned
  embedding model in terms of semantic search quality. For production-grade retrieval quality,
  switch `EMBEDDINGS_PROVIDER` to `openai` (or add a local sentence-transformers provider
  following the same `EmbeddingsProvider` interface).
- The sample documents are synthetic and small; this project is meant to demonstrate the
  pipeline's architecture, not to serve as a real compliance tool.
- There is no authentication on the API endpoints — add an auth layer (API key, OAuth2) before
  exposing this beyond local/demo use.

## License

MIT
