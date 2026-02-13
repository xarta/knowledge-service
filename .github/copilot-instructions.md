# knowledge-service — GitHub Copilot Instructions

## Project Overview

knowledge-service is a Dockerised FastAPI service that manages document ingestion into SeekDB (vector database) and provides a RAG query interface. It composes the Normalised Semantic Chunker and vLLM embedding endpoints to split, enrich, embed, and store documents.

This service was extracted from the doc-sanitiser monolith's `source/knowledge/` modules to create a reusable, independently testable component.

## Key Rules

- **Use `python3`** not `python`.
- **British spelling** — `sanitise`, `analyse`, `colour`, etc.
- **No real infrastructure in source** — never put real IPs, hostnames, LXC IDs, or API keys in committed code. All loaded from environment variables.
- **TDD approach** — write tests first when implementing new features.
- **Stdlib HTTP clients** — all HTTP client code uses `urllib.request` only. No `requests`, no `httpx`.
- **FastAPI + pydantic** — web framework. Use `pydantic.BaseModel` for request/response models.
- **Run tests with unittest** — `PYTHONPATH=. python3 -m unittest discover tests -v`.

## Project Structure

```
knowledge-service/
├── app.py                    # FastAPI endpoints: /, /health, /ingest, /query, /collections
├── source/
│   ├── seekdb_client.py      # SeekDB HTTP client (stdlib urllib)
│   ├── ingestion.py          # Ingestion pipeline: chunk → enrich → embed → store
│   ├── query.py              # RAG query pipeline: rewrite → search → rerank
│   ├── chunker_client.py     # HTTP client for Normalised Semantic Chunker
│   ├── embedding_client.py   # HTTP client for vLLM embedding endpoint
│   ├── llm_client.py         # HTTP client for vLLM chat (query rewriting, contextual headers)
│   └── registry.py           # Collection registry management
├── tests/
│   ├── __init__.py
│   ├── test_api.py           # FastAPI endpoint tests (mocked backends)
│   ├── test_ingestion.py     # Ingestion pipeline unit tests
│   └── test_query.py         # RAG query unit tests
├── tools/
│   └── check_service.py      # Health check + integration tests
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── LICENSE
└── README.md
```

## Running Tests

```bash
# Unit tests (fast, no external services needed)
PYTHONPATH=. python3 -m unittest discover tests -v

# Integration tests against live service (reads endpoint from .env)
python3 tools/check_service.py --test

# Health check only
python3 tools/check_service.py

# Full integration test suite
python3 tools/check_service.py --all
```

## Upstream Services

| Service | Purpose | Required |
|---------|---------|----------|
| SeekDB | Vector database for storage and search | Required |
| Embedding endpoint (vLLM) | Generate embeddings for chunks and queries | Required |
| Normalised Semantic Chunker | Split documents into semantic chunks | Required |
| LLM endpoint (vLLM) | Query rewriting, contextual chunk headers | Optional |
| Reranker endpoint (vLLM) | Rerank retrieval results | Optional |

## Relationship to doc-sanitiser

This service was extracted from `doc-sanitiser/source/knowledge/` (ingestion.py, query.py, seekdb.py, registry.py, contextual.py). The doc-sanitiser orchestrator calls this service's HTTP API instead of importing those modules directly. See `_plans/BUILD-PLAN.md` for extraction details.

## API Design

### POST /ingest

Accepts file contents + optional analysis metadata, chunks via the semantic chunker, embeds, and stores in SeekDB.

### POST /query

CRAG-style RAG pipeline: optional query rewriting → hybrid search → optional reranking → results with metadata.

### GET /collections

Lists all collections managed by this service instance.

### DELETE /collections/{name}

Deletes a collection and its registry entry.
