# knowledge-service

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Dockerised FastAPI service that manages document ingestion into a vector database (SeekDB) and provides a RAG (Retrieval-Augmented Generation) query interface. It composes external chunking and embedding services to split, enrich, embed, and store documents — then exposes search and retrieval for downstream consumers.

## ⚠️ AI-Generated Content Notice

This project was **generated with AI assistance** and should be treated accordingly:

- **Not production-ready**: Created for a specific homelab environment.
- **May contain bugs**: AI-generated code can have subtle issues.
- **Author's Python experience**: The author is not an experienced Python programmer.

### AI Tools Used

- GitHub Copilot (Claude models)
- Local vLLM instances for validation

### Licensing Note

Released under the **MIT License**. Given the AI-generated nature:
- The author makes no claims about originality
- Use at your own risk
- If you discover any copyright concerns, please open an issue

---

## How It Works

The Knowledge Service sits between document analysis pipelines and the vector database. Callers POST documents (as JSON text) to be ingested, or POST queries to retrieve relevant context via RAG.

### Ingestion Flow

1. Receives file contents + optional analysis metadata (duplication flags, contradiction flags, file roles)
2. Calls the **Normalised Semantic Chunker** to split documents into appropriately-sized semantic chunks
3. Enriches each chunk with metadata (source file, role, visibility, analysis tags)
4. Embeds chunks via the remote **vLLM embedding endpoint**
5. Stores enriched, embedded chunks in **SeekDB** (vector database)
6. Supports public + private collection tiers (same content, different visibility metadata)

### Query Flow

1. Receives a natural-language query + target collection(s)
2. Optionally rewrites the query via LLM for better retrieval
3. Performs hybrid search (vector + keyword) against SeekDB
4. Optionally reranks results via a dedicated reranker endpoint
5. Returns ranked, relevant chunks with metadata

## Prerequisites

- **Docker** on the host
- **SeekDB** — vector database instance accessible via HTTP
- **Embedding endpoint** — OpenAI-compatible embedding API (e.g., vLLM)
- **Normalised Semantic Chunker** — chunking service accessible via HTTP
- **(Optional) Reranker endpoint** — for improved retrieval quality
- **(Optional) LLM endpoint** — for query rewriting and contextual chunking

## Quick Start

### 1. Build the image

```bash
docker build -t knowledge-service:latest .
```

### 2. Create a Compose file

```yaml
# compose.yaml with made-up values
services:
  knowledge-service:
    image: knowledge-service:latest
    container_name: knowledge-service
    restart: unless-stopped
    ports:
      - "8209:8000"
    env_file:
      - secrets.env
    environment:
      LOG_LEVEL: INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Start the service

```bash
docker compose up -d
```

### 4. Check health

```bash
curl http://localhost:8209/health
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service info and version |
| `GET` | `/health` | Health check (tests SeekDB + embedding connectivity) |
| `POST` | `/ingest` | Ingest documents into a SeekDB collection |
| `POST` | `/query` | RAG query against one or more collections |
| `GET` | `/collections` | List managed collections |
| `DELETE` | `/collections/{name}` | Delete a collection |

## Project Structure

```
knowledge-service/
├── app.py                    # FastAPI endpoints
├── source/
│   ├── seekdb_client.py      # SeekDB HTTP client (stdlib urllib)
│   ├── ingestion.py          # Ingestion pipeline: chunk → enrich → embed → store
│   ├── query.py              # RAG query pipeline: rewrite → search → rerank
│   ├── chunker_client.py     # HTTP client for Normalised Semantic Chunker
│   ├── embedding_client.py   # HTTP client for vLLM embedding endpoint
│   ├── llm_client.py         # HTTP client for vLLM chat (query rewriting)
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
# Unit tests (no external services needed)
PYTHONPATH=. python3 -m unittest discover tests -v

# Health check against live service (reads endpoint from .env)
python3 tools/check_service.py

# Integration tests
python3 tools/check_service.py --test

# Full test suite
python3 tools/check_service.py --all
```

## Environment Variables

All service endpoints and credentials are loaded from environment variables.
See `.env.example` for client-side configuration.
The server-side config (vLLM endpoints, SeekDB URL, chunker URL, API keys) is injected via `secrets.env` at deploy time — never committed to source.

## Related Projects

This service is part of the [xarta](https://github.com/xarta) document analysis ecosystem:

- [Normalized-Semantic-Chunker](https://github.com/xarta/Normalized-Semantic-Chunker) — embedding-based text chunking service (upstream dependency)
- [gitleaks-validator](https://github.com/xarta/gitleaks-validator) — Dockerised gitleaks scanning API
- [Agentic-Chunker](https://github.com/xarta/Agentic-Chunker) — LLM-driven proposition chunking service

## License

MIT — see [LICENSE](LICENSE).
