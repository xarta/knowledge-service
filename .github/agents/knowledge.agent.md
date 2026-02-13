---
description: "Knowledge Service agent — health checks, integration testing, and service management"
---

# @knowledge agent

You help manage the Knowledge Service — a Dockerised FastAPI service for document ingestion and RAG querying via SeekDB.

## Quick Commands

| Action | Command |
|--------|---------|
| Health check | `python3 tools/check_service.py` |
| Integration tests | `python3 tools/check_service.py --test` |
| Full test suite | `python3 tools/check_service.py --all` |
| Unit tests | `PYTHONPATH=. python3 -m unittest discover tests -v` |

## Service Dependencies

- SeekDB (vector database)
- Embedding endpoint (vLLM)
- Normalised Semantic Chunker
- LLM endpoint (optional, for query rewriting)
- Reranker endpoint (optional)

## Deployment

See the project README for Docker Compose setup. Service endpoints and credentials are configured via environment variables (see `.env.example`).
