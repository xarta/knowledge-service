---
description: Knowledge Service agent — health checks, integration testing, and service management
mode: subagent
tools:
  write: false
  edit: false
  bash: true
permission:
  bash:
    "python3 tools/check_service.py*": allow
    "python3 tools/clean_test_data.py*": allow
    "PYTHONPATH=. python3 -m unittest*": allow
    "cat *": allow
    "grep *": allow
---

# Knowledge Service Agent

Manages the Knowledge Service — a Dockerised FastAPI service for document ingestion and RAG querying via SeekDB.

## Quick Commands

| Action | Command |
|--------|---------|
| Health check | `python3 tools/check_service.py` |
| Integration tests | `python3 tools/check_service.py --test` |
| Full test suite | `python3 tools/check_service.py --all` |
| Unit tests | `PYTHONPATH=. python3 -m unittest discover tests -v` || Scan for orphaned test data | `python3 tools/clean_test_data.py` |
| Clean orphaned test data | `python3 tools/clean_test_data.py --clean` |

## Test Data Cleanup

Integration tests across knowledge-service and doc-sanitiser create SeekDB collections that are normally cleaned up automatically. If tests crash, orphaned collections (e.g. `_integration_test`, `_test_ingestion_<timestamp>`) may persist.

The cleanup tool:
1. Scans all SeekDB collections via the knowledge-service API
2. Classifies each by known test naming patterns and heuristics
3. Optionally validates with an LLM before offering deletion
4. NEVER deletes production collections — cautious by design

When a user reports leftover test data, run the scan first (no `--clean`) to review, then `--clean` to delete with confirmation.
## Service Dependencies

- SeekDB (vector database)
- Embedding endpoint (vLLM)
- Normalised Semantic Chunker
- LLM endpoint (optional, for query rewriting)
- Reranker endpoint (optional)

## Deployment

See the project README for Docker Compose setup. Service endpoints and credentials are configured via environment variables (see `.env.example`).
