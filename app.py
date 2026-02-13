"""
Knowledge Service — FastAPI application.

Provides HTTP endpoints for document ingestion and RAG query against SeekDB.

Endpoints:
  GET  /          — service info
  GET  /health    — health check (SeekDB, embedding, chunker, LLM, reranker)
  POST /ingest    — ingest documents into SeekDB
  POST /query     — query the knowledge base
  GET  /collections — list collections
  DELETE /collections/{name} — delete a collection

Environment variables:
  SEEKDB_BASE_URL     — SeekDB HTTP API endpoint
  SEEKDB_API_KEY      — SeekDB Bearer token
  SEEKDB_DATABASE     — SeekDB database name
  EMBEDDING_BASE_URL  — vLLM embedding endpoint
  EMBEDDING_API_KEY   — Embedding Bearer token
  CHUNKER_URL         — Normalised Semantic Chunker endpoint
  VLLM_BASE_URL       — vLLM LLM endpoint (optional, for query rewriting + contextual headers)
  VLLM_API_KEY        — LLM Bearer token (optional)
  RERANKER_BASE_URL   — Reranker endpoint (optional)
  RERANKER_API_KEY    — Reranker Bearer token (optional)
  LOG_LEVEL           — Logging level (default: INFO)
"""

import logging
import os
import sys
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from source.chunker_client import ChunkerClient, ChunkerConfig, ChunkerError
from source.embedding_client import EmbeddingClient, EmbeddingConfig, EmbeddingError
from source.ingestion import IngestionEngine
from source.llm_client import LLMClient, LLMConfig, LLMError
from source.models import (
    CollectionInfo,
    CollectionsResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    QueryResult,
)
from source.query import QueryEngine
from source.seekdb_client import SeekDBClient, SeekDBConfig, SeekDBError

# Configure logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Knowledge Service",
    description="Document ingestion and RAG query service for SeekDB",
    version="1.0.0",
)

# Global clients (initialized on startup)
seekdb_client: SeekDBClient = None
embedding_client: EmbeddingClient = None
chunker_client: ChunkerClient = None
llm_client: LLMClient = None
ingestion_engine: IngestionEngine = None
query_engine: QueryEngine = None


@app.on_event("startup")
async def startup_event():
    """Initialize clients on startup."""
    global seekdb_client, embedding_client, chunker_client, llm_client
    global ingestion_engine, query_engine

    logger.info("Initialising knowledge-service...")

    # Required clients
    try:
        seekdb_config = SeekDBConfig.from_env()
        seekdb_client = SeekDBClient(seekdb_config)
        logger.info(f"SeekDB client configured: {seekdb_config.base_url}")
    except Exception as exc:
        logger.error(f"SeekDB configuration failed: {exc}")
        sys.exit(1)

    try:
        embedding_config = EmbeddingConfig.from_env()
        embedding_client = EmbeddingClient(embedding_config)
        logger.info(f"Embedding client configured: {embedding_config.embedding_base_url}")
    except Exception as exc:
        logger.error(f"Embedding configuration failed: {exc}")
        sys.exit(1)

    try:
        chunker_config = ChunkerConfig.from_env()
        chunker_client = ChunkerClient(chunker_config)
        logger.info(f"Chunker client configured: {chunker_config.base_url}")
    except Exception as exc:
        logger.error(f"Chunker configuration failed: {exc}")
        sys.exit(1)

    # Optional LLM client
    try:
        llm_config = LLMConfig.from_env()
        if llm_config:
            llm_client = LLMClient(llm_config)
            logger.info(f"LLM client configured: {llm_config.base_url}")
        else:
            logger.info("LLM client not configured (VLLM_BASE_URL not set)")
    except Exception as exc:
        logger.warning(f"LLM configuration failed: {exc}")

    # Initialize engines
    ingestion_engine = IngestionEngine(
        chunker_client=chunker_client,
        embedding_client=embedding_client,
        seekdb_client=seekdb_client,
        llm_client=llm_client,
    )

    query_engine = QueryEngine(
        seekdb_client=seekdb_client,
        embedding_client=embedding_client,
        llm_client=llm_client,
    )

    logger.info("knowledge-service startup complete")


# ===================================================================
# Endpoints
# ===================================================================

@app.get("/")
async def root():
    """Service info."""
    return {
        "service": "knowledge-service",
        "version": "1.0.0",
        "status": "running",
        "description": "Document ingestion and RAG query service for SeekDB",
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — tests all upstream dependencies."""
    status = "healthy"
    dependencies: Dict[str, any] = {}

    # SeekDB
    try:
        seekdb_test = seekdb_client.test_connection()
        dependencies["seekdb"] = {
            "connected": seekdb_test["connected"],
            "endpoint": seekdb_test["endpoint"],
            "error": seekdb_test.get("error"),
        }
        if not seekdb_test["connected"]:
            status = "degraded"
    except Exception as exc:
        dependencies["seekdb"] = {"connected": False, "error": str(exc)}
        status = "unhealthy"

    # Embedding
    try:
        embedding_test = embedding_client.test_connection()
        dependencies["embedding"] = {
            "connected": embedding_test["embedding_connected"],
            "model": embedding_test["embedding_model"],
            "errors": embedding_test.get("errors", []),
        }
        if not embedding_test["embedding_connected"]:
            status = "degraded"
    except Exception as exc:
        dependencies["embedding"] = {"connected": False, "error": str(exc)}
        status = "unhealthy"

    # Chunker
    try:
        chunker_test = chunker_client.test_connection()
        dependencies["chunker"] = {
            "connected": chunker_test["connected"],
            "endpoint": chunker_test["endpoint"],
            "error": chunker_test.get("error"),
        }
        if not chunker_test["connected"]:
            status = "degraded"
    except Exception as exc:
        dependencies["chunker"] = {"connected": False, "error": str(exc)}
        status = "unhealthy"

    # LLM (optional)
    if llm_client:
        try:
            llm_test = llm_client.test_connection()
            dependencies["llm"] = {
                "connected": llm_test["connected"],
                "model": llm_test["model"],
                "error": llm_test.get("error"),
            }
        except Exception as exc:
            dependencies["llm"] = {"connected": False, "error": str(exc)}
    else:
        dependencies["llm"] = {"status": "not_configured"}

    # Reranker (optional)
    if embedding_client.config.reranker_base_url:
        try:
            reranker_test = embedding_client.test_connection()
            dependencies["reranker"] = {
                "connected": reranker_test["reranker_connected"],
                "model": reranker_test["reranker_model"],
            }
        except Exception as exc:
            dependencies["reranker"] = {"connected": False, "error": str(exc)}
    else:
        dependencies["reranker"] = {"status": "not_configured"}

    return HealthResponse(status=status, dependencies=dependencies)


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest documents into SeekDB."""
    try:
        result = ingestion_engine.ingest(
            files=request.files,
            collection_name=request.collection_name,
            visibility=request.visibility,
            metadata=request.metadata,
            contextualise=request.contextualise,
            chunker_config=request.chunker_config.dict() if request.chunker_config else None,
        )

        return IngestResponse(**result)

    except (SeekDBError, EmbeddingError, ChunkerError) as exc:
        logger.error(f"Ingestion failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during ingestion")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the knowledge base."""
    try:
        result = query_engine.query(
            question=request.question,
            collections=request.collections,
            n_results=request.n_results,
            rerank=request.rerank,
            rewrite=request.rewrite,
        )

        # Convert to Pydantic models
        results = [
            QueryResult(
                collection=r["collection"],
                document=r["document"],
                metadata=r["metadata"],
                distance=r["distance"],
                score=r.get("score"),
            )
            for r in result["results"]
        ]

        return QueryResponse(
            results=results,
            rewritten_query=result.get("rewritten_query"),
            total_results=result["total_results"],
        )

    except (SeekDBError, EmbeddingError, LLMError) as exc:
        logger.error(f"Query failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
    except Exception as exc:
        logger.exception("Unexpected error during query")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")


@app.get("/collections", response_model=CollectionsResponse)
async def list_collections():
    """List all collections in the database."""
    try:
        response = seekdb_client.list_collections()
        collections = response.get("collections", [])

        # Get counts
        collection_infos = []
        for coll in collections:
            name = coll["name"]
            try:
                count_response = seekdb_client.count(name)
                count = count_response.get("count", 0)
            except Exception:
                count = 0

            # Infer visibility from first item metadata (if available)
            visibility = "unknown"
            try:
                items = seekdb_client.get(name, ids=None)
                if items.get("metadatas") and len(items["metadatas"]) > 0:
                    visibility = items["metadatas"][0].get("visibility", "unknown")
            except Exception:
                pass

            collection_infos.append(
                CollectionInfo(name=name, count=count, visibility=visibility)
            )

        return CollectionsResponse(collections=collection_infos)

    except SeekDBError as exc:
        logger.error(f"List collections failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/collections/{name}", status_code=204)
async def delete_collection(name: str):
    """Delete a collection."""
    try:
        seekdb_client.delete_collection(name)
        logger.info(f"Deleted collection '{name}'")
        return JSONResponse(status_code=204, content={})
    except SeekDBError as exc:
        logger.error(f"Delete collection failed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
