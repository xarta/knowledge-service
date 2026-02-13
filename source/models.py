"""
Pydantic models for knowledge-service API.

Defines request and response schemas for all endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ===================================================================
# Ingestion Models
# ===================================================================

class ChunkerConfig(BaseModel):
    """Configuration for the semantic chunker (optional)."""
    buffer_size: Optional[int] = None
    breakpoint_percentile_threshold: Optional[int] = None


class FileMetadata(BaseModel):
    """Per-file analysis metadata from doc-sanitiser."""
    file_role: Optional[str] = None
    has_duplicates: Optional[bool] = None
    has_contradictions: Optional[bool] = None
    duplication_group: Optional[int] = None
    topics: Optional[List[str]] = None


class IngestRequest(BaseModel):
    """Request to ingest documents into SeekDB."""
    files: Dict[str, str] = Field(..., description="Map of relative path → content")
    collection_name: Optional[str] = Field(None, description="Collection name (auto-generated if omitted)")
    visibility: str = Field("public", description="Visibility tier: 'public' or 'private'")
    metadata: Optional[Dict[str, FileMetadata]] = Field(None, description="Per-file metadata")
    contextualise: bool = Field(False, description="Generate LLM context headers per chunk")
    chunker_config: Optional[ChunkerConfig] = Field(None, description="Chunker configuration")


class IngestResponse(BaseModel):
    """Response from ingestion."""
    collection: str
    visibility: str
    chunks_ingested: int
    files_processed: int
    duration_seconds: float


# ===================================================================
# Query Models
# ===================================================================

class QueryRequest(BaseModel):
    """Request to query the knowledge base."""
    question: str = Field(..., description="Natural-language query")
    collections: Optional[List[str]] = Field(None, description="Target collections (all if omitted)")
    n_results: int = Field(5, description="Number of results to return")
    rerank: bool = Field(False, description="Apply reranker to results")
    rewrite: bool = Field(False, description="Rewrite query via LLM")


class QueryResult(BaseModel):
    """Single search result."""
    collection: str
    document: str
    metadata: Dict[str, Any]
    distance: float
    score: Optional[float] = None  # Reranker score, if applied


class QueryResponse(BaseModel):
    """Response from query."""
    results: List[QueryResult]
    rewritten_query: Optional[str] = None
    total_results: int


# ===================================================================
# Collection Models
# ===================================================================

class CollectionInfo(BaseModel):
    """Information about a collection."""
    name: str
    count: int
    visibility: str


class CollectionsResponse(BaseModel):
    """Response from list collections endpoint."""
    collections: List[CollectionInfo]


# ===================================================================
# Health Models
# ===================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    dependencies: Dict[str, Any]
