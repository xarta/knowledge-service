"""
Document ingestion logic for knowledge-service.

Handles the full ingestion pipeline:
  1. Chunk documents via Normalised Semantic Chunker
  2. Enrich chunks with metadata
  3. Optionally generate contextual headers via LLM
  4. Embed chunks via vLLM embedding endpoint
  5. Store in SeekDB

Zero doc-sanitiser dependencies — all data comes via API requests.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from source.chunker_client import ChunkerClient
from source.embedding_client import EmbeddingClient
from source.llm_client import LLMClient, LLMConfig
from source.seekdb_client import SeekDBClient

logger = logging.getLogger(__name__)


# ===================================================================
# Ingestion Engine
# ===================================================================

class IngestionEngine:
    """Orchestrates document ingestion into SeekDB.

    Usage::

        engine = IngestionEngine(chunker, embedding, seekdb, llm)
        result = engine.ingest(files={
            "doc1.md": "# Title\\n\\nContent...",
        }, collection_name="my_docs")
    """

    def __init__(
        self,
        chunker_client: ChunkerClient,
        embedding_client: EmbeddingClient,
        seekdb_client: SeekDBClient,
        llm_client: Optional[LLMClient] = None,
    ):
        self.chunker = chunker_client
        self.embedding = embedding_client
        self.seekdb = seekdb_client
        self.llm = llm_client

    def ingest(
        self,
        files: Dict[str, str],
        collection_name: Optional[str] = None,
        visibility: str = "public",
        metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        contextualise: bool = False,
        chunker_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest documents into SeekDB.

        Args:
            files: Map of relative path → content.
            collection_name: Collection name (auto-gen if None).
            visibility: "public" or "private".
            metadata: Per-file metadata dicts.
            contextualise: Generate LLM context headers.
            chunker_config: Chunker configuration (buffer_size, etc.).

        Returns:
            Dict with keys: collection, visibility, chunks_ingested, files_processed, duration_seconds.
        """
        start_time = time.time()

        # Auto-generate collection name if omitted
        if not collection_name:
            # Use first path component as collection name
            first_path = list(files.keys())[0] if files else "default"
            collection_name = first_path.split("/")[0].replace(".", "_")

        logger.info(f"Starting ingestion for collection '{collection_name}' ({len(files)} files)")

        # Ensure collection exists
        self._ensure_collection(collection_name)

        # Chunk all files
        all_chunks = []
        for filepath, content in files.items():
            logger.debug(f"Chunking {filepath}")
            chunks = self._chunk_file(filepath, content, chunker_config or {})

            # Enrich chunks with metadata
            file_metadata = (metadata or {}).get(filepath, {})
            for chunk in chunks:
                enriched = self._enrich_chunk(chunk, filepath, visibility, file_metadata)
                all_chunks.append(enriched)

        logger.info(f"Chunked {len(files)} files into {len(all_chunks)} chunks")

        # Contextualise (optional)
        if contextualise and self.llm:
            logger.info("Generating contextual headers")
            all_chunks = self._contextualise_chunks(all_chunks)

        # Embed chunks
        logger.info(f"Embedding {len(all_chunks)} chunks")
        all_chunks = self._embed_chunks(all_chunks)

        # Write to SeekDB
        logger.info(f"Writing {len(all_chunks)} chunks to SeekDB")
        self._write_chunks(collection_name, all_chunks)

        duration = time.time() - start_time
        logger.info(f"Ingestion complete: {len(all_chunks)} chunks in {duration:.2f}s")

        return {
            "collection": collection_name,
            "visibility": visibility,
            "chunks_ingested": len(all_chunks),
            "files_processed": len(files),
            "duration_seconds": duration,
        }

    def _ensure_collection(self, name: str) -> None:
        """Create collection if it doesn't exist."""
        try:
            collections = self.seekdb.list_collections()
            existing = [c["name"] for c in collections.get("collections", [])]
            if name not in existing:
                self.seekdb.create_collection(name)
                logger.debug(f"Created collection '{name}'")
        except Exception as exc:
            logger.warning(f"Collection check failed: {exc}")

    def _chunk_file(
        self,
        filepath: str,
        content: str,
        chunker_config: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Chunk a single file via the semantic chunker."""
        try:
            chunks = self.chunker.chunk_document(
                filename=filepath.split("/")[-1],
                content=content,
                **chunker_config,
            )
            return chunks
        except Exception as exc:
            logger.error(f"Chunking failed for {filepath}: {exc}")
            # Fallback: single chunk
            return [{"text": content, "start_index": 0, "end_index": len(content)}]

    def _enrich_chunk(
        self,
        chunk: Dict[str, Any],
        filepath: str,
        visibility: str,
        file_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Add metadata to a chunk."""
        chunk_id = str(uuid4())

        enriched = {
            "id": chunk_id,
            "text": chunk["text"],
            "source_file": filepath,
            "visibility": visibility,
            "start_index": chunk.get("start_index", 0),
            "end_index": chunk.get("end_index", len(chunk["text"])),
        }

        # Merge file metadata
        for key, value in file_metadata.items():
            enriched[key] = value

        return enriched

    def _contextualise_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate LLM context headers for chunks (optional)."""
        if not self.llm:
            return chunks

        # Group chunks by source file
        files_map: Dict[str, List[Dict[str, Any]]] = {}
        for chunk in chunks:
            source = chunk["source_file"]
            if source not in files_map:
                files_map[source] = []
            files_map[source].append(chunk)

        # Generate context per file
        contextualised = []
        for source, file_chunks in files_map.items():
            full_doc = " ".join([c["text"] for c in file_chunks])

            prompt = (
                f"Provide a brief 1-2 sentence context for this document (from {source}):\n\n"
                f"{full_doc[:2000]}"  # Truncate to avoid huge payloads
            )

            try:
                context_header = self.llm.ask(prompt, "You are a helpful assistant.")
                for chunk in file_chunks:
                    chunk["context_header"] = context_header
                    contextualised.append(chunk)
            except Exception as exc:
                logger.warning(f"Contextualisation failed for {source}: {exc}")
                contextualised.extend(file_chunks)

        return contextualised

    def _embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed all chunks in batches."""
        texts = [c["text"] for c in chunks]

        try:
            result = self.embedding.embed(texts)
            for chunk, embedding in zip(chunks, result.embeddings):
                chunk["embedding"] = embedding
        except Exception as exc:
            logger.error(f"Embedding failed: {exc}")
            # Continue without embeddings
            for chunk in chunks:
                chunk["embedding"] = []

        return chunks

    def _write_chunks(self, collection: str, chunks: List[Dict[str, Any]]) -> None:
        """Write chunks to SeekDB in batches."""
        batch_size = 32

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            ids = [c["id"] for c in batch]
            documents = [c["text"] for c in batch]
            embeddings = [c["embedding"] for c in batch]
            metadatas = [
                {k: v for k, v in c.items() if k not in ("id", "text", "embedding")}
                for c in batch
            ]

            try:
                self.seekdb.add(
                    collection=collection,
                    ids=ids,
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            except Exception as exc:
                logger.error(f"SeekDB write failed for batch {i//batch_size}: {exc}")
