"""Unit tests for the ingestion engine.

Tests cover the full ingestion pipeline: chunking, enrichment,
contextualisation, embedding, and writing to SeekDB.
"""

import unittest
from unittest.mock import MagicMock, call, patch

from source.ingestion import IngestionEngine


# ===================================================================
# Ingestion Pipeline
# ===================================================================

class TestIngestReturnShape(unittest.TestCase):
    """Tests for IngestionEngine.ingest() return value."""

    def setUp(self):
        self.mock_chunker = MagicMock()
        self.mock_embedding = MagicMock()
        self.mock_seekdb = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=self.mock_chunker,
            embedding_client=self.mock_embedding,
            seekdb_client=self.mock_seekdb,
        )

        # Standard mock responses
        self.mock_seekdb.list_collections.return_value = {"collections": []}
        self.mock_seekdb.create_collection.return_value = {}
        self.mock_seekdb.add.return_value = {}
        self.mock_chunker.chunk_document.return_value = [
            {"text": "Chunk one.", "start_index": 0, "end_index": 10},
        ]
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3]]
        )

    def test_returns_expected_keys(self):
        """Result dict includes all required fields."""
        result = self.engine.ingest(
            files={"doc.md": "Chunk one."},
            collection_name="test",
        )
        for key in ("collection", "visibility", "chunks_ingested",
                     "files_processed", "duration_seconds"):
            self.assertIn(key, result)

    def test_counts_files_processed(self):
        """files_processed matches number of input files."""
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        result = self.engine.ingest(
            files={"a.md": "Content A", "b.md": "Content B"},
            collection_name="test",
        )
        self.assertEqual(result["files_processed"], 2)

    def test_counts_chunks_ingested(self):
        """chunks_ingested reflects chunker output."""
        self.mock_chunker.chunk_document.return_value = [
            {"text": "Chunk A", "start_index": 0, "end_index": 7},
            {"text": "Chunk B", "start_index": 8, "end_index": 15},
        ]
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1], [0.2]]
        )

        result = self.engine.ingest(
            files={"a.md": "Chunk A\nChunk B"},
            collection_name="test",
        )
        self.assertEqual(result["chunks_ingested"], 2)

    def test_duration_is_positive(self):
        """duration_seconds is a non-negative float."""
        result = self.engine.ingest(
            files={"doc.md": "Content"},
            collection_name="test",
        )
        self.assertIsInstance(result["duration_seconds"], float)
        self.assertGreaterEqual(result["duration_seconds"], 0)


class TestCollectionNaming(unittest.TestCase):
    """Tests for collection name generation."""

    def setUp(self):
        self.mock_chunker = MagicMock()
        self.mock_embedding = MagicMock()
        self.mock_seekdb = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=self.mock_chunker,
            embedding_client=self.mock_embedding,
            seekdb_client=self.mock_seekdb,
        )

        self.mock_seekdb.list_collections.return_value = {"collections": []}
        self.mock_seekdb.add.return_value = {}
        self.mock_chunker.chunk_document.return_value = [
            {"text": "text", "start_index": 0, "end_index": 4},
        ]
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1]]
        )

    def test_explicit_collection_name(self):
        """Explicit name is used as-is."""
        result = self.engine.ingest(
            files={"doc.md": "text"},
            collection_name="my_collection",
        )
        self.assertEqual(result["collection"], "my_collection")

    def test_auto_generates_from_first_path(self):
        """When collection_name is None, uses first path component."""
        result = self.engine.ingest(
            files={"guides/setup.md": "text"},
        )
        self.assertEqual(result["collection"], "guides")

    def test_auto_replaces_dots_with_underscores(self):
        """Dots in auto-generated names are replaced with underscores."""
        result = self.engine.ingest(
            files={"config.md": "text"},
        )
        self.assertEqual(result["collection"], "config_md")


class TestVisibility(unittest.TestCase):
    """Tests for visibility parameter handling."""

    def setUp(self):
        self.mock_chunker = MagicMock()
        self.mock_embedding = MagicMock()
        self.mock_seekdb = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=self.mock_chunker,
            embedding_client=self.mock_embedding,
            seekdb_client=self.mock_seekdb,
        )

        self.mock_seekdb.list_collections.return_value = {"collections": []}
        self.mock_seekdb.add.return_value = {}
        self.mock_chunker.chunk_document.return_value = [
            {"text": "text", "start_index": 0, "end_index": 4},
        ]
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1]]
        )

    def test_default_is_public(self):
        result = self.engine.ingest(files={"doc.md": "text"})
        self.assertEqual(result["visibility"], "public")

    def test_private_visibility(self):
        result = self.engine.ingest(
            files={"doc.md": "text"},
            visibility="private",
        )
        self.assertEqual(result["visibility"], "private")


class TestEnsureCollection(unittest.TestCase):
    """Tests for IngestionEngine._ensure_collection()."""

    def setUp(self):
        self.mock_seekdb = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=self.mock_seekdb,
        )

    def test_creates_new_collection(self):
        """Creates collection when it does not exist."""
        self.mock_seekdb.list_collections.return_value = {"collections": []}
        self.engine._ensure_collection("new_docs")
        self.mock_seekdb.create_collection.assert_called_once_with("new_docs")

    def test_skips_existing_collection(self):
        """Does not create collection when already present."""
        self.mock_seekdb.list_collections.return_value = {
            "collections": [{"name": "existing_docs"}],
        }
        self.engine._ensure_collection("existing_docs")
        self.mock_seekdb.create_collection.assert_not_called()

    def test_handles_list_error(self):
        """Continues gracefully if listing fails."""
        self.mock_seekdb.list_collections.side_effect = Exception("Timeout")
        # Should not raise
        self.engine._ensure_collection("test")


# ===================================================================
# Chunk Enrichment
# ===================================================================

class TestEnrichChunk(unittest.TestCase):
    """Tests for IngestionEngine._enrich_chunk()."""

    def setUp(self):
        self.engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=MagicMock(),
        )

    def _sample_chunk(self):
        return {"text": "hello world", "start_index": 0, "end_index": 11}

    def test_adds_source_file(self):
        enriched = self.engine._enrich_chunk(
            self._sample_chunk(), "readme.md", "public", {},
        )
        self.assertEqual(enriched["source_file"], "readme.md")

    def test_adds_visibility(self):
        enriched = self.engine._enrich_chunk(
            self._sample_chunk(), "doc.md", "private", {},
        )
        self.assertEqual(enriched["visibility"], "private")

    def test_generates_uuid_id(self):
        enriched = self.engine._enrich_chunk(
            self._sample_chunk(), "doc.md", "public", {},
        )
        self.assertIn("id", enriched)
        # UUID format: 8-4-4-4-12 = 36 chars
        self.assertEqual(len(enriched["id"]), 36)

    def test_preserves_text(self):
        enriched = self.engine._enrich_chunk(
            self._sample_chunk(), "doc.md", "public", {},
        )
        self.assertEqual(enriched["text"], "hello world")

    def test_preserves_start_end_index(self):
        enriched = self.engine._enrich_chunk(
            self._sample_chunk(), "doc.md", "public", {},
        )
        self.assertEqual(enriched["start_index"], 0)
        self.assertEqual(enriched["end_index"], 11)

    def test_merges_file_metadata(self):
        metadata = {"file_role": "guide", "topics": ["setup", "deploy"]}
        enriched = self.engine._enrich_chunk(
            self._sample_chunk(), "doc.md", "public", metadata,
        )
        self.assertEqual(enriched["file_role"], "guide")
        self.assertEqual(enriched["topics"], ["setup", "deploy"])

    def test_unique_ids_per_chunk(self):
        """Successive calls generate different IDs."""
        chunk = self._sample_chunk()
        id_a = self.engine._enrich_chunk(chunk, "a.md", "public", {})["id"]
        id_b = self.engine._enrich_chunk(chunk, "b.md", "public", {})["id"]
        self.assertNotEqual(id_a, id_b)


# ===================================================================
# Chunking
# ===================================================================

class TestChunkFile(unittest.TestCase):
    """Tests for IngestionEngine._chunk_file()."""

    def setUp(self):
        self.mock_chunker = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=self.mock_chunker,
            embedding_client=MagicMock(),
            seekdb_client=MagicMock(),
        )

    def test_passes_filename_and_content(self):
        """Filename is extracted from path and sent to chunker."""
        self.mock_chunker.chunk_document.return_value = [
            {"text": "chunk", "start_index": 0, "end_index": 5},
        ]
        self.engine._chunk_file("guides/setup.md", "content", {})
        self.mock_chunker.chunk_document.assert_called_once_with(
            filename="setup.md",
            content="content",
        )

    def test_fallback_on_chunker_error(self):
        """Returns single chunk with full content when chunker fails."""
        self.mock_chunker.chunk_document.side_effect = Exception("Chunker down")
        chunks = self.engine._chunk_file("doc.md", "Full content here", {})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["text"], "Full content here")

    def test_fallback_chunk_has_indices(self):
        """Fallback chunk includes start_index and end_index."""
        self.mock_chunker.chunk_document.side_effect = Exception("Error")
        chunks = self.engine._chunk_file("doc.md", "Hello", {})
        self.assertEqual(chunks[0]["start_index"], 0)
        self.assertEqual(chunks[0]["end_index"], 5)

    def test_passes_chunker_config(self):
        """Additional chunker kwargs are forwarded."""
        self.mock_chunker.chunk_document.return_value = []
        self.engine._chunk_file(
            "doc.md", "text",
            {"buffer_size": 3, "breakpoint_percentile_threshold": 90},
        )
        self.mock_chunker.chunk_document.assert_called_once_with(
            filename="doc.md",
            content="text",
            buffer_size=3,
            breakpoint_percentile_threshold=90,
        )


# ===================================================================
# Embedding
# ===================================================================

class TestEmbedChunks(unittest.TestCase):
    """Tests for IngestionEngine._embed_chunks()."""

    def setUp(self):
        self.mock_embedding = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=self.mock_embedding,
            seekdb_client=MagicMock(),
        )

    def test_attaches_embeddings(self):
        """Each chunk gets its embedding vector."""
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1, 0.2], [0.3, 0.4]]
        )
        chunks = [{"text": "a"}, {"text": "b"}]
        result = self.engine._embed_chunks(chunks)
        self.assertEqual(result[0]["embedding"], [0.1, 0.2])
        self.assertEqual(result[1]["embedding"], [0.3, 0.4])

    def test_failure_gives_empty_embeddings(self):
        """On embedding failure, chunks get empty lists."""
        self.mock_embedding.embed.side_effect = Exception("Model offline")
        chunks = [{"text": "a"}, {"text": "b"}]
        result = self.engine._embed_chunks(chunks)
        self.assertEqual(result[0]["embedding"], [])
        self.assertEqual(result[1]["embedding"], [])

    def test_sends_all_texts_in_single_call(self):
        """All chunk texts are sent in one embed() call."""
        self.mock_embedding.embed.return_value = MagicMock(
            embeddings=[[0.1], [0.2], [0.3]]
        )
        chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        self.engine._embed_chunks(chunks)
        self.mock_embedding.embed.assert_called_once_with(["a", "b", "c"])


# ===================================================================
# Contextualisation
# ===================================================================

class TestContextualisation(unittest.TestCase):
    """Tests for IngestionEngine._contextualise_chunks()."""

    def test_without_llm_returns_unchanged(self):
        """Without LLM client, chunks are returned as-is."""
        engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=MagicMock(),
            llm_client=None,
        )
        chunks = [{"text": "hello", "source_file": "a.md"}]
        result = engine._contextualise_chunks(chunks)
        self.assertEqual(result, chunks)

    def test_with_llm_adds_context_header(self):
        """With LLM, context_header is added to each chunk."""
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "This document covers testing."
        engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=MagicMock(),
            llm_client=mock_llm,
        )
        chunks = [
            {"text": "Test content", "source_file": "tests.md"},
        ]
        result = engine._contextualise_chunks(chunks)
        self.assertEqual(
            result[0]["context_header"],
            "This document covers testing.",
        )

    def test_groups_chunks_by_source_file(self):
        """All chunks from the same file share a context header."""
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "About file A."
        engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=MagicMock(),
            llm_client=mock_llm,
        )
        chunks = [
            {"text": "Part 1", "source_file": "a.md"},
            {"text": "Part 2", "source_file": "a.md"},
        ]
        result = engine._contextualise_chunks(chunks)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["context_header"], "About file A.")
        self.assertEqual(result[1]["context_header"], "About file A.")

    def test_llm_failure_preserves_chunks(self):
        """When LLM fails, chunks are preserved without context_header."""
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = Exception("LLM timeout")
        engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=MagicMock(),
            llm_client=mock_llm,
        )
        chunks = [
            {"text": "Content", "source_file": "doc.md"},
        ]
        result = engine._contextualise_chunks(chunks)
        self.assertEqual(len(result), 1)
        self.assertNotIn("context_header", result[0])


# ===================================================================
# Writing to SeekDB
# ===================================================================

class TestWriteChunks(unittest.TestCase):
    """Tests for IngestionEngine._write_chunks()."""

    def setUp(self):
        self.mock_seekdb = MagicMock()
        self.engine = IngestionEngine(
            chunker_client=MagicMock(),
            embedding_client=MagicMock(),
            seekdb_client=self.mock_seekdb,
        )

    def _make_chunks(self, count):
        """Create N minimal chunks for testing."""
        return [
            {"id": f"id-{i}", "text": f"text-{i}", "embedding": [0.1]}
            for i in range(count)
        ]

    def test_single_batch(self):
        """Fewer than 32 chunks written in one batch."""
        self.engine._write_chunks("coll", self._make_chunks(5))
        self.assertEqual(self.mock_seekdb.add.call_count, 1)

    def test_two_batches(self):
        """64 chunks produce exactly 2 batches."""
        self.engine._write_chunks("coll", self._make_chunks(64))
        self.assertEqual(self.mock_seekdb.add.call_count, 2)

    def test_partial_final_batch(self):
        """33 chunks produce 2 batches (32 + 1)."""
        self.engine._write_chunks("coll", self._make_chunks(33))
        self.assertEqual(self.mock_seekdb.add.call_count, 2)

    def test_empty_chunks(self):
        """No chunks means no write calls."""
        self.engine._write_chunks("coll", [])
        self.mock_seekdb.add.assert_not_called()

    def test_batch_excludes_internal_keys(self):
        """id, text, embedding are excluded from metadata."""
        chunks = [
            {
                "id": "id-1",
                "text": "content",
                "embedding": [0.1],
                "source_file": "doc.md",
                "visibility": "public",
            },
        ]
        self.engine._write_chunks("coll", chunks)
        call_kwargs = self.mock_seekdb.add.call_args
        metadatas = call_kwargs.kwargs.get("metadatas") or call_kwargs[1].get("metadatas")
        # If called positionally, check the right arg
        if metadatas is None:
            _, kwargs = self.mock_seekdb.add.call_args
            metadatas = kwargs.get("metadatas", [])
        if metadatas:
            meta = metadatas[0]
            self.assertNotIn("id", meta)
            self.assertNotIn("text", meta)
            self.assertNotIn("embedding", meta)
            self.assertIn("source_file", meta)
            self.assertIn("visibility", meta)

    def test_continues_on_write_error(self):
        """A failed batch does not prevent subsequent batches."""
        self.mock_seekdb.add.side_effect = [Exception("Write failed"), None]
        # Should not raise — logs error and continues
        self.engine._write_chunks("coll", self._make_chunks(64))
        self.assertEqual(self.mock_seekdb.add.call_count, 2)


if __name__ == "__main__":
    unittest.main()
