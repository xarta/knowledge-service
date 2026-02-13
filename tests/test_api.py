"""Unit tests for knowledge-service FastAPI endpoints."""

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient


def _make_client():
    """Create a TestClient with all external dependencies mocked.

    Patches config and client classes so the startup event runs
    without requiring real network connections or environment variables.
    """
    with patch("app.SeekDBConfig"), \
         patch("app.SeekDBClient"), \
         patch("app.EmbeddingConfig"), \
         patch("app.EmbeddingClient"), \
         patch("app.ChunkerConfig"), \
         patch("app.ChunkerClient"), \
         patch("app.LLMConfig"), \
         patch("app.LLMClient"):
        from app import app
        return TestClient(app)


# ===================================================================
# GET /
# ===================================================================

class TestRootEndpoint(unittest.TestCase):
    """Tests for GET /."""

    def setUp(self):
        self.client = _make_client()

    def test_returns_service_info(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["service"], "knowledge-service")
        self.assertEqual(data["version"], "1.0.0")
        self.assertEqual(data["status"], "running")

    def test_includes_description(self):
        resp = self.client.get("/")
        data = resp.json()
        self.assertIn("description", data)
        self.assertIsInstance(data["description"], str)


# ===================================================================
# GET /health
# ===================================================================

class TestHealthEndpoint(unittest.TestCase):
    """Tests for GET /health."""

    def setUp(self):
        self.client = _make_client()

        # Set up properly-structured mock clients for health checks
        import app as app_module

        app_module.seekdb_client = MagicMock()
        app_module.seekdb_client.test_connection.return_value = {
            "connected": True,
            "endpoint": "http://test-host",
            "error": None,
        }

        app_module.embedding_client = MagicMock()
        app_module.embedding_client.test_connection.return_value = {
            "embedding_connected": True,
            "reranker_connected": False,
            "embedding_model": "test-embed-model",
            "reranker_model": None,
            "errors": [],
        }
        app_module.embedding_client.config.reranker_base_url = ""

        app_module.chunker_client = MagicMock()
        app_module.chunker_client.test_connection.return_value = {
            "connected": True,
            "endpoint": "http://test-host",
            "error": None,
        }

        # LLM not configured
        app_module.llm_client = None

    def test_returns_status_and_dependencies(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("dependencies", data)

    def test_healthy_when_all_connected(self):
        resp = self.client.get("/health")
        data = resp.json()
        self.assertEqual(data["status"], "healthy")

    def test_includes_seekdb(self):
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("seekdb", data["dependencies"])
        self.assertTrue(data["dependencies"]["seekdb"]["connected"])

    def test_includes_embedding(self):
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("embedding", data["dependencies"])
        self.assertTrue(data["dependencies"]["embedding"]["connected"])

    def test_includes_chunker(self):
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("chunker", data["dependencies"])
        self.assertTrue(data["dependencies"]["chunker"]["connected"])

    def test_llm_not_configured(self):
        resp = self.client.get("/health")
        data = resp.json()
        self.assertIn("llm", data["dependencies"])
        self.assertEqual(data["dependencies"]["llm"]["status"], "not_configured")

    def test_degraded_when_seekdb_disconnected(self):
        import app as app_module
        app_module.seekdb_client.test_connection.return_value = {
            "connected": False,
            "endpoint": "http://test-host",
            "error": "Connection refused",
        }
        resp = self.client.get("/health")
        data = resp.json()
        self.assertEqual(data["status"], "degraded")


# ===================================================================
# POST /ingest
# ===================================================================

class TestIngestEndpoint(unittest.TestCase):
    """Tests for POST /ingest."""

    def setUp(self):
        self.client = _make_client()

    def test_requires_files(self):
        """Missing 'files' field triggers validation error."""
        resp = self.client.post("/ingest", json={})
        self.assertEqual(resp.status_code, 422)

    def test_with_valid_payload(self):
        """Ingest with mocked engine returns expected shape."""
        import app as app_module
        app_module.ingestion_engine = MagicMock()
        app_module.ingestion_engine.ingest.return_value = {
            "collection": "test_docs",
            "visibility": "public",
            "chunks_ingested": 3,
            "files_processed": 1,
            "duration_seconds": 0.5,
        }

        resp = self.client.post("/ingest", json={
            "files": {"doc.md": "# Test\n\nContent here."},
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["collection"], "test_docs")
        self.assertEqual(data["chunks_ingested"], 3)
        self.assertEqual(data["files_processed"], 1)

    def test_with_collection_name(self):
        """Explicit collection_name is forwarded to engine."""
        import app as app_module
        mock_engine = MagicMock()
        mock_engine.ingest.return_value = {
            "collection": "my_collection",
            "visibility": "public",
            "chunks_ingested": 2,
            "files_processed": 1,
            "duration_seconds": 0.3,
        }
        app_module.ingestion_engine = mock_engine

        resp = self.client.post("/ingest", json={
            "files": {"readme.md": "Hello world"},
            "collection_name": "my_collection",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["collection"], "my_collection")

    def test_with_visibility_private(self):
        """Private visibility is forwarded to engine."""
        import app as app_module
        mock_engine = MagicMock()
        mock_engine.ingest.return_value = {
            "collection": "docs",
            "visibility": "private",
            "chunks_ingested": 1,
            "files_processed": 1,
            "duration_seconds": 0.1,
        }
        app_module.ingestion_engine = mock_engine

        resp = self.client.post("/ingest", json={
            "files": {"doc.md": "Content"},
            "visibility": "private",
        })
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["visibility"], "private")

    def test_engine_seekdb_error_returns_500(self):
        """SeekDBError from engine produces 500."""
        import app as app_module
        from source.seekdb_client import SeekDBError
        mock_engine = MagicMock()
        mock_engine.ingest.side_effect = SeekDBError("Connection refused")
        app_module.ingestion_engine = mock_engine

        resp = self.client.post("/ingest", json={
            "files": {"doc.md": "Content"},
        })
        self.assertEqual(resp.status_code, 500)

    def test_engine_embedding_error_returns_500(self):
        """EmbeddingError from engine produces 500."""
        import app as app_module
        from source.embedding_client import EmbeddingError
        mock_engine = MagicMock()
        mock_engine.ingest.side_effect = EmbeddingError("Model offline")
        app_module.ingestion_engine = mock_engine

        resp = self.client.post("/ingest", json={
            "files": {"doc.md": "Content"},
        })
        self.assertEqual(resp.status_code, 500)

    def test_engine_chunker_error_returns_500(self):
        """ChunkerError from engine produces 500."""
        import app as app_module
        from source.chunker_client import ChunkerError
        mock_engine = MagicMock()
        mock_engine.ingest.side_effect = ChunkerError("Chunker unreachable")
        app_module.ingestion_engine = mock_engine

        resp = self.client.post("/ingest", json={
            "files": {"doc.md": "Content"},
        })
        self.assertEqual(resp.status_code, 500)


# ===================================================================
# POST /query
# ===================================================================

class TestQueryEndpoint(unittest.TestCase):
    """Tests for POST /query."""

    def setUp(self):
        self.client = _make_client()

    def test_requires_question(self):
        """Missing 'question' field triggers validation error."""
        resp = self.client.post("/query", json={})
        self.assertEqual(resp.status_code, 422)

    def test_with_valid_question(self):
        """Query with mocked engine returns expected shape."""
        import app as app_module
        mock_engine = MagicMock()
        mock_engine.query.return_value = {
            "results": [
                {
                    "collection": "docs",
                    "document": "How to configure the service.",
                    "metadata": {"source_file": "readme.md"},
                    "distance": 0.15,
                    "score": None,
                },
            ],
            "rewritten_query": None,
            "total_results": 1,
        }
        app_module.query_engine = mock_engine

        resp = self.client.post("/query", json={
            "question": "How do I configure the service?",
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_results"], 1)
        self.assertEqual(len(data["results"]), 1)
        self.assertEqual(data["results"][0]["collection"], "docs")

    def test_with_rewrite_flag(self):
        """Query with rewrite=True returns rewritten_query."""
        import app as app_module
        mock_engine = MagicMock()
        mock_engine.query.return_value = {
            "results": [],
            "rewritten_query": "configuration setup guide parameters",
            "total_results": 0,
        }
        app_module.query_engine = mock_engine

        resp = self.client.post("/query", json={
            "question": "How to config?",
            "rewrite": True,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["rewritten_query"], "configuration setup guide parameters")

    def test_with_explicit_collections(self):
        """Target collections are forwarded to engine."""
        import app as app_module
        mock_engine = MagicMock()
        mock_engine.query.return_value = {
            "results": [],
            "rewritten_query": None,
            "total_results": 0,
        }
        app_module.query_engine = mock_engine

        resp = self.client.post("/query", json={
            "question": "test",
            "collections": ["project_docs", "guides"],
        })
        self.assertEqual(resp.status_code, 200)
        # Verify collections were passed through
        call_kwargs = mock_engine.query.call_args
        self.assertEqual(call_kwargs.kwargs.get("collections"), ["project_docs", "guides"])

    def test_engine_error_returns_500(self):
        """Engine exceptions produce 500 responses."""
        import app as app_module
        from source.seekdb_client import SeekDBError
        mock_engine = MagicMock()
        mock_engine.query.side_effect = SeekDBError("Database offline")
        app_module.query_engine = mock_engine

        resp = self.client.post("/query", json={
            "question": "test query",
        })
        self.assertEqual(resp.status_code, 500)


# ===================================================================
# GET /collections and DELETE /collections/{name}
# ===================================================================

class TestCollectionsEndpoint(unittest.TestCase):
    """Tests for GET /collections and DELETE /collections/{name}."""

    def setUp(self):
        self.client = _make_client()

    def test_list_collections(self):
        """GET /collections returns expected shape."""
        import app as app_module
        mock_seekdb = MagicMock()
        mock_seekdb.list_collections.return_value = {
            "collections": [{"name": "docs"}],
        }
        mock_seekdb.count.return_value = {"count": 42}
        mock_seekdb.get.return_value = {
            "metadatas": [{"visibility": "public"}],
        }
        app_module.seekdb_client = mock_seekdb

        resp = self.client.get("/collections")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("collections", data)
        self.assertEqual(len(data["collections"]), 1)
        self.assertEqual(data["collections"][0]["name"], "docs")
        self.assertEqual(data["collections"][0]["count"], 42)
        self.assertEqual(data["collections"][0]["visibility"], "public")

    def test_list_empty_collections(self):
        """GET /collections with no collections returns empty list."""
        import app as app_module
        mock_seekdb = MagicMock()
        mock_seekdb.list_collections.return_value = {"collections": []}
        app_module.seekdb_client = mock_seekdb

        resp = self.client.get("/collections")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["collections"], [])

    def test_delete_collection(self):
        """DELETE /collections/{name} returns 204."""
        import app as app_module
        mock_seekdb = MagicMock()
        mock_seekdb.delete_collection.return_value = {}
        app_module.seekdb_client = mock_seekdb

        resp = self.client.delete("/collections/test_docs")
        self.assertEqual(resp.status_code, 204)

    def test_delete_collection_error_returns_500(self):
        """SeekDBError on delete produces 500."""
        import app as app_module
        from source.seekdb_client import SeekDBError
        mock_seekdb = MagicMock()
        mock_seekdb.delete_collection.side_effect = SeekDBError("Not found")
        app_module.seekdb_client = mock_seekdb

        resp = self.client.delete("/collections/nonexistent")
        self.assertEqual(resp.status_code, 500)

    def test_list_collections_error_returns_500(self):
        """SeekDBError on list produces 500."""
        import app as app_module
        from source.seekdb_client import SeekDBError
        mock_seekdb = MagicMock()
        mock_seekdb.list_collections.side_effect = SeekDBError("Connection lost")
        app_module.seekdb_client = mock_seekdb

        resp = self.client.get("/collections")
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
