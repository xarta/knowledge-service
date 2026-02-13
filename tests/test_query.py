"""Unit tests for the query engine, helper functions, and configuration.

Covers:
  - QueryEngine query, rewrite, search, and rerank logic
  - Pure helper functions (cosine_similarity, strip_think_tags, etc.)
  - Config from_env validation for all client configs
  - Error class attributes
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from source.query import QueryEngine


# ===================================================================
# QueryEngine.query()
# ===================================================================

class TestQueryResults(unittest.TestCase):
    """Tests for QueryEngine.query() result shape and content."""

    def setUp(self):
        self.mock_seekdb = MagicMock()
        self.mock_embedding = MagicMock()
        self.mock_embedding.config = MagicMock()
        self.mock_embedding.config.reranker_base_url = ""
        self.engine = QueryEngine(
            seekdb_client=self.mock_seekdb,
            embedding_client=self.mock_embedding,
        )

    def test_returns_expected_keys(self):
        """Result includes results, rewritten_query, total_results."""
        self.mock_seekdb.list_collections.return_value = {"collections": []}
        result = self.engine.query(question="test")
        self.assertIn("results", result)
        self.assertIn("rewritten_query", result)
        self.assertIn("total_results", result)

    def test_no_rewrite_by_default(self):
        """rewritten_query is None when rewrite=False."""
        self.mock_seekdb.list_collections.return_value = {"collections": []}
        result = self.engine.query(question="test")
        self.assertIsNone(result["rewritten_query"])

    def test_empty_collections_gives_empty_results(self):
        """No collections means no results."""
        self.mock_seekdb.list_collections.return_value = {"collections": []}
        result = self.engine.query(question="anything")
        self.assertEqual(result["total_results"], 0)
        self.assertEqual(result["results"], [])

    def test_returns_search_results(self):
        """Hybrid search results are returned."""
        self.mock_seekdb.list_collections.return_value = {
            "collections": [{"name": "docs"}],
        }
        self.mock_seekdb.hybrid_search.return_value = {
            "ids": [["id-1"]],
            "documents": [["Some relevant text"]],
            "metadatas": [[{"source_file": "guide.md"}]],
            "distances": [[0.15]],
        }
        result = self.engine.query(question="How to setup?")
        self.assertEqual(result["total_results"], 1)
        self.assertEqual(result["results"][0]["document"], "Some relevant text")
        self.assertEqual(result["results"][0]["collection"], "docs")

    def test_explicit_collections_skips_listing(self):
        """Providing collections avoids list_collections call."""
        self.mock_seekdb.hybrid_search.return_value = {
            "ids": [["id-1"]],
            "documents": [["Text"]],
            "metadatas": [[{}]],
            "distances": [[0.3]],
        }
        result = self.engine.query(
            question="test",
            collections=["my_collection"],
        )
        self.mock_seekdb.list_collections.assert_not_called()
        self.assertEqual(result["total_results"], 1)

    def test_limits_to_n_results(self):
        """Results are capped at n_results."""
        self.mock_seekdb.list_collections.return_value = {
            "collections": [{"name": "docs"}],
        }
        self.mock_seekdb.hybrid_search.return_value = {
            "ids": [["id-1", "id-2", "id-3"]],
            "documents": [["A", "B", "C"]],
            "metadatas": [[{}, {}, {}]],
            "distances": [[0.1, 0.2, 0.3]],
        }
        result = self.engine.query(question="test", n_results=2)
        self.assertEqual(result["total_results"], 2)

    def test_results_sorted_by_distance(self):
        """Results are sorted ascending by distance."""
        self.mock_seekdb.list_collections.return_value = {
            "collections": [{"name": "docs"}],
        }
        self.mock_seekdb.hybrid_search.return_value = {
            "ids": [["id-1", "id-2"]],
            "documents": [["Far", "Near"]],
            "metadatas": [[{}, {}]],
            "distances": [[0.9, 0.1]],
        }
        result = self.engine.query(question="test")
        self.assertLessEqual(
            result["results"][0]["distance"],
            result["results"][1]["distance"],
        )


# ===================================================================
# Query Rewriting
# ===================================================================

class TestQueryRewriting(unittest.TestCase):
    """Tests for QueryEngine._rewrite_query()."""

    def test_calls_llm(self):
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "expanded search terms"
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=MagicMock(),
            llm_client=mock_llm,
        )
        result = engine._rewrite_query("How to config?")
        self.assertEqual(result, "expanded search terms")
        mock_llm.ask.assert_called_once()

    def test_without_llm_returns_original(self):
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=MagicMock(),
            llm_client=None,
        )
        result = engine._rewrite_query("original query")
        self.assertEqual(result, "original query")

    def test_failure_returns_original(self):
        mock_llm = MagicMock()
        mock_llm.ask.side_effect = Exception("LLM offline")
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=MagicMock(),
            llm_client=mock_llm,
        )
        result = engine._rewrite_query("test query")
        self.assertEqual(result, "test query")

    def test_strips_whitespace(self):
        mock_llm = MagicMock()
        mock_llm.ask.return_value = "  trimmed result  "
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=MagicMock(),
            llm_client=mock_llm,
        )
        result = engine._rewrite_query("test")
        self.assertEqual(result, "trimmed result")


# ===================================================================
# Collection Search
# ===================================================================

class TestSearchCollection(unittest.TestCase):
    """Tests for QueryEngine._search_collection()."""

    def test_returns_formatted_results(self):
        mock_seekdb = MagicMock()
        mock_seekdb.hybrid_search.return_value = {
            "ids": [["id-1", "id-2"]],
            "documents": [["Doc A", "Doc B"]],
            "metadatas": [[{"source_file": "a.md"}, {"source_file": "b.md"}]],
            "distances": [[0.1, 0.3]],
        }
        engine = QueryEngine(
            seekdb_client=mock_seekdb,
            embedding_client=MagicMock(),
        )
        results = engine._search_collection("docs", "test", 5)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["collection"], "docs")
        self.assertEqual(results[0]["document"], "Doc A")
        self.assertIsNone(results[0]["score"])

    def test_returns_empty_on_error(self):
        mock_seekdb = MagicMock()
        mock_seekdb.hybrid_search.side_effect = Exception("Connection refused")
        engine = QueryEngine(
            seekdb_client=mock_seekdb,
            embedding_client=MagicMock(),
        )
        results = engine._search_collection("docs", "test", 5)
        self.assertEqual(results, [])

    def test_includes_metadata(self):
        mock_seekdb = MagicMock()
        mock_seekdb.hybrid_search.return_value = {
            "ids": [["id-1"]],
            "documents": [["Text"]],
            "metadatas": [[{"source_file": "doc.md", "visibility": "public"}]],
            "distances": [[0.2]],
        }
        engine = QueryEngine(
            seekdb_client=mock_seekdb,
            embedding_client=MagicMock(),
        )
        results = engine._search_collection("coll", "query", 5)
        self.assertEqual(results[0]["metadata"]["source_file"], "doc.md")

    def test_handles_empty_response(self):
        mock_seekdb = MagicMock()
        mock_seekdb.hybrid_search.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        engine = QueryEngine(
            seekdb_client=mock_seekdb,
            embedding_client=MagicMock(),
        )
        results = engine._search_collection("coll", "query", 5)
        self.assertEqual(results, [])


# ===================================================================
# Reranking
# ===================================================================

class TestReranking(unittest.TestCase):
    """Tests for QueryEngine._rerank_results()."""

    def test_attaches_scores(self):
        mock_embedding = MagicMock()
        mock_embedding.config.reranker_base_url = "http://test-host/v1"
        mock_embedding.rerank.return_value = MagicMock(scores=[0.9, 0.3])
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=mock_embedding,
        )
        results = [
            {"document": "A", "distance": 0.1, "score": None},
            {"document": "B", "distance": 0.2, "score": None},
        ]
        reranked = engine._rerank_results("query", results)
        self.assertEqual(reranked[0]["score"], 0.9)

    def test_sorts_by_score_descending(self):
        mock_embedding = MagicMock()
        mock_embedding.config.reranker_base_url = "http://test-host/v1"
        mock_embedding.rerank.return_value = MagicMock(scores=[0.2, 0.8])
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=mock_embedding,
        )
        results = [
            {"document": "Low", "distance": 0.1, "score": None},
            {"document": "High", "distance": 0.2, "score": None},
        ]
        reranked = engine._rerank_results("query", results)
        self.assertEqual(reranked[0]["document"], "High")

    def test_empty_results(self):
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=MagicMock(),
        )
        result = engine._rerank_results("query", [])
        self.assertEqual(result, [])

    def test_failure_preserves_results(self):
        mock_embedding = MagicMock()
        mock_embedding.config.reranker_base_url = "http://test-host/v1"
        mock_embedding.rerank.side_effect = Exception("Reranker offline")
        engine = QueryEngine(
            seekdb_client=MagicMock(),
            embedding_client=mock_embedding,
        )
        results = [
            {"document": "A", "distance": 0.1, "score": None},
        ]
        reranked = engine._rerank_results("query", results)
        self.assertEqual(len(reranked), 1)


# ===================================================================
# Pure Helper Functions
# ===================================================================

class TestStripThinkTags(unittest.TestCase):
    """Tests for strip_think_tags()."""

    def test_removes_think_block(self):
        from source.llm_client import strip_think_tags
        text = "before<think>internal thought</think>after"
        self.assertEqual(strip_think_tags(text), "beforeafter")

    def test_multiline_think_block(self):
        from source.llm_client import strip_think_tags
        text = "start<think>\nthinking\nmore lines\n</think>end"
        self.assertEqual(strip_think_tags(text), "startend")

    def test_no_tags(self):
        from source.llm_client import strip_think_tags
        self.assertEqual(strip_think_tags("plain text"), "plain text")

    def test_multiple_think_blocks(self):
        from source.llm_client import strip_think_tags
        text = "a<think>x</think>b<think>y</think>c"
        self.assertEqual(strip_think_tags(text), "abc")

    def test_empty_string(self):
        from source.llm_client import strip_think_tags
        self.assertEqual(strip_think_tags(""), "")


class TestStripMarkdownFences(unittest.TestCase):
    """Tests for strip_markdown_fences()."""

    def test_strips_json_fence(self):
        from source.llm_client import strip_markdown_fences
        text = '```json\n{"key": "value"}\n```'
        self.assertEqual(strip_markdown_fences(text), '{"key": "value"}')

    def test_strips_plain_fence(self):
        from source.llm_client import strip_markdown_fences
        text = "```\nplain content\n```"
        self.assertEqual(strip_markdown_fences(text), "plain content")

    def test_no_fences(self):
        from source.llm_client import strip_markdown_fences
        self.assertEqual(strip_markdown_fences("normal text"), "normal text")

    def test_inline_backticks_preserved(self):
        from source.llm_client import strip_markdown_fences
        text = "Use `code` inline"
        self.assertEqual(strip_markdown_fences(text), "Use `code` inline")


class TestCosineSimilarity(unittest.TestCase):
    """Tests for cosine_similarity()."""

    def test_identical_vectors(self):
        from source.embedding_client import cosine_similarity
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0,
        )

    def test_orthogonal_vectors(self):
        from source.embedding_client import cosine_similarity
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0,
        )

    def test_opposite_vectors(self):
        from source.embedding_client import cosine_similarity
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [-1.0, 0.0]), -1.0,
        )

    def test_empty_vectors(self):
        from source.embedding_client import cosine_similarity
        self.assertEqual(cosine_similarity([], []), 0.0)

    def test_mismatched_lengths(self):
        from source.embedding_client import cosine_similarity
        self.assertEqual(cosine_similarity([1.0], [1.0, 0.0]), 0.0)

    def test_zero_vector(self):
        from source.embedding_client import cosine_similarity
        self.assertEqual(cosine_similarity([0.0, 0.0], [1.0, 0.0]), 0.0)


# ===================================================================
# Config from_env Validation
# ===================================================================

class TestSeekDBConfigFromEnv(unittest.TestCase):
    """Tests for SeekDBConfig.from_env()."""

    def test_missing_base_url_raises(self):
        from source.seekdb_client import SeekDBConfig
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                SeekDBConfig.from_env()

    def test_missing_api_key_raises(self):
        from source.seekdb_client import SeekDBConfig
        with patch.dict(os.environ, {"SEEKDB_BASE_URL": "http://test-host"}, clear=True):
            with self.assertRaises(ValueError):
                SeekDBConfig.from_env()

    def test_missing_database_raises(self):
        from source.seekdb_client import SeekDBConfig
        env = {
            "SEEKDB_BASE_URL": "http://test-host",
            "SEEKDB_API_KEY": "test-key",
        }
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaises(ValueError):
                SeekDBConfig.from_env()

    def test_valid_config(self):
        from source.seekdb_client import SeekDBConfig
        env = {
            "SEEKDB_BASE_URL": "http://test-host",
            "SEEKDB_API_KEY": "test-key",
            "SEEKDB_DATABASE": "test-db",
        }
        with patch.dict(os.environ, env, clear=True):
            config = SeekDBConfig.from_env()
            self.assertEqual(config.base_url, "http://test-host")
            self.assertEqual(config.database, "test-db")


class TestEmbeddingConfigFromEnv(unittest.TestCase):
    """Tests for EmbeddingConfig.from_env()."""

    def test_missing_base_url_raises(self):
        from source.embedding_client import EmbeddingConfig
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                EmbeddingConfig.from_env()

    def test_valid_config(self):
        from source.embedding_client import EmbeddingConfig
        env = {"EMBEDDING_BASE_URL": "http://test-host/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = EmbeddingConfig.from_env()
            self.assertEqual(config.embedding_base_url, "http://test-host/v1")

    def test_optional_reranker(self):
        from source.embedding_client import EmbeddingConfig
        env = {"EMBEDDING_BASE_URL": "http://test-host/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = EmbeddingConfig.from_env()
            self.assertEqual(config.reranker_base_url, "")


class TestChunkerConfigFromEnv(unittest.TestCase):
    """Tests for ChunkerConfig.from_env()."""

    def test_missing_url_raises(self):
        from source.chunker_client import ChunkerConfig
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                ChunkerConfig.from_env()

    def test_valid_config(self):
        from source.chunker_client import ChunkerConfig
        env = {"CHUNKER_URL": "http://test-host"}
        with patch.dict(os.environ, env, clear=True):
            config = ChunkerConfig.from_env()
            self.assertEqual(config.base_url, "http://test-host")


class TestLLMConfigFromEnv(unittest.TestCase):
    """Tests for LLMConfig.from_env()."""

    def test_returns_none_when_not_configured(self):
        from source.llm_client import LLMConfig
        with patch.dict(os.environ, {}, clear=True):
            config = LLMConfig.from_env()
            self.assertIsNone(config)

    def test_returns_config_when_set(self):
        from source.llm_client import LLMConfig
        env = {"VLLM_BASE_URL": "http://test-host/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
            self.assertIsNotNone(config)
            self.assertEqual(config.base_url, "http://test-host/v1")

    def test_optional_api_key(self):
        from source.llm_client import LLMConfig
        env = {"VLLM_BASE_URL": "http://test-host/v1"}
        with patch.dict(os.environ, env, clear=True):
            config = LLMConfig.from_env()
            self.assertIsNone(config.api_key)


# ===================================================================
# Error Classes
# ===================================================================

class TestErrorClasses(unittest.TestCase):
    """Tests for custom error class attributes."""

    def test_seekdb_error_attributes(self):
        from source.seekdb_client import SeekDBError
        err = SeekDBError("test error", status_code=500, response_body='{"detail":"fail"}')
        self.assertEqual(str(err), "test error")
        self.assertEqual(err.status_code, 500)
        self.assertEqual(err.response_body, '{"detail":"fail"}')

    def test_embedding_error_attributes(self):
        from source.embedding_client import EmbeddingError
        err = EmbeddingError("embed fail", status_code=429)
        self.assertEqual(str(err), "embed fail")
        self.assertEqual(err.status_code, 429)
        self.assertIsNone(err.response_body)

    def test_chunker_error_attributes(self):
        from source.chunker_client import ChunkerError
        err = ChunkerError("chunk fail")
        self.assertEqual(str(err), "chunk fail")
        self.assertIsNone(err.status_code)

    def test_llm_error_attributes(self):
        from source.llm_client import LLMError
        err = LLMError("llm fail", status_code=503, response_body="unavailable")
        self.assertEqual(err.status_code, 503)
        self.assertEqual(err.response_body, "unavailable")


if __name__ == "__main__":
    unittest.main()
