"""
RAG query logic for knowledge-service.

Implements the CRAG (Corrective RAG) pipeline:
  1. Optional query rewriting via LLM
  2. Hybrid search against SeekDB (vector + keyword)
  3. Optional reranking
  4. Return ranked, relevant chunks
"""

import logging
from typing import Any, Dict, List, Optional

from source.embedding_client import EmbeddingClient
from source.llm_client import LLMClient
from source.seekdb_client import SeekDBClient

logger = logging.getLogger(__name__)


# ===================================================================
# Query Engine
# ===================================================================

class QueryEngine:
    """Orchestrates RAG queries against SeekDB.

    Usage::

        engine = QueryEngine(seekdb, embedding, llm)
        results = engine.query(
            question="How do I configure the API?",
            collections=["my_docs"],
            n_results=5,
            rerank=True,
        )
    """

    def __init__(
        self,
        seekdb_client: SeekDBClient,
        embedding_client: EmbeddingClient,
        llm_client: Optional[LLMClient] = None,
    ):
        self.seekdb = seekdb_client
        self.embedding = embedding_client
        self.llm = llm_client

    def query(
        self,
        question: str,
        collections: Optional[List[str]] = None,
        n_results: int = 5,
        rerank: bool = False,
        rewrite: bool = False,
    ) -> Dict[str, Any]:
        """Query the knowledge base.

        Args:
            question: Natural-language query.
            collections: Target collections (all if None).
            n_results: Number of results to return.
            rerank: Apply reranker to results.
            rewrite: Rewrite query via LLM.

        Returns:
            Dict with keys: results, rewritten_query, total_results.
        """
        logger.info(f"Query: '{question}' (collections: {collections}, n_results: {n_results})")

        # Query rewriting (optional)
        rewritten_query = None
        search_query = question

        if rewrite and self.llm:
            try:
                rewritten_query = self._rewrite_query(question)
                search_query = rewritten_query
                logger.info(f"Rewritten query: '{rewritten_query}'")
            except Exception as exc:
                logger.warning(f"Query rewriting failed: {exc}")

        # Determine target collections
        if not collections:
            try:
                all_collections = self.seekdb.list_collections()
                collections = [c["name"] for c in all_collections.get("collections", [])]
            except Exception:
                collections = []

        # Search each collection
        all_results = []
        for collection in collections:
            try:
                results = self._search_collection(collection, search_query, n_results)
                all_results.extend(results)
            except Exception as exc:
                logger.warning(f"Search failed for collection '{collection}': {exc}")

        # Sort by distance (lower = better)
        all_results.sort(key=lambda r: r["distance"])

        # Take top-N
        all_results = all_results[:n_results]

        # Rerank (optional)
        if rerank and self.embedding.config.reranker_base_url:
            try:
                all_results = self._rerank_results(search_query, all_results)
            except Exception as exc:
                logger.warning(f"Reranking failed: {exc}")

        logger.info(f"Query complete: {len(all_results)} results")

        return {
            "results": all_results,
            "rewritten_query": rewritten_query,
            "total_results": len(all_results),
        }

    def _rewrite_query(self, question: str) -> str:
        """Rewrite query via LLM for better retrieval."""
        if not self.llm:
            return question

        prompt = (
            "Rewrite the following query to improve semantic search retrieval. "
            "Expand abbreviations, add relevant keywords, preserve intent. "
            "Return ONLY the rewritten query, no explanation."
        )

        try:
            rewritten = self.llm.ask(question, prompt)
            return rewritten.strip()
        except Exception:
            return question

    def _search_collection(
        self,
        collection: str,
        query: str,
        n_results: int,
    ) -> List[Dict[str, Any]]:
        """Search a single collection via hybrid search."""
        try:
            response = self.seekdb.hybrid_search(
                collection=collection,
                query_texts=[query],  # Vector component
                query_text=query,  # Keyword component
                n_results=n_results,
            )

            # Transform to result format
            results = []
            ids = response.get("ids", [[]])[0]
            documents = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0]

            for i in range(len(ids)):
                results.append({
                    "collection": collection,
                    "document": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "distance": distances[i] if i < len(distances) else 1.0,
                    "score": None,
                })

            return results

        except Exception as exc:
            logger.error(f"Search failed for collection '{collection}': {exc}")
            return []

    def _rerank_results(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank results via the reranker endpoint."""
        if not results:
            return results

        candidates = [r["document"] for r in results]

        try:
            rerank_result = self.embedding.rerank(query, candidates)

            # Attach scores and re-sort
            for result, score in zip(results, rerank_result.scores):
                result["score"] = score

            results.sort(key=lambda r: r.get("score", 0.0), reverse=True)

        except Exception as exc:
            logger.warning(f"Reranking failed: {exc}")

        return results
