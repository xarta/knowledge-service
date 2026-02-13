"""
Embedding and reranker client for knowledge-service.

OpenAI-compatible embedding and reranker client. Zero external dependencies —
uses only ``urllib.request`` and ``json`` from the standard library.

Features:
  - ``embed(texts)`` for batch embedding requests
  - ``rerank(query, candidates)`` for pairwise reranker scoring
  - ``cosine_similarity(a, b)`` for manual vector comparison
  - Auto model detection from ``/v1/models``
  - Retry with exponential backoff (429) and flat retry (5xx)
  - Environment variable configuration
"""

import json
import logging
import math
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===================================================================
# Pure helper functions
# ===================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculate cosine similarity between two vectors.

    Pure computation — no numpy or external dependencies.

    Args:
        vec_a: First vector.
        vec_b: Second vector.

    Returns:
        Similarity score between -1 and 1. Returns 0.0 for empty or
        mismatched vectors.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ===================================================================
# EmbeddingError
# ===================================================================

class EmbeddingError(Exception):
    """Embedding / reranker API error with optional HTTP status and body.

    Attributes:
        status_code: HTTP status code (None for non-HTTP errors).
        response_body: Raw response body from the endpoint.
    """

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_body: Optional[str] = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


# ===================================================================
# Data classes
# ===================================================================

@dataclass
class EmbeddingResult:
    """Result from a batch embedding request.

    Attributes:
        embeddings: List of embedding vectors, one per input text.
        model: Model name that produced the embeddings.
        usage: Token usage dictionary from the API.
    """
    embeddings: List[List[float]]
    model: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankResult:
    """Result from a reranking request.

    Attributes:
        scores: Relevance scores for each candidate (same order as input).
        model: Model name that produced the scores.
    """
    scores: List[float]
    model: str = ""


# ===================================================================
# EmbeddingConfig
# ===================================================================

@dataclass
class EmbeddingConfig:
    """Configuration for embedding and reranker endpoints.

    Attributes:
        embedding_base_url: Base URL for embedding endpoint (e.g. ``http://host:8000/v1``).
        embedding_api_key: Bearer token for embedding endpoint.
        embedding_model: Model name. None triggers auto-detection.
        reranker_base_url: Base URL for reranker endpoint (e.g. ``http://host:8001/v1``).
        reranker_api_key: Bearer token for reranker endpoint.
        reranker_model: Model name. None triggers auto-detection.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    embedding_base_url: str = ""
    embedding_api_key: Optional[str] = None
    embedding_model: Optional[str] = None
    reranker_base_url: str = ""
    reranker_api_key: Optional[str] = None
    reranker_model: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Build EmbeddingConfig from environment variables.

        Requires:
            EMBEDDING_BASE_URL: Embedding endpoint
            EMBEDDING_API_KEY: Embedding Bearer token

        Optional:
            EMBEDDING_MODEL: Model name (auto-detect if omitted)
            RERANKER_BASE_URL: Reranker endpoint
            RERANKER_API_KEY: Reranker Bearer token
            RERANKER_MODEL: Model name (auto-detect if omitted)

        Returns:
            Configured EmbeddingConfig.

        Raises:
            ValueError: If required env vars are missing.
        """
        embedding_base_url = os.getenv("EMBEDDING_BASE_URL", "").strip()
        embedding_api_key = os.getenv("EMBEDDING_API_KEY", "").strip() or None
        embedding_model = os.getenv("EMBEDDING_MODEL", "").strip() or None

        if not embedding_base_url:
            raise ValueError("EMBEDDING_BASE_URL environment variable is required")

        reranker_base_url = os.getenv("RERANKER_BASE_URL", "").strip()
        reranker_api_key = os.getenv("RERANKER_API_KEY", "").strip() or None
        reranker_model = os.getenv("RERANKER_MODEL", "").strip() or None

        return cls(
            embedding_base_url=embedding_base_url,
            embedding_api_key=embedding_api_key,
            embedding_model=embedding_model,
            reranker_base_url=reranker_base_url,
            reranker_api_key=reranker_api_key,
            reranker_model=reranker_model,
        )


# ===================================================================
# EmbeddingClient
# ===================================================================

class EmbeddingClient:
    """Client for OpenAI-compatible embedding and reranker endpoints.

    Zero external dependencies — uses ``urllib.request`` + ``json``.

    Usage::

        cfg = EmbeddingConfig.from_env()
        client = EmbeddingClient(cfg)
        result = client.embed(["hello world", "test"])
        sim = cosine_similarity(result.embeddings[0], result.embeddings[1])
    """

    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._detected_embedding_model: Optional[str] = None
        self._detected_reranker_model: Optional[str] = None

    @property
    def config(self) -> EmbeddingConfig:
        """Current configuration."""
        return self._config

    @property
    def embedding_model(self) -> str:
        """Embedding model name — auto-detected if not configured.

        Raises:
            EmbeddingError: If auto-detection fails.
        """
        if self._config.embedding_model:
            return self._config.embedding_model

        if self._detected_embedding_model:
            return self._detected_embedding_model

        self._detected_embedding_model = self._detect_model(
            self._config.embedding_base_url,
            self._config.embedding_api_key,
        )
        logger.info(f"Auto-detected embedding model: {self._detected_embedding_model}")
        return self._detected_embedding_model

    @property
    def reranker_model(self) -> str:
        """Reranker model name — auto-detected if not configured.

        Raises:
            EmbeddingError: If auto-detection fails.
        """
        if self._config.reranker_model:
            return self._config.reranker_model

        if self._detected_reranker_model:
            return self._detected_reranker_model

        self._detected_reranker_model = self._detect_model(
            self._config.reranker_base_url,
            self._config.reranker_api_key,
        )
        logger.info(f"Auto-detected reranker model: {self._detected_reranker_model}")
        return self._detected_reranker_model

    def embed(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for a list of texts.

        Sends a single batch request to the embedding endpoint.
        Response embeddings are re-ordered by ``index`` to match
        the input order.

        Args:
            texts: List of strings to embed.

        Returns:
            EmbeddingResult with one embedding per input text.

        Raises:
            EmbeddingError: On API errors after retries exhausted.
        """
        if not texts:
            return EmbeddingResult(embeddings=[], model="", usage={})

        payload = {
            "input": texts,
            "model": self.embedding_model,
        }

        raw = self._request(
            base_url=self._config.embedding_base_url,
            path="/embeddings",
            payload=payload,
            api_key=self._config.embedding_api_key,
        )

        # Extract embeddings, re-order by index
        embeddings: List[Optional[List[float]]] = [None] * len(texts)
        for item in raw.get("data", []):
            idx = item.get("index", 0)
            if 0 <= idx < len(texts):
                embeddings[idx] = item.get("embedding", [])

        # Replace any None entries with empty lists
        final_embeddings = [e if e is not None else [] for e in embeddings]

        return EmbeddingResult(
            embeddings=final_embeddings,
            model=raw.get("model", self.embedding_model),
            usage=raw.get("usage", {}),
        )

    def rerank(self, query: str, candidates: List[str]) -> RerankResult:
        """Score candidates against a query using the reranker endpoint.

        The reranker uses pairwise scoring (``text_1``/``text_2`` format).
        Each candidate is scored individually against the query.

        Args:
            query: The query or reference text.
            candidates: List of candidate texts to score.

        Returns:
            RerankResult with one score per candidate (same order).

        Raises:
            EmbeddingError: On API errors after retries exhausted.
        """
        if not candidates:
            return RerankResult(scores=[], model="")

        if not self._config.reranker_base_url:
            # Reranker not configured — return zeros
            return RerankResult(scores=[0.0] * len(candidates), model="")

        scores: List[float] = []

        for candidate in candidates:
            payload = {
                "model": self.reranker_model,
                "text_1": query,
                "text_2": candidate,
            }

            try:
                raw = self._request(
                    base_url=self._config.reranker_base_url,
                    path="/score",
                    payload=payload,
                    api_key=self._config.reranker_api_key,
                )

                score_data = raw.get("data", [])
                if score_data:
                    scores.append(score_data[0].get("score", 0.0))
                else:
                    scores.append(0.0)

            except EmbeddingError:
                # Individual candidate failure → score 0.0, continue
                scores.append(0.0)

        return RerankResult(
            scores=scores,
            model=self.reranker_model if self._config.reranker_base_url else "",
        )

    def similarity(self, text_a: str, text_b: str) -> float:
        """Calculate semantic similarity between two texts.

        Embeds both texts and computes cosine similarity.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score between -1 and 1.
        """
        result = self.embed([text_a, text_b])

        if len(result.embeddings) < 2:
            return 0.0

        return cosine_similarity(result.embeddings[0], result.embeddings[1])

    def batch_similarity(self, query: str, candidates: List[str]) -> List[float]:
        """Calculate similarity between a query and multiple candidates.

        More efficient than calling ``similarity()`` repeatedly — uses
        a single batch embedding request.

        Args:
            query: Query text.
            candidates: List of candidate texts.

        Returns:
            List of similarity scores (same order as candidates).
        """
        if not candidates:
            return []

        all_texts = [query] + candidates
        result = self.embed(all_texts)

        if len(result.embeddings) < 2:
            return [0.0] * len(candidates)

        query_embedding = result.embeddings[0]
        return [
            cosine_similarity(query_embedding, cand_emb)
            for cand_emb in result.embeddings[1:]
        ]

    def test_connection(self) -> Dict[str, Any]:
        """Test connectivity to embedding and reranker endpoints.

        Returns:
            Dict with keys: ``embedding_connected``, ``reranker_connected``,
            ``embedding_model``, ``reranker_model``, ``errors``.
        """
        result: Dict[str, Any] = {
            "embedding_connected": False,
            "reranker_connected": False,
            "embedding_model": None,
            "reranker_model": None,
            "errors": [],
        }

        # Test embedding
        if self._config.embedding_base_url:
            try:
                result["embedding_model"] = self.embedding_model
                result["embedding_connected"] = True
            except Exception as exc:
                result["errors"].append(f"Embedding: {exc}")

        # Test reranker
        if self._config.reranker_base_url:
            try:
                result["reranker_model"] = self.reranker_model
                result["reranker_connected"] = True
            except Exception as exc:
                result["errors"].append(f"Reranker: {exc}")

        return result

    # --- Private methods ---

    def _detect_model(self, base_url: str, api_key: Optional[str]) -> str:
        """Hit ``/v1/models`` and return the first served model ID.

        Raises:
            EmbeddingError: If the endpoint is unreachable or returns no models.
        """
        if not base_url:
            raise EmbeddingError("No base URL configured for model detection")

        url = f"{base_url}/models"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if models:
                    return models[0]["id"]
                raise EmbeddingError("No models returned from /v1/models")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else ""
            raise EmbeddingError(
                f"Model detection failed: HTTP {exc.code}",
                status_code=exc.code,
                response_body=body,
            )
        except urllib.error.URLError as exc:
            raise EmbeddingError(f"Model detection failed: {exc.reason}")

    def _request(
        self,
        base_url: str,
        path: str,
        payload: Dict[str, Any],
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP POST request with retry logic.

        Args:
            base_url: Base URL (e.g. ``http://host:8000/v1``).
            path: API path (e.g. ``/embeddings``).
            payload: JSON-serialisable request body.
            api_key: Bearer token for authentication.

        Returns:
            Parsed JSON response.

        Raises:
            EmbeddingError: On non-transient errors or after retries exhausted.
        """
        url = f"{base_url}{path}"

        # Determine endpoint type for logging
        if "/score" in path:
            endpoint_type = "reranker"
        elif "/embeddings" in path:
            endpoint_type = "embedding"
        else:
            endpoint_type = "embedding"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        data = json.dumps(payload).encode("utf-8")
        last_error: Optional[EmbeddingError] = None

        for attempt in range(self._config.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))
                    return response_data

            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8") if exc.fp else ""

                if exc.code == 429:
                    # Rate limited → exponential backoff
                    wait = (2 ** attempt) * self._config.retry_base_delay
                    logger.warning(f"{endpoint_type.capitalize()} rate limited (429), waiting {wait:.1f}s")
                    time.sleep(wait)
                    last_error = EmbeddingError(
                        f"Rate limited: HTTP 429",
                        status_code=429,
                        response_body=body,
                    )
                elif exc.code >= 500:
                    # Server error → flat retry
                    logger.warning(f"{endpoint_type.capitalize()} server error ({exc.code}), retrying")
                    time.sleep(self._config.retry_base_delay)
                    last_error = EmbeddingError(
                        f"Server error: HTTP {exc.code}",
                        status_code=exc.code,
                        response_body=body,
                    )
                else:
                    # Client error (4xx except 429) → immediate fail
                    raise EmbeddingError(
                        f"API error: HTTP {exc.code} — {body}",
                        status_code=exc.code,
                        response_body=body,
                    )

            except urllib.error.URLError as exc:
                # Connection error → flat retry
                logger.warning(f"{endpoint_type.capitalize()} connection error: {exc.reason}, retrying")
                time.sleep(self._config.retry_base_delay)
                last_error = EmbeddingError(f"Connection error: {exc.reason}")

            except json.JSONDecodeError as exc:
                raise EmbeddingError(f"Invalid JSON response: {exc}")

        # Retries exhausted
        if last_error:
            raise last_error
        raise EmbeddingError("Max retries exceeded")
