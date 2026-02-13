"""
SeekDB client for knowledge-service.

Zero-dependency HTTP client for the SeekDB HTTP API wrapper.
Uses only ``urllib.request`` and ``json`` from the standard library.

Features:
  - Health check (unauthenticated)
  - Database and collection management
  - Item CRUD (add, get, update, delete)
  - Vector search and hybrid search
  - Retry with exponential backoff (429) and flat retry (5xx)
  - Bearer token authentication
  - Environment variable configuration
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===================================================================
# SeekDBError
# ===================================================================

class SeekDBError(Exception):
    """SeekDB API error with optional HTTP status and response body.

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
# SeekDBConfig
# ===================================================================

@dataclass
class SeekDBConfig:
    """Configuration for SeekDB client.

    Attributes:
        base_url: Base URL for SeekDB HTTP API.
        api_key: Bearer token for authentication.
        database: Default database name.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    base_url: str
    api_key: str
    database: str
    timeout: int = 60
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "SeekDBConfig":
        """Build SeekDBConfig from environment variables.

        Required:
            SEEKDB_BASE_URL: SeekDB HTTP API endpoint
            SEEKDB_API_KEY: Bearer token
            SEEKDB_DATABASE: Database name

        Returns:
            Configured SeekDBConfig.

        Raises:
            ValueError: If required env vars are missing.
        """
        base_url = os.getenv("SEEKDB_BASE_URL", "").strip()
        api_key = os.getenv("SEEKDB_API_KEY", "").strip()
        database = os.getenv("SEEKDB_DATABASE", "").strip()

        if not base_url:
            raise ValueError("SEEKDB_BASE_URL environment variable is required")
        if not api_key:
            raise ValueError("SEEKDB_API_KEY environment variable is required")
        if not database:
            raise ValueError("SEEKDB_DATABASE environment variable is required")

        return cls(
            base_url=base_url,
            api_key=api_key,
            database=database,
        )


# ===================================================================
# SeekDBClient
# ===================================================================

class SeekDBClient:
    """HTTP client for the SeekDB API wrapper.

    Usage::

        config = SeekDBConfig.from_env()
        client = SeekDBClient(config)

        # Health check (no auth required)
        health = client.health()

        # Add items
        client.add("articles", ids=["id-1"], documents=["Hello world"])

        # Search
        results = client.query("articles", query_texts=["hello"], n_results=5)
    """

    def __init__(self, config: SeekDBConfig):
        self._config = config

    @property
    def config(self) -> SeekDBConfig:
        """Current configuration."""
        return self._config

    # --- Health ---

    def health(self) -> Dict[str, Any]:
        """Check SeekDB API health.

        No authentication required.

        Returns:
            Parsed JSON health response.

        Raises:
            SeekDBError: On connection or parse errors.
        """
        return self._get("/health", auth=False)

    def test_connection(self) -> Dict[str, Any]:
        """Test connectivity to the SeekDB endpoint.

        Returns:
            Dict with keys: ``connected``, ``endpoint``, ``error``.
        """
        result: Dict[str, Any] = {
            "connected": False,
            "endpoint": self._config.base_url,
            "error": None,
        }
        try:
            self.health()
            result["connected"] = True
        except Exception as exc:
            result["error"] = str(exc)
        return result

    # --- Database operations ---

    def list_databases(self) -> Dict[str, Any]:
        """List all databases.

        Returns:
            Parsed JSON with ``databases`` key.

        Raises:
            SeekDBError: On API errors.
        """
        return self._get("/databases")

    # --- Collection operations ---

    def list_collections(self, database: Optional[str] = None) -> Dict[str, Any]:
        """List collections in a database.

        Args:
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON with ``collections`` key (list of objects).

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        return self._get(f"/databases/{db}/collections")

    def create_collection(
        self,
        name: str,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new collection.

        Args:
            name: Collection name.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        return self._post(f"/databases/{db}/collections", {"name": name})

    def delete_collection(
        self,
        name: str,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete a collection.

        Args:
            name: Collection name.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        return self._delete(f"/databases/{db}/collections/{name}")

    def count(
        self,
        collection: str,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get item count in a collection.

        Args:
            collection: Collection name.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON with ``count`` key.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        return self._get(f"/databases/{db}/collections/{collection}/count")

    # --- Item CRUD ---

    def add(
        self,
        collection: str,
        ids: List[str],
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Add items to a collection.

        Args:
            collection: Collection name.
            ids: Item identifiers.
            documents: Document texts.
            metadatas: Optional metadata dicts per item.
            embeddings: Optional pre-computed embeddings per item.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        payload: Dict[str, Any] = {
            "ids": ids,
            "documents": documents,
        }
        if metadatas is not None:
            payload["metadatas"] = metadatas
        if embeddings is not None:
            payload["embeddings"] = embeddings
        return self._post(
            f"/databases/{db}/collections/{collection}/add", payload
        )

    def get(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get items from a collection by ID or filter.

        Args:
            collection: Collection name.
            ids: Optional list of item IDs to retrieve.
            where: Optional filter dict.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON with ids, documents, metadatas.
            The ``results`` wrapper from the API is automatically unwrapped.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        payload: Dict[str, Any] = {}
        if ids is not None:
            payload["ids"] = ids
        if where is not None:
            payload["where"] = where
        response = self._post(
            f"/databases/{db}/collections/{collection}/get", payload
        )
        return response.get("results", response)

    def update(
        self,
        collection: str,
        ids: List[str],
        documents: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        embeddings: Optional[List[List[float]]] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update items in a collection.

        Args:
            collection: Collection name.
            ids: Item identifiers to update.
            documents: Optional updated document texts.
            metadatas: Optional updated metadata dicts.
            embeddings: Optional updated embeddings.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        payload: Dict[str, Any] = {"ids": ids}
        if documents is not None:
            payload["documents"] = documents
        if metadatas is not None:
            payload["metadatas"] = metadatas
        if embeddings is not None:
            payload["embeddings"] = embeddings
        return self._post(
            f"/databases/{db}/collections/{collection}/update", payload
        )

    def delete(
        self,
        collection: str,
        ids: List[str],
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Delete items from a collection.

        Args:
            collection: Collection name.
            ids: Item identifiers to delete.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        return self._post(
            f"/databases/{db}/collections/{collection}/delete", {"ids": ids}
        )

    # --- Search ---

    def query(
        self,
        collection: str,
        query_texts: List[str],
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Vector search in a collection.

        Args:
            collection: Collection name.
            query_texts: Query texts to embed and search.
            n_results: Number of results to return.
            where: Optional filter dict.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON with ids, documents, metadatas, distances.
            The ``results`` wrapper from the API is automatically unwrapped.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        payload: Dict[str, Any] = {
            "query_texts": query_texts,
            "n_results": n_results,
        }
        if where is not None:
            payload["where"] = where
        response = self._post(
            f"/databases/{db}/collections/{collection}/query", payload
        )
        return response.get("results", response)

    def hybrid_search(
        self,
        collection: str,
        query_texts: List[str],
        query_text: str,
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        database: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Hybrid (vector + keyword) search in a collection.

        Args:
            collection: Collection name.
            query_texts: Query texts for vector search component.
            query_text: Query text for keyword search component.
            n_results: Number of results to return.
            where: Optional filter dict.
            database: Database name. Defaults to config database.

        Returns:
            Parsed JSON with ids, documents, metadatas, distances.
            The ``results`` wrapper from the API is automatically unwrapped.

        Raises:
            SeekDBError: On API errors.
        """
        db = database or self._config.database
        payload: Dict[str, Any] = {
            "query_texts": query_texts,
            "query_text": query_text,
            "n_results": n_results,
        }
        if where is not None:
            payload["where"] = where
        response = self._post(
            f"/databases/{db}/collections/{collection}/hybrid_search", payload
        )
        return response.get("results", response)

    # --- Private HTTP methods ---

    def _base_url(self) -> str:
        """Return base URL with trailing slash stripped."""
        return self._config.base_url.rstrip("/")

    def _auth_headers(self) -> Dict[str, str]:
        """Build authentication headers."""
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"
        return headers

    def _noauth_headers(self) -> Dict[str, str]:
        """Build headers without authentication."""
        return {"Content-Type": "application/json"}

    def _get(self, path: str, auth: bool = True) -> Dict[str, Any]:
        """HTTP GET with retry logic.

        Args:
            path: URL path (appended to base_url).
            auth: Whether to include auth headers.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On errors after retries.
        """
        url = f"{self._base_url()}{path}"
        headers = self._auth_headers() if auth else self._noauth_headers()
        return self._request_with_retry(url, headers=headers, data=None)

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """HTTP POST with retry logic.

        Args:
            path: URL path (appended to base_url).
            payload: JSON-serialisable request body.

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On errors after retries.
        """
        url = f"{self._base_url()}{path}"
        headers = self._auth_headers()
        data = json.dumps(payload).encode("utf-8")
        return self._request_with_retry(url, headers=headers, data=data)

    def _delete(self, path: str) -> Dict[str, Any]:
        """HTTP DELETE with retry logic.

        Args:
            path: URL path (appended to base_url).

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On errors after retries.
        """
        url = f"{self._base_url()}{path}"
        headers = self._auth_headers()
        return self._request_with_retry(
            url, headers=headers, data=None, method="DELETE"
        )

    def _request_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        data: Optional[bytes],
        method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request with retry logic.

        Retry strategy:
          - 429 → exponential backoff (2^attempt × base_delay)
          - 5xx → flat retry (base_delay)
          - 4xx (except 429) → immediate failure
          - Connection errors → flat retry

        Args:
            url: Full URL.
            headers: HTTP headers dict.
            data: Request body bytes (None for GET/DELETE).
            method: HTTP method override (e.g. "DELETE").

        Returns:
            Parsed JSON response.

        Raises:
            SeekDBError: On non-transient errors or after retries exhausted.
        """
        last_error: Optional[SeekDBError] = None

        for attempt in range(self._config.max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers=headers)
                if method:
                    req.get_method = lambda m=method: m

                with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
                    response_data = json.loads(resp.read().decode("utf-8"))
                    return response_data

            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8") if exc.fp else ""

                if exc.code == 429:
                    # Rate limited → exponential backoff
                    wait = (2 ** attempt) * self._config.retry_base_delay
                    logger.warning(f"Rate limited (429), waiting {wait:.1f}s")
                    time.sleep(wait)
                    last_error = SeekDBError(
                        f"Rate limited: HTTP 429",
                        status_code=429,
                        response_body=body,
                    )
                elif exc.code >= 500:
                    # Server error → flat retry
                    logger.warning(f"Server error ({exc.code}), retrying")
                    time.sleep(self._config.retry_base_delay)
                    last_error = SeekDBError(
                        f"Server error: HTTP {exc.code}",
                        status_code=exc.code,
                        response_body=body,
                    )
                else:
                    # Client error (4xx except 429) → immediate fail
                    raise SeekDBError(
                        f"API error: HTTP {exc.code} — {body}",
                        status_code=exc.code,
                        response_body=body,
                    )

            except urllib.error.URLError as exc:
                # Connection error → flat retry
                logger.warning(f"Connection error: {exc.reason}, retrying")
                time.sleep(self._config.retry_base_delay)
                last_error = SeekDBError(f"Connection error: {exc.reason}")

            except json.JSONDecodeError as exc:
                raise SeekDBError(f"Invalid JSON response: {exc}")

        # Retries exhausted
        if last_error:
            raise last_error
        raise SeekDBError("Max retries exceeded")
