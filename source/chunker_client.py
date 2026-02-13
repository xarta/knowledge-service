"""
HTTP client for the Normalised Semantic Chunker service.

This client wraps the RESTful API of the semantic chunker deployed as
a separate Docker service. Provides document chunking via multipart
file upload.

Zero external dependencies — uses only ``urllib.request`` from the standard library.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


# ===================================================================
# ChunkerError
# ===================================================================

class ChunkerError(Exception):
    """Chunker API error with optional HTTP status and response body.

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
# ChunkerConfig
# ===================================================================

@dataclass
class ChunkerConfig:
    """Configuration for the Normalised Semantic Chunker client.

    Attributes:
        base_url: Chunker service base URL (e.g. ``http://host:8101``).
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
    """

    base_url: str
    timeout: int = 120
    max_retries: int = 3
    retry_base_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "ChunkerConfig":
        """Build ChunkerConfig from environment variables.

        Requires:
            CHUNKER_URL: Chunker service endpoint

        Returns:
            Configured ChunkerConfig.

        Raises:
            ValueError: If CHUNKER_URL is not set.
        """
        base_url = os.getenv("CHUNKER_URL", "").strip()

        if not base_url:
            raise ValueError("CHUNKER_URL environment variable is required")

        return cls(base_url=base_url)


# ===================================================================
# ChunkerClient
# ===================================================================

class ChunkerClient:
    """HTTP client for the Normalised Semantic Chunker service.

    Usage::

        config = ChunkerConfig.from_env()
        client = ChunkerClient(config)

        chunks = client.chunk_document(
            filename="document.md",
            content="# Title\\n\\nContent here...",
        )

        for chunk in chunks:
            print(chunk["text"])
    """

    def __init__(self, config: ChunkerConfig):
        self._config = config

    @property
    def config(self) -> ChunkerConfig:
        """Current configuration."""
        return self._config

    def health(self) -> Dict[str, Any]:
        """Check chunker service health.

        Returns:
            Parsed JSON health response.

        Raises:
            ChunkerError: On connection or parse errors.
        """
        url = f"{self._config.base_url.rstrip('/')}/"
        return self._get(url)

    def test_connection(self) -> Dict[str, Any]:
        """Test connectivity to the chunker endpoint.

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

    def chunk_document(
        self,
        filename: str,
        content: str,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Chunk a document via the semantic chunker.

        Sends a multipart file upload to the chunker's
        ``/normalized_semantic_chunker/`` endpoint. Returns chunk dicts.

        Args:
            filename: Document filename (e.g. ``document.md``).
            content: Document text content.
            **kwargs: Additional chunker config (buffer_size, breakpoint_percentile_threshold, etc.).

        Returns:
            List of chunk dicts with keys: ``text``, ``start_index``, ``end_index``.

        Raises:
            ChunkerError: On API errors after retries exhausted.
        """
        url = f"{self._config.base_url.rstrip('/')}/normalized_semantic_chunker/"

        # Build multipart/form-data payload
        boundary = f"----FormBoundary{uuid4().hex}"
        parts = []

        # File part
        file_headers = (
            f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
            f'Content-Type: text/plain\r\n\r\n'
        )
        parts.append(f"--{boundary}\r\n{file_headers}{content}\r\n")

        # Additional fields (chunker config)
        for key, value in kwargs.items():
            field_headers = f'Content-Disposition: form-data; name="{key}"\r\n\r\n'
            parts.append(f"--{boundary}\r\n{field_headers}{value}\r\n")

        parts.append(f"--{boundary}--\r\n")

        body = "".join(parts).encode("utf-8")

        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        }

        response = self._request_with_retry(url, headers=headers, data=body)

        # Response format: {"file_name": "<name>", "num_chunks": N, "chunks": [{"text": ..., "start_index": ..., "end_index": ...}]}
        chunks = response.get("chunks", [])
        return chunks

    # --- Private HTTP methods ---

    def _get(self, url: str) -> Dict[str, Any]:
        """HTTP GET with retry logic.

        Args:
            url: Full URL.

        Returns:
            Parsed JSON response.

        Raises:
            ChunkerError: On errors after retries.
        """
        return self._request_with_retry(url, headers={}, data=None)

    def _request_with_retry(
        self,
        url: str,
        headers: Dict[str, str],
        data: Optional[bytes],
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
            data: Request body bytes (None for GET).

        Returns:
            Parsed JSON response.

        Raises:
            ChunkerError: On non-transient errors or after retries exhausted.
        """
        last_error: Optional[ChunkerError] = None

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
                    logger.warning(f"Chunker rate limited (429), waiting {wait:.1f}s")
                    time.sleep(wait)
                    last_error = ChunkerError(
                        f"Rate limited: HTTP 429",
                        status_code=429,
                        response_body=body,
                    )
                elif exc.code >= 500:
                    # Server error → flat retry
                    logger.warning(f"Chunker server error ({exc.code}), retrying")
                    time.sleep(self._config.retry_base_delay)
                    last_error = ChunkerError(
                        f"Server error: HTTP {exc.code}",
                        status_code=exc.code,
                        response_body=body,
                    )
                else:
                    # Client error (4xx except 429) → immediate fail
                    raise ChunkerError(
                        f"API error: HTTP {exc.code} — {body}",
                        status_code=exc.code,
                        response_body=body,
                    )

            except urllib.error.URLError as exc:
                # Connection error → flat retry
                logger.warning(f"Chunker connection error: {exc.reason}, retrying")
                time.sleep(self._config.retry_base_delay)
                last_error = ChunkerError(f"Connection error: {exc.reason}")

            except json.JSONDecodeError as exc:
                raise ChunkerError(f"Invalid JSON response: {exc}")

        # Retries exhausted
        if last_error:
            raise last_error
        raise ChunkerError("Max retries exceeded")
