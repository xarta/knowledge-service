"""
LLM client for knowledge-service.

OpenAI-compatible LLM client. Zero external dependencies —
uses only ``urllib.request`` and ``json`` from the standard library.

Features:
  - ``chat()`` for full message-list requests
  - ``ask(content, prompt)`` convenience for simple use cases
  - Auto model detection from ``/v1/models``
  - Retry with exponential backoff (429) and flat retry (5xx)
  - ``<think>`` tag stripping (safety net for Qwen3 and similar)
  - ``/no_think`` directive appending (configurable)
  - Markdown fence stripping from responses
  - Environment variable configuration
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ===================================================================
# Pure helper functions
# ===================================================================

def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` blocks from LLM output.

    Applied as a safety net — even when ``/no_think`` is used, some
    models occasionally emit think tags.

    Args:
        text: Raw LLM response text.

    Returns:
        Text with all think-tag blocks removed.
    """
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


def strip_markdown_fences(text: str) -> str:
    """Remove wrapping markdown code fences from LLM output.

    Handles ````` ```json ```, ````` ```markdown ```, and plain ````` ``` `````.
    Only strips when fences wrap the *entire* content (leading/trailing).

    Args:
        text: Raw or think-stripped LLM response text.

    Returns:
        Text with outer fences removed, if present.
    """
    stripped = text.strip()
    # Match: optional language tag, newline, content, closing fence
    match = re.match(
        r"^```(?:\w+)?\s*\n(.*?)\n\s*```\s*$",
        stripped,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1)
    return text


# ===================================================================
# LLMError
# ===================================================================

class LLMError(Exception):
    """LLM API error with optional HTTP status and response body.

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
# LLMConfig
# ===================================================================

@dataclass
class LLMConfig:
    """Configuration for an OpenAI-compatible LLM endpoint.

    Attributes:
        base_url: Base URL ending in ``/v1`` (e.g. ``http://host:8000/v1``).
        api_key: Bearer token for authentication (None = no auth).
        model: Model name. None triggers auto-detection from ``/v1/models``.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts for transient errors.
        retry_base_delay: Base delay in seconds for exponential backoff.
        no_think: Whether to append ``/no_think`` to user messages (Qwen3).
    """

    base_url: str = ""
    api_key: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 3
    retry_base_delay: float = 1.0
    no_think: bool = True

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build LLMConfig from environment variables.

        Uses:
            VLLM_BASE_URL: LLM endpoint (required)
            VLLM_API_KEY: Bearer token (optional)
            LLM_MODEL: Model name (optional, auto-detect if omitted)
            LLM_TEMPERATURE: Sampling temperature (optional, default 0.3)
            LLM_MAX_TOKENS: Max response tokens (optional, default 4096)

        Returns:
            Configured LLMConfig (or None if base_url not set).
        """
        base_url = os.getenv("VLLM_BASE_URL", "").strip()
        if not base_url:
            # LLM is optional — return None
            return None

        api_key = os.getenv("VLLM_API_KEY", "").strip() or None
        model = os.getenv("LLM_MODEL", "").strip() or None
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        max_tokens = int(os.getenv("LLM_MAX_TOKENS", "4096"))

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )


# ===================================================================
# LLMClient
# ===================================================================

class LLMClient:
    """OpenAI-compatible LLM client using only the standard library.

    Usage::

        cfg = LLMConfig.from_env()
        if cfg:
            client = LLMClient(cfg)
            answer = client.ask("Some document text", "Analyse for secrets.")
    """

    def __init__(self, config: LLMConfig):
        self._config = config
        self._detected_model: Optional[str] = None

    @property
    def config(self) -> LLMConfig:
        """Current configuration."""
        return self._config

    @property
    def model(self) -> str:
        """Model name — auto-detected from ``/v1/models`` if not configured.

        Raises:
            LLMError: If auto-detection fails.
        """
        if self._config.model:
            return self._config.model

        if self._detected_model:
            return self._detected_model

        self._detected_model = self._detect_model()
        return self._detected_model

    def chat(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a chat completion request.

        Args:
            messages: Message list, e.g. ``[{"role": "user", "content": "..."}]``.
            system: Optional system prompt (prepended to messages).
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            Assistant response text (think-tags and fences stripped).

        Raises:
            LLMError: On API errors after retries are exhausted.
        """
        # Build final message list
        full_messages: List[Dict[str, str]] = []
        if system:
            full_messages.append({"role": "system", "content": system})

        # Deep-copy messages so we don't mutate caller's data
        for msg in messages:
            full_messages.append(dict(msg))

        # Append /no_think to last user message if enabled
        if self._config.no_think:
            self._append_no_think(full_messages)

        payload = {
            "model": self.model,
            "messages": full_messages,
            "temperature": temperature if temperature is not None else self._config.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self._config.max_tokens,
        }

        raw = self._request("/chat/completions", payload)

        content = raw["choices"][0]["message"]["content"]

        # Safety-net post-processing
        content = strip_think_tags(content)
        content = strip_markdown_fences(content)

        return content

    def ask(self, content: str, prompt: str, **kwargs: Any) -> str:
        """Convenience method: ``ask(content, prompt)`` → ``chat(user=content, system=prompt)``.

        Args:
            content: User message (the document/text to analyse).
            prompt: System prompt (instructions for the LLM).
            **kwargs: Passed through to ``chat()`` (temperature, max_tokens).

        Returns:
            Assistant response text.
        """
        return self.chat(
            messages=[{"role": "user", "content": content}],
            system=prompt,
            **kwargs,
        )

    def test_connection(self) -> Dict[str, Any]:
        """Test connectivity to the LLM endpoint.

        Returns:
            Dict with keys: ``connected``, ``endpoint``, ``model``, ``error``.
        """
        result: Dict[str, Any] = {
            "connected": False,
            "endpoint": self._config.base_url,
            "model": None,
            "error": None,
        }

        try:
            result["model"] = self.model
            result["connected"] = True
        except Exception as exc:
            result["error"] = str(exc)

        return result

    def _detect_model(self) -> str:
        """Hit ``/v1/models`` and return the first served model ID.

        Raises:
            LLMError: If the endpoint is unreachable or returns no models.
        """
        url = f"{self._config.base_url}/models"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if models:
                    model_id = models[0]["id"]
                    logger.info(f"Auto-detected LLM model: {model_id}")
                    return model_id
                raise LLMError("No models returned from /v1/models")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8") if exc.fp else ""
            raise LLMError(
                f"Model detection failed: HTTP {exc.code}",
                status_code=exc.code,
                response_body=body,
            )
        except urllib.error.URLError as exc:
            raise LLMError(f"Model detection failed: {exc.reason}")

    def _append_no_think(self, messages: List[Dict[str, str]]) -> None:
        """Append ``/no_think`` to the last user message if not already present."""
        for msg in reversed(messages):
            if msg["role"] == "user":
                if not msg["content"].rstrip().endswith("/no_think"):
                    msg["content"] = msg["content"] + " /no_think"
                break

    def _request(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make an HTTP POST request with retry logic.

        Args:
            path: API path (e.g. ``/chat/completions``).
            payload: JSON-serialisable request body.

        Returns:
            Parsed JSON response.

        Raises:
            LLMError: On non-transient errors or after retries exhausted.
        """
        url = f"{self._config.base_url}{path}"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        data = json.dumps(payload).encode("utf-8")
        last_error: Optional[LLMError] = None

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
                    logger.warning(f"LLM rate limited (429), waiting {wait:.1f}s")
                    time.sleep(wait)
                    last_error = LLMError(
                        f"Rate limited: HTTP 429",
                        status_code=429,
                        response_body=body,
                    )
                elif exc.code >= 500:
                    # Server error → flat retry
                    logger.warning(f"LLM server error ({exc.code}), retrying")
                    time.sleep(self._config.retry_base_delay)
                    last_error = LLMError(
                        f"Server error: HTTP {exc.code}",
                        status_code=exc.code,
                        response_body=body,
                    )
                else:
                    # Client error (4xx except 429) → immediate fail
                    raise LLMError(
                        f"API error: HTTP {exc.code} — {body}",
                        status_code=exc.code,
                        response_body=body,
                    )

            except urllib.error.URLError as exc:
                # Connection error → flat retry
                logger.warning(f"LLM connection error: {exc.reason}, retrying")
                time.sleep(self._config.retry_base_delay)
                last_error = LLMError(f"Connection error: {exc.reason}")

            except json.JSONDecodeError as exc:
                raise LLMError(f"Invalid JSON response: {exc}")

        # Retries exhausted
        if last_error:
            raise last_error
        raise LLMError("Max retries exceeded")
