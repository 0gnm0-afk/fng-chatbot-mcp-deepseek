"""Minimal NVIDIA-hosted DeepSeek client for grounded interpretations."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, cast

import httpx

NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
NVIDIA_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_DEEPSEEK_MODEL = "deepseek-ai/deepseek-v4-flash"
NVIDIA_REASONING_EFFORT = "none"
DEEPSEEK_MAX_TOKENS = 256
DEEPSEEK_TEMPERATURE = 0.2
DEEPSEEK_TIMEOUT_SECONDS = 20.0

# The shared NVIDIA pool rejects bursts with 503 while a worker is saturated, so retries
# only help when they are spread out; retrying immediately hits the same busy worker.
DEEPSEEK_MAX_ATTEMPTS = 4
DEEPSEEK_BACKOFF_BASE_SECONDS = 1.0
DEEPSEEK_BACKOFF_MAX_SECONDS = 8.0
DEEPSEEK_BACKOFF_JITTER_RATIO = 0.25
# Retries must stay well inside the 90s MCP tool budget shared with the CNN fetch.
DEEPSEEK_TOTAL_BUDGET_SECONDS = 45.0

_DEEPSEEK_TRANSPORT_REASON_CODES = frozenset(
    {
        "transport_error",
        "transport_connect_timeout",
        "transport_read_timeout",
        "transport_write_timeout",
        "transport_pool_timeout",
        "transport_connect_error",
        "transport_read_error",
        "transport_write_error",
        "transport_close_error",
        "transport_proxy_error",
        "transport_network_error",
    }
)

# Fixed vocabulary distilled from the provider error body. The raw body is never kept so a
# key, token, or request identifier can never travel with the failure metadata.
_DEEPSEEK_PROVIDER_REASONS = frozenset(
    {
        "resource_exhausted",
        "rate_limited",
        "model_unavailable",
        "unauthorized",
        "invalid_request",
        "unspecified",
    }
)

_PROVIDER_REASON_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "resource_exhausted",
        ("resourceexhausted", "resource exhausted", "request limit reached", "capacity"),
    ),
    ("rate_limited", ("rate limit", "too many requests", "quota", "throttl")),
    ("unauthorized", ("unauthorized", "invalid api key", "forbidden", "authentication")),
    ("model_unavailable", ("model not found", "no such model", "unknown model", "not found")),
    ("invalid_request", ("invalid", "bad request", "unsupported", "malformed")),
)


class DeepSeekError(RuntimeError):
    """Base error for the DeepSeek boundary."""


class DeepSeekConfigurationError(DeepSeekError):
    """The client cannot make a safe request with its current configuration."""


class DeepSeekTransportError(DeepSeekError):
    """The DeepSeek endpoint could not be reached."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "transport_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = (
            reason_code if reason_code in _DEEPSEEK_TRANSPORT_REASON_CODES else "transport_error"
        )


class DeepSeekApiError(DeepSeekError):
    """DeepSeek rejected the request."""

    def __init__(self, status_code: int, *, provider_reason: str = "unspecified") -> None:
        super().__init__(f"DeepSeek chat completion returned HTTP {status_code}")
        self.status_code = status_code
        self.provider_reason = (
            provider_reason if provider_reason in _DEEPSEEK_PROVIDER_REASONS else "unspecified"
        )


class DeepSeekResponseError(DeepSeekError):
    """DeepSeek returned an unusable completion response."""


class DeepSeekHttpResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class DeepSeekHttpSession(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: httpx.Timeout,
    ) -> DeepSeekHttpResponse: ...


@dataclass(frozen=True, slots=True)
class DeepSeekJsonCompletion:
    """Validated response envelope; content JSON is checked by the next layer."""

    content: str
    model: str
    finish_reason: str


class DeepSeekClient:
    """Call DeepSeek V4 Flash through NVIDIA's OpenAI-compatible endpoint."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        session: DeepSeekHttpSession | None = None,
        timeout_seconds: float = DEEPSEEK_TIMEOUT_SECONDS,
        max_attempts: int = DEEPSEEK_MAX_ATTEMPTS,
        backoff_base_seconds: float = DEEPSEEK_BACKOFF_BASE_SECONDS,
        total_budget_seconds: float = DEEPSEEK_TOTAL_BUDGET_SECONDS,
        sleep: Callable[[float], None] | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be greater than zero")
        if backoff_base_seconds <= 0:
            raise ValueError("backoff_base_seconds must be greater than zero")
        if total_budget_seconds <= 0:
            raise ValueError("total_budget_seconds must be greater than zero")
        configured_key = api_key if api_key is not None else os.getenv(NVIDIA_API_KEY_ENV)
        self._api_key = configured_key.strip() if configured_key else ""
        self._owns_session = session is None
        self._session: DeepSeekHttpSession = session or httpx.Client()
        self._timeout = httpx.Timeout(timeout_seconds)
        self._max_attempts = max_attempts
        self._backoff_base_seconds = backoff_base_seconds
        self._total_budget_seconds = total_budget_seconds
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> DeepSeekJsonCompletion:
        """Request a short non-thinking completion whose content must be a JSON object."""

        if not self._api_key:
            raise DeepSeekConfigurationError(f"{NVIDIA_API_KEY_ENV} is not configured")
        if not system_prompt.strip() or not user_prompt.strip():
            raise DeepSeekConfigurationError("DeepSeek prompts must not be empty")

        response: DeepSeekHttpResponse | None = None
        started_at = self._clock()
        for attempt in range(1, self._max_attempts + 1):
            try:
                response = self._session.post(
                    NVIDIA_CHAT_COMPLETIONS_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                    json={
                        "model": NVIDIA_DEEPSEEK_MODEL,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "reasoning_effort": NVIDIA_REASONING_EFFORT,
                        "response_format": {"type": "json_object"},
                        "stream": False,
                        "max_tokens": DEEPSEEK_MAX_TOKENS,
                        "temperature": DEEPSEEK_TEMPERATURE,
                    },
                    timeout=self._timeout,
                )
                if _is_retryable_http_status(response.status_code) and self._wait_before_retry(
                    attempt=attempt,
                    started_at=started_at,
                ):
                    continue
                break
            except httpx.ReadTimeout as error:
                if self._wait_before_retry(attempt=attempt, started_at=started_at):
                    continue
                raise DeepSeekTransportError(
                    "DeepSeek chat completion request failed",
                    reason_code="transport_read_timeout",
                ) from error
            except httpx.RequestError as error:
                raise DeepSeekTransportError(
                    "DeepSeek chat completion request failed",
                    reason_code=_transport_reason_code(error),
                ) from error

        if response is None:  # Defensive guard; the loop either returns a response or raises.
            raise DeepSeekTransportError("DeepSeek chat completion request failed")

        if not 200 <= response.status_code < 300:
            raise DeepSeekApiError(
                response.status_code,
                provider_reason=_provider_reason(response),
            )
        try:
            payload = response.json()
        except ValueError as error:
            raise DeepSeekResponseError("DeepSeek response body is not valid JSON") from error
        return _parse_completion(payload)

    def _wait_before_retry(self, *, attempt: int, started_at: float) -> bool:
        """Sleep before the next attempt; return False when this attempt must be the last."""

        if attempt >= self._max_attempts:
            return False
        delay = min(
            self._backoff_base_seconds * 2 ** (attempt - 1),
            DEEPSEEK_BACKOFF_MAX_SECONDS,
        )
        # Jitter keeps concurrent previews from retrying into the same saturated worker.
        delay += delay * DEEPSEEK_BACKOFF_JITTER_RATIO * random.random()
        if self._clock() - started_at + delay >= self._total_budget_seconds:
            return False
        self._sleep(delay)
        return True

    def close(self) -> None:
        """Close the internally owned HTTP session."""

        if self._owns_session and isinstance(self._session, httpx.Client):
            self._session.close()

    def __enter__(self) -> DeepSeekClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _parse_completion(payload: object) -> DeepSeekJsonCompletion:
    if not isinstance(payload, Mapping):
        raise DeepSeekResponseError("DeepSeek response JSON must be an object")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], Mapping):
        raise DeepSeekResponseError("DeepSeek response has no completion choice")

    choice = cast(Mapping[str, object], choices[0])
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise DeepSeekResponseError("DeepSeek completion did not finish normally")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise DeepSeekResponseError("DeepSeek completion has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise DeepSeekResponseError("DeepSeek completion content is empty")

    return DeepSeekJsonCompletion(
        content=content,
        model=NVIDIA_DEEPSEEK_MODEL,
        finish_reason=finish_reason,
    )


def _provider_reason(response: DeepSeekHttpResponse) -> str:
    """Distill a rejection body into a fixed code, discarding the provider text itself."""

    try:
        payload = response.json()
    except (ValueError, TypeError):
        return "unspecified"
    if not isinstance(payload, Mapping):
        return "unspecified"

    detail: object = payload.get("error", payload)
    if isinstance(detail, Mapping):
        detail = detail.get("message") or detail.get("detail") or detail.get("type")
    if not isinstance(detail, str):
        detail = payload.get("detail") or payload.get("message")
    if not isinstance(detail, str):
        return "unspecified"

    lowered = detail.casefold()
    for reason, needles in _PROVIDER_REASON_PATTERNS:
        if any(needle in lowered for needle in needles):
            return reason
    return "unspecified"


def _transport_reason_code(error: httpx.RequestError) -> str:
    if isinstance(error, httpx.ConnectTimeout):
        return "transport_connect_timeout"
    if isinstance(error, httpx.ReadTimeout):
        return "transport_read_timeout"
    if isinstance(error, httpx.WriteTimeout):
        return "transport_write_timeout"
    if isinstance(error, httpx.PoolTimeout):
        return "transport_pool_timeout"
    if isinstance(error, httpx.ConnectError):
        return "transport_connect_error"
    if isinstance(error, httpx.ReadError):
        return "transport_read_error"
    if isinstance(error, httpx.WriteError):
        return "transport_write_error"
    if isinstance(error, httpx.CloseError):
        return "transport_close_error"
    if isinstance(error, httpx.ProxyError):
        return "transport_proxy_error"
    if isinstance(error, httpx.NetworkError):
        return "transport_network_error"
    return "transport_error"


def _is_retryable_http_status(status_code: int) -> bool:
    return status_code == 429 or 500 <= status_code < 600
