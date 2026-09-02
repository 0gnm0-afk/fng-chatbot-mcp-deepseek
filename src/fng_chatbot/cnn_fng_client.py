"""HTTP client for CNN Fear & Greed data with bounded retries and TTL cache."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from secrets import choice
from typing import Protocol, cast

import httpx

from fng_chatbot.errors import (
    CnnBlockedError,
    CnnHttpError,
    CnnNetworkError,
    CnnPayloadError,
    CnnSchemaError,
    CnnTimeoutError,
)
from fng_chatbot.models import FearGreedSnapshot
from fng_chatbot.normalizer import NormalizationError, normalize_cnn_payload

CNN_FNG_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
DEFAULT_CACHE_TTL_SECONDS = 300.0
_RETRYABLE_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})
_BLOCKED_HTTP_STATUSES = frozenset({401, 403, 418, 429})
_BROWSER_USER_AGENTS = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:140.0) Gecko/20100101 Firefox/140.0",
)


class HttpResponse(Protocol):
    """Small response surface required by the client."""

    status_code: int

    def json(self) -> object: ...


class HttpSession(Protocol):
    """Injectable synchronous HTTP session."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout: httpx.Timeout,
    ) -> HttpResponse: ...


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    snapshot: FearGreedSnapshot
    stored_at: float


class CnnFearGreedClient:
    """Fetch and normalize CNN Fear & Greed data."""

    def __init__(
        self,
        *,
        session: HttpSession | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        max_retries: int = 2,
        backoff_base: float = 0.25,
        cache_ttl: float = DEFAULT_CACHE_TTL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        utc_now: Callable[[], datetime] | None = None,
        user_agent_factory: Callable[[], str] | None = None,
    ) -> None:
        _validate_configuration(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=max_retries,
            backoff_base=backoff_base,
            cache_ttl=cache_ttl,
        )
        self._owns_session = session is None
        self._session: HttpSession = session or httpx.Client()
        self._timeout = httpx.Timeout(
            connect=connect_timeout,
            read=read_timeout,
            write=read_timeout,
            pool=connect_timeout,
        )
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._cache_ttl = cache_ttl
        self._monotonic = monotonic
        self._sleep = sleep
        self._utc_now = utc_now or (lambda: datetime.now(UTC))
        self._user_agent_factory = user_agent_factory or _random_browser_user_agent
        self._cache: _CacheEntry | None = None

    def get_snapshot(self, *, force_refresh: bool = False) -> FearGreedSnapshot:
        """Return a normalized snapshot, using this instance's fresh cache when possible."""

        now = self._monotonic()
        if not force_refresh and self._cache_is_fresh(now):
            assert self._cache is not None
            return _with_cache_flag(self._cache.snapshot, cached=True)

        payload = self._request_json()
        fetched_at = self._utc_now().astimezone(UTC).isoformat()
        try:
            snapshot = normalize_cnn_payload(payload, fetched_at=fetched_at)
        except NormalizationError as error:
            raise CnnSchemaError(f"CNN response schema is invalid: {error}") from error

        snapshot = _with_cache_flag(snapshot, cached=False)
        if self._cache_ttl > 0:
            self._cache = _CacheEntry(snapshot=snapshot, stored_at=self._monotonic())
        return snapshot

    def clear_cache(self) -> None:
        """Discard the in-memory cache for this client instance."""

        self._cache = None

    def close(self) -> None:
        """Close the internally owned HTTP session."""

        if self._owns_session and isinstance(self._session, httpx.Client):
            self._session.close()

    def __enter__(self) -> CnnFearGreedClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _cache_is_fresh(self, now: float) -> bool:
        return (
            self._cache_ttl > 0
            and self._cache is not None
            and now - self._cache.stored_at < self._cache_ttl
        )

    def _request_json(self) -> Mapping[str, object]:
        headers = _browser_headers(self._user_agent_factory())
        for attempt in range(self._max_retries + 1):
            try:
                response = self._session.get(
                    CNN_FNG_URL,
                    headers=headers,
                    timeout=self._timeout,
                )
            except httpx.TimeoutException as error:
                if self._can_retry(attempt):
                    self._backoff(attempt)
                    continue
                raise CnnTimeoutError(
                    f"CNN request timed out after {attempt + 1} attempt(s)"
                ) from error
            except httpx.RequestError as error:
                if self._can_retry(attempt):
                    self._backoff(attempt)
                    continue
                raise CnnNetworkError(
                    f"CNN request failed after {attempt + 1} attempt(s)"
                ) from error

            status_code = response.status_code
            if status_code in _RETRYABLE_HTTP_STATUSES and self._can_retry(attempt):
                self._backoff(attempt)
                continue
            if status_code in _BLOCKED_HTTP_STATUSES:
                raise CnnBlockedError(status_code)
            if not 200 <= status_code < 300:
                raise CnnHttpError(status_code)

            try:
                payload = response.json()
            except ValueError as error:
                raise CnnPayloadError("CNN response body is not valid JSON") from error
            if not isinstance(payload, Mapping):
                raise CnnPayloadError("CNN response JSON must be an object")
            return cast(Mapping[str, object], payload)

        raise AssertionError("retry loop exited unexpectedly")

    def _can_retry(self, attempt: int) -> bool:
        return attempt < self._max_retries

    def _backoff(self, attempt: int) -> None:
        self._sleep(self._backoff_base * (2**attempt))


def _with_cache_flag(snapshot: FearGreedSnapshot, *, cached: bool) -> FearGreedSnapshot:
    return replace(
        snapshot,
        data_quality=replace(snapshot.data_quality, cached=cached, stale=False),
    )


def _random_browser_user_agent() -> str:
    return choice(_BROWSER_USER_AGENTS)


def _browser_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
        "Referer": "https://www.cnn.com/",
        "Origin": "https://www.cnn.com",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }


def _validate_configuration(
    *,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    backoff_base: float,
    cache_ttl: float,
) -> None:
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("connect_timeout and read_timeout must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries must be zero or greater")
    if backoff_base < 0:
        raise ValueError("backoff_base must be zero or greater")
    if cache_ttl < 0:
        raise ValueError("cache_ttl must be zero or greater")
