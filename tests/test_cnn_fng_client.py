import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from fng_chatbot.cnn_fng_client import CNN_FNG_URL, CnnFearGreedClient
from fng_chatbot.errors import (
    CnnBlockedError,
    CnnHttpError,
    CnnNetworkError,
    CnnPayloadError,
    CnnSchemaError,
    CnnTimeoutError,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"
FIXED_NOW = datetime(2026, 7, 20, 9, tzinfo=UTC)


def load_payload() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


class FakeResponse:
    def __init__(
        self,
        status_code: int = 200,
        payload: object | None = None,
        json_error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = load_payload() if payload is None else payload
        self._json_error = json_error

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, *outcomes: FakeResponse | Exception) -> None:
        self.outcomes = list(outcomes)
        self.urls: list[str] = []
        self.headers: list[dict[str, str]] = []
        self.timeouts: list[httpx.Timeout] = []

    @property
    def call_count(self) -> int:
        return len(self.urls)

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: httpx.Timeout,
    ) -> FakeResponse:
        self.urls.append(url)
        self.headers.append(headers)
        self.timeouts.append(timeout)
        if not self.outcomes:
            raise AssertionError("fake session has no remaining outcome")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def make_client(
    session: FakeSession,
    *,
    clock: Callable[[], float] | None = None,
    sleep: Callable[[float], None] | None = None,
    **kwargs: object,
) -> CnnFearGreedClient:
    return CnnFearGreedClient(
        session=session,
        monotonic=clock or ManualClock(),
        sleep=sleep or (lambda _: None),
        utc_now=lambda: FIXED_NOW,
        user_agent_factory=lambda: "test-browser-user-agent",
        **kwargs,
    )


def test_success_returns_normalized_snapshot_with_separate_timeouts() -> None:
    session = FakeSession(FakeResponse())
    client = make_client(session, connect_timeout=1.5, read_timeout=7.25)

    snapshot = client.get_snapshot()

    assert snapshot.composite.score == 37.0
    assert len(snapshot.indicators) == 7
    assert snapshot.fetched_at == "2026-07-20T09:00:00Z"
    assert snapshot.data_quality.cached is False
    assert session.urls == [CNN_FNG_URL]
    assert session.timeouts[0].connect == 1.5
    assert session.timeouts[0].read == 7.25


def test_request_uses_browser_headers_from_reference_strategy() -> None:
    session = FakeSession(FakeResponse())
    client = make_client(session)

    client.get_snapshot()

    assert session.headers == [
        {
            "User-Agent": "test-browser-user-agent",
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
            "Referer": "https://www.cnn.com/",
            "Origin": "https://www.cnn.com",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }
    ]


def test_timeout_retries_with_exponential_backoff_then_raises() -> None:
    session = FakeSession(
        httpx.ReadTimeout("read timed out"),
        httpx.ReadTimeout("read timed out"),
        httpx.ReadTimeout("read timed out"),
    )
    delays: list[float] = []
    client = make_client(
        session,
        max_retries=2,
        backoff_base=0.25,
        sleep=delays.append,
    )

    with pytest.raises(CnnTimeoutError, match=r"after 3 attempt\(s\)"):
        client.get_snapshot()

    assert session.call_count == 3
    assert delays == [0.25, 0.5]


def test_network_error_is_distinct_from_timeout() -> None:
    session = FakeSession(httpx.ConnectError("connection failed"))
    client = make_client(session, max_retries=0)

    with pytest.raises(CnnNetworkError, match="after 1 attempt"):
        client.get_snapshot()


@pytest.mark.parametrize("status_code", [401, 403, 418])
def test_access_rejection_is_reported_as_blocked_without_retry(status_code: int) -> None:
    session = FakeSession(FakeResponse(status_code=status_code))
    client = make_client(session, max_retries=2)

    with pytest.raises(CnnBlockedError) as error:
        client.get_snapshot()

    assert error.value.status_code == status_code
    assert session.call_count == 1


def test_rate_limit_retries_then_reports_blocked() -> None:
    session = FakeSession(FakeResponse(status_code=429), FakeResponse(status_code=429))
    delays: list[float] = []
    client = make_client(session, max_retries=1, backoff_base=0.1, sleep=delays.append)

    with pytest.raises(CnnBlockedError) as error:
        client.get_snapshot()

    assert error.value.status_code == 429
    assert session.call_count == 2
    assert delays == [0.1]


def test_server_error_retries_then_raises_http_error() -> None:
    session = FakeSession(FakeResponse(status_code=500), FakeResponse(status_code=500))
    delays: list[float] = []
    client = make_client(session, max_retries=1, backoff_base=0.5, sleep=delays.append)

    with pytest.raises(CnnHttpError) as error:
        client.get_snapshot()

    assert error.value.status_code == 500
    assert session.call_count == 2
    assert delays == [0.5]


def test_non_retryable_http_error_fails_immediately() -> None:
    session = FakeSession(FakeResponse(status_code=404))
    client = make_client(session, max_retries=2)

    with pytest.raises(CnnHttpError) as error:
        client.get_snapshot()

    assert error.value.status_code == 404
    assert session.call_count == 1


def test_invalid_json_has_distinct_payload_error() -> None:
    session = FakeSession(FakeResponse(json_error=ValueError("invalid JSON")))
    client = make_client(session)

    with pytest.raises(CnnPayloadError, match="not valid JSON"):
        client.get_snapshot()


def test_non_object_json_has_distinct_payload_error() -> None:
    session = FakeSession(FakeResponse(payload=[]))
    client = make_client(session)

    with pytest.raises(CnnPayloadError, match="must be an object"):
        client.get_snapshot()


def test_schema_error_preserves_explainable_field_path() -> None:
    payload = load_payload()
    del payload["safe_haven_demand"]
    session = FakeSession(FakeResponse(payload=payload))
    client = make_client(session)

    with pytest.raises(CnnSchemaError, match="payload.safe_haven_demand"):
        client.get_snapshot()


def test_same_instance_uses_cache_before_300_seconds() -> None:
    clock = ManualClock()
    session = FakeSession(FakeResponse())
    client = make_client(session, clock=clock)

    first = client.get_snapshot()
    clock.advance(299.9)
    second = client.get_snapshot()

    assert first.data_quality.cached is False
    assert second.data_quality.cached is True
    assert first.composite == second.composite
    assert session.call_count == 1


def test_cache_expires_at_300_seconds() -> None:
    clock = ManualClock()
    session = FakeSession(FakeResponse(), FakeResponse())
    client = make_client(session, clock=clock)

    client.get_snapshot()
    clock.advance(300)
    refreshed = client.get_snapshot()

    assert refreshed.data_quality.cached is False
    assert session.call_count == 2


def test_zero_ttl_disables_cache() -> None:
    session = FakeSession(FakeResponse(), FakeResponse())
    client = make_client(session, cache_ttl=0)

    client.get_snapshot()
    second = client.get_snapshot()

    assert second.data_quality.cached is False
    assert session.call_count == 2


def test_force_refresh_and_clear_cache_bypass_existing_entry() -> None:
    session = FakeSession(FakeResponse(), FakeResponse(), FakeResponse())
    client = make_client(session)

    client.get_snapshot()
    forced = client.get_snapshot(force_refresh=True)
    client.clear_cache()
    cleared = client.get_snapshot()

    assert forced.data_quality.cached is False
    assert cleared.data_quality.cached is False
    assert session.call_count == 3


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("connect_timeout", 0),
        ("read_timeout", -1),
        ("max_retries", -1),
        ("backoff_base", -0.1),
        ("cache_ttl", -1),
    ],
)
def test_rejects_invalid_configuration(option: str, value: float) -> None:
    with pytest.raises(ValueError):
        make_client(FakeSession(), **{option: value})
