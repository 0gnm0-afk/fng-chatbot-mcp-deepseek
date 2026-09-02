from collections.abc import Mapping

import httpx
import pytest

from fng_chatbot.deepseek import (
    DEEPSEEK_BACKOFF_BASE_SECONDS,
    DEEPSEEK_BACKOFF_JITTER_RATIO,
    DEEPSEEK_BACKOFF_MAX_SECONDS,
    DEEPSEEK_MAX_ATTEMPTS,
    DEEPSEEK_MAX_TOKENS,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT_SECONDS,
    NVIDIA_API_KEY_ENV,
    NVIDIA_CHAT_COMPLETIONS_URL,
    NVIDIA_DEEPSEEK_MODEL,
    NVIDIA_REASONING_EFFORT,
    DeepSeekApiError,
    DeepSeekClient,
    DeepSeekConfigurationError,
    DeepSeekResponseError,
    DeepSeekTransportError,
)


class FakeResponse:
    def __init__(
        self,
        payload: object,
        *,
        status_code: int = 200,
        json_error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self) -> object:
        if self._json_error is not None:
            raise self._json_error
        return self._payload


class FakeSession:
    def __init__(self, outcome: FakeResponse | Exception) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: httpx.Timeout,
    ) -> FakeResponse:
        self.calls.append(
            {
                "url": url,
                "headers": dict(headers),
                "json": dict(json),
                "timeout": timeout,
            }
        )
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class SequenceSession(FakeSession):
    def __init__(self, outcomes: list[FakeResponse | Exception]) -> None:
        if not outcomes:
            raise ValueError("outcomes must not be empty")
        super().__init__(outcomes[-1])
        self._outcomes = list(outcomes)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: Mapping[str, object],
        timeout: httpx.Timeout,
    ) -> FakeResponse:
        self.outcome = self._outcomes.pop(0)
        return super().post(url, headers=headers, json=json, timeout=timeout)


class SleepRecorder:
    """Capture backoff delays so retry tests never wait in real time."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, delay: float) -> None:
        self.delays.append(delay)

    @property
    def elapsed(self) -> float:
        return sum(self.delays)


def resource_exhausted_response() -> FakeResponse:
    """The exact shape NVIDIA returns when the shared worker pool is saturated."""

    return FakeResponse(
        {
            "error": {
                "message": "ResourceExhausted: Worker local total request limit reached (48/48)",
                "type": "Service Unavailable",
                "code": 503,
            }
        },
        status_code=503,
    )


def successful_payload(content: str = '{"lines":["설명"]}') -> dict[str, object]:
    return {
        "model": NVIDIA_DEEPSEEK_MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
    }


def test_requests_fixed_lightweight_model_with_non_thinking_json_contract() -> None:
    session = FakeSession(FakeResponse(successful_payload()))
    client = DeepSeekClient(api_key="fake-deepseek-key", session=session, timeout_seconds=7.5)

    completion = client.complete_json(
        system_prompt="Return JSON only.",
        user_prompt="Explain the supplied factors.",
    )

    assert completion.content == '{"lines":["설명"]}'
    assert completion.model == NVIDIA_DEEPSEEK_MODEL
    assert completion.finish_reason == "stop"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == NVIDIA_CHAT_COMPLETIONS_URL
    assert call["headers"] == {
        "Authorization": "Bearer fake-deepseek-key",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    assert call["json"] == {
        "model": NVIDIA_DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": "Explain the supplied factors."},
        ],
        "reasoning_effort": NVIDIA_REASONING_EFFORT,
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": DEEPSEEK_MAX_TOKENS,
        "temperature": DEEPSEEK_TEMPERATURE,
    }
    assert call["timeout"].read == 7.5


def test_reads_api_key_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(NVIDIA_API_KEY_ENV, "environment-key")
    session = FakeSession(FakeResponse(successful_payload()))
    client = DeepSeekClient(session=session)

    client.complete_json(system_prompt="system", user_prompt="user")

    assert session.calls[0]["headers"]["Authorization"] == "Bearer environment-key"
    assert session.calls[0]["timeout"].read == DEEPSEEK_TIMEOUT_SECONDS


def test_missing_api_key_fails_without_network_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NVIDIA_API_KEY_ENV, raising=False)
    session = FakeSession(FakeResponse(successful_payload()))
    client = DeepSeekClient(session=session)

    with pytest.raises(DeepSeekConfigurationError, match=NVIDIA_API_KEY_ENV):
        client.complete_json(system_prompt="system", user_prompt="user")

    assert session.calls == []


def test_legacy_deepseek_key_is_not_sent_to_nvidia(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(NVIDIA_API_KEY_ENV, raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "legacy-provider-key")
    session = FakeSession(FakeResponse(successful_payload()))
    client = DeepSeekClient(session=session)

    with pytest.raises(DeepSeekConfigurationError, match=NVIDIA_API_KEY_ENV):
        client.complete_json(system_prompt="system", user_prompt="user")

    assert session.calls == []


@pytest.mark.parametrize("field", ["system_prompt", "user_prompt"])
def test_rejects_empty_prompts_before_network_call(field: str) -> None:
    session = FakeSession(FakeResponse(successful_payload()))
    client = DeepSeekClient(api_key="fake-deepseek-key", session=session)
    arguments = {"system_prompt": "system", "user_prompt": "user"}
    arguments[field] = "  "

    with pytest.raises(DeepSeekConfigurationError, match="prompts"):
        client.complete_json(**arguments)

    assert session.calls == []


@pytest.mark.parametrize(
    ("request_error", "reason_code"),
    [
        (httpx.ConnectTimeout("connect timed out"), "transport_connect_timeout"),
        (httpx.ReadTimeout("read timed out"), "transport_read_timeout"),
        (httpx.WriteTimeout("write timed out"), "transport_write_timeout"),
        (httpx.PoolTimeout("pool timed out"), "transport_pool_timeout"),
        (httpx.ConnectError("connection failed"), "transport_connect_error"),
        (httpx.ReadError("read failed"), "transport_read_error"),
        (httpx.WriteError("write failed"), "transport_write_error"),
        (httpx.CloseError("close failed"), "transport_close_error"),
        (httpx.ProxyError("proxy failed"), "transport_proxy_error"),
        (httpx.NetworkError("network failed"), "transport_network_error"),
        (httpx.RequestError("request failed"), "transport_error"),
    ],
)
def test_transport_error_has_safe_reason_code_without_exposing_api_key(
    request_error: httpx.RequestError,
    reason_code: str,
) -> None:
    session = FakeSession(request_error)
    client = DeepSeekClient(api_key="sensitive-test-key", session=session, sleep=SleepRecorder())

    with pytest.raises(DeepSeekTransportError) as error:
        client.complete_json(system_prompt="system", user_prompt="user")

    assert error.value.reason_code == reason_code
    assert "sensitive-test-key" not in str(error.value)


def test_unknown_transport_reason_code_falls_back_to_generic_code() -> None:
    error = DeepSeekTransportError(
        "sensitive transport detail",
        reason_code="sensitive-provider-detail",
    )

    assert error.reason_code == "transport_error"


def test_read_timeout_retries_once_and_returns_second_response() -> None:
    session = SequenceSession(
        [
            httpx.ReadTimeout("read timed out"),
            FakeResponse(successful_payload()),
        ]
    )
    recorder = SleepRecorder()
    client = DeepSeekClient(
        api_key="fake-deepseek-key",
        session=session,
        sleep=recorder,
    )

    completion = client.complete_json(system_prompt="system", user_prompt="user")

    assert completion.finish_reason == "stop"
    assert len(session.calls) == 2
    assert len(recorder.delays) == 1


def test_non_read_transport_error_is_not_retried() -> None:
    session = FakeSession(httpx.ConnectError("connection failed"))
    client = DeepSeekClient(api_key="fake-deepseek-key", session=session)

    with pytest.raises(DeepSeekTransportError):
        client.complete_json(system_prompt="system", user_prompt="user")

    assert len(session.calls) == 1


@pytest.mark.parametrize("status_code", [429, 500, 503])
def test_retryable_http_status_retries_once_and_returns_second_response(
    status_code: int,
) -> None:
    session = SequenceSession(
        [
            FakeResponse({}, status_code=status_code),
            FakeResponse(successful_payload()),
        ]
    )
    recorder = SleepRecorder()
    client = DeepSeekClient(
        api_key="fake-deepseek-key",
        session=session,
        sleep=recorder,
    )

    completion = client.complete_json(system_prompt="system", user_prompt="user")

    assert completion.finish_reason == "stop"
    assert len(session.calls) == 2
    assert len(recorder.delays) == 1


def test_non_retryable_http_status_is_not_retried() -> None:
    session = FakeSession(FakeResponse({}, status_code=401))
    client = DeepSeekClient(api_key="fake-deepseek-key", session=session)

    with pytest.raises(DeepSeekApiError):
        client.complete_json(system_prompt="system", user_prompt="user")

    assert len(session.calls) == 1


@pytest.mark.parametrize("status_code", [401, 429, 500])
def test_non_success_status_has_sanitized_api_error(status_code: int) -> None:
    session = FakeSession(FakeResponse({}, status_code=status_code))
    client = DeepSeekClient(api_key="sensitive-test-key", session=session, sleep=SleepRecorder())

    with pytest.raises(DeepSeekApiError) as error:
        client.complete_json(system_prompt="system", user_prompt="user")

    assert error.value.status_code == status_code
    assert "sensitive-test-key" not in str(error.value)


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, json_error=ValueError("invalid JSON")),
        FakeResponse([]),
        FakeResponse({}),
        FakeResponse({"choices": [{"finish_reason": "length", "message": {}}]}),
        FakeResponse({"choices": [{"finish_reason": "stop", "message": {}}]}),
    ],
    ids=["invalid-json", "non-object", "no-choice", "not-stopped", "empty-content"],
)
def test_rejects_unusable_completion_responses(response: FakeResponse) -> None:
    client = DeepSeekClient(
        api_key="fake-deepseek-key",
        session=FakeSession(response),
    )

    with pytest.raises(DeepSeekResponseError):
        client.complete_json(system_prompt="system", user_prompt="user")


def test_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        DeepSeekClient(api_key="fake-deepseek-key", timeout_seconds=0)


def test_rejects_non_positive_max_attempts() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        DeepSeekClient(api_key="fake-deepseek-key", max_attempts=0)


@pytest.mark.parametrize("field", ["backoff_base_seconds", "total_budget_seconds"])
def test_rejects_non_positive_retry_pacing(field: str) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        DeepSeekClient(api_key="fake-deepseek-key", **{field: 0})


def test_saturated_pool_is_retried_up_to_the_attempt_limit_with_growing_backoff() -> None:
    session = SequenceSession([resource_exhausted_response() for _ in range(DEEPSEEK_MAX_ATTEMPTS)])
    recorder = SleepRecorder()
    client = DeepSeekClient(
        api_key="fake-deepseek-key",
        session=session,
        sleep=recorder,
    )

    with pytest.raises(DeepSeekApiError) as error:
        client.complete_json(system_prompt="system", user_prompt="user")

    assert error.value.status_code == 503
    assert len(session.calls) == DEEPSEEK_MAX_ATTEMPTS
    assert len(recorder.delays) == DEEPSEEK_MAX_ATTEMPTS - 1
    # Each wait is longer than the last, so a retry never lands on the same busy worker.
    assert recorder.delays == sorted(recorder.delays)
    assert recorder.delays[0] >= DEEPSEEK_BACKOFF_BASE_SECONDS


def test_backoff_delay_stays_within_jittered_exponential_bounds() -> None:
    session = SequenceSession([resource_exhausted_response() for _ in range(DEEPSEEK_MAX_ATTEMPTS)])
    recorder = SleepRecorder()
    client = DeepSeekClient(api_key="fake-deepseek-key", session=session, sleep=recorder)

    with pytest.raises(DeepSeekApiError):
        client.complete_json(system_prompt="system", user_prompt="user")

    for index, delay in enumerate(recorder.delays):
        base = min(DEEPSEEK_BACKOFF_BASE_SECONDS * 2**index, DEEPSEEK_BACKOFF_MAX_SECONDS)
        assert base <= delay <= base * (1 + DEEPSEEK_BACKOFF_JITTER_RATIO)


def test_retries_stop_once_the_total_budget_would_be_exceeded() -> None:
    session = SequenceSession([resource_exhausted_response() for _ in range(DEEPSEEK_MAX_ATTEMPTS)])
    recorder = SleepRecorder()
    client = DeepSeekClient(
        api_key="fake-deepseek-key",
        session=session,
        sleep=recorder,
        total_budget_seconds=1.5,
    )

    with pytest.raises(DeepSeekApiError):
        client.complete_json(system_prompt="system", user_prompt="user")

    # The budget is smaller than the second backoff, so only one retry fits.
    assert len(session.calls) == 2
    assert len(recorder.delays) == 1


def test_slow_attempts_do_not_overrun_the_mcp_tool_budget() -> None:
    session = SequenceSession([resource_exhausted_response() for _ in range(DEEPSEEK_MAX_ATTEMPTS)])
    recorder = SleepRecorder()
    ticks = iter([0.0, 44.5, 89.0, 133.5, 178.0])
    client = DeepSeekClient(
        api_key="fake-deepseek-key",
        session=session,
        sleep=recorder,
        clock=lambda: next(ticks),
    )

    with pytest.raises(DeepSeekApiError):
        client.complete_json(system_prompt="system", user_prompt="user")

    assert recorder.delays == []
    assert len(session.calls) == 1


@pytest.mark.parametrize(
    ("body", "provider_reason"),
    [
        (
            {
                "error": {
                    "message": (
                        "ResourceExhausted: Worker local total request limit reached (48/48)"
                    ),
                    "type": "Service Unavailable",
                    "code": 503,
                }
            },
            "resource_exhausted",
        ),
        ({"error": {"message": "Rate limit exceeded for this key"}}, "rate_limited"),
        ({"error": {"message": "Invalid API key provided"}}, "unauthorized"),
        ({"detail": "Model not found"}, "model_unavailable"),
        ({"message": "Unsupported parameter: reasoning_effort"}, "invalid_request"),
        ({"error": {"message": "something entirely new"}}, "unspecified"),
        ({}, "unspecified"),
        ([], "unspecified"),
    ],
    ids=[
        "resource-exhausted",
        "rate-limited",
        "unauthorized",
        "model-unavailable",
        "invalid-request",
        "unknown-message",
        "empty-object",
        "non-object",
    ],
)
def test_rejection_body_is_distilled_into_a_fixed_provider_reason(
    body: object,
    provider_reason: str,
) -> None:
    session = FakeSession(FakeResponse(body, status_code=503))
    client = DeepSeekClient(api_key="sensitive-test-key", session=session, sleep=SleepRecorder())

    with pytest.raises(DeepSeekApiError) as error:
        client.complete_json(system_prompt="system", user_prompt="user")

    assert error.value.provider_reason == provider_reason


def test_provider_reason_never_leaks_the_body_or_the_api_key() -> None:
    session = FakeSession(
        FakeResponse(
            {"error": {"message": "ResourceExhausted for key sensitive-test-key at worker 12"}},
            status_code=503,
        )
    )
    client = DeepSeekClient(api_key="sensitive-test-key", session=session, sleep=SleepRecorder())

    with pytest.raises(DeepSeekApiError) as error:
        client.complete_json(system_prompt="system", user_prompt="user")

    assert error.value.provider_reason == "resource_exhausted"
    assert "sensitive-test-key" not in str(error.value)
    assert "worker 12" not in str(error.value)


def test_unparsable_rejection_body_falls_back_to_unspecified_reason() -> None:
    session = FakeSession(
        FakeResponse({}, status_code=503, json_error=ValueError("invalid JSON")),
    )
    client = DeepSeekClient(api_key="fake-deepseek-key", session=session, sleep=SleepRecorder())

    with pytest.raises(DeepSeekApiError) as error:
        client.complete_json(system_prompt="system", user_prompt="user")

    assert error.value.provider_reason == "unspecified"


def test_unknown_provider_reason_falls_back_to_generic_reason() -> None:
    error = DeepSeekApiError(503, provider_reason="sensitive-provider-detail")

    assert error.provider_reason == "unspecified"
