import json
from collections.abc import Mapping
from pathlib import Path

import httpx
import pytest

from fng_chatbot.message_renderer import TELEGRAM_PARSE_MODE
from fng_chatbot.normalizer import normalize_cnn_payload
from fng_chatbot.report import build_static_report
from fng_chatbot.telegram import (
    TELEGRAM_API_BASE_URL,
    TelegramApiError,
    TelegramBotClient,
    TelegramConfigurationError,
    TelegramReportService,
    TelegramSendResult,
    TelegramTransportError,
    preview_telegram_report,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"


class FakeSender:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    def send_message(
        self,
        text: str,
        *,
        parse_mode: str,
        idempotency_key: str,
    ) -> TelegramSendResult:
        self.calls.append(
            {
                "text": text,
                "parse_mode": parse_mode,
                "idempotency_key": idempotency_key,
            }
        )
        return TelegramSendResult(message_id=123, idempotency_key=idempotency_key)


class FakeResponse:
    def __init__(self, payload: object, status_code: int = 200) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, outcomes: list[FakeResponse | Exception]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object],
        timeout: httpx.Timeout,
    ) -> FakeResponse:
        self.calls.append({"url": url, "json": dict(json), "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def load_report_inputs():
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)
    snapshot = normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")
    return snapshot, build_static_report(snapshot)


def test_preview_has_no_sender_or_network_side_effect() -> None:
    snapshot, comment = load_report_inputs()
    sender = FakeSender()
    service = TelegramReportService(sender)

    preview = service.preview(snapshot, comment)
    direct_preview = preview_telegram_report(snapshot, comment)

    assert preview == direct_preview
    assert preview.parse_mode == TELEGRAM_PARSE_MODE
    assert preview.character_count == len(preview.text)
    assert sender.calls == []


def test_send_is_explicit_and_passes_idempotency_key_to_sender() -> None:
    snapshot, comment = load_report_inputs()
    sender = FakeSender()
    service = TelegramReportService(sender)

    result = service.send(snapshot, comment, idempotency_key="daily-report:2026-07-20")

    assert result == TelegramSendResult(
        message_id=123,
        idempotency_key="daily-report:2026-07-20",
    )
    assert len(sender.calls) == 1
    assert sender.calls[0]["parse_mode"] == TELEGRAM_PARSE_MODE
    assert sender.calls[0]["idempotency_key"] == "daily-report:2026-07-20"


def test_send_requires_sender_and_nonempty_idempotency_key() -> None:
    snapshot, comment = load_report_inputs()

    with pytest.raises(TelegramConfigurationError, match="sender is not configured"):
        TelegramReportService().send(
            snapshot,
            comment,
            idempotency_key="daily-report:2026-07-20",
        )

    sender = FakeSender()
    with pytest.raises(TelegramConfigurationError, match="idempotency_key"):
        TelegramReportService(sender).send(snapshot, comment, idempotency_key="  ")
    assert sender.calls == []


def test_bot_client_posts_send_message_once() -> None:
    session = FakeSession([FakeResponse({"ok": True, "result": {"message_id": 456}})])
    client = TelegramBotClient(
        bot_token="fake-bot-token",
        chat_id="fake-chat-id",
        session=session,
    )

    result = client.send_message(
        "안전한 메시지",
        parse_mode=TELEGRAM_PARSE_MODE,
        idempotency_key="daily-report:2026-07-20",
    )

    assert result.message_id == 456
    assert result.idempotency_key == "daily-report:2026-07-20"
    assert len(session.calls) == 1
    assert session.calls[0]["url"] == (f"{TELEGRAM_API_BASE_URL}/botfake-bot-token/sendMessage")
    assert session.calls[0]["json"] == {
        "chat_id": "fake-chat-id",
        "text": "안전한 메시지",
        "parse_mode": "MarkdownV2",
    }
    assert "idempotency_key" not in session.calls[0]["json"]


@pytest.mark.parametrize(
    ("token", "chat_id"),
    [("", "fake-chat-id"), ("fake-bot-token", "")],
)
def test_bot_client_rejects_missing_configuration_without_network(
    token: str,
    chat_id: str,
) -> None:
    session = FakeSession([])
    client = TelegramBotClient(bot_token=token, chat_id=chat_id, session=session)

    with pytest.raises(TelegramConfigurationError, match="token and chat ID"):
        client.send_message(
            "message",
            parse_mode=TELEGRAM_PARSE_MODE,
            idempotency_key="daily-report:2026-07-20",
        )

    assert session.calls == []


@pytest.mark.parametrize("text", ["", "x" * 4097])
def test_bot_client_rejects_invalid_length_without_network(text: str) -> None:
    session = FakeSession([])
    client = TelegramBotClient(
        bot_token="fake-bot-token",
        chat_id="fake-chat-id",
        session=session,
    )

    with pytest.raises(TelegramConfigurationError, match="1 to 4096"):
        client.send_message(
            text,
            parse_mode=TELEGRAM_PARSE_MODE,
            idempotency_key="daily-report:2026-07-20",
        )

    assert session.calls == []


def test_bot_client_does_not_retry_ambiguous_transport_failure() -> None:
    session = FakeSession([httpx.ReadTimeout("timed out")])
    client = TelegramBotClient(
        bot_token="fake-bot-token",
        chat_id="fake-chat-id",
        session=session,
    )

    with pytest.raises(TelegramTransportError, match="request failed"):
        client.send_message(
            "message",
            parse_mode=TELEGRAM_PARSE_MODE,
            idempotency_key="daily-report:2026-07-20",
        )

    assert len(session.calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse({}, status_code=429),
        FakeResponse({"ok": False}),
        FakeResponse(ValueError("not json")),
        FakeResponse({"ok": True, "result": {}}),
    ],
    ids=["http-error", "api-rejection", "invalid-json", "missing-message-id"],
)
def test_bot_client_reports_api_failures(response: FakeResponse) -> None:
    session = FakeSession([response])
    client = TelegramBotClient(
        bot_token="fake-bot-token",
        chat_id="fake-chat-id",
        session=session,
    )

    with pytest.raises(TelegramApiError):
        client.send_message(
            "message",
            parse_mode=TELEGRAM_PARSE_MODE,
            idempotency_key="daily-report:2026-07-20",
        )

    assert len(session.calls) == 1
