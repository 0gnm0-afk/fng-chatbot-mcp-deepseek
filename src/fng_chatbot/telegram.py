"""Telegram preview and explicit send boundaries."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

import httpx

from fng_chatbot.interpretation import ReportInterpretation
from fng_chatbot.message_renderer import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    TELEGRAM_PARSE_MODE,
    render_telegram_message,
)
from fng_chatbot.models import FearGreedSnapshot
from fng_chatbot.report import StaticReport

TELEGRAM_API_BASE_URL = "https://api.telegram.org"
_TELEGRAM_BOT_URL_PATTERN = re.compile(
    r"(?P<prefix>https?://api\.telegram\.org/bot)[^/\s'\"]+(?P<suffix>/sendMessage)"
)


def _redact_telegram_bot_url(value: object) -> object:
    rendered = str(value)
    redacted = _TELEGRAM_BOT_URL_PATTERN.sub(
        r"\g<prefix>[REDACTED]\g<suffix>",
        rendered,
    )
    return redacted if redacted != rendered else value


class _TelegramBotUrlRedactionFilter(logging.Filter):
    """Remove Bot API credentials before httpx formats a log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = _redact_telegram_bot_url(record.msg)
        if isinstance(record.args, tuple):
            record.args = tuple(_redact_telegram_bot_url(value) for value in record.args)
        elif isinstance(record.args, Mapping):
            record.args = {
                key: _redact_telegram_bot_url(value) for key, value in record.args.items()
            }
        return True


_TELEGRAM_BOT_URL_REDACTION_FILTER = _TelegramBotUrlRedactionFilter()


def _install_httpx_telegram_redaction() -> None:
    logger = logging.getLogger("httpx")
    if _TELEGRAM_BOT_URL_REDACTION_FILTER not in logger.filters:
        logger.addFilter(_TELEGRAM_BOT_URL_REDACTION_FILTER)


class TelegramError(RuntimeError):
    """Base error for the external Telegram boundary."""


class TelegramConfigurationError(TelegramError):
    """Telegram sending was requested without complete configuration."""


class TelegramTransportError(TelegramError):
    """Telegram could not be reached."""


class TelegramApiError(TelegramError):
    """Telegram rejected a send request or returned an invalid response."""


@dataclass(frozen=True, slots=True)
class TelegramPreview:
    text: str
    parse_mode: str
    character_count: int


@dataclass(frozen=True, slots=True)
class TelegramSendResult:
    message_id: int
    idempotency_key: str


class TelegramSender(Protocol):
    def send_message(
        self,
        text: str,
        *,
        parse_mode: str,
        idempotency_key: str,
    ) -> TelegramSendResult: ...


class TelegramHttpResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class TelegramHttpSession(Protocol):
    def post(
        self,
        url: str,
        *,
        json: Mapping[str, object],
        timeout: httpx.Timeout,
    ) -> TelegramHttpResponse: ...


def preview_telegram_report(
    snapshot: FearGreedSnapshot,
    comment: StaticReport,
    interpretation: ReportInterpretation | None = None,
) -> TelegramPreview:
    """Render a preview without constructing or calling an HTTP client."""

    text = render_telegram_message(snapshot, comment, interpretation)
    return TelegramPreview(
        text=text,
        parse_mode=TELEGRAM_PARSE_MODE,
        character_count=len(text),
    )


class TelegramReportService:
    """Keep preview pure and require an explicit sender for delivery."""

    def __init__(self, sender: TelegramSender | None = None) -> None:
        self._sender = sender

    def preview(
        self,
        snapshot: FearGreedSnapshot,
        comment: StaticReport,
        interpretation: ReportInterpretation | None = None,
    ) -> TelegramPreview:
        return preview_telegram_report(snapshot, comment, interpretation)

    def send(
        self,
        snapshot: FearGreedSnapshot,
        comment: StaticReport,
        *,
        idempotency_key: str,
        interpretation: ReportInterpretation | None = None,
    ) -> TelegramSendResult:
        if self._sender is None:
            raise TelegramConfigurationError("Telegram sender is not configured")
        key = idempotency_key.strip()
        if not key:
            raise TelegramConfigurationError("idempotency_key must not be empty")
        preview = self.preview(snapshot, comment, interpretation)
        return self.send_preview(preview, idempotency_key=key)

    def send_preview(
        self,
        preview: TelegramPreview,
        *,
        idempotency_key: str,
    ) -> TelegramSendResult:
        """Send the exact previously reviewed preview text."""

        if self._sender is None:
            raise TelegramConfigurationError("Telegram sender is not configured")
        key = idempotency_key.strip()
        if not key:
            raise TelegramConfigurationError("idempotency_key must not be empty")
        if preview.character_count != len(preview.text):
            raise TelegramConfigurationError("Telegram preview character count is invalid")
        return self._sender.send_message(
            preview.text,
            parse_mode=preview.parse_mode,
            idempotency_key=key,
        )


class TelegramBotClient:
    """Send an already-rendered message through Telegram Bot API sendMessage."""

    def __init__(
        self,
        *,
        bot_token: str | None = None,
        chat_id: str | None = None,
        session: TelegramHttpSession | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        _install_httpx_telegram_redaction()
        self._bot_token = bot_token if bot_token is not None else os.getenv("TELEGRAM_BOT_TOKEN")
        self._chat_id = chat_id if chat_id is not None else os.getenv("TELEGRAM_CHAT_ID")
        self._owns_session = session is None
        self._session: TelegramHttpSession = session or httpx.Client()
        self._timeout = httpx.Timeout(timeout_seconds)

    def send_message(
        self,
        text: str,
        *,
        parse_mode: str,
        idempotency_key: str,
    ) -> TelegramSendResult:
        """Send exactly once; retry and duplicate prevention belong to later orchestration."""

        if not self._bot_token or not self._chat_id:
            raise TelegramConfigurationError("Telegram bot token and chat ID are required")
        if not text or len(text) > TELEGRAM_MAX_MESSAGE_LENGTH:
            raise TelegramConfigurationError("Telegram text must contain 1 to 4096 characters")
        if parse_mode != TELEGRAM_PARSE_MODE:
            raise TelegramConfigurationError("Telegram parse mode must be MarkdownV2")

        try:
            response = self._session.post(
                f"{TELEGRAM_API_BASE_URL}/bot{self._bot_token}/sendMessage",
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=self._timeout,
            )
        except httpx.RequestError as error:
            raise TelegramTransportError("Telegram sendMessage request failed") from error

        if not 200 <= response.status_code < 300:
            raise TelegramApiError(f"Telegram sendMessage returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as error:
            raise TelegramApiError("Telegram sendMessage returned invalid JSON") from error
        message_id = _extract_message_id(payload)
        return TelegramSendResult(
            message_id=message_id,
            idempotency_key=idempotency_key,
        )

    def close(self) -> None:
        if self._owns_session and isinstance(self._session, httpx.Client):
            self._session.close()

    def __enter__(self) -> TelegramBotClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _extract_message_id(payload: object) -> int:
    if not isinstance(payload, Mapping) or payload.get("ok") is not True:
        raise TelegramApiError("Telegram rejected sendMessage")
    result = payload.get("result")
    if not isinstance(result, Mapping):
        raise TelegramApiError("Telegram sendMessage response has no result")
    message_id = result.get("message_id")
    if isinstance(message_id, bool) or not isinstance(message_id, int):
        raise TelegramApiError("Telegram sendMessage response has no message ID")
    return message_id
