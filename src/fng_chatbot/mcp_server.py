"""MCP tools for structured market reports and guarded Telegram delivery."""

from __future__ import annotations

import hashlib
import hmac
import os
from typing import Protocol

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from fng_chatbot.cnn_fng_client import CnnFearGreedClient
from fng_chatbot.deepseek import (
    DeepSeekApiError,
    DeepSeekClient,
    DeepSeekConfigurationError,
    DeepSeekResponseError,
    DeepSeekTransportError,
)
from fng_chatbot.interpretation import (
    DeepSeekInterpreter,
    InterpretationProvider,
    InterpretationValidationError,
    ReportInterpretation,
    build_interpretation_context,
    rules_fallback_metadata,
)
from fng_chatbot.market_data import (
    MarketOverview,
    MarketOverviewClient,
    MarketOverviewProvider,
)
from fng_chatbot.message_renderer import render_morning_telegram_message
from fng_chatbot.models import FearGreedSnapshot
from fng_chatbot.report import StaticReport, build_static_report
from fng_chatbot.telegram import (
    TELEGRAM_PARSE_MODE,
    TelegramBotClient,
    TelegramPreview,
    TelegramReportService,
)


class SnapshotProvider(Protocol):
    def get_snapshot(self, *, force_refresh: bool = False) -> FearGreedSnapshot: ...


class FngMcpService:
    """Application service shared by MCP transport tools and unit tests."""

    def __init__(
        self,
        snapshot_provider: SnapshotProvider,
        *,
        telegram_service: TelegramReportService | None = None,
        market_overview_provider: MarketOverviewProvider | None = None,
        interpretation_provider: InterpretationProvider | None = None,
        allow_telegram_send: bool = False,
    ) -> None:
        self._snapshot_provider = snapshot_provider
        self._telegram_service = telegram_service or TelegramReportService()
        self._market_overview_provider = market_overview_provider
        self._interpretation_provider = interpretation_provider
        self._allow_telegram_send = allow_telegram_send

    def get_fear_greed_report(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[str, object]:
        """Fetch one snapshot and return normalized facts plus a rules-based report."""

        snapshot, comment = self._build_report(force_refresh=force_refresh)
        return _serialize_report(snapshot, comment)

    def preview_telegram_report(
        self,
        *,
        force_refresh: bool = False,
    ) -> dict[str, object]:
        """Build a report and Telegram preview without sending anything."""

        snapshot, comment = self._build_report(force_refresh=force_refresh)
        interpretation, interpretation_metadata = self._build_interpretation(
            snapshot,
            comment,
        )
        preview = self._telegram_service.preview(snapshot, comment, interpretation)
        result: dict[str, object] = {
            "report": _serialize_report(snapshot, comment),
            "interpretation": interpretation_metadata,
        }
        if self._market_overview_provider is not None:
            try:
                market_overview = self._market_overview_provider.get_overview()
            except Exception:  # A market add-on must never hide the core F&G report.
                market_overview = MarketOverview.unavailable(
                    reason="market_overview_provider_failed"
                )
            preview_text = render_morning_telegram_message(
                preview.text,
                market_overview,
                snapshot.composite.to_dict(),
            )
            preview = TelegramPreview(
                text=preview_text,
                parse_mode=preview.parse_mode,
                character_count=len(preview_text),
            )
            result["market_overview"] = market_overview.to_dict()

        result["telegram"] = {
            "text": preview.text,
            "parse_mode": preview.parse_mode,
            "character_count": preview.character_count,
            "preview_hash": hash_telegram_preview(preview.text),
            "sent": False,
        }
        return result

    def send_telegram_report(
        self,
        *,
        idempotency_key: str,
        force_refresh: bool = False,
    ) -> dict[str, object]:
        """Build and immediately send the current preview when delivery is enabled."""

        if not self._allow_telegram_send:
            raise PermissionError("Telegram sending is disabled by MCP_ALLOW_TELEGRAM_SEND")
        result = self.preview_telegram_report(force_refresh=force_refresh)
        telegram = result.get("telegram")
        if not isinstance(telegram, dict):
            raise RuntimeError("Telegram preview metadata is missing")
        preview_text = telegram.get("text")
        preview_hash = telegram.get("preview_hash")
        if not isinstance(preview_text, str) or not isinstance(preview_hash, str):
            raise RuntimeError("Telegram preview text or hash is missing")
        expected_hash = hash_telegram_preview(preview_text)
        if not preview_hash or not hmac.compare_digest(preview_hash, expected_hash):
            raise ValueError("preview_hash does not match preview_text")
        preview = TelegramPreview(
            text=preview_text,
            parse_mode=TELEGRAM_PARSE_MODE,
            character_count=len(preview_text),
        )
        send_result = self._telegram_service.send_preview(
            preview,
            idempotency_key=idempotency_key,
        )
        result["telegram"] = {
            **telegram,
            "sent": True,
            "message_id": send_result.message_id,
            "idempotency_key": send_result.idempotency_key,
            "preview_hash": expected_hash,
        }
        return result

    def _build_report(
        self,
        *,
        force_refresh: bool,
    ) -> tuple[FearGreedSnapshot, StaticReport]:
        snapshot = self._snapshot_provider.get_snapshot(force_refresh=force_refresh)
        return snapshot, build_static_report(snapshot)

    def _build_interpretation(
        self,
        snapshot: FearGreedSnapshot,
        comment: StaticReport,
    ) -> tuple[ReportInterpretation | None, dict[str, object]]:
        if self._interpretation_provider is None:
            return None, rules_fallback_metadata(reason="provider_not_configured")
        try:
            context = build_interpretation_context(snapshot, comment)
            interpretation = self._interpretation_provider.interpret(context)
        except Exception as error:  # The optional AI add-on must never hide the core report.
            return None, rules_fallback_metadata(reason=_interpretation_failure_reason(error))
        return interpretation, interpretation.to_dict()


def hash_telegram_preview(text: str) -> str:
    """Return a stable digest used to bind send to an exact preview."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def create_mcp_server(service: FngMcpService) -> FastMCP:
    """Register the three public tools on a local FastMCP server."""

    server = FastMCP(
        "F&G Market Sentiment",
        instructions=(
            "Inspect the seven-indicator report and data quality first. "
            "Send the current Telegram report by default unless the user requests preview-only."
        ),
        json_response=True,
    )

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    def get_fear_greed_report(
        force_refresh: bool = False,
    ) -> dict[str, object]:
        """Get CNN Fear & Greed facts and a deterministic rules-based summary."""

        return service.get_fear_greed_report(force_refresh=force_refresh)

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=True,
            destructiveHint=False,
            idempotentHint=True,
            openWorldHint=True,
        )
    )
    def preview_telegram_report(
        force_refresh: bool = False,
    ) -> dict[str, object]:
        """Render and hash the current rules-based Telegram report."""

        return service.preview_telegram_report(force_refresh=force_refresh)

    @server.tool(
        annotations=ToolAnnotations(
            readOnlyHint=False,
            destructiveHint=False,
            idempotentHint=False,
            openWorldHint=True,
        )
    )
    def send_telegram_report(
        idempotency_key: str,
        force_refresh: bool = False,
    ) -> dict[str, object]:
        """Build, hash, and immediately send the current Telegram report."""

        return service.send_telegram_report(
            idempotency_key=idempotency_key,
            force_refresh=force_refresh,
        )

    return server


def create_default_service() -> FngMcpService:
    """Build the environment-configured service without making network calls."""

    return FngMcpService(
        CnnFearGreedClient(),
        telegram_service=TelegramReportService(TelegramBotClient()),
        market_overview_provider=MarketOverviewClient(),
        interpretation_provider=DeepSeekInterpreter(DeepSeekClient()),
        allow_telegram_send=_env_flag("MCP_ALLOW_TELEGRAM_SEND", default=True),
    )


def _serialize_report(
    snapshot: FearGreedSnapshot,
    comment: StaticReport,
) -> dict[str, object]:
    return {
        "source": snapshot.source,
        "fetched_at": snapshot.fetched_at,
        "overall": snapshot.composite.to_dict(),
        "indicators": [indicator.to_dict() for indicator in snapshot.indicators],
        "fear_drivers": [factor.to_dict() for factor in comment.fear_drivers],
        "buffer": None if comment.buffer is None else comment.buffer.to_dict(),
        "fear_note": comment.fear_note,
        "summary": comment.summary,
        "summary_source": comment.summary_source,
        "data_quality": snapshot.data_quality.to_dict(),
    }


def _env_flag(name: str, *, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _interpretation_failure_reason(error: Exception) -> str:
    if isinstance(error, DeepSeekConfigurationError):
        return "configuration_error"
    if isinstance(error, DeepSeekTransportError):
        return error.reason_code
    if isinstance(error, DeepSeekApiError):
        if error.provider_reason == "unspecified":
            return f"http_{error.status_code}"
        return f"http_{error.status_code}_{error.provider_reason}"
    if isinstance(error, DeepSeekResponseError):
        return "response_error"
    if isinstance(error, InterpretationValidationError):
        return error.reason_code
    return "provider_error"


mcp = create_mcp_server(create_default_service())


def main() -> None:
    """Run the local stdio MCP server."""

    mcp.run()


if __name__ == "__main__":
    main()
