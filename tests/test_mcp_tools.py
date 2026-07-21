import json
from pathlib import Path

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from fng_chatbot.deepseek import (
    NVIDIA_DEEPSEEK_MODEL,
    DeepSeekApiError,
    DeepSeekConfigurationError,
    DeepSeekResponseError,
    DeepSeekTransportError,
)
from fng_chatbot.errors import CnnNetworkError
from fng_chatbot.interpretation import (
    InterpretationContext,
    InterpretationValidationError,
    ReportInterpretation,
)
from fng_chatbot.market_data import MarketOverview, MarketQuote
from fng_chatbot.mcp_server import (
    FngMcpService,
    create_mcp_server,
    hash_telegram_preview,
)
from fng_chatbot.message_renderer import escape_markdown_v2
from fng_chatbot.normalizer import normalize_cnn_payload
from fng_chatbot.telegram import (
    TelegramReportService,
    TelegramSendResult,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"


@pytest.fixture
def anyio_backend():
    return "asyncio"


def load_snapshot():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")


class FakeSnapshotProvider:
    def __init__(self, error: Exception | None = None) -> None:
        self.snapshot = load_snapshot()
        self.error = error
        self.calls: list[bool] = []

    def get_snapshot(self, *, force_refresh: bool = False):
        self.calls.append(force_refresh)
        if self.error is not None:
            raise self.error
        return self.snapshot


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
        return TelegramSendResult(message_id=789, idempotency_key=idempotency_key)


class FakeMarketOverviewProvider:
    def __init__(self) -> None:
        self.calls = 0

    def get_overview(self) -> MarketOverview:
        self.calls += 1
        return MarketOverview(
            quotes=(
                MarketQuote(
                    id="kospi",
                    name_ko="코스피",
                    price=2795.46,
                    change_percent=0.66,
                    unit="포인트",
                    as_of="2026-07-17T18:05:00+09:00",
                    source="Naver Finance",
                ),
            ),
            missing_quote_ids=("bitcoin",),
            issues=("bitcoin: unavailable",),
            fetched_at="2026-07-20T00:00:00Z",
        )


class FakeInterpretationProvider:
    def __init__(
        self,
        *,
        lines: tuple[str, ...] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.lines = lines or (
            "선택된 공포 요인들은 위험 회피 심리가 강하다는 뜻입니다.",
            "완충 요인은 공포가 시장 전반으로 확산되는 압력을 일부 제한합니다.",
        )
        self.error = error
        self.calls: list[InterpretationContext] = []

    def interpret(self, context: InterpretationContext) -> ReportInterpretation:
        self.calls.append(context)
        if self.error is not None:
            raise self.error
        return ReportInterpretation(
            lines=self.lines,
            referenced_indicator_ids=context.selected_indicator_ids,
            model=NVIDIA_DEEPSEEK_MODEL,
        )


def build_service(
    *,
    snapshot_provider: FakeSnapshotProvider | None = None,
    sender: FakeSender | None = None,
    market_overview_provider: FakeMarketOverviewProvider | None = None,
    interpretation_provider: FakeInterpretationProvider | None = None,
    allow_send: bool = False,
):
    snapshot_provider = snapshot_provider or FakeSnapshotProvider()
    service = FngMcpService(
        snapshot_provider,
        telegram_service=TelegramReportService(sender),
        market_overview_provider=market_overview_provider,
        interpretation_provider=interpretation_provider,
        allow_telegram_send=allow_send,
    )
    return service, snapshot_provider


def test_get_report_contains_complete_deterministic_contract() -> None:
    interpretations = FakeInterpretationProvider()
    service, snapshots = build_service(interpretation_provider=interpretations)

    report = service.get_fear_greed_report(force_refresh=True)

    assert report["source"] == "CNN Fear & Greed"
    assert report["overall"]["score"] == 37.0
    assert len(report["indicators"]) == 7
    assert len({item["id"] for item in report["indicators"]}) == 7
    assert report["fear_drivers"]
    assert report["buffer"] is not None
    assert report["summary_source"] == "rules"
    assert "interpretation_context" not in report
    assert report["data_quality"] == {
        "complete": True,
        "missing_fields": [],
        "cached": False,
        "stale": False,
    }
    assert snapshots.calls == [True]
    assert interpretations.calls == []


def test_every_selected_factor_is_grounded_in_returned_indicators() -> None:
    service, _ = build_service()

    report = service.get_fear_greed_report()

    indicators = {item["id"]: item for item in report["indicators"]}
    factors = [*report["fear_drivers"], report["buffer"]]
    for factor in factors:
        source = indicators[factor["indicator_id"]]
        assert factor["indicator"] == source["name_ko"]
        assert factor["score"] == source["score"]
        assert factor["rating"] == source["rating"]


def test_same_snapshot_always_returns_the_same_report() -> None:
    service, _ = build_service()

    assert service.get_fear_greed_report() == service.get_fear_greed_report()


def test_cnn_failure_is_propagated_as_report_failure() -> None:
    snapshots = FakeSnapshotProvider(CnnNetworkError("CNN unavailable"))
    service, _ = build_service(snapshot_provider=snapshots)

    with pytest.raises(CnnNetworkError, match="CNN unavailable"):
        service.get_fear_greed_report()


def test_preview_is_read_only_and_returns_hash_for_exact_text() -> None:
    sender = FakeSender()
    service, _ = build_service(sender=sender)

    result = service.preview_telegram_report()

    telegram = result["telegram"]
    assert telegram["sent"] is False
    assert telegram["parse_mode"] == "MarkdownV2"
    assert telegram["character_count"] == len(telegram["text"])
    assert telegram["preview_hash"] == hash_telegram_preview(telegram["text"])
    assert result["report"]["summary_source"] == "rules"
    assert result["interpretation"]["status"] == "rules_fallback"
    assert result["interpretation"]["reason"] == "provider_not_configured"
    assert sender.calls == []


def test_preview_applies_deepseek_interpretation_and_hashes_final_text() -> None:
    interpretations = FakeInterpretationProvider()
    service, _ = build_service(interpretation_provider=interpretations)

    result = service.preview_telegram_report()

    metadata = result["interpretation"]
    telegram = result["telegram"]
    assert metadata == {
        "status": "applied",
        "source": "deepseek",
        "model": NVIDIA_DEEPSEEK_MODEL,
        "lines": list(interpretations.lines),
        "referenced_indicator_ids": [
            "stock_price_strength",
            "safe_haven_demand",
            "put_call_options",
        ],
    }
    assert len(interpretations.calls) == 1
    assert interpretations.calls[0].selected_indicator_ids == tuple(
        metadata["referenced_indicator_ids"]
    )
    assert all(escape_markdown_v2(line) in telegram["text"] for line in interpretations.lines)
    assert NVIDIA_DEEPSEEK_MODEL not in telegram["text"]
    assert telegram["preview_hash"] == hash_telegram_preview(telegram["text"])


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (DeepSeekConfigurationError("sensitive configuration detail"), "configuration_error"),
        (DeepSeekTransportError("sensitive transport detail"), "transport_error"),
        (
            DeepSeekTransportError(
                "sensitive transport detail",
                reason_code="transport_read_timeout",
            ),
            "transport_read_timeout",
        ),
        (DeepSeekApiError(429), "http_429"),
        (
            DeepSeekApiError(503, provider_reason="resource_exhausted"),
            "http_503_resource_exhausted",
        ),
        (DeepSeekApiError(401, provider_reason="unauthorized"), "http_401_unauthorized"),
        (
            DeepSeekApiError(503, provider_reason="sensitive-provider-detail"),
            "http_503",
        ),
        (DeepSeekResponseError("sensitive response detail"), "response_error"),
        (InterpretationValidationError("sensitive validation detail"), "validation_error"),
        (
            InterpretationValidationError(
                "sensitive validation detail",
                reason_code="validation_id_mismatch",
            ),
            "validation_id_mismatch",
        ),
        (RuntimeError("sensitive unexpected detail"), "provider_error"),
    ],
)
def test_preview_falls_back_without_exposing_provider_error(
    error: Exception,
    reason: str,
) -> None:
    interpretations = FakeInterpretationProvider(error=error)
    service, _ = build_service(interpretation_provider=interpretations)

    result = service.preview_telegram_report()

    assert result["interpretation"] == {
        "status": "rules_fallback",
        "source": "rules",
        "model": None,
        "reason": reason,
        "lines": [],
        "referenced_indicator_ids": [],
    }
    assert "sensitive" not in str(result)
    assert "설명: AI 보조" not in result["telegram"]["text"]
    assert result["telegram"]["preview_hash"] == hash_telegram_preview(result["telegram"]["text"])


def test_preview_combines_market_quotes_with_fng_report() -> None:
    markets = FakeMarketOverviewProvider()
    service, _ = build_service(market_overview_provider=markets)

    result = service.preview_telegram_report()

    text = result["telegram"]["text"]
    assert text.startswith("🌅 *아침 시장 브리핑*")
    assert "코스피" in text
    assert "📊 *F&G 시장심리*" in text
    assert "*전일 대비 심리*" in text
    assert result["market_overview"]["data_quality"]["missing_quote_ids"] == ["bitcoin"]
    assert result["telegram"]["preview_hash"] == hash_telegram_preview(text)
    assert markets.calls == 1


def test_send_is_disabled_by_default_even_with_confirmation() -> None:
    sender = FakeSender()
    service, _ = build_service(sender=sender)
    text = "reviewed preview"

    with pytest.raises(PermissionError, match="disabled"):
        service.send_telegram_report(
            preview_text=text,
            preview_hash=hash_telegram_preview(text),
            idempotency_key="report:2026-07-20",
            confirm_send=True,
        )

    assert sender.calls == []


def test_send_requires_confirmation_and_matching_preview_hash() -> None:
    sender = FakeSender()
    service, _ = build_service(sender=sender, allow_send=True)
    text = "reviewed preview"

    with pytest.raises(PermissionError, match="confirm_send=true"):
        service.send_telegram_report(
            preview_text=text,
            preview_hash=hash_telegram_preview(text),
            idempotency_key="report:2026-07-20",
        )
    with pytest.raises(ValueError, match="does not match"):
        service.send_telegram_report(
            preview_text=text,
            preview_hash="wrong",
            idempotency_key="report:2026-07-20",
            confirm_send=True,
        )

    assert sender.calls == []


def test_send_delivers_exact_reviewed_preview() -> None:
    sender = FakeSender()
    service, _ = build_service(sender=sender, allow_send=True)
    preview = service.preview_telegram_report()["telegram"]

    result = service.send_telegram_report(
        preview_text=preview["text"],
        preview_hash=preview["preview_hash"],
        idempotency_key="report:2026-07-20",
        confirm_send=True,
    )

    assert result["sent"] is True
    assert result["message_id"] == 789
    assert sender.calls[0]["text"] == preview["text"]


@pytest.mark.anyio
async def test_mcp_client_lists_three_tools_and_calls_structured_report() -> None:
    service, _ = build_service()
    server = create_mcp_server(service)

    async with create_connected_server_and_client_session(server, raise_exceptions=True) as session:
        listed = await session.list_tools()
        by_name = {tool.name: tool for tool in listed.tools}
        result = await session.call_tool("get_fear_greed_report", {})

    assert set(by_name) == {
        "get_fear_greed_report",
        "preview_telegram_report",
        "send_telegram_report",
    }
    assert by_name["get_fear_greed_report"].annotations.readOnlyHint is True
    assert by_name["preview_telegram_report"].annotations.readOnlyHint is True
    assert by_name["send_telegram_report"].annotations.readOnlyHint is False
    assert "confirm_send" in by_name["send_telegram_report"].inputSchema["properties"]
    assert "provider" not in by_name["get_fear_greed_report"].inputSchema["properties"]
    preview_properties = by_name["preview_telegram_report"].inputSchema["properties"]
    assert set(preview_properties) == {"force_refresh"}
    assert result.isError is False
    assert result.structuredContent["summary_source"] == "rules"
    assert len(result.structuredContent["indicators"]) == 7


@pytest.mark.anyio
async def test_mcp_client_receives_tool_error_only_when_cnn_report_fails() -> None:
    snapshots = FakeSnapshotProvider(CnnNetworkError("CNN unavailable"))
    service, _ = build_service(snapshot_provider=snapshots)
    server = create_mcp_server(service)

    async with create_connected_server_and_client_session(server) as session:
        result = await session.call_tool("get_fear_greed_report", {})

    assert result.isError is True
    assert "CNN unavailable" in result.content[0].text
