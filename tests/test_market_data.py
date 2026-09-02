import json
from collections.abc import Mapping
from pathlib import Path

import httpx

from fng_chatbot.market_data import (
    NAVER_KOSPI_URL,
    YAHOO_CHART_BASE_URL,
    MarketOverviewClient,
)

FIXTURES = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload: object, *, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> object:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(
        self,
        naver_response: FakeResponse,
        yahoo_response: FakeResponse,
        *,
        failed_symbol: str | None = None,
    ) -> None:
        self.naver_response = naver_response
        self.yahoo_response = yahoo_response
        self.failed_symbol = failed_symbol
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout,
    ) -> FakeResponse:
        self.calls.append(url)
        assert headers and headers["Accept"] == "application/json"
        assert isinstance(timeout, httpx.Timeout)
        if url == NAVER_KOSPI_URL:
            return self.naver_response
        assert url.startswith(YAHOO_CHART_BASE_URL)
        assert params == {"range": "5d", "interval": "1d"}
        if self.failed_symbol and url.endswith(self.failed_symbol):
            return FakeResponse({}, status_code=503)
        return self.yahoo_response


def load_fixture(name: str) -> object:
    with (FIXTURES / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_collects_naver_kospi_and_six_yahoo_quotes() -> None:
    session = FakeSession(
        FakeResponse(load_fixture("naver_kospi_basic.json")),
        FakeResponse(load_fixture("yahoo_chart_quote.json")),
    )

    overview = MarketOverviewClient(session=session).get_overview()

    assert len(overview.quotes) == 7
    assert overview.complete is True
    assert overview.missing_quote_ids == ()
    assert set(overview.to_dict()) == {"quotes", "data_quality", "fetched_at"}
    kospi = overview.quotes[0]
    assert kospi.id == "kospi"
    assert kospi.price == 2795.46
    assert kospi.change_percent == 0.66
    assert kospi.source == "Naver Finance"
    assert overview.quotes[1].change_percent == (25.5 / 1400.0) * 100
    assert set(overview.to_dict()["data_quality"]) == {
        "complete",
        "missing_quote_ids",
        "issues",
    }
    assert len(session.calls) == 7


def test_naver_failure_uses_yahoo_kospi_fallback() -> None:
    session = FakeSession(
        FakeResponse({}, status_code=404),
        FakeResponse(load_fixture("yahoo_chart_quote.json")),
    )

    overview = MarketOverviewClient(session=session).get_overview()

    kospi = next(quote for quote in overview.quotes if quote.id == "kospi")
    assert kospi.source == "Yahoo Finance"
    assert overview.complete is True
    assert any(url.endswith("%5EKS11") for url in session.calls)


def test_one_quote_failure_is_reported_without_hiding_other_quotes() -> None:
    session = FakeSession(
        FakeResponse(load_fixture("naver_kospi_basic.json")),
        FakeResponse(load_fixture("yahoo_chart_quote.json")),
        failed_symbol="BTC-USD",
    )

    overview = MarketOverviewClient(session=session).get_overview()

    assert len(overview.quotes) == 6
    assert overview.complete is False
    assert overview.missing_quote_ids == ("bitcoin",)
    assert overview.issues == ("bitcoin: unavailable",)
    assert {quote.id for quote in overview.quotes} == {
        "kospi",
        "usd_krw",
        "nasdaq",
        "nasdaq_futures",
        "sp500",
        "sp500_futures",
    }


def test_invalid_json_isolated_as_missing_quote() -> None:
    session = FakeSession(
        FakeResponse(ValueError("not json")),
        FakeResponse(ValueError("not json")),
    )

    overview = MarketOverviewClient(session=session).get_overview()

    assert overview.quotes == ()
    assert overview.complete is False
    assert "kospi" in overview.missing_quote_ids
    assert "usd_krw" in overview.missing_quote_ids
