"""Small, failure-isolated clients for the morning market overview."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import quote

import httpx

NAVER_KOSPI_URL = "https://m.stock.naver.com/api/index/KOSPI/basic"
YAHOO_CHART_BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


class MarketDataError(RuntimeError):
    """One external market quote could not be validated."""


@dataclass(frozen=True, slots=True)
class MarketQuote:
    """One display-ready quote with a comparable daily percentage change."""

    id: str
    name_ko: str
    price: float
    change_percent: float
    unit: str
    as_of: str
    source: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name_ko": self.name_ko,
            "price": self.price,
            "change_percent": self.change_percent,
            "unit": self.unit,
            "as_of": self.as_of,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class MarketOverview:
    """Best-effort collection; one failed provider never hides healthy quotes."""

    quotes: tuple[MarketQuote, ...]
    missing_quote_ids: tuple[str, ...]
    issues: tuple[str, ...]
    fetched_at: str

    @property
    def complete(self) -> bool:
        return not self.missing_quote_ids

    def to_dict(self) -> dict[str, object]:
        return {
            "quotes": [item.to_dict() for item in self.quotes],
            "data_quality": {
                "complete": self.complete,
                "missing_quote_ids": list(self.missing_quote_ids),
                "issues": list(self.issues),
            },
            "fetched_at": self.fetched_at,
        }

    @classmethod
    def unavailable(cls, *, reason: str) -> MarketOverview:
        quote_ids = tuple(item[0] for item in YAHOO_QUOTES)
        return cls(
            quotes=(),
            missing_quote_ids=("kospi", *quote_ids),
            issues=(reason,),
            fetched_at=_now_iso(),
        )


class MarketOverviewProvider(Protocol):
    def get_overview(self) -> MarketOverview: ...


class MarketHttpResponse(Protocol):
    status_code: int

    def json(self) -> object: ...


class MarketHttpSession(Protocol):
    def get(
        self,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
        headers: Mapping[str, str] | None = None,
        timeout: httpx.Timeout,
    ) -> MarketHttpResponse: ...


YAHOO_QUOTES: tuple[tuple[str, str, str, str], ...] = (
    ("usd_krw", "달러·원", "KRW=X", "원"),
    ("bitcoin", "비트코인", "BTC-USD", "달러"),
    ("nasdaq", "나스닥", "^IXIC", "포인트"),
    ("nasdaq_futures", "나스닥 선물", "NQ=F", "포인트"),
    ("sp500", "S&P 500", "^GSPC", "포인트"),
    ("sp500_futures", "S&P 500 선물", "ES=F", "포인트"),
)


class MarketOverviewClient:
    """Collect Naver KOSPI and Yahoo quotes using only the existing HTTP stack."""

    def __init__(
        self,
        *,
        session: MarketHttpSession | None = None,
        timeout_seconds: float = 6.0,
    ) -> None:
        self._session = session
        self._timeout = httpx.Timeout(timeout_seconds)
        self._headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (compatible; MorningMarketReport/1.0)",
        }

    def get_overview(self) -> MarketOverview:
        """Return every valid quote and explicit quality metadata for failures."""

        if self._session is not None:
            return self._collect(self._session)
        try:
            with httpx.Client(follow_redirects=True) as session:
                return self._collect(session)
        except httpx.RequestError:
            return MarketOverview.unavailable(reason="market_transport_unavailable")

    def _collect(self, session: MarketHttpSession) -> MarketOverview:
        quotes: list[MarketQuote] = []
        missing: list[str] = []
        issues: list[str] = []

        try:
            quotes.append(self._get_kospi(session))
        except (MarketDataError, httpx.RequestError):
            try:
                quotes.append(self._get_yahoo_quote(session, "kospi", "코스피", "^KS11", "포인트"))
            except (MarketDataError, httpx.RequestError):
                missing.append("kospi")
                issues.append("kospi: unavailable")

        for quote_id, name_ko, symbol, unit in YAHOO_QUOTES:
            try:
                quotes.append(self._get_yahoo_quote(session, quote_id, name_ko, symbol, unit))
            except (MarketDataError, httpx.RequestError):
                missing.append(quote_id)
                issues.append(f"{quote_id}: unavailable")

        return MarketOverview(
            quotes=tuple(quotes),
            missing_quote_ids=tuple(missing),
            issues=tuple(issues),
            fetched_at=_now_iso(),
        )

    def _get_kospi(self, session: MarketHttpSession) -> MarketQuote:
        payload = self._get_json(session, NAVER_KOSPI_URL)
        if not isinstance(payload, Mapping):
            raise MarketDataError("Naver KOSPI response must be an object")
        return MarketQuote(
            id="kospi",
            name_ko="코스피",
            price=_number(payload.get("closePrice"), "closePrice"),
            change_percent=_number(payload.get("fluctuationsRatio"), "fluctuationsRatio"),
            unit="포인트",
            as_of=_required_text(payload.get("localTradedAt"), "localTradedAt"),
            source="Naver Finance",
        )

    def _get_yahoo_quote(
        self,
        session: MarketHttpSession,
        quote_id: str,
        name_ko: str,
        symbol: str,
        unit: str,
    ) -> MarketQuote:
        url = f"{YAHOO_CHART_BASE_URL}/{quote(symbol, safe='')}"
        payload = self._get_json(
            session,
            url,
            params={"range": "5d", "interval": "1d"},
        )
        meta = _yahoo_meta(payload)
        price = _number(meta.get("regularMarketPrice"), "regularMarketPrice")
        previous_close = _number(meta.get("chartPreviousClose"), "chartPreviousClose")
        if previous_close == 0:
            raise MarketDataError("Yahoo previous close must not be zero")
        timestamp = meta.get("regularMarketTime")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
            raise MarketDataError("Yahoo regularMarketTime must be a timestamp")
        as_of = datetime.fromtimestamp(float(timestamp), tz=UTC).isoformat().replace("+00:00", "Z")
        return MarketQuote(
            id=quote_id,
            name_ko=name_ko,
            price=price,
            change_percent=((price - previous_close) / previous_close) * 100,
            unit=unit,
            as_of=as_of,
            source="Yahoo Finance",
        )

    def _get_json(
        self,
        session: MarketHttpSession,
        url: str,
        *,
        params: Mapping[str, str] | None = None,
    ) -> object:
        response = session.get(
            url,
            params=params,
            headers=self._headers,
            timeout=self._timeout,
        )
        if not 200 <= response.status_code < 300:
            raise MarketDataError(f"market provider returned HTTP {response.status_code}")
        try:
            return response.json()
        except ValueError as error:
            raise MarketDataError("market provider returned invalid JSON") from error


def _yahoo_meta(payload: object) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise MarketDataError("Yahoo response must be an object")
    chart = payload.get("chart")
    if not isinstance(chart, Mapping) or chart.get("error") is not None:
        raise MarketDataError("Yahoo chart response contains an error")
    results = chart.get("result")
    if not isinstance(results, list) or not results or not isinstance(results[0], Mapping):
        raise MarketDataError("Yahoo chart result is missing")
    meta = results[0].get("meta")
    if not isinstance(meta, Mapping):
        raise MarketDataError("Yahoo chart metadata is missing")
    return meta


def _number(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise MarketDataError(f"{field} must be numeric")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.replace(",", "").strip())
        except ValueError as error:
            raise MarketDataError(f"{field} must be numeric") from error
    else:
        raise MarketDataError(f"{field} must be numeric")
    if number != number or number in {float("inf"), float("-inf")}:
        raise MarketDataError(f"{field} must be finite")
    return number


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MarketDataError(f"{field} must be non-empty text")
    return value.strip()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
