"""Convert CNN-shaped payloads into the project's validated data contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from fng_chatbot.models import (
    VALID_RATINGS,
    CompositeReading,
    DataQuality,
    FearGreedSnapshot,
    IndicatorReading,
    Rating,
    RawCnnComposite,
    RawCnnMetric,
    RawCnnSnapshot,
)

SOURCE_NAME = "CNN Fear & Greed"

_RAW_METRIC_KEYS = (
    "market_momentum_sp500",
    "market_momentum_sp125",
    "stock_price_strength",
    "stock_price_breadth",
    "put_call_options",
    "market_volatility_vix",
    "market_volatility_vix_50",
    "safe_haven_demand",
    "junk_bond_demand",
)

_INDICATOR_SPECS = (
    (
        "market_momentum",
        "시장 모멘텀",
        "market_momentum_sp500",
        "S&P 500과 125일 이동평균 비교",
    ),
    (
        "stock_price_strength",
        "주가 강도",
        "stock_price_strength",
        "NYSE 52주 신고가와 신저가 비교",
    ),
    (
        "stock_price_breadth",
        "시장 폭",
        "stock_price_breadth",
        "상승·하락 종목의 거래량 비교",
    ),
    (
        "put_call_options",
        "풋·콜 옵션",
        "put_call_options",
        "5일 평균 풋·콜 비율",
    ),
    (
        "market_volatility",
        "시장 변동성",
        "market_volatility_vix",
        "VIX와 50일 이동평균 비교",
    ),
    (
        "safe_haven_demand",
        "안전자산 수요",
        "safe_haven_demand",
        "주식과 채권의 20일 수익률 차이",
    ),
    (
        "junk_bond_demand",
        "정크본드 수요",
        "junk_bond_demand",
        "정크본드와 우량채권 수익률 스프레드",
    ),
)


class NormalizationError(ValueError):
    """Raised when a CNN-shaped payload violates the expected data contract."""


def parse_cnn_payload(payload: Mapping[str, object]) -> RawCnnSnapshot:
    """Extract a CNN-shaped payload without converting its scalar values."""

    root = _require_mapping(payload, "payload")
    composite_data = _require_mapping(
        _required(root, "fear_and_greed", "payload"), "fear_and_greed"
    )
    composite = RawCnnComposite(
        score=_required(composite_data, "score", "fear_and_greed"),
        rating=_required(composite_data, "rating", "fear_and_greed"),
        timestamp=_required(composite_data, "timestamp", "fear_and_greed"),
        previous_close=_required(composite_data, "previous_close", "fear_and_greed"),
        previous_1_week=_required(composite_data, "previous_1_week", "fear_and_greed"),
        previous_1_month=_required(composite_data, "previous_1_month", "fear_and_greed"),
        previous_1_year=_required(composite_data, "previous_1_year", "fear_and_greed"),
    )

    metrics: dict[str, RawCnnMetric] = {}
    for key in _RAW_METRIC_KEYS:
        metric_data = _require_mapping(_required(root, key, "payload"), key)
        raw_series = metric_data.get("data", [])
        if not isinstance(raw_series, (list, tuple)):
            raise NormalizationError(f"{key}.data must be an array")
        metrics[key] = RawCnnMetric(
            score=_required(metric_data, "score", key),
            rating=_required(metric_data, "rating", key),
            timestamp=_required(metric_data, "timestamp", key),
            data=tuple(raw_series),
        )

    return RawCnnSnapshot(composite=composite, metrics=metrics)


def normalize_snapshot(
    raw: RawCnnSnapshot,
    *,
    fetched_at: object | None = None,
) -> FearGreedSnapshot:
    """Validate and normalize an already extracted raw CNN snapshot."""

    composite = CompositeReading(
        score=_to_score(raw.composite.score, "fear_and_greed.score"),
        rating=_to_rating(raw.composite.rating, "fear_and_greed.rating"),
        updated_at=_to_timestamp(raw.composite.timestamp, "fear_and_greed.timestamp"),
        previous_close=_to_score(raw.composite.previous_close, "fear_and_greed.previous_close"),
        previous_1_week=_to_score(raw.composite.previous_1_week, "fear_and_greed.previous_1_week"),
        previous_1_month=_to_score(
            raw.composite.previous_1_month, "fear_and_greed.previous_1_month"
        ),
        previous_1_year=_to_score(raw.composite.previous_1_year, "fear_and_greed.previous_1_year"),
    )

    validated_metrics: dict[str, tuple[float, Rating, str]] = {}
    for key, metric in raw.metrics.items():
        validated_metrics[key] = (
            _to_score(metric.score, f"{key}.score"),
            _to_rating(metric.rating, f"{key}.rating"),
            _to_timestamp(metric.timestamp, f"{key}.timestamp"),
        )

    indicators = tuple(
        IndicatorReading(
            id=indicator_id,
            name_ko=name_ko,
            score=validated_metrics[source_key][0],
            rating=validated_metrics[source_key][1],
            updated_at=validated_metrics[source_key][2],
            basis=basis,
        )
        for indicator_id, name_ko, source_key, basis in _INDICATOR_SPECS
    )

    normalized_fetched_at = (
        datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
        if fetched_at is None
        else _to_timestamp(fetched_at, "fetched_at")
    )

    return FearGreedSnapshot(
        source=SOURCE_NAME,
        fetched_at=normalized_fetched_at,
        composite=composite,
        indicators=indicators,
        data_quality=DataQuality(complete=True, missing_fields=()),
    )


def normalize_cnn_payload(
    payload: Mapping[str, object],
    *,
    fetched_at: object | None = None,
) -> FearGreedSnapshot:
    """Extract, validate, and normalize one CNN-shaped response."""

    return normalize_snapshot(parse_cnn_payload(payload), fetched_at=fetched_at)


def _required(mapping: Mapping[str, object], key: str, path: str) -> object:
    if key not in mapping:
        raise NormalizationError(f"missing required field: {path}.{key}")
    return mapping[key]


def _require_mapping(value: object, path: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise NormalizationError(f"{path} must be an object")
    return cast(Mapping[str, object], value)


def _to_score(value: object, path: str) -> float:
    if isinstance(value, bool):
        raise NormalizationError(f"{path} must be a number between 0 and 100")
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise NormalizationError(f"{path} must be a number between 0 and 100") from error
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise NormalizationError(f"{path} must be between 0 and 100; got {value!r}")
    return score


def _to_rating(value: object, path: str) -> Rating:
    if not isinstance(value, str):
        raise NormalizationError(f"{path} must be one of {sorted(VALID_RATINGS)}")
    rating = " ".join(value.strip().lower().split())
    if rating not in VALID_RATINGS:
        raise NormalizationError(f"{path} must be one of {sorted(VALID_RATINGS)}; got {value!r}")
    return cast(Rating, rating)


def _to_timestamp(value: object, path: str) -> str:
    parsed: datetime
    if isinstance(value, bool):
        raise NormalizationError(f"{path} must be an ISO-8601 or Unix timestamp")

    if isinstance(value, (int, float)):
        parsed = _timestamp_from_number(float(value), path)
    elif isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
            except ValueError as error:
                raise NormalizationError(
                    f"{path} must be an ISO-8601 or Unix timestamp; got {value!r}"
                ) from error
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
        else:
            parsed = _timestamp_from_number(numeric, path)
    else:
        raise NormalizationError(f"{path} must be an ISO-8601 or Unix timestamp")

    return parsed.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _timestamp_from_number(value: float, path: str) -> datetime:
    if not math.isfinite(value):
        raise NormalizationError(f"{path} must be a finite Unix timestamp")
    seconds = value / 1000 if abs(value) >= 100_000_000_000 else value
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OSError, OverflowError, ValueError) as error:
        raise NormalizationError(f"{path} is outside the supported timestamp range") from error
