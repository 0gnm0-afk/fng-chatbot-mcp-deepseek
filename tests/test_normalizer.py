import json
from copy import deepcopy
from pathlib import Path

import pytest

from fng_chatbot.normalizer import (
    NormalizationError,
    normalize_cnn_payload,
    parse_cnn_payload,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"
EXPECTED_INDICATOR_IDS = [
    "market_momentum",
    "stock_price_strength",
    "stock_price_breadth",
    "put_call_options",
    "market_volatility",
    "safe_haven_demand",
    "junk_bond_demand",
]


def load_payload() -> dict[str, object]:
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def test_normalizes_composite_history_and_exactly_seven_indicators() -> None:
    result = normalize_cnn_payload(load_payload(), fetched_at="2026-07-20T09:00:00+09:00")

    assert result.source == "CNN Fear & Greed"
    assert result.fetched_at == "2026-07-20T00:00:00Z"
    assert result.composite.score == 37.0
    assert result.composite.rating == "fear"
    assert result.composite.updated_at == "2026-01-01T00:00:00Z"
    assert result.composite.previous_close == 41.0
    assert result.composite.previous_1_week == 46.0
    assert result.composite.previous_1_month == 32.0
    assert result.composite.previous_1_year == 74.0
    assert [indicator.id for indicator in result.indicators] == EXPECTED_INDICATOR_IDS
    assert len(result.indicators) == 7
    assert result.data_quality.complete is True
    assert result.data_quality.missing_fields == ()


def test_comparison_metrics_remain_raw_and_are_not_double_counted() -> None:
    payload = load_payload()
    raw = parse_cnn_payload(payload)
    normalized = normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")

    assert len(raw.metrics) == 9
    assert "market_momentum_sp125" in raw.metrics
    assert "market_volatility_vix_50" in raw.metrics
    normalized_ids = {indicator.id for indicator in normalized.indicators}
    assert "market_momentum_sp125" not in normalized_ids
    assert "market_volatility_vix_50" not in normalized_ids


def test_long_cnn_series_is_excluded_from_normalized_result() -> None:
    payload = load_payload()
    momentum = payload["market_momentum_sp500"]
    assert isinstance(momentum, dict)
    momentum["data"] = [{"x": index, "y": index % 101} for index in range(500)]

    raw = parse_cnn_payload(payload)
    normalized = normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")

    assert len(raw.metrics["market_momentum_sp500"].data) == 500
    assert '"data"' not in json.dumps(normalized.to_dict())


def test_missing_required_top_level_field_has_explainable_error() -> None:
    payload = load_payload()
    del payload["safe_haven_demand"]

    with pytest.raises(
        NormalizationError,
        match=r"missing required field: payload\.safe_haven_demand",
    ):
        normalize_cnn_payload(payload)


def test_missing_required_composite_field_has_explainable_error() -> None:
    payload = load_payload()
    composite = payload["fear_and_greed"]
    assert isinstance(composite, dict)
    del composite["previous_1_month"]

    with pytest.raises(
        NormalizationError,
        match=r"missing required field: fear_and_greed\.previous_1_month",
    ):
        normalize_cnn_payload(payload)


@pytest.mark.parametrize(
    ("field", "invalid_score"),
    [("fear_and_greed", -0.1), ("stock_price_strength", 100.1)],
)
def test_rejects_scores_outside_zero_to_one_hundred(field: str, invalid_score: float) -> None:
    payload = load_payload()
    metric = payload[field]
    assert isinstance(metric, dict)
    metric["score"] = invalid_score

    with pytest.raises(NormalizationError, match=r"must be between 0 and 100"):
        normalize_cnn_payload(payload)


def test_rejects_unknown_rating() -> None:
    payload = load_payload()
    metric = payload["put_call_options"]
    assert isinstance(metric, dict)
    metric["rating"] = "optimistic"

    with pytest.raises(NormalizationError, match=r"put_call_options\.rating must be one of"):
        normalize_cnn_payload(payload)


@pytest.mark.parametrize(
    ("timestamp", "expected"),
    [
        (1767225600, "2026-01-01T00:00:00Z"),
        ("1767225600000", "2026-01-01T00:00:00Z"),
        ("2026-01-01T09:00:00+09:00", "2026-01-01T00:00:00Z"),
    ],
)
def test_accepts_numeric_and_string_timestamp_formats(timestamp: object, expected: str) -> None:
    payload = deepcopy(load_payload())
    composite = payload["fear_and_greed"]
    assert isinstance(composite, dict)
    composite["timestamp"] = timestamp

    result = normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")

    assert result.composite.updated_at == expected
