from fng_chatbot.driver_selector import NO_CLEAR_FEAR_FACTOR, select_report_drivers
from fng_chatbot.models import IndicatorReading, Rating


def indicator(indicator_id: str, score: float, rating: Rating) -> IndicatorReading:
    return IndicatorReading(
        id=indicator_id,
        name_ko=indicator_id,
        score=score,
        rating=rating,
        updated_at="2026-01-01T00:00:00Z",
        basis="test basis",
    )


def test_selects_two_lowest_fear_drivers_and_highest_buffer() -> None:
    selection = select_report_drivers(
        [
            indicator("market_momentum", 35, "fear"),
            indicator("stock_price_strength", 18, "extreme fear"),
            indicator("stock_price_breadth", 27, "fear"),
            indicator("put_call_options", 64, "greed"),
            indicator("market_volatility", 58, "neutral"),
        ]
    )

    assert [item.id for item in selection.fear_drivers] == [
        "stock_price_strength",
        "stock_price_breadth",
    ]
    assert selection.buffer is not None
    assert selection.buffer.id == "put_call_options"
    assert selection.fear_note is None


def test_does_not_fill_missing_candidate_slots() -> None:
    selection = select_report_drivers(
        [
            indicator("market_momentum", 35, "fear"),
            indicator("stock_price_strength", 20, "fear"),
            indicator("stock_price_breadth", 55, "fear"),
        ]
    )

    assert len(selection.fear_drivers) == 2
    assert selection.buffer is None


def test_ties_use_fixed_indicator_order_regardless_of_input_order() -> None:
    readings = [
        indicator("safe_haven_demand", 20, "fear"),
        indicator("stock_price_strength", 20, "fear"),
        indicator("market_momentum", 20, "fear"),
        indicator("junk_bond_demand", 60, "neutral"),
        indicator("market_volatility", 60, "neutral"),
    ]

    forward = select_report_drivers(readings)
    reverse = select_report_drivers(reversed(readings))

    assert [item.id for item in forward.fear_drivers] == [
        "market_momentum",
        "stock_price_strength",
    ]
    assert forward == reverse
    assert forward.buffer is not None
    assert forward.buffer.id == "market_volatility"


def test_all_greed_has_no_fear_driver_and_explicit_note() -> None:
    selection = select_report_drivers(
        [
            indicator("market_momentum", 80, "greed"),
            indicator("stock_price_strength", 90, "extreme greed"),
        ]
    )

    assert selection.fear_drivers == ()
    assert selection.fear_note == NO_CLEAR_FEAR_FACTOR
    assert selection.buffer is not None
    assert selection.buffer.id == "stock_price_strength"


def test_one_fear_driver_is_not_padded_with_neutral_indicator() -> None:
    selection = select_report_drivers(
        [
            indicator("market_momentum", 30, "fear"),
            indicator("stock_price_strength", 50, "neutral"),
        ]
    )

    assert [item.id for item in selection.fear_drivers] == ["market_momentum"]
    assert selection.buffer is not None
    assert selection.buffer.id == "stock_price_strength"
