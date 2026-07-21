"""Select the report's main fear and buffer factors deterministically."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from fng_chatbot.models import IndicatorReading

NO_CLEAR_FEAR_FACTOR = "뚜렷한 공포 요인 없음"

_FEAR_RATINGS = frozenset({"fear", "extreme fear"})
_BUFFER_RATINGS = frozenset({"neutral", "greed", "extreme greed"})
_INDICATOR_ORDER = (
    "market_momentum",
    "stock_price_strength",
    "stock_price_breadth",
    "put_call_options",
    "market_volatility",
    "safe_haven_demand",
    "junk_bond_demand",
)
_ORDER_INDEX = {indicator_id: index for index, indicator_id in enumerate(_INDICATOR_ORDER)}


@dataclass(frozen=True, slots=True)
class DriverSelection:
    """Deterministic candidates selected from normalized indicators."""

    fear_drivers: tuple[IndicatorReading, ...]
    buffer: IndicatorReading | None
    fear_note: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "fear_drivers": [indicator.to_dict() for indicator in self.fear_drivers],
            "buffer": None if self.buffer is None else self.buffer.to_dict(),
            "fear_note": self.fear_note,
        }


def select_report_drivers(indicators: Iterable[IndicatorReading]) -> DriverSelection:
    """Select report factors from the seven normalized indicators."""

    readings = tuple(indicators)
    fear_drivers = tuple(
        sorted(
            (reading for reading in readings if reading.rating in _FEAR_RATINGS),
            key=lambda reading: (
                reading.score,
                _order_of(reading),
                reading.id,
            ),
        )[:2]
    )
    buffers = sorted(
        (reading for reading in readings if reading.rating in _BUFFER_RATINGS),
        key=lambda reading: (
            -reading.score,
            _order_of(reading),
            reading.id,
        ),
    )

    return DriverSelection(
        fear_drivers=fear_drivers,
        buffer=buffers[0] if buffers else None,
        fear_note=None if fear_drivers else NO_CLEAR_FEAR_FACTOR,
    )


def _order_of(indicator: IndicatorReading) -> int:
    return _ORDER_INDEX.get(indicator.id, len(_INDICATOR_ORDER))
