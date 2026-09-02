"""Raw CNN and normalized market-sentiment data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Rating: TypeAlias = Literal[
    "extreme fear",
    "fear",
    "neutral",
    "greed",
    "extreme greed",
]

VALID_RATINGS: frozenset[str] = frozenset(
    {"extreme fear", "fear", "neutral", "greed", "extreme greed"}
)


@dataclass(frozen=True, slots=True)
class RawCnnMetric:
    """One untrusted metric exactly as extracted from a CNN response."""

    score: object
    rating: object
    timestamp: object
    data: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RawCnnComposite:
    """Untrusted composite values exactly as extracted from a CNN response."""

    score: object
    rating: object
    timestamp: object
    previous_close: object
    previous_1_week: object
    previous_1_month: object
    previous_1_year: object


@dataclass(frozen=True, slots=True)
class RawCnnSnapshot:
    """CNN-shaped source model, including comparison metrics and series data."""

    composite: RawCnnComposite
    metrics: dict[str, RawCnnMetric]


@dataclass(frozen=True, slots=True)
class CompositeReading:
    """Validated composite Fear & Greed reading."""

    score: float
    rating: Rating
    updated_at: str
    previous_close: float
    previous_1_week: float
    previous_1_month: float
    previous_1_year: float

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "rating": self.rating,
            "updated_at": self.updated_at,
            "previous_close": self.previous_close,
            "previous_1_week": self.previous_1_week,
            "previous_1_month": self.previous_1_month,
            "previous_1_year": self.previous_1_year,
        }


@dataclass(frozen=True, slots=True)
class IndicatorReading:
    """One of the seven normalized conceptual indicators."""

    id: str
    name_ko: str
    score: float
    rating: Rating
    updated_at: str
    basis: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "name_ko": self.name_ko,
            "score": self.score,
            "rating": self.rating,
            "updated_at": self.updated_at,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class DataQuality:
    """Quality metadata attached to a normalized snapshot."""

    complete: bool
    missing_fields: tuple[str, ...]
    cached: bool = False
    stale: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "complete": self.complete,
            "missing_fields": list(self.missing_fields),
            "cached": self.cached,
            "stale": self.stale,
        }


@dataclass(frozen=True, slots=True)
class FearGreedSnapshot:
    """Validated project-owned model, independent of CNN's response shape."""

    source: str
    fetched_at: str
    composite: CompositeReading
    indicators: tuple[IndicatorReading, ...]
    data_quality: DataQuality

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "fetched_at": self.fetched_at,
            "composite": self.composite.to_dict(),
            "indicators": [indicator.to_dict() for indicator in self.indicators],
            "data_quality": self.data_quality.to_dict(),
        }
