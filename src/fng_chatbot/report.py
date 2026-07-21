"""Generate a deterministic informational report from normalized facts."""

from __future__ import annotations

from dataclasses import dataclass

from fng_chatbot.driver_selector import DriverSelection, select_report_drivers
from fng_chatbot.models import FearGreedSnapshot, IndicatorReading, Rating

_RATING_LABELS: dict[Rating, str] = {
    "extreme fear": "극단적 공포",
    "fear": "공포",
    "neutral": "중립",
    "greed": "탐욕",
    "extreme greed": "극단적 탐욕",
}


@dataclass(frozen=True, slots=True)
class ReportFactor:
    """One selected indicator and its deterministic explanation."""

    indicator_id: str
    indicator: str
    score: float
    rating: Rating
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return {
            "indicator_id": self.indicator_id,
            "indicator": self.indicator,
            "score": self.score,
            "rating": self.rating,
            "explanation": self.explanation,
        }


@dataclass(frozen=True, slots=True)
class StaticReport:
    """Structured report produced entirely by local Python rules."""

    fear_drivers: tuple[ReportFactor, ...]
    buffer: ReportFactor | None
    fear_note: str | None
    summary: str
    summary_source: str = "rules"

    def to_dict(self) -> dict[str, object]:
        return {
            "fear_drivers": [factor.to_dict() for factor in self.fear_drivers],
            "buffer": None if self.buffer is None else self.buffer.to_dict(),
            "fear_note": self.fear_note,
            "summary": self.summary,
            "summary_source": self.summary_source,
        }


def build_static_report(snapshot: FearGreedSnapshot) -> StaticReport:
    """Build an informational report only from validated snapshot facts."""

    selection = select_report_drivers(snapshot.indicators)
    fear_drivers = tuple(_fear_factor(indicator) for indicator in selection.fear_drivers)
    buffer = None if selection.buffer is None else _buffer_factor(selection.buffer)

    return StaticReport(
        fear_drivers=fear_drivers,
        buffer=buffer,
        fear_note=selection.fear_note,
        summary=_summary(snapshot, selection),
    )


def _fear_factor(indicator: IndicatorReading) -> ReportFactor:
    label = _RATING_LABELS[indicator.rating]
    return ReportFactor(
        indicator_id=indicator.id,
        indicator=indicator.name_ko,
        score=indicator.score,
        rating=indicator.rating,
        explanation=(
            f"{indicator.name_ko}: {indicator.score:.1f}점({label})으로 "
            "주요 공포 요인에 해당합니다."
        ),
    )


def _buffer_factor(indicator: IndicatorReading) -> ReportFactor:
    label = _RATING_LABELS[indicator.rating]
    return ReportFactor(
        indicator_id=indicator.id,
        indicator=indicator.name_ko,
        score=indicator.score,
        rating=indicator.rating,
        explanation=(
            f"{indicator.name_ko}: {indicator.score:.1f}점({label})으로 "
            "비공포 등급 지표 중 가장 높은 완충 요인입니다."
        ),
    )


def _summary(snapshot: FearGreedSnapshot, selection: DriverSelection) -> str:
    composite = snapshot.composite
    opening = f"종합지수는 {composite.score:.1f}점({_RATING_LABELS[composite.rating]})입니다."

    if selection.fear_drivers:
        fear_names = ", ".join(indicator.name_ko for indicator in selection.fear_drivers)
        fear_sentence = f"주요 공포 요인은 {fear_names}입니다."
    else:
        fear_sentence = "뚜렷한 공포 요인 없음으로 분류됩니다."

    buffer_sentence = (
        f"완충 요인은 {selection.buffer.name_ko}입니다."
        if selection.buffer is not None
        else "뚜렷한 완충 요인은 없습니다."
    )
    return " ".join((opening, fear_sentence, buffer_sentence))
