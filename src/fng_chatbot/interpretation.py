"""Build and validate internal interpretations of Python-selected factors."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from fng_chatbot.deepseek import DeepSeekJsonCompletion
from fng_chatbot.models import FearGreedSnapshot, Rating
from fng_chatbot.report import StaticReport

INTERPRETATION_MAX_LINES = 2
INTERPRETATION_MAX_CHARACTERS = 300

DEEPSEEK_INTERPRETATION_SYSTEM_PROMPT = "\n".join(
    (
        "당신은 시장심리 지표 해석 보조자입니다.",
        "입력에는 Python이 이미 선택한 공포 요인과 완충 요인만 있습니다.",
        (
            "요인을 새로 선택·추가·누락·재정렬하지 말고, "
            "각 요인이 현재 시장심리에서 무엇을 뜻하는지만 한국어로 설명하세요."
        ),
        "뉴스·기사·외부 사실·가격 전망·수익률 예측·매수·매도·종목 추천·URL은 사용하지 마세요.",
        "요인 이름만 반복해서 나열하지 말고 공포 심리 또는 완충 작용의 의미를 설명하세요.",
        "반드시 JSON 객체만 반환하세요. 키는 lines와 referenced_indicator_ids 두 개만 사용하세요.",
        "lines는 1~2개의 한 줄 문자열 배열이며 줄바꿈을 포함한 전체 길이는 300자 이하여야 합니다.",
        "referenced_indicator_ids는 입력 ID 전체를 입력 순서 그대로 담아야 합니다.",
    )
)

_INTERPRETATION_VALIDATION_REASON_CODES = frozenset(
    {
        "validation_error",
        "validation_invalid_json",
        "validation_json_not_object",
        "validation_invalid_keys",
        "validation_lines_not_array",
        "validation_ids_not_array",
        "validation_line_count",
        "validation_invalid_line",
        "validation_too_long",
        "validation_id_mismatch",
        "validation_missing_model",
    }
)


class InterpretationValidationError(ValueError):
    """A generated interpretation does not satisfy the grounded preview contract."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "validation_error",
    ) -> None:
        super().__init__(message)
        self.reason_code = (
            reason_code
            if reason_code in _INTERPRETATION_VALIDATION_REASON_CODES
            else "validation_error"
        )


class JsonCompletionProvider(Protocol):
    """Small completion surface consumed by the interpretation layer."""

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> DeepSeekJsonCompletion: ...


class InterpretationProvider(Protocol):
    """Generate one validated interpretation for a selected-factor context."""

    def interpret(self, context: InterpretationContext) -> ReportInterpretation: ...


@dataclass(frozen=True, slots=True)
class InterpretationFactor:
    """One Python-selected factor supplied to DeepSeek."""

    role: Literal["fear_driver", "buffer"]
    indicator_id: str
    name_ko: str
    score: float
    rating: Rating
    basis: str

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "indicator_id": self.indicator_id,
            "name_ko": self.name_ko,
            "score": self.score,
            "rating": self.rating,
            "basis": self.basis,
        }


@dataclass(frozen=True, slots=True)
class InterpretationContext:
    """Facts supplied internally to an interpretation provider."""

    overall_score: float
    overall_rating: Rating
    factors: tuple[InterpretationFactor, ...]

    @property
    def selected_indicator_ids(self) -> tuple[str, ...]:
        return tuple(factor.indicator_id for factor in self.factors)

    def to_dict(self) -> dict[str, object]:
        return {
            "overall": {
                "score": self.overall_score,
                "rating": self.overall_rating,
            },
            "selected_indicator_ids": list(self.selected_indicator_ids),
            "factors": [factor.to_dict() for factor in self.factors],
            "output_contract": {
                "maximum_lines": INTERPRETATION_MAX_LINES,
                "maximum_characters": INTERPRETATION_MAX_CHARACTERS,
                "required_keys": ["lines", "referenced_indicator_ids"],
            },
        }


@dataclass(frozen=True, slots=True)
class ReportInterpretation:
    """Validated DeepSeek output that can be rendered into a reviewed preview."""

    lines: tuple[str, ...]
    referenced_indicator_ids: tuple[str, ...]
    model: str
    source: str = "deepseek"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": "applied",
            "source": self.source,
            "model": self.model,
            "lines": list(self.lines),
            "referenced_indicator_ids": list(self.referenced_indicator_ids),
        }


def build_interpretation_context(
    snapshot: FearGreedSnapshot,
    report: StaticReport,
) -> InterpretationContext:
    """Build the exact grounded facts a lightweight interpreter may explain."""

    indicators = {indicator.id: indicator for indicator in snapshot.indicators}
    factors: list[InterpretationFactor] = []
    for report_factor in report.fear_drivers:
        source = indicators[report_factor.indicator_id]
        factors.append(
            InterpretationFactor(
                role="fear_driver",
                indicator_id=source.id,
                name_ko=source.name_ko,
                score=source.score,
                rating=source.rating,
                basis=source.basis,
            )
        )
    if report.buffer is not None:
        source = indicators[report.buffer.indicator_id]
        factors.append(
            InterpretationFactor(
                role="buffer",
                indicator_id=source.id,
                name_ko=source.name_ko,
                score=source.score,
                rating=source.rating,
                basis=source.basis,
            )
        )

    return InterpretationContext(
        overall_score=snapshot.composite.score,
        overall_rating=snapshot.composite.rating,
        factors=tuple(factors),
    )


def build_deepseek_prompts(context: InterpretationContext) -> tuple[str, str]:
    """Serialize only the factors that deterministic Python rules selected."""

    user_payload = {
        "selected_factors": [factor.to_dict() for factor in context.factors],
    }
    return (
        DEEPSEEK_INTERPRETATION_SYSTEM_PROMPT,
        json.dumps(
            user_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


class DeepSeekInterpreter:
    """Generate and validate an explanation without changing selected factors."""

    def __init__(self, completion_provider: JsonCompletionProvider) -> None:
        self._completion_provider = completion_provider

    def interpret(self, context: InterpretationContext) -> ReportInterpretation:
        system_prompt, user_prompt = build_deepseek_prompts(context)
        completion = self._completion_provider.complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return parse_deepseek_interpretation(
            context=context,
            content=completion.content,
            model=completion.model,
        )


def parse_deepseek_interpretation(
    *,
    context: InterpretationContext,
    content: str,
    model: str,
) -> ReportInterpretation:
    """Parse DeepSeek JSON and apply the grounded interpretation contract."""

    try:
        payload = json.loads(content)
    except (TypeError, json.JSONDecodeError) as error:
        raise InterpretationValidationError(
            "interpretation content is not valid JSON",
            reason_code="validation_invalid_json",
        ) from error
    if not isinstance(payload, Mapping):
        raise InterpretationValidationError(
            "interpretation JSON must be an object",
            reason_code="validation_json_not_object",
        )
    if set(payload) != {"lines", "referenced_indicator_ids"}:
        raise InterpretationValidationError(
            "interpretation JSON must contain only lines and referenced_indicator_ids",
            reason_code="validation_invalid_keys",
        )

    lines = payload["lines"]
    referenced_indicator_ids = payload["referenced_indicator_ids"]
    if not isinstance(lines, list) or any(not isinstance(line, str) for line in lines):
        raise InterpretationValidationError(
            "interpretation lines must be a string array",
            reason_code="validation_lines_not_array",
        )
    if not isinstance(referenced_indicator_ids, list) or any(
        not isinstance(indicator_id, str) for indicator_id in referenced_indicator_ids
    ):
        raise InterpretationValidationError(
            "referenced_indicator_ids must be a string array",
            reason_code="validation_ids_not_array",
        )

    return validate_interpretation(
        context=context,
        lines=cast(list[str], lines),
        referenced_indicator_ids=cast(list[str], referenced_indicator_ids),
        model=model,
    )


def validate_interpretation(
    *,
    context: InterpretationContext,
    lines: list[str],
    referenced_indicator_ids: list[str],
    model: str,
) -> ReportInterpretation:
    """Validate structured agent output against the current Python-selected report."""

    normalized_lines = tuple(line.strip() for line in lines)
    if not 1 <= len(normalized_lines) <= INTERPRETATION_MAX_LINES:
        raise InterpretationValidationError(
            "interpretation must contain one or two lines",
            reason_code="validation_line_count",
        )
    if any(not line or "\n" in line or "\r" in line for line in normalized_lines):
        raise InterpretationValidationError(
            "interpretation lines must be non-empty single lines",
            reason_code="validation_invalid_line",
        )
    if len("\n".join(normalized_lines)) > INTERPRETATION_MAX_CHARACTERS:
        raise InterpretationValidationError(
            "interpretation exceeds 300 characters",
            reason_code="validation_too_long",
        )

    referenced_ids = tuple(referenced_indicator_ids)
    if referenced_ids != context.selected_indicator_ids:
        raise InterpretationValidationError(
            "referenced_indicator_ids must exactly match the Python-selected factors",
            reason_code="validation_id_mismatch",
        )
    if not model.strip():
        raise InterpretationValidationError(
            "interpretation model must not be empty",
            reason_code="validation_missing_model",
        )

    return ReportInterpretation(
        lines=normalized_lines,
        referenced_indicator_ids=referenced_ids,
        model=model,
    )


def rules_fallback_metadata(*, reason: str) -> dict[str, object]:
    """Describe a preview that contains only the deterministic Python summary."""

    return {
        "status": "rules_fallback",
        "source": "rules",
        "model": None,
        "reason": reason,
        "lines": [],
        "referenced_indicator_ids": [],
    }
