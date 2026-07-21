"""Render deterministic market reports as safe Telegram MarkdownV2."""

from __future__ import annotations

from collections.abc import Mapping

from fng_chatbot.interpretation import ReportInterpretation
from fng_chatbot.market_data import MarketOverview, MarketQuote
from fng_chatbot.models import FearGreedSnapshot
from fng_chatbot.report import ReportFactor, StaticReport

TELEGRAM_PARSE_MODE = "MarkdownV2"
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

_MARKDOWN_V2_SPECIALS = frozenset(r"_*[]()~`>#+-=|{}.!\\")
_EXPLANATION_LIMIT = 300
_SUMMARY_LIMIT = 600


class MessageRenderError(ValueError):
    """A Telegram message cannot be rendered within the safe contract."""


def escape_markdown_v2(text: str) -> str:
    """Escape Telegram MarkdownV2 syntax in untrusted dynamic text."""

    return "".join(
        f"\\{character}" if character in _MARKDOWN_V2_SPECIALS else character for character in text
    )


def render_telegram_message(
    snapshot: FearGreedSnapshot,
    comment: StaticReport,
    interpretation: ReportInterpretation | None = None,
) -> str:
    """Create one bounded informational message without any network access."""

    lines = [
        "📊 *F&G 시장심리*",
        f"종합: *{escape_markdown_v2(_format_score(snapshot.composite.score))}* "
        f"\\({escape_markdown_v2(snapshot.composite.rating)}\\)",
        f"기준 시각: {escape_markdown_v2(snapshot.composite.updated_at)}",
        "",
        "*주요 공포 요인*",
    ]

    if comment.fear_drivers:
        for factor in comment.fear_drivers:
            lines.extend(_render_factor(factor))
    else:
        lines.append(
            escape_markdown_v2(
                _shorten(comment.fear_note or "뚜렷한 공포 요인 없음", _EXPLANATION_LIMIT)
            )
        )

    lines.extend(["", "*완충 요인*"])
    if comment.buffer is None:
        lines.append("뚜렷한 완충 요인 없음")
    else:
        lines.extend(_render_factor(comment.buffer))

    lines.extend(
        [
            "",
            "*한 줄 해석*",
            escape_markdown_v2(_shorten(comment.summary, _SUMMARY_LIMIT)),
        ]
    )
    if interpretation is not None:
        lines.extend(escape_markdown_v2(line) for line in interpretation.lines)
    lines.extend(
        [
            "",
            f"데이터: {escape_markdown_v2(snapshot.source)}",
            f"요약: {escape_markdown_v2(_source_label(comment.summary_source))}",
        ]
    )
    if interpretation is not None:
        lines.append("설명: AI 보조")
    if snapshot.data_quality.cached or snapshot.data_quality.stale:
        states = []
        if snapshot.data_quality.cached:
            states.append("캐시")
        if snapshot.data_quality.stale:
            states.append("오래된 데이터")
        lines.append(f"품질: {escape_markdown_v2(', '.join(states))}")

    message = "\n".join(lines)
    if not message or len(message) > TELEGRAM_MAX_MESSAGE_LENGTH:
        raise MessageRenderError("rendered Telegram message exceeds the safe length limit")
    return message


def render_morning_telegram_message(
    fear_greed_message: str,
    overview: MarketOverview,
    overall: Mapping[str, object],
) -> str:
    """Add best-effort market quotes and a neutral day-over-day sentiment comparison."""

    lines = ["🌅 *아침 시장 브리핑*", "", "*주요 시장지표*"]
    if overview.quotes:
        lines.extend(_render_market_quote(quote) for quote in overview.quotes)
    else:
        lines.append("현재 확인 가능한 시장지표가 없습니다")

    if overview.missing_quote_ids:
        missing = ", ".join(overview.missing_quote_ids)
        lines.append(f"누락: {escape_markdown_v2(missing)}")

    lines.extend(
        [
            "",
            fear_greed_message,
        ]
    )
    comparison = _render_sentiment_comparison(overall)
    if comparison:
        lines.extend(["", "*전일 대비 심리*", comparison])

    message = "\n".join(lines)
    if not message or len(message) > TELEGRAM_MAX_MESSAGE_LENGTH:
        raise MessageRenderError("rendered morning Telegram message exceeds the safe length limit")
    return message


def _render_factor(factor: ReportFactor) -> list[str]:
    name = escape_markdown_v2(_shorten(factor.indicator, 100))
    score = escape_markdown_v2(_format_score(factor.score))
    rating = escape_markdown_v2(factor.rating)
    explanation = escape_markdown_v2(_shorten(factor.explanation, _EXPLANATION_LIMIT))
    return [f"• *{name}* — {score} \\({rating}\\)", f"  {explanation}"]


def _render_market_quote(quote: MarketQuote) -> str:
    name = escape_markdown_v2(quote.name_ko)
    price = escape_markdown_v2(_format_market_price(quote))
    change = escape_markdown_v2(f"{quote.change_percent:+.2f}%")
    return f"• *{name}*: {price} \\({change}\\)"


def _format_market_price(quote: MarketQuote) -> str:
    decimals = 0 if quote.id == "bitcoin" else 2
    return f"{quote.price:,.{decimals}f}{quote.unit}"


def _render_sentiment_comparison(overall: Mapping[str, object]) -> str | None:
    score = overall.get("score")
    previous = overall.get("previous_close")
    rating = overall.get("rating")
    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or isinstance(previous, bool)
        or not isinstance(previous, (int, float))
        or not isinstance(rating, str)
    ):
        return None
    change = float(score) - float(previous)
    rating_label = {
        "extreme fear": "극단적 공포",
        "fear": "공포",
        "neutral": "중립",
        "greed": "탐욕",
        "extreme greed": "극단적 탐욕",
    }.get(rating, rating)
    text = (
        f"전일 {_format_score(float(previous))} → 오늘 {_format_score(float(score))} "
        f"({change:+.2f}), 현재 {rating_label} 구간"
    )
    return escape_markdown_v2(text)


def _shorten(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1].rstrip() + "…"


def _format_score(score: float) -> str:
    if score.is_integer():
        return str(int(score))
    return f"{score:.2f}".rstrip("0").rstrip(".")


def _source_label(source: str) -> str:
    return {"rules": "Python 규칙"}[source]
