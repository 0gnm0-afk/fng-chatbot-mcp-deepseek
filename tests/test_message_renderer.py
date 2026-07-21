import json
from dataclasses import replace
from pathlib import Path

from fng_chatbot.driver_selector import NO_CLEAR_FEAR_FACTOR
from fng_chatbot.interpretation import ReportInterpretation
from fng_chatbot.message_renderer import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    escape_markdown_v2,
    render_telegram_message,
)
from fng_chatbot.normalizer import normalize_cnn_payload
from fng_chatbot.report import StaticReport, build_static_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"


def load_snapshot():
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)
    return normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")


def load_comment():
    return build_static_report(load_snapshot())


def test_renders_fixture_as_bounded_informational_message() -> None:
    snapshot = load_snapshot()
    message = render_telegram_message(snapshot, load_comment())

    assert message.startswith("📊 *F&G 시장심리*")
    assert "종합: *37* \\(fear\\)" in message
    assert "*주요 공포 요인*" in message
    assert "주가 강도" in message
    assert "안전자산 수요" in message
    assert "*완충 요인*" in message
    assert "풋·콜 옵션" in message
    assert "*한 줄 해석*" in message
    assert "요약: Python 규칙" in message
    assert len(message) <= TELEGRAM_MAX_MESSAGE_LENGTH


def test_escapes_every_markdown_v2_special_character() -> None:
    specials = "_*[]()~`>#+-=|{}.!\\"

    escaped = escape_markdown_v2(specials)

    assert escaped == "".join(f"\\{character}" for character in specials)
    assert escape_markdown_v2("a_b.c!") == r"a\_b\.c\!"


def test_dynamic_markdown_is_escaped_inside_message() -> None:
    comment = load_comment()
    first = replace(
        comment.fear_drivers[0],
        indicator="지표_[테스트]",
        explanation="설명 *강조* (임의).",
    )
    modified = replace(
        comment,
        fear_drivers=(first, *comment.fear_drivers[1:]),
        summary="요약_테스트 [링크](주소)!",
    )

    message = render_telegram_message(load_snapshot(), modified)

    assert r"지표\_\[테스트\]" in message
    assert r"설명 \*강조\* \(임의\)\." in message
    assert r"요약\_테스트 \[링크\]\(주소\)\!" in message


def test_long_report_text_is_shortened_before_markdown_escaping() -> None:
    comment = load_comment()
    long_special_text = "_*[]()!" * 2000
    fear_drivers = tuple(
        replace(factor, explanation=long_special_text) for factor in comment.fear_drivers
    )
    buffer = replace(comment.buffer, explanation=long_special_text)
    modified = replace(
        comment,
        fear_drivers=fear_drivers,
        buffer=buffer,
        summary=long_special_text,
    )

    message = render_telegram_message(load_snapshot(), modified)

    assert "…" in message
    assert len(message) <= TELEGRAM_MAX_MESSAGE_LENGTH


def test_missing_factors_are_stated_without_inventing_them() -> None:
    comment = StaticReport(
        fear_drivers=(),
        buffer=None,
        fear_note=NO_CLEAR_FEAR_FACTOR,
        summary="제공된 지표에는 뚜렷한 공포 요인이 없습니다.",
        summary_source="rules",
    )

    message = render_telegram_message(load_snapshot(), comment)

    assert NO_CLEAR_FEAR_FACTOR in message
    assert "뚜렷한 완충 요인 없음" in message


def test_same_input_always_renders_the_same_preview() -> None:
    snapshot = load_snapshot()
    comment = load_comment()

    assert render_telegram_message(snapshot, comment) == render_telegram_message(snapshot, comment)


def test_renders_agent_interpretation_after_rules_summary_without_model_name() -> None:
    snapshot = load_snapshot()
    comment = load_comment()
    interpretation = ReportInterpretation(
        lines=(
            "공포 요인이 여러 지표에서 확인돼 위험 회피 심리가 두드러집니다.",
            "완충 요인은 일부 심리 압력을 제한하고 있습니다.",
        ),
        referenced_indicator_ids=(
            "stock_price_strength",
            "safe_haven_demand",
            "put_call_options",
        ),
        model="deepseek-ai/deepseek-v4-flash",
    )

    message = render_telegram_message(snapshot, comment, interpretation)

    summary_position = message.index("종합지수는")
    interpretation_position = message.index("공포 요인이")
    assert summary_position < interpretation_position
    assert "설명: AI 보조" in message
    assert r"deepseek\-v4\-flash" not in message
