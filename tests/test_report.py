import json
from dataclasses import replace
from pathlib import Path

from fng_chatbot.normalizer import normalize_cnn_payload
from fng_chatbot.report import build_static_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"


def normalized_fixture():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")


def test_fixture_builds_expected_static_report() -> None:
    report = build_static_report(normalized_fixture())

    assert [factor.indicator_id for factor in report.fear_drivers] == [
        "stock_price_strength",
        "safe_haven_demand",
    ]
    assert report.buffer is not None
    assert report.buffer.indicator_id == "put_call_options"
    assert report.fear_note is None
    assert report.summary_source == "rules"
    assert "종합지수는 37.0점(공포)" in report.summary
    assert "주요 공포 요인은 주가 강도, 안전자산 수요" in report.summary
    assert "완충 요인은 풋·콜 옵션" in report.summary


def test_same_input_always_produces_same_report() -> None:
    snapshot = normalized_fixture()

    assert build_static_report(snapshot) == build_static_report(snapshot)


def test_report_requires_no_model_configuration() -> None:
    report = build_static_report(normalized_fixture())

    assert report.summary
    assert report.summary_source == "rules"


def test_all_greed_report_states_no_clear_fear_factor() -> None:
    snapshot = normalized_fixture()
    greedy_indicators = tuple(
        replace(indicator, score=80 + index, rating="greed")
        for index, indicator in enumerate(snapshot.indicators)
    )
    greedy_composite = replace(
        snapshot.composite,
        score=85,
        rating="extreme greed",
    )
    greedy_snapshot = replace(
        snapshot,
        composite=greedy_composite,
        indicators=greedy_indicators,
    )

    report = build_static_report(greedy_snapshot)

    assert report.fear_drivers == ()
    assert report.fear_note == "뚜렷한 공포 요인 없음"
    assert "뚜렷한 공포 요인 없음" in report.summary


def test_static_report_contains_no_trading_advice_or_forecast_language() -> None:
    rendered = json.dumps(build_static_report(normalized_fixture()).to_dict(), ensure_ascii=False)
    forbidden_phrases = ("매수", "매도", "종목 추천", "가격 전망", "BUY", "SELL")

    assert not any(phrase in rendered for phrase in forbidden_phrases)
