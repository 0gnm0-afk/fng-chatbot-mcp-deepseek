import json
from pathlib import Path

import pytest

from fng_chatbot.deepseek import NVIDIA_DEEPSEEK_MODEL, DeepSeekJsonCompletion
from fng_chatbot.interpretation import (
    DEEPSEEK_INTERPRETATION_SYSTEM_PROMPT,
    INTERPRETATION_MAX_CHARACTERS,
    DeepSeekInterpreter,
    InterpretationValidationError,
    build_deepseek_prompts,
    build_interpretation_context,
    parse_deepseek_interpretation,
    validate_interpretation,
)
from fng_chatbot.normalizer import normalize_cnn_payload
from fng_chatbot.report import build_static_report

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "cnn_graphdata_minimal.json"


def load_context():
    payload = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    snapshot = normalize_cnn_payload(payload, fetched_at="2026-07-20T00:00:00Z")
    report = build_static_report(snapshot)
    return snapshot, report, build_interpretation_context(snapshot, report)


def valid_input(context):
    return {
        "context": context,
        "lines": [
            "주가 강도와 안전자산 수요가 공포 수준이어서 위험 회피 심리가 두드러집니다.",
            "풋·콜 옵션은 중립이어서 공포가 모든 지표로 확산된 상태는 아닙니다.",
        ],
        "referenced_indicator_ids": list(context.selected_indicator_ids),
        "model": NVIDIA_DEEPSEEK_MODEL,
    }


class FakeCompletionProvider:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, str]] = []

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> DeepSeekJsonCompletion:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return DeepSeekJsonCompletion(
            content=self.content,
            model=NVIDIA_DEEPSEEK_MODEL,
            finish_reason="stop",
        )


def test_context_contains_only_python_selected_factors() -> None:
    snapshot, report, context = load_context()

    assert context == build_interpretation_context(snapshot, report)
    assert context.selected_indicator_ids == (
        "stock_price_strength",
        "safe_haven_demand",
        "put_call_options",
    )
    assert context.to_dict()["output_contract"]["maximum_characters"] == 300


def test_prompt_contains_only_python_selected_factors() -> None:
    snapshot, _, context = load_context()

    system_prompt, user_prompt = build_deepseek_prompts(context)
    payload = json.loads(user_prompt)

    assert system_prompt == DEEPSEEK_INTERPRETATION_SYSTEM_PROMPT
    assert "JSON" in system_prompt
    assert set(payload) == {"selected_factors"}
    assert [factor["indicator_id"] for factor in payload["selected_factors"]] == list(
        context.selected_indicator_ids
    )
    assert "overall" not in payload
    assert all(
        indicator.id not in user_prompt
        for indicator in snapshot.indicators
        if indicator.id not in context.selected_indicator_ids
    )


def test_interpreter_calls_provider_and_validates_json() -> None:
    _, _, context = load_context()
    content = json.dumps(
        {
            "lines": [
                "주가 강도와 안전자산 수요의 위축은 위험 회피 심리가 강하다는 뜻입니다.",
                "풋·콜 옵션의 중립은 공포가 파생시장 전반으로 번지지는 않았음을 보여줍니다.",
            ],
            "referenced_indicator_ids": list(context.selected_indicator_ids),
        },
        ensure_ascii=False,
    )
    provider = FakeCompletionProvider(content)

    interpretation = DeepSeekInterpreter(provider).interpret(context)

    assert interpretation.model == NVIDIA_DEEPSEEK_MODEL
    assert interpretation.source == "deepseek"
    assert interpretation.referenced_indicator_ids == context.selected_indicator_ids
    assert len(provider.calls) == 1
    assert json.loads(provider.calls[0]["user_prompt"])["selected_factors"]


def test_validates_two_grounded_lines_and_records_model_metadata() -> None:
    _, _, context = load_context()

    interpretation = validate_interpretation(**valid_input(context))

    assert len(interpretation.lines) == 2
    assert interpretation.referenced_indicator_ids == context.selected_indicator_ids
    assert interpretation.to_dict()["model"] == NVIDIA_DEEPSEEK_MODEL
    assert interpretation.to_dict()["source"] == "deepseek"


@pytest.mark.parametrize(
    ("field", "value", "message", "reason_code"),
    [
        ("lines", ["하나", "둘", "셋"], "one or two lines", "validation_line_count"),
        ("lines", [""], "non-empty single lines", "validation_invalid_line"),
        (
            "lines",
            ["첫 줄\n둘째 줄"],
            "non-empty single lines",
            "validation_invalid_line",
        ),
        (
            "lines",
            ["가" * (INTERPRETATION_MAX_CHARACTERS + 1)],
            "exceeds 300 characters",
            "validation_too_long",
        ),
        (
            "lines",
            ["가" * 150, "나" * 150],
            "exceeds 300 characters",
            "validation_too_long",
        ),
        (
            "referenced_indicator_ids",
            ["market_momentum"],
            "exactly match",
            "validation_id_mismatch",
        ),
        ("model", "", "must not be empty", "validation_missing_model"),
    ],
)
def test_rejects_ungrounded_or_unsafe_output(
    field: str,
    value: object,
    message: str,
    reason_code: str,
) -> None:
    _, _, context = load_context()
    arguments = valid_input(context)
    arguments[field] = value

    with pytest.raises(InterpretationValidationError, match=message) as error:
        validate_interpretation(**arguments)

    assert error.value.reason_code == reason_code


@pytest.mark.parametrize(
    ("content", "message", "reason_code"),
    [
        ("not-json", "not valid JSON", "validation_invalid_json"),
        ("[]", "must be an object", "validation_json_not_object"),
        ('{"lines":[]}', "contain only", "validation_invalid_keys"),
        (
            '{"lines":[],"referenced_indicator_ids":[],"extra":true}',
            "contain only",
            "validation_invalid_keys",
        ),
        (
            '{"lines":"설명","referenced_indicator_ids":[]}',
            "string array",
            "validation_lines_not_array",
        ),
        (
            '{"lines":["설명"],"referenced_indicator_ids":"id"}',
            "string array",
            "validation_ids_not_array",
        ),
    ],
    ids=["invalid-json", "array", "missing-key", "extra-key", "lines-type", "ids-type"],
)
def test_rejects_invalid_deepseek_json_shape(
    content: str,
    message: str,
    reason_code: str,
) -> None:
    _, _, context = load_context()

    with pytest.raises(InterpretationValidationError, match=message) as error:
        parse_deepseek_interpretation(
            context=context,
            content=content,
            model=NVIDIA_DEEPSEEK_MODEL,
        )

    assert error.value.reason_code == reason_code


def test_unknown_validation_reason_code_falls_back_to_generic_code() -> None:
    error = InterpretationValidationError(
        "sensitive validation detail",
        reason_code="sensitive-provider-detail",
    )

    assert error.reason_code == "validation_error"
