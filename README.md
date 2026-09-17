# F&G 알림봇 — Python이 요인을 고르고, LLM이 설명합니다

기존 [F&G 알림봇](https://github.com/0gnm0-afk/fng-chatbot)에 지표 해석을 추가한 프로젝트입니다. **LLM이 설명의 근거까지 임의로 고르지 않도록 Python이 요인을 먼저 선정하고, 선택된 지표의 의미만 짧게 설명하도록 구성했습니다.**

## 1. 누가 사용하는가?

미국 지수투자를 하면서 CNN Fear & Greed Index와 주요 시장 시세를 확인하는 개인투자자를 위한 도구입니다. 제가 사용하던 알림봇을 확장하면서 시작했습니다. 현재 공개본은 로컬 MCP 서버로 실행하며, 결과를 미리보기로 확인하거나 Telegram으로 받을 수 있습니다.

## 2. 어떤 실제 문제가 있는가?

F&G 종합점수와 공포·탐욕 구간을 알아도, 그 안의 어떤 지표가 공포 쪽에 있고 어떤 지표는 다른 모습을 보이는지까지 한눈에 파악하기는 어려웠습니다. 점수와 함께 구성 지표의 의미를 짧게 읽고 싶었습니다.

## 3. 기존에는 어떻게 처리했는가?

먼저 만든 봇은 F&G 종합점수와 환율·비트코인·주가지수·선물 시세를 모아 Telegram으로 보내는 역할을 했습니다. 수치는 자동으로 받아볼 수 있었지만, 그 의미를 해석하는 일은 사용자의 몫이었습니다.

이 프로젝트는 기존 알림의 용도를 이어가면서 F&G 구성 지표와 보조 설명을 추가했습니다. 현재 실행 방식은 기존 Flask 웹서버형 공개 봇과 달리 로컬 MCP입니다. NVIDIA는 모델 추론을 제공하며 봇 전체를 호스팅하는 것은 아닙니다.

## 4. 무엇이 병목이었는가?

수집한 숫자와 등급을 사람이 이해할 수 있는 짧은 설명으로 연결하는 부분이었습니다. 그런데 이 설명을 LLM에 넓게 맡기면 제공하지 않은 뉴스나 그럴듯한 시장 원인을 덧붙일 수 있다는 문제가 있었습니다.

따라서 단순히 AI 설명을 추가하는 것보다 **설명에 사용할 근거를 먼저 정하고, 모델이 맡을 범위를 좁히는 것**을 중요하게 봤습니다.

## 5. AI가 필요한 부분은 어디인가?

선택된 지표의 이름·점수·등급·설명 기준을 한국어 문장으로 풀어 쓰는 부분에 사용했습니다. 당시 NVIDIA에서 모델 API를 무료로 체험할 수 있다는 소식을 계기로 연결했습니다. 이는 제작 계기이며 현재 무료 이용을 보장하는 설명은 아닙니다.

현재 코드는 NVIDIA API의 `deepseek-ai/deepseek-v4-flash`를 호출합니다. 모델에는 뉴스 조사나 가격 예측을 맡기지 않고, 전달된 요인의 의미를 1~2줄로 설명하도록 요청합니다.

**기본 보고서 생성에 AI가 필수인 것은 아닙니다.** Python만으로도 점수·요인·기본 요약을 만들며, LLM은 이를 읽기 쉬운 설명으로 보완하는 역할입니다.

## 6. 규칙이나 일반 코드로 처리할 부분은 어디인가?

데이터 조회·입력 검증·요인 선정·기본 요약·응답 검사·메시지 발송은 Python이 담당합니다. CNN의 F&G 구성 지표 7개 중 다음 기준으로 요인을 고릅니다.

- 공포 요인: `fear`·`extreme fear` 등급에서 점수가 낮은 최대 2개.
- 완충 요인: `neutral`·`greed`·`extreme greed` 등급에서 점수가 가장 높은 1개.
- 해당 지표가 없으면 없다고 표시하며, 동점은 고정 순서로 처리합니다.

아래는 선정 코드를 이해하기 쉽게 줄인 예시입니다. 실제 코드에는 동점 처리와 자료형 정의가 추가되어 있습니다.

```python
fear = sorted(
    [x for x in indicators if x.rating in {"fear", "extreme fear"}],
    key=lambda x: x.score,
)[:2]

buffer = max(
    [x for x in indicators if x.rating in {"neutral", "greed", "extreme greed"}],
    key=lambda x: x.score,
    default=None,
)
```

선정한 요인의 이름·점수·등급과 지표의 설명 기준만 모델에 전달합니다. 예를 들어 시장 변동성에는 “VIX와 50일 이동평균 비교”라는 기준을 함께 줍니다. 모델이 전달받지 않은 요인이나 뉴스를 새로 추가하지 않도록 프롬프트에서 제한합니다.

```text
CNN 구성 지표 수집·검증
    → Python이 설명할 요인 선정
    → 선택된 요인만 NVIDIA API에 전달
    → LLM이 짧은 해석 작성
    → Python이 응답 검사
    → 기본 보고서에 AI 설명 추가
```

공포·완충은 이 선택 규칙에 따른 분류입니다. 시장 움직임의 실제 원인이나 손실 상쇄 효과를 입증한 것은 아닙니다.

## 7. 결과를 어떻게 검증하는가?

모델의 답변을 그대로 붙이지 않고 JSON 구조와 필수 필드, 1~2줄·300자 제한, 참조 지표 ID와 순서가 Python의 선정 목록과 같은지 확인합니다.

```python
# 실제 검증 로직을 줄인 예시
if not 1 <= len(lines) <= 2:
    raise ValueError("설명 줄 수 초과")
if len("\n".join(lines)) > 300:
    raise ValueError("설명 길이 초과")
if tuple(referenced_indicator_ids) != selected_indicator_ids:
    raise ValueError("선정한 지표와 응답의 참조 목록이 다름")
```

API 호출이나 응답 검사가 실패하면 AI 설명을 제외하고 Python 규칙 보고서를 유지합니다. 테스트에는 선택한 요인만 전달되는지, 잘못된 JSON·길이 초과·지표 ID 불일치를 거부하는지 확인하는 항목이 있습니다.

다만 이 검사는 **설명 문장의 사실성까지 보장하지 않습니다.** 올바른 지표 ID를 반환하면서도 부정확하게 설명할 가능성은 남아 있습니다. 뉴스·전망 금지는 프롬프트 지시이며, 그 위반을 의미 수준에서 모두 판별하는 장치는 아닙니다.

## 8. 사용자는 어디에서 판단하거나 승인하는가?

사용자는 보고서 내용을 읽고 투자에 어떻게 활용할지 판단합니다. 이 도구에는 자동 주문 기능이 없습니다.

발송 제어는 현재 구현 그대로 구분해야 합니다. `preview_telegram_report`는 발송 없이 확인하는 도구이고, `send_telegram_report`는 메시지를 구성해 전송합니다. 포함된 보고서 Skill은 **기본적으로 발송하며 매번 별도 승인을 묻지는 않습니다.**

확인만 하려면 “미리보기만, 전송하지 마”라고 요청하거나 `MCP_ALLOW_TELEGRAM_SEND=false`로 설정합니다. 내부 해시 검사는 전송할 원문의 일치를 확인하는 장치이며, 사용자가 내용을 승인했다는 증거는 아닙니다.

## 9. 실제로 무엇이 달라졌는가?

| 기존 공개 봇 | 현재 공개 코드 |
|---|---|
| 종합점수와 구간 중심 | F&G 구성 지표 7개와 선정 요인까지 확인 |
| 정해진 알림 문구 | Python 기본 요약에 선택 요인 해석 추가 |
| 시장 데이터 조회 실패 항목 생략 | 시장 데이터 누락 항목 표시 |
| Flask 발송 경로 | 로컬 MCP 조회·미리보기·발송 도구 |

모델 API를 사용할 수 없어도 기본 보고서는 남습니다. 즉, 지표를 받아보는 도구에서 **어떤 지표를 설명 대상으로 삼았는지 함께 확인할 수 있는 도구**로 확장했습니다.

이는 현재 코드에서 확인한 기능 변화입니다. 환각 감소율·이해도 개선·시간 절감·투자 수익 개선은 측정하지 않았으며 성과 수치로 제시하지 않습니다. 이 문서에는 실제 모델 출력으로 검증하지 않은 예시 해석을 수신 결과처럼 싣지 않았습니다.

## 10. 아직 해결하지 못한 것은 무엇인가?

**입력을 제한하자 해석도 비슷한 틀에 머무르는 문제가 있었습니다.** 실제로 사용하면서 비슷한 요인이 반복해서 선정되면 LLM의 설명도 크게 달라지지 않고, 점수와 등급을 풀어 쓴 뻔한 해석이 반복된다고 느꼈습니다.

환각을 줄이려고 설명할 근거를 좁혔지만, 그만큼 모델이 제공할 수 있는 맥락과 해석의 폭도 좁아졌습니다. 결과적으로 Python 규칙 요약에 비해 LLM 설명이 얼마나 더 유용한지 다시 확인할 필요가 생겼습니다. 이는 실사용 중 관찰한 한계이며, 환각 감소나 유용성의 차이를 수치로 측정한 결과는 아닙니다.

남은 과제는 **근거를 제한하면서도 읽을 가치가 있는 설명을 만드는 것**입니다. 표현만 다양하게 바꾸기보다, 검증 가능한 변화 정보가 추가로 필요한지, 또는 이 범위에서는 규칙 기반 설명만으로 충분한지 비교하려 합니다.

그 밖에도 다음 한계가 남아 있습니다.

- 응답 형식 검사만으로 문장 내용의 정확성이나 근거 없는 설명을 완전히 판별할 수 없습니다.
- 낮은 점수 2개·비공포 최고점 1개라는 규칙이 시장심리를 설명하는 최선의 방법인지 검증하지 않았습니다.
- 웹 데이터 제공 방식의 변경·누락과 모델 API의 오류·이용 제한에 영향을 받습니다.

다음에는 실제 출력 표본에서 근거 일치·반복성·설명 유용성을 평가하고, Python 규칙 요약만 있을 때와 비교하려 합니다.

## 설치

Windows PowerShell에서 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

시스템에 `python` 명령이 없다면 설치된 Python 실행 파일로 첫 명령을 실행한다.

## 환경변수

`.env.example`은 변수 이름을 보여주는 예시일 뿐이며 자동으로 읽히지 않는다. 실제 키·토큰·채팅 ID는 저장소 파일이나 MCP 설정에 기록하지 말고 Windows **사용자 환경변수**에 저장한다. Windows의 `시스템 속성 → 고급 → 환경 변수 → 사용자 변수`에서 다음 이름을 추가한 뒤 Codex 또는 Claude Code를 완전히 다시 시작한다.

| 환경변수 | 용도 | 없을 때 |
|---|---|---|
| `NVIDIA_API_KEY` | NVIDIA hosted DeepSeek 보조 설명 | Python 규칙 문장으로 fallback |
| `TELEGRAM_BOT_TOKEN` | Telegram Bot API 인증 | 실제 전송 불가 |
| `TELEGRAM_CHAT_ID` | Telegram 수신 대상 | 실제 전송 불가 |
| `MCP_ALLOW_TELEGRAM_SEND` | 실제 전송 서버 잠금 | 기본 `true`, `false`면 전송 차단 |
| `CNN_FNG_CACHE_TTL_SECONDS` | CNN 메모리 캐시 초 | 기본 `300` |

Codex 프로젝트 설정은 변수 이름을 MCP 프로세스로 전달하고, Claude 프로젝트 설정은 `${VAR}` 참조로 부모 환경의 값을 MCP 프로세스에 전달한다. 설정 파일에는 실제 값이 들어가지 않는다.

DeepSeek 연결은 NVIDIA API Catalog에서 새 키를 발급해 `NVIDIA_API_KEY` 사용자 환경변수로 저장하고 앱을 재시작한 뒤 `preview_telegram_report` 또는 `send_telegram_report`를 호출할 때 이루어진다. 프로젝트는 `https://integrate.api.nvidia.com/v1/chat/completions`와 `deepseek-ai/deepseek-v4-flash`를 고정 사용하며, `get_fear_greed_report`는 키가 있어도 외부 모델을 호출하지 않는다. `DEEPSEEK_API_KEY`나 `OPENAI_API_KEY`는 이 경로에서 사용하지 않는다.

기본 Telegram 즉시 전송에는 다음 조건이 필요하다.

- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`가 환경에 존재
- `MCP_ALLOW_TELEGRAM_SEND`가 기본 `true`이거나 명시적으로 `true`
- 내부에서 생성한 미리보기 원문과 SHA-256이 일치

보고서 Skill은 별도 승인 대화 없이 `send_telegram_report`를 호출한다. 전송을 막으려면 `MCP_ALLOW_TELEGRAM_SEND=false`로 앱을 재시작하거나 요청에 “미리보기만”, “전송하지 마”를 명시한다.

## Codex 연결

프로젝트를 신뢰한 상태에서 `.codex/config.toml`이 로컬 stdio MCP를 시작한다. 저장소 루트를 Codex 작업 폴더로 열면 프로젝트 설정과 `fear-greed-report` Skill을 함께 사용할 수 있다.

1. 위 설치를 완료한다.
2. Codex에서 이 프로젝트를 신뢰한다.
3. 설정과 Skill 탐색은 작업 시작 시 수행되므로 Codex에서 새 작업을 시작한다.
4. MCP 서버 목록에서 `fng_chatbot`을 확인한다.

Codex용 Skill은 `.agents/skills/fear-greed-report/`에 있다.

자동 인식이 되지 않으면 현재 작업 폴더에서 `codex mcp list`를 실행해 `fng_chatbot`이 `enabled`인지 확인한다. 기존 작업은 새 설정을 동적으로 다시 읽지 않으므로 반드시 새 작업에서 `$fear-greed-report`를 호출한다.

## Claude Code 연결

`.mcp.json`이 `${CLAUDE_PROJECT_DIR:-.}/.venv/Scripts/python.exe`로 같은 stdio MCP를 시작한다.

1. Claude Code가 없다면 Windows PowerShell에서 `irm https://claude.ai/install.ps1 | iex`로 설치하고 `claude --version`을 확인한다.
2. 위 Python 설치를 완료한다.
3. PowerShell에서 이 `fng-chatbot` 프로젝트 루트로 이동한 뒤 `claude`를 실행한다.
4. 프로젝트 MCP 사용을 승인한다.
5. `/mcp`에서 `fng-chatbot`을 확인한다.
6. `/fear-greed-report 현재 F&G 보고서를 조회해줘.`로 기본 즉시 전송을 실행한다. 미리보기만 필요하면 `Telegram 전송은 하지 마`를 덧붙인다.

Claude Code용 Skill은 `.claude/skills/fear-greed-report/`에 있다.

## MCP 도구

### `get_fear_greed_report`

종합점수, 정확히 7개 지표, Python이 선정한 공포·완충 요인, 요약 출처와 데이터 품질을 반환한다. 읽기 전용이며 DeepSeek를 호출하지 않는다.

### `preview_telegram_report`

시장시세와 F&G 분석을 합친 Telegram 원문과 `preview_hash`를 반환한다. DeepSeek 키가 있으면 선택 요인의 의미 설명을 검증해 덧붙이고, 실패하면 규칙 전용 문장을 유지한다. NVIDIA 요청은 20초 timeout을 사용하며, read timeout·HTTP 429·5xx에 대해 대기 간격을 두고 제한적으로 재시도한다. `interpretation` 메타데이터에는 적용/fallback 상태와 모델 또는 검증 단계·transport 유형을 구분하는 안전한 사유 코드가 들어가지만 모델명과 오류 원문은 Telegram 본문에 표시하지 않는다. 메시지를 보내지 않는다.

### `send_telegram_report`

`idempotency_key`와 선택적 `force_refresh`를 받아 현재 보고서·시장시세·DeepSeek 설명을 한 번 렌더링하고, 내부 SHA-256 검증 후 같은 원문을 즉시 전송한다. 결과에는 실제 전송한 원문·해시·`message_id`가 함께 들어간다.

## 직접 실행

```powershell
.\.venv\Scripts\python.exe -m fng_chatbot.mcp_server
```

stdio는 MCP 클라이언트와 통신하는 전송 채널이므로 터미널에서 실행하면 일반 대화형 프롬프트가 나타나지 않는다.

## 검증

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Skill 검증은 저장소의 두 Skill 본문 일치 검사와 Codex Skill 기본 검증을 함께 수행한다.

## 문서

- `설계.md` — 데이터·규칙 기반 보고서·MCP·Skill 경계
- `AGENTS.md`, `CLAUDE.md` — 에이전트별 저장소 진입점
