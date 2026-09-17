# F&G 알림봇 — Python 요인 선정과 LLM 해석

기존 [F&G 알림봇](https://github.com/0gnm0-afk/fng-chatbot)에 시장심리 지표의 해석을 추가한 프로젝트입니다. **Python이 설명할 요인을 먼저 고르고, LLM은 선택된 요인의 의미를 짧게 풀어 쓰도록 역할을 나눴습니다.**

현재 공개본은 CNN Fear & Greed Index와 주요 시장 데이터를 조회하고, NVIDIA API의 DeepSeek를 통해 보조 설명을 생성하는 로컬 MCP 도구입니다. 결과는 미리보기로 확인하거나 Telegram으로 받을 수 있습니다.

## 왜 확장했나

기존 봇으로 F&G 점수와 시장 시세를 받아보고 있었지만, 숫자와 공포·탐욕 구간만으로는 구성 지표의 의미를 이해하기 어려웠습니다. 어떤 지표가 공포 쪽에 있고, 어떤 지표는 다른 모습을 보이는지 설명을 함께 읽고 싶었습니다.

당시 NVIDIA에서 모델 API를 무료로 체험할 수 있다는 소식을 계기로, 기존 알림에 LLM 해석을 붙였습니다. 현재도 무료로 이용할 수 있다는 보장을 뜻하지는 않습니다.

이 과정에서 중요하게 본 문제는 **LLM이 제공되지 않은 뉴스나 그럴듯한 이유를 만들어내는 것**이었습니다. 그래서 설명할 요인의 선정까지 모델에 맡기지 않고, Python 규칙으로 선정한 지표만 전달하도록 구성했습니다.

## 기존 봇에서 달라진 점

| 구분 | 기존 공개 봇 | 이 프로젝트 |
|---|---|---|
| F&G 정보 | 종합점수와 심리 구간 | 종합점수와 7개 구성 지표 |
| 요약 | 정해진 구간·알림 문구 | Python이 공포·완충 요인을 고르고 기본 요약 생성 |
| AI 역할 | 없음 | 선택된 요인의 의미를 짧게 설명 |
| 실행 방식 | Flask 발송 경로 | 로컬 MCP의 조회·미리보기·발송 도구 |

환율·비트코인·코스피·나스닥·S&P500·나스닥 선물·S&P500 선물을 함께 보는 용도는 이어갑니다. 현재 코드는 네이버·Yahoo의 웹 데이터 주소에서 JSON을 가져옵니다. 이 시장 데이터 7종과 **F&G 구성 지표 7개는 별개**입니다.

## LLM이 설명할 범위를 어떻게 제한했나

```text
CNN 데이터 수집·입력 검증
  → Python이 7개 구성 지표에서 공포·완충 요인 선정
  → 선택한 요인의 이름·점수·등급·설명 기준만 NVIDIA API에 전달
  → DeepSeek가 한국어 설명 생성
  → 응답 형식·길이·참조 지표 ID 검사
  → 기본 요약에 AI 설명 추가
     실패하면 Python 규칙 보고서 유지
```

### 1. 요인 선정은 Python이 담당

F&G의 구성 지표는 시장 모멘텀, 주가 강도, 시장 폭, 풋·콜 옵션, 시장 변동성, 안전자산 수요, 정크본드 수요입니다. CNN이 제공하는 점수와 등급을 읽어 다음 규칙을 적용합니다.

- **공포 요인:** `fear`·`extreme fear` 지표 중 점수가 낮은 순서로 최대 2개를 선택합니다.
- **완충 요인:** `neutral`·`greed`·`extreme greed` 지표 중 점수가 가장 높은 1개를 선택합니다.
- 해당 지표가 없으면 없다고 표시합니다. 빈자리를 모델이 새 근거로 채우게 하지 않습니다.

여기서 공포·완충은 보고서의 선택 규칙에 따른 이름입니다. 실제 시장 움직임의 원인이나 손실을 상쇄하는 효과를 입증한 것은 아닙니다.

### 2. LLM에는 선택한 요인만 전달

모델은 지표를 새로 고르거나 시장 전체의 원인을 추론하는 대신, 전달받은 요인이 시장심리에서 어떤 의미인지 설명하도록 요청받습니다. 뉴스·외부 사실·가격 전망·매수·매도 추천을 추가하지 않도록 프롬프트에서 제한합니다.

현재 코드는 `NVIDIA_API_KEY`를 사용해 NVIDIA의 모델 API를 호출합니다. 지정 모델은 `deepseek-ai/deepseek-v4-flash`입니다. 모델을 직접 학습하거나 개인 컴퓨터에서 실행하는 구조는 아닙니다.

### 3. 응답을 검사하고 실패하면 기본 보고서 유지

설명은 **1~2줄, 전체 300자 이내**로 제한합니다. JSON 구조와 필수 필드를 확인하고, 모델이 반환한 참조 지표 ID와 순서가 Python이 선택한 목록과 같은지 검사합니다.

키가 없거나 API 호출·응답 검사가 실패하면 AI 설명을 덧붙이지 않고 Python 규칙 보고서를 유지합니다. 모델 API에 문제가 생겨도 기본 보고서를 확인할 수 있도록 했습니다.

다만 이 검사는 **설명 문장의 사실성까지 검증하지는 않습니다.** 올바른 ID를 반환하면서 부정확한 문장을 쓸 가능성은 남아 있습니다. 근거 없는 설명을 줄이려는 설계이며, 환각을 완전히 제거했다고 주장하지 않습니다.

## 어떤 결과를 확인할 수 있나

보고서는 종합점수와 기준 시각, 선택된 공포·완충 요인과 점수, Python 기본 요약을 보여줍니다. AI 설명이 적용되면 보조 설명과 적용 표시가 추가됩니다. 시장시세 일부를 가져오지 못하면 누락 항목을 표시합니다.

예를 들어 Python이 주가 강도·안전자산 수요·풋콜 옵션을 선택했다면, 모델 입력에는 그 세 지표의 정보만 들어갑니다. 선택하지 않은 지표나 뉴스는 전달하지 않습니다. 이는 처리 방식의 예시이며 실제 현재 시장 분석이나 모델 실행 결과는 아닙니다.

| 도구 | 확인할 결과 |
|---|---|
| `get_fear_greed_report` | 구성 지표·선정 요인·규칙 요약. NVIDIA 모델 호출 없음 |
| `preview_telegram_report` | 시장시세와 보조 해석을 포함한 메시지 미리보기. 발송 없음 |
| `send_telegram_report` | 메시지를 구성해 Telegram으로 발송 |

기존 Google Cloud 예약 봇과 달리, 이 저장소는 사용자의 컴퓨터에서 MCP 서버를 실행하는 구성이며 모델 추론만 NVIDIA API에 요청합니다. 포함된 보고서 Skill은 기본적으로 발송 도구를 호출하므로, 확인만 할 때는 **“미리보기만, 전송하지 마”**라고 요청하거나 전송 설정을 꺼야 합니다.

## 검증과 남은 과제

테스트에는 선택한 요인만 프롬프트에 전달되는지, 잘못된 JSON·길이 초과·참조 ID 불일치를 거부하는지 확인하는 항목이 있습니다. 이러한 입력·응답 검사는 모델 해석의 정확성이나 투자 성과를 보장하지 않습니다.

다음 과제는 반복적이거나 과도하게 단정적인 설명을 실제 표본으로 평가하는 것입니다. 현재 README에서는 환각 감소율이나 해석 정확도 개선을 수치로 주장하지 않습니다. 자동 주문이나 매매 판단을 맡기는 도구로도 사용하지 않습니다.

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
