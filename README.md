# fng-chatbot

CNN Fear & Greed 종합점수와 7개 구성 지표, 주요 시장시세를 수집해 근거가 검증된 시장심리 보고서를 만드는 로컬 MCP 도구다. Codex와 Claude Code에서 같은 Python 코어와 MCP 도구를 사용하며, Skill이 조회·검수·Telegram 미리보기·승인 전송 순서를 안내한다.

```text
사용자 요청
  → 프로젝트 Skill
  → 로컬 stdio MCP
  → CNN·시장시세 수집과 정규화
  → Python 규칙으로 공포·완충 요인 선정
  → Telegram 미리보기에서 DeepSeek가 선택 요인의 의미만 설명
     (키 누락·API·검증 실패 시 Python 규칙 문장으로 fallback)
  → Telegram 미리보기
  → 명시적 승인 뒤 동일 원문 전송
```

## 주요 기능

- CNN 종합점수와 정확히 7개 구성 지표 수집·정규화
- 네이버 코스피와 Yahoo 주요 시세의 항목별 실패 격리
- Python 규칙으로 재현 가능한 공포·완충 요인과 정보성 요약 생성
- NVIDIA hosted `deepseek-ai/deepseek-v4-flash` 비사고 모드가 Python 선정 요인의 의미를 1~2줄·300자 이내로 설명
- DeepSeek 키 누락·네트워크·HTTP·응답·검증 실패 시 규칙 전용 미리보기로 복귀
- Telegram MarkdownV2 미리보기와 SHA-256 결속
- 로컬 FastMCP 도구 3개
- Codex·Claude Code 프로젝트 Skill 제공

자동 매매, 종목 추천, 매수·매도 신호, 제공되지 않은 뉴스나 전망은 지원하지 않는다.

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
| `MCP_ALLOW_TELEGRAM_SEND` | 실제 전송 서버 잠금 | 기본 `false`, 전송 차단 |
| `CNN_FNG_CACHE_TTL_SECONDS` | CNN 메모리 캐시 초 | 기본 `300` |

Codex 프로젝트 설정은 변수 이름을 MCP 프로세스로 전달하고, Claude 프로젝트 설정은 `${VAR}` 참조로 부모 환경의 값을 MCP 프로세스에 전달한다. 설정 파일에는 실제 값이 들어가지 않는다.

DeepSeek 연결은 NVIDIA API Catalog에서 새 키를 발급해 `NVIDIA_API_KEY` 사용자 환경변수로 저장하고 앱을 재시작한 뒤 `preview_telegram_report`를 호출할 때 이루어진다. 프로젝트는 `https://integrate.api.nvidia.com/v1/chat/completions`와 `deepseek-ai/deepseek-v4-flash`를 고정 사용하며, `get_fear_greed_report`는 키가 있어도 외부 모델을 호출하지 않는다. `DEEPSEEK_API_KEY`나 `OPENAI_API_KEY`는 이 경로에서 사용하지 않는다.

실제 Telegram 전송에는 다음 조건이 모두 필요하다.

- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`가 환경에 존재
- `MCP_ALLOW_TELEGRAM_SEND=true`
- 사용자가 현재 대화에서 표시된 미리보기를 명시적으로 승인
- 전송 요청의 원문과 SHA-256이 미리보기 결과와 일치

처음에는 `MCP_ALLOW_TELEGRAM_SEND=false`를 유지한 채 미리보기만 확인한다. 실제 전송을 원할 때에만 사용자 환경변수를 `true`로 바꾸고 앱을 재시작한 다음, 표시된 미리보기를 현재 대화에서 다시 명시적으로 승인한다.

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
6. `/fear-greed-report 현재 F&G 보고서를 조회해줘. Telegram 전송은 하지 마.`로 Skill을 직접 실행한다.

Claude Code용 Skill은 `.claude/skills/fear-greed-report/`에 있다.

## MCP 도구

### `get_fear_greed_report`

종합점수, 정확히 7개 지표, Python이 선정한 공포·완충 요인, 요약 출처와 데이터 품질을 반환한다. 읽기 전용이며 DeepSeek를 호출하지 않는다.

### `preview_telegram_report`

시장시세와 F&G 분석을 합친 Telegram 원문과 `preview_hash`를 반환한다. DeepSeek 키가 있으면 선택 요인의 의미 설명을 검증해 덧붙이고, 실패하면 규칙 전용 문장을 유지한다. NVIDIA 요청은 20초 timeout을 사용하고 read timeout·HTTP 429·5xx에만 한 번 더 시도한다. `interpretation` 메타데이터에는 적용/fallback 상태와 모델 또는 검증 단계·transport 유형을 구분하는 안전한 사유 코드가 들어가지만 모델명과 오류 원문은 Telegram 본문에 표시하지 않는다. 메시지를 보내지 않는다.

### `send_telegram_report`

검수한 `preview_text`, 일치하는 `preview_hash`, 실행 식별자, `confirm_send=true`를 받아 같은 원문만 전송한다. 기본적으로 비활성화돼 있다.

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
