# fng-chatbot — Codex 진입점

작업 전 `STATUS.md`, `설계.md`, `DECISIONS.md`를 읽고 이번 작업을 `STATUS.md` 작업 큐에 `[진행중]`으로 선기록한다. 작업 단위가 끝날 때마다 최근 체크포인트와 다음 시작 지점을 갱신하고, 정상 종료 시 `[진행중]`을 남기지 않는다.

## 먼저 볼 파일

- `src/fng_chatbot/mcp_server.py` — MCP 도구와 전송 경계
- `src/fng_chatbot/normalizer.py` — 정확히 7개 지표 계약
- `src/fng_chatbot/report.py` — 규칙 기반 요약
- `src/fng_chatbot/deepseek.py` — NVIDIA hosted DeepSeek REST·인증·응답 경계
- `src/fng_chatbot/interpretation.py` — 선택 요인 전용 프롬프트와 출력 검증
- `.agents/skills/fear-greed-report/SKILL.md` — Codex 분석 절차
- `.codex/config.toml` — 프로젝트 stdio MCP 연결

## 저장소 규칙

- CNN 원본 응답과 내부 정규화 모델을 분리한다.
- 결과의 구성 지표는 중복 없이 정확히 7개다.
- 요인 선정과 규칙 요약은 7개 지표를 사용하는 결정적 Python 규칙으로 만든다.
- DeepSeek는 Telegram 미리보기에서 Python 선정 요인의 의미만 설명하며, 조회 도구에서는 호출하지 않는다.
- 모델은 NVIDIA hosted `deepseek-ai/deepseek-v4-flash` 비사고 모드로 고정하고 공급자·모델 선택 계층을 추가하지 않는다.
- DeepSeek 실패는 규칙 fallback으로 격리하고 모델명·오류 원문은 Telegram 본문에 넣지 않는다.
- 실제 외부 API와 Telegram은 테스트에서 mock 또는 fixture로 대체한다.
- 키·토큰·채팅 ID·인증 헤더를 코드, fixture, 로그, 문서, MCP 설정에 넣지 않는다.
- 자동 매매, 종목 추천, 매수·매도 권유를 추가하지 않는다.
- Telegram 전송은 표시된 미리보기의 명시적 승인과 SHA-256 일치를 요구한다.
- MCP 도구를 바꾸면 두 Skill, 두 MCP 설정, 계약 테스트를 함께 확인한다.
- 공개 전 개인 식별 정보, 내부 출처, 로컬 절대 경로를 전체 검색한다.

## 검증 명령

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

Codex Skill을 수정하면 `skill-creator`의 `quick_validate.py`도 실행한다.
