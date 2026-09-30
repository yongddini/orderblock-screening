# Obscreen — Claude Code 개발 가이드

오더블록(Order Block) 기반 **국내주식 스크리닝 웹서비스**. 매일 장 마감 뒤 코스피·코스닥·ETF를
스크리닝해 오더블록 근처 종목을 SQLite에 저장하고, Flask 웹 화면으로 보여 준다.
서버는 `/home/rocky/orderblock`(gunicorn + 매일 20:30 KST cron `daily_screening.sh`).

**이 저장소에서 Claude Code의 역할은 「개발자」다.** 완료 판단·상태 관리·다음 작업 제안은
PM 러너가, 머지는 사용자가 한다.

## 역할 분담

- **Claude Code (로컬)** = 개발. 아래 「개발 워크플로우」만 수행한다.
- **PM 러너** = In Review 검토 · 상태 관리 · 다음 Todo 제안 · 보고. 코드는 건드리지 않는다.
- **사용자** = 이슈 승인(**Approved** 상태로 옮김) · PR 머지 · 서버 배포.

## Linear · GitHub

- Linear 팀 **Obscreen** (key: `OBS`). 원격 `github.com/yongddini/orderblock-screening`
  (기본 브랜치 `main`).
- Linear ↔ GitHub 연동으로 브랜치/PR/머지가 이슈 상태를 움직인다. ⚠️ **In Review 자동 이동이
  안 된 적이 있다**(OBS-1) — PR을 연 뒤 이슈 상태를 다시 읽고, 안 옮겨졌으면 직접 옮긴다.

## 개발 워크플로우

1. **Approved 이슈만 개발한다.** Backlog/Todo/Rejected는 절대 개발하지 않는다. `blockedBy`가
   아직 Done이 아니면 착수하지 않는다.
2. **착수 = 브랜치 생성.** 이슈의 Linear 제안 브랜치명(`yu04038/obs-N-...`)으로 `origin/main`에서
   만들고 이슈를 **In Progress**로 옮긴다. `main`에서 직접 작업하지 않는다.
3. 이슈의 완료 기준을 충족하도록 개발한다. 「범위 밖」은 건드리지 않는다. 이슈가 「사용자 확인」을
   요구하는 결정은 임의로 정하지 말고 PR·이슈 코멘트에 권고안을 적어 묻는다.
4. 커밋 메시지 앞에 이슈 식별자: `OBS-2: pyproject·uv 전환`.
5. **완료 = push + PR.** PR 제목·본문에 `OBS-N`을 넣고, 본문에 변경 파일 · 완료 기준 체크리스트 ·
   품질 게이트 결과를 적는다. CI가 초록불이 아니면 리뷰 대상이 아니다.
6. PM이 변경요청하면 **같은 브랜치/PR에 추가 커밋**한다.
7. **PR은 자동 머지하지 않는다.** 머지는 사용자가 한다. 이슈를 Done으로 옮기지 않는다.

## 품질 게이트 (커밋/리뷰요청 전 필수)

```bash
uv sync --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

- 통과 판정은 exit 코드가 아니라 **완주 증거**다 — pytest는 `N passed in ...` 요약 줄이 찍혀야 한다.
- 선택: `uv run pre-commit install`로 커밋 훅을 건다.
- **mypy는 점진 적용이다** — `pyproject.toml`의 `[tool.mypy] files`에 적힌 경로(새 코드)만
  strict로 검사하고, 레거시 모듈은 `ignore_errors` override로 둔다. **새 패키지를 만들면
  `files`에 추가**하고, 레거시 모듈을 정리하면 override 목록에서 뺀다. ruff도 같은 방식이다
  (레거시 모듈은 `per-file-ignores`).
- 테스트는 **외부 API(pykrx·FinanceDataReader·yfinance)를 부르지 않는다** — 고정 입력을 쓴다.
  테스트는 `tests/conftest.py`가 `DB_PATH`를 임시 파일로 박은 뒤 앱을 import한다(레거시 앱은
  import 순간 DB를 만든다). `conftest.py`가 모든 테스트에서 TCP·DNS를 막는다.
- **현행 동작 스냅샷(OBS-3)** — `tests/snapshots/*.json`이 탐지·스크리닝·API 출력을 고정한다.
  **동작을 일부러 바꾼 PR에서만** `UPDATE_SNAPSHOTS=1 uv run pytest`로 갱신하고 PR 본문에
  이유를 적는다. 구조 정리처럼 동작이 안 바뀌어야 하는 PR에서 스냅샷이 바뀌면 버그다
  (자세한 것은 `tests/README.md`).

## 프로젝트 구조 (리모델링 전 — 평평한 레거시 레이아웃)

```
app_production.py     # Flask 앱(gunicorn 진입점) · 라우트 · 차트 API · init_db()
screening_core.py     # 스크리닝 실행·DB 저장(오더블록 + 외국인/기관 매매)
collect_data.py       # cron 진입점 CLI (--all/--screening/--investor [YYYYMMDD])
stock_screener.py     # 종목 스크리너(병렬)
realtime_detector.py  # 오더블록 탐지기(룩어헤드 제거 버전)
orderblock_info.py    # 오더블록 자료구조
indicators.py         # ATR·스윙 등 지표
data_provider.py      # 시세 조회(pykrx·FinanceDataReader)
templates/            # Flask 템플릿(index.html · investor.html)
index.html            # 루트의 옛 사본(templates/index.html과 내용이 다르다 — 리모델링 3에서 정리)
daily_screening.sh    # 서버 cron 스크립트(서버 경로 하드코딩)
tests/                # pytest — 스모크 + 현행 동작 고정 특성 테스트(OBS-3, tests/README.md)
```

- 의존성은 `pyproject.toml` + `uv.lock`이 정본이다. `requirements.txt`는 서버 배포가 아직 쓰므로
  리모델링 5(배포)까지 남겨 둔다.
- 설정은 환경변수(`.env.example` 참고). API 키·시크릿은 코드에 하드코딩하지 않는다.

## 리모델링 방향 (사용자 결정 2026-09-30)

순서: 1 토대(OBS-2) → 2 현행 동작 고정 회귀 테스트(OBS-3) → 3 패키지 구조 정리 →
4 탐지기 교체 → 5 서버 배포 변경. **아래층이 굳은 뒤 위층.**

- **탐지기 = AlphaBlock 재사용.** 오더블록 탐지 로직은 `~/AlphaBlock`(암호화폐 자동매매 저장소)의
  탐지기를 가져다 쓴다. 공유 방식(복사/패키지화)은 해당 이슈에서 사용자 확인 후 정한다.
  ⚠️ `~/AlphaBlock`은 **읽기만** 한다 — 쓰기·브랜치 변경·프로세스 종료 금지(다른 자동 루틴이 돈다).
- **웹 = 기존 도메인에서 Flask 유지**, 관리하기 쉽게 정리한다(프레임워크 교체 아님).
- 탐지기를 바꾸기 전에 **현행 동작을 테스트로 먼저 고정**한다(리모델링 2).

## 안전 규칙

- 서버(`/home/rocky/orderblock`, 운영 도메인)에 직접 접속·배포하지 않는다(로컬에서 SSH 불가).
  서버 작업은 서버에서 돌릴 스크립트와 런북으로만 넘긴다. 도메인·DNS·TLS는 건드리지 않는다.
- `.env`·DB 파일·로그는 커밋하지 않는다(`.gitignore`).
- 테스트 실패/완료 기준 미충족 상태로 PR을 열지 않는다. PR을 자동 머지하지 않는다.
