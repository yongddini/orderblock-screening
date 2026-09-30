# Obscreen — Claude Code 개발 가이드

오더블록(Order Block) 기반 **국내주식 스크리닝 웹서비스**. 매일 장 마감 뒤 코스피·코스닥·ETF를
스크리닝해 오더블록 근처 종목을 SQLite에 저장하고, Flask 웹 화면으로 보여 준다.
서버는 `/home/rocky/orderblock`(gunicorn `app_production:app`는 `venv310` 가상환경 · 매일 20:05 KST cron
`daily_screening.sh`).

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
- **mypy strict · ruff 전체 규칙이 전 코드에 걸린다**(OBS-4가 레거시 예외 목록을 없앴다).
  pandas는 타입 스텁이 없어 `pd.DataFrame`이 `Any`다 — 시세 조회가 `None`을 돌려줄 수 있어도
  반환 타입에 `| None`을 적지 않았다(`obscreen/data/provider.py` 모듈 설명).
- 테스트는 **외부 API(pykrx·FinanceDataReader·yfinance)를 부르지 않는다** — 고정 입력을 쓴다.
  테스트는 `tests/conftest.py`가 `DB_PATH`를 임시 파일로 박은 뒤 앱을 import한다(레거시 앱은
  import 순간 DB를 만든다). `conftest.py`가 모든 테스트에서 TCP·DNS를 막는다.
- **현행 동작 스냅샷(OBS-3)** — `tests/snapshots/*.json`이 탐지·스크리닝·API 출력을 고정한다.
  **동작을 일부러 바꾼 PR에서만** `UPDATE_SNAPSHOTS=1 uv run pytest`로 갱신하고 PR 본문에
  이유를 적는다. 구조 정리처럼 동작이 안 바뀌어야 하는 PR에서 스냅샷이 바뀌면 버그다
  (자세한 것은 `tests/README.md`).

## 프로젝트 구조 (OBS-4 — `obscreen` 패키지)

```
obscreen/
  config.py            # 설정(pydantic-settings, `.env`) — DB 경로·로그·포트·근접도·탐지기 값
  cli.py · __main__.py # `obscreen collect` · `obscreen serve` (서버 cron: `python3 -m obscreen collect`)
  collect.py           # 수집 본체(옛 collect_data.py — 인자·종료 코드 동일)
  data/provider.py     # 시세 조회(pykrx·FinanceDataReader)
  data/store.py        # SQLite 스키마(init_db)
  detect/base.py       # 탐지기 프로토콜 — 리모델링 4(OBS-5)가 갈아끼울 자리
  detect/realtime.py · orderblock.py · indicators.py   # 현행 탐지기·자료구조·지표
  screening/core.py    # 스크리닝 실행·DB 저장(오더블록 + 외국인/기관) — **유일한** run_and_save_screening
  screening/screener.py# 종목 스크리너(병렬)
  web/app.py           # 조합 루트: create_app() · DB_PATH · 탐지기 클래스 · init_db()
  web/deps.py          # Blueprint가 조합 루트 전역을 **호출 시점에** 읽는 창구
  web/pages.py · screening_api.py · investor_api.py · chart_api.py   # Blueprint
templates/ · static/    # Flask 템플릿·정적 파일(위치·URL 불변 — /static/...)
daily_screening.sh      # 서버 cron — 스크립트 폴더 기준으로 돈다(경로 하드코딩 없음)
app_production.py 등    # 호환 별칭(아래)
tests/                  # pytest — 스모크 + 현행 동작 고정 특성 테스트(OBS-3, tests/README.md)
```

- **루트의 옛 이름은 별칭이다** — `app_production`·`screening_core`·`stock_screener`·
  `realtime_detector`·`orderblock_info`·`indicators`·`data_provider`·`collect_data`를 import하면
  `sys.modules`를 바꿔 끼워 **패키지 모듈 객체 자체**를 받는다. 그래서 gunicorn
  `app_production:app`·`python3 collect_data.py`·특성 테스트의 monkeypatch가 그대로 동작한다.
  **새 코드는 `obscreen.*`를 직접 import한다**(패키지 안에서 옛 이름을 쓰면 안 된다).
- ⚠️ **모듈 전역을 import 시점에 복사하지 말 것** — 테스트는 `app_production.DB_PATH`·
  `app_production.RealtimeOrderBlockDetector`를 바꿔 끼운다. Blueprint는 `obscreen.web.deps`로
  호출 시점에 읽는다(`from obscreen.web.app import DB_PATH`로 복사하면 조용히 옛 값을 쓴다).
- 설정 기본값은 옮기기 전 코드의 숫자와 같다. 옛 환경변수 이름(`DB_PATH`·`SCREENING_TOP_N`·
  `SCREENING_ETF_N`·`FLASK_ENV`)은 그대로, 새 값은 `OBSCREEN_*`(`.env.example`).
- 의존성은 `pyproject.toml` + `uv.lock`이 정본이다. `requirements.txt`는 서버 배포가 아직 쓰므로
  리모델링 5(배포)까지 남겨 둔다.
  ⚠️ OBS-4가 `pydantic-settings`를 더했다(두 파일 모두) — 서버는 이 커밋을 받은 뒤
  `pip install -r requirements.txt`를 한 번 해야 cron·gunicorn이 뜬다.
- 설정은 환경변수(`.env.example` 참고). API 키·시크릿은 코드에 하드코딩하지 않는다.

- **차트는 Lightweight Charts 한 가지다(OBS-10)** — AlphaBlock과 같은 v5.2.0을 `static/vendor/`에
  벤더링하고 `static/js/ob_chart.js`가 캔들 + 존 박스(캔버스 프리미티브 **하나**) + RSI 보조창을
  그린다. 존마다 시리즈를 만들지 말 것(AlphaBlock에서 2,000개에 브라우저가 멈췄다). 순수 함수는
  `tests/test_chart_frontend.py`가 node로 검사한다. plotly `create_chart_html*`·실험 라우트는
  OBS-4가 지웠다(사용자 결정 — plotly 의존성도 제거).

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
