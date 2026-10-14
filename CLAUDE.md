# Obscreen — Claude Code 개발 가이드

오더블록(Order Block) 기반 **국내주식 스크리닝 웹서비스**. 매일 장 마감 뒤 코스피·코스닥·ETF를
스크리닝해 오더블록 근처 종목을 SQLite에 저장하고, Flask 웹 화면으로 보여 준다.
서버는 `/home/rocky/orderblock`(git 체크아웃 · nginx 뒤 gunicorn `app_production:app` · 매일 20:05 KST
`daily_screening.sh`). OBS-6 이후 운영 표준은 **systemd**(웹 `orderblock.service` = `.venv` gunicorn ·
수집 `obscreen-collect.timer`)이고 배포는 `./scripts/deploy.sh` 한 줄이다 — 런북
[`docs/ops/obs-6-deploy.md`](docs/ops/obs-6-deploy.md). 서버 전환(§5)은 사용자가 하므로 그 전까지는
옛 구성(`venv310` gunicorn · crontab)이 돌고 있을 수 있다.

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
  detect/base.py       # 탐지기 프로토콜(스크리너·차트가 기대는 면)
  detect/alphablock/   # AlphaBlock 탐지기 **복사본**(OBS-5) — 손으로 고치지 말 것(아래)
  detect/adapter.py    # 국장 OHLCV ↔ AlphaBlock 어댑터 · `AlphaBlockOrderBlockDetector`
  detect/orderblock.py # `OrderBlockInfo` — 스크리너·차트가 쓰는 존 자료구조
  screening/core.py    # 스크리닝 실행·DB 저장(오더블록 + 외국인/기관) — **유일한** run_and_save_screening
  screening/screener.py# 종목 스크리너(병렬)
  web/app.py           # 조합 루트: create_app() · DB_PATH · 탐지기 클래스 · init_db()
  web/deps.py          # Blueprint가 조합 루트 전역을 **호출 시점에** 읽는 창구
  web/pages.py · screening_api.py · investor_api.py · chart_api.py   # Blueprint
templates/ · static/    # Flask 템플릿·정적 파일(위치·URL 불변 — /static/...)
daily_screening.sh      # 매일 수집 — 스크립트 폴더 기준으로 돈다(경로 하드코딩 없음)
scripts/                # 서버 운영(OBS-6): deploy.sh · install-systemd.sh · server-inventory.sh ·
                        #   health-check.sh · db-backup.sh · lib/common.sh · systemd/*.in(유닛 템플릿)
app_production.py 등    # 호환 별칭(아래)
tests/                  # pytest — 스모크 + 현행 동작 고정 특성 테스트(OBS-3, tests/README.md)
```

- **루트의 옛 이름은 별칭이다** — `app_production`·`screening_core`·`stock_screener`·
  `orderblock_info`·`data_provider`·`collect_data`를 import하면
  `sys.modules`를 바꿔 끼워 **패키지 모듈 객체 자체**를 받는다. 그래서 gunicorn
  `app_production:app`·`python3 collect_data.py`·특성 테스트의 monkeypatch가 그대로 동작한다.
  **새 코드는 `obscreen.*`를 직접 import한다**(패키지 안에서 옛 이름을 쓰면 안 된다).
- ⚠️ **모듈 전역을 import 시점에 복사하지 말 것** — 테스트는 `app_production.DB_PATH`·
  `app_production.AlphaBlockOrderBlockDetector`를 바꿔 끼운다. Blueprint는 `obscreen.web.deps`로
  호출 시점에 읽는다(`from obscreen.web.app import DB_PATH`로 복사하면 조용히 옛 값을 쓴다).
- 설정 기본값은 옮기기 전 코드의 숫자와 같다. 옛 환경변수 이름(`DB_PATH`·`SCREENING_TOP_N`·
  `SCREENING_ETF_N`·`FLASK_ENV`)은 그대로, 새 값은 `OBSCREEN_*`(`.env.example`).
- 의존성은 `pyproject.toml` + `uv.lock`이 정본이다. 배포(OBS-6)는 `uv sync --frozen --no-dev`로
  `.venv`(파이썬 3.11)를 만든다. `requirements.txt`는 **서버가 systemd 전환(런북 §5)을 끝내고 안정될
  때까지** 남긴다 — 전환 전 `venv310` 경로와 롤백(§7b)이 아직 쓴다.
  ⚠️ OBS-4가 `pydantic-settings`를 더했다(두 파일 모두) — 서버는 이 커밋을 받은 뒤
  `pip install -r requirements.txt`를 한 번 해야 cron·gunicorn이 뜬다.
- 설정은 환경변수(`.env.example` 참고). API 키·시크릿은 코드에 하드코딩하지 않는다.

- 📌 **운영 스크립트(OBS-6)** — 서버 작업은 전부 `scripts/`의 셸 스크립트 + 런북으로 넘긴다(로컬에서
  SSH 불가). 지키는 것:
  - **서버 경로를 쓰지 않는다** — 저장소 위치는 `scripts/lib/common.sh`가 자기 위치에서 구하고, systemd
    유닛은 `scripts/systemd/*.in` 템플릿의 `@REPO_DIR@` 등을 `install-systemd.sh`가 채운다
    (`test_no_hardcoded_server_path_in_code_or_scripts`가 `.in`·`.timer`도 본다).
  - **nginx·도메인·TLS는 읽기만 한다** — 새 웹 유닛은 기존 유닛의 gunicorn `--bind`를 그대로 옮긴다.
  - **수집 유닛은 `/bin/bash daily_screening.sh`로 부른다**(실행 비트에 기대지 않음 — OBS-13 사고).
    운영 스크립트도 실행 비트(100755)로 커밋한다(`tests/test_ops_scripts.py`가 인덱스 모드를 건다).
  - **비밀은 `.env`(chmod 600) 하나** — 유닛이 `EnvironmentFile`로 읽어 pykrx(`KRX_ID`·`KRX_PW`)까지
    전달된다. 점검·설치 출력은 `mask_secrets`로 값을 가린다.
  - **crontab과 timer가 동시에 돌지 않게** — `install-systemd.sh`는 crontab에 `daily_screening.sh`
    활성 줄이 있으면 타이머를 켜지 않고 멈춘다.
  - 셸 스크립트는 `bash -n` + **shellcheck(`-S warning`, dev 의존성 `shellcheck-py`)** 를 통과해야
    한다(pytest가 건다). `deploy.sh`는 가짜 `uv`·`sudo`·`systemctl`·`curl`로 호출 순서·실패 시 멈춤을
    테스트한다 — 새 단계를 넣으면 그 테스트도 고친다.

- 🚨 **`daily_screening.sh`는 실행 비트(100755)로 커밋돼 있어야 한다(OBS-13)** — crontab이 스크립트를
  직접 실행한다. 2026-09-30 서버를 git 체크아웃으로 바꿨을 때 644로 내려와 **2주간
  `Permission denied`로 스크리닝이 안 돌았다**. `tests/test_structure.py`가 git 인덱스 모드를 건다.
  수집이 실패하면 스크립트는 **종료 코드 1**로 끝나고 stderr에 한 줄 낸다(옛 코드는 무조건 `exit 0`).
  서버 반영·crontab 대안은 `docs/ops/obs-13-daily-screening-permission.md`.

- 🚨 **수급 수집 실패는 조용히 넘어가지 않는다(OBS-11)** — `run_and_save_investor_data`는 실패하면
  (pykrx 없음·KRX 로그인 실패·빈 응답·예외·저장 0건) `InvestorDataError`를 올리고, `obscreen collect`
  (`--all`·`--investor`)는 **종료 코드 1**로 끝나며 「완료」를 찍지 않는다. 그날의 기존 행은 **새
  데이터를 확보한 뒤** 한 트랜잭션에서 지우고 쓴다(가져오기 전에 지우지 말 것). KRX 로그인 문제는
  stderr에 `❌ [KRX 로그인 실패] …` **한 줄**로 남는다(ID·비밀번호 값은 찍지 않는다 — 설정 여부만).
  `/api/investor-trading`·`/api/investor-dates` 응답의 `freshness`(최근 수집일·직전 영업일·빠진
  영업일 수·`stale`·`warning`)가 화면 경고의 근거다 — 판단은 서버 한 곳에서만 한다. 옛 코드는 실패를
  삼켜 2026-08~09 두 달간 수집이 죽어 있는 것을 아무도 몰랐다.

- 📌 **탐지기 = AlphaBlock 복사본(OBS-5 · 사용자 결정 「복사 이식」)** — `obscreen/detect/alphablock/`은
  AlphaBlock `strategy/models.py`·`strategy/order_blocks.py`를 특정 커밋(`UPSTREAM_COMMIT`)에서
  **그대로** 복사한 것이다(다른 것은 `order_blocks.py`의 import 한 줄). 국장 데이터 접점(날짜 인덱스
  ↔ `open_time` ms, `OrderBlockInfo` 변환)은 `obscreen/detect/adapter.py`에만 둔다.
  - **복사본을 손으로 고치지 않는다** — `tests/test_alphablock_parity.py`가 해시(항상)와 AlphaBlock
    그 커밋 파일(로컬, `ALPHABLOCK_REPO` 기본 `~/AlphaBlock` — **읽기만** 한다, `git show`)로 대조하고,
    원본 코드로 만든 골든(`tests/fixtures/alphablock_golden/`)과 존 대장이 완전히 같은지 CI에서도 건다.
    AlphaBlock 수정을 받아들이는 갱신 절차는 `obscreen/detect/alphablock/__init__.py` 설명.
  - **탐지 값의 기본값은 AlphaBlock `OrderBlockParams()`** — 스윙 10 · ATR(Wilder 10) 배수 3.5 ·
    꼬리 무효화 · 병합 안 함 · 방향별 최신 3개(`zone_count="low"`). 설정·스크리너 기본값이
    `adapter.DEFAULT_PARAMS`에서 나온다(숫자를 두 곳에 적지 않는다). 국장 조정은 별도 이슈.
  - **스크리너·차트가 받는 존 = AlphaBlock 렌더 뷰**(마지막 봉 시점, 아직 소멸 안 한 존 중 방향별
    최신 3개). 존 개수 규칙은 탐지기 한 곳에만 있다 — 스크리너·차트에서 `[:3]`로 다시 자르지 말 것
    (OBS-5 전에는 일봉만 잘라 주봉과 규칙이 달랐다).
  - 옛 탐지기 대비 변화는 `docs/obs-5-detector-change.md`(옛 스냅샷 ↔ 새 스냅샷에서 만든다 —
    `tests/test_obs5_change_table.py`가 둘이 맞는지 건다).

- **차트는 Lightweight Charts 한 가지다(OBS-10)** — AlphaBlock과 같은 v5.2.0을 `static/vendor/`에
  벤더링하고 `static/js/ob_chart.js`가 캔들 + 존 박스(캔버스 프리미티브 **하나**) + RSI 보조창을
  그린다. 존마다 시리즈를 만들지 말 것(AlphaBlock에서 2,000개에 브라우저가 멈췄다). 순수 함수는
  `tests/test_chart_frontend.py`가 node로 검사한다. plotly `create_chart_html*`·실험 라우트는
  OBS-4가 지웠다(사용자 결정 — plotly 의존성도 제거).

## 리모델링 방향 (사용자 결정 2026-09-30)

순서: 1 토대(OBS-2) → 2 현행 동작 고정 회귀 테스트(OBS-3) → 3 패키지 구조 정리 →
4 탐지기 교체 → 5 서버 배포 변경. **아래층이 굳은 뒤 위층.**

- **탐지기 = AlphaBlock 재사용.** 오더블록 탐지 로직은 `~/AlphaBlock`(암호화폐 자동매매 저장소)의
  탐지기를 가져다 쓴다. OBS-5가 **복사 이식**으로 옮겼다(위 📌 문단).
  ⚠️ `~/AlphaBlock`은 **읽기만** 한다 — 쓰기·브랜치 변경·프로세스 종료 금지(다른 자동 루틴이 돈다).
- **웹 = 기존 도메인에서 Flask 유지**, 관리하기 쉽게 정리한다(프레임워크 교체 아님).
- 탐지기를 바꾸기 전에 **현행 동작을 테스트로 먼저 고정**한다(리모델링 2).

## 안전 규칙

- 서버(`/home/rocky/orderblock`, 운영 도메인)에 직접 접속·배포하지 않는다(로컬에서 SSH 불가).
  서버 작업은 서버에서 돌릴 스크립트와 런북으로만 넘긴다. 도메인·DNS·TLS는 건드리지 않는다.
- `.env`·DB 파일·로그는 커밋하지 않는다(`.gitignore`).
- 테스트 실패/완료 기준 미충족 상태로 PR을 열지 않는다. PR을 자동 머지하지 않는다.
