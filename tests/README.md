# 테스트 — 현행 동작 고정(특성 테스트, OBS-3)

리모델링 3(구조 정리)·4(탐지기 교체) **전에** 지금 코드가 무엇을 내는지 못 박아 둔
테스트다. 목적은 둘이다.

1. 구조 정리가 결과를 **조용히** 바꾸지 않았는지 기계가 확인한다.
2. 탐지기 교체가 결과를 **얼마나** 바꿨는지 스냅샷 diff로 숫자로 본다.

⚠️ 스냅샷은 **옳은 값이 아니라 지금 값**이다. 버그로 보이는 동작도 그대로 고정했고
(아래 「고정한 버그 후보」), 고칠 때는 그 테스트를 **의도적으로** 뒤집는다.

## 실행

```bash
uv run pytest            # 전부(네트워크 호출 0 — conftest가 TCP·DNS를 막는다)
```

## 구성

| 파일 | 고정하는 것 |
| -- | -- |
| `test_characterization_detector.py` | 탐지기(`realtime_detector`) 원시·최종 존(위·아래·시작/무효화 시각·상태·거래량) × 일봉/주봉 × Wick/Close × 병합 켬/끔, `indicators` 값, 일봉→주봉 변환, **합성 입력 경계**(무효화 부등호·ATR 필터) |
| `test_characterization_screening.py` | 근접도 분류(`classify_position`) 경계, 종목별 `check_proximity(_weekly)`, 종목 선정(시총 상위·ETF 제외 규칙), 시장 스크리닝, `run_and_save_screening` DB 행 · 추천 플래그(bit1·bit2) · 주말/공휴일 |
| `test_characterization_investor.py` | 수급(외국인·기관) 저장 → 임시 SQLite → 표·API 왕복 |
| `test_characterization_api.py` | `/api/screening/*`·`/api/chart-data(-weekly)/*`·`/api/stock/*`·`/health` 응답 모양(키·타입) + 차트 오더블록 값 |
| `test_characterization_wiring.py` | 운영 경로가 **넘기는 설정값** — `run_and_save_screening`의 `StockScreener` 3개(근접도 3/1/5%·ATR 배수 2.0 등)와 차트 API의 `RealtimeOrderBlockDetector` 인자를 생성자 스파이로 기록(기본값 포함). 필터를 **푸는** 변화는 출력 스냅샷으로 안 잡히기 때문(OBS-4가 이 숫자들을 설정으로 옮긴다) + 근접도를 값으로 거는 합성 존(현재가 3.2% 아래) |
| `support.py` | 가짜 FinanceDataReader·pykrx, 스냅샷 비교기 |

## 고정 입력(`tests/fixtures/`)

- `ohlcv/<종목코드>.csv` — FinanceDataReader에서 **한 번** 받은 실제 일봉(2015-01-02 ~
  2026-09-18, 2026-09-30 수집). 코스피 대형(005930·000660·035420·051910)·소형(003490·012800),
  코스닥 대형(247540·086520·196170·041510)·소형(228670), ETF(069500·229200·360750).
  주봉은 저장하지 않고 운영 코드(`convert_to_weekly`)로 만든다 — 그 변환 결과 자체를
  `weekly_bars_*.json` 스냅샷으로 고정한다.
- `listing_krx.csv`·`listing_etf.csv` — 종목 목록. **시가총액은 손으로 넣은 값**이다
  (2026-09-30 실측 `fdr.StockListing("KRX")`의 `Marcap`이 전부 비어 있었다 — 아래 참고).
  시세 없는 종목(999990)·레버리지/인버스 ETF·거래 0 ETF는 경로를 태우려고 넣었다.
- `investor/*.csv` — pykrx 모양을 손으로 만든 수급 표(작고 결정적). 빈 종목명·시세 없는
  종목을 일부러 넣었다.
- 스크리닝 기준일은 **2026-09-14**(월).

## 스냅샷 갱신 — 의도한 변경일 때만

```bash
UPDATE_SNAPSHOTS=1 uv run pytest     # tests/snapshots/*.json 을 다시 쓴다
git diff --stat tests/snapshots/     # 무엇이 바뀌었나
```

- 기본 실행은 **비교만** 한다. 스냅샷이 없거나 다르면 실패하고, 어긋난 경로와 값을 찍는다.
- 갱신은 **동작을 일부러 바꾼 PR**에서만 하고, PR 본문에 「어느 스냅샷이 왜 바뀌었나」를
  적는다. 구조 정리(리모델링 3)처럼 동작이 안 바뀌어야 하는 PR에서 스냅샷이 바뀌면 그건
  갱신할 일이 아니라 **버그**다.
- 부동소수는 상대 1e-9 안에서 같으면 같다고 본다(플랫폼 끝자리 잡음만 흡수).

## 고정한 버그 후보(기록만 — 수정은 별도 이슈)

1. **평일 공휴일에도 스크리닝이 돈다** — `screening_core`의 `if KR_HOLIDAYS and today in
   KR_HOLIDAYS`: `holidays.SouthKorea()`는 연도를 지연 생성해 막 만든 객체의 `bool()`이
   거짓이라 `in` 검사까지 가지 않는다(2026-09-25 추석에 실제로 돈다).
2. **수집 전 `/api/investor-trading`이 500** — `investor_trading` 테이블을 `init_db()`가 아니라
   첫 수집이 만든다. 빈 DB에서는 `no such table`로 500(404가 아님).
3. **종목명이 빈 수급 행이 조용히 빠지고 순위에 구멍이 난다** — 빈 칸이 NaN으로 읽히고 NaN이
   참이라 이름 조회 폴백을 건너뛴 뒤 NOT NULL에 걸린다. 순위는 그 행 자리까지 센다.
4. **실측 종목 목록의 시가총액이 비어 있다** — 2026-09-30 `fdr.StockListing("KRX")`의
   `Marcap`이 2,873행 전부 NaN이었다. 그러면 「시총 상위 400」이 정렬이 안 된 목록 앞 400개가
   된다. 운영 서버에서도 그런지는 확인 필요(테스트는 손으로 넣은 시총을 쓴다).
5. `app_production.py`가 `screening_core.run_and_save_screening`을 import한 뒤 **같은 이름의
   자기 함수로 덮어쓴다**(두 벌). cron(`collect_data.py`)은 `screening_core` 판을 쓴다 —
   이 테스트도 그쪽을 고정한다.
6. **일봉과 주봉의 존 개수 규칙이 다르다** — 일봉 `check_proximity`는 최신 3개 존만 보는데
   (`[:3]`, 「Zone Count: Low」) 주봉 `check_proximity_weekly`에는 그 절단이 없다.
