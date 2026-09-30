# 벤더링한 서드파티 파일

CDN에 기대지 않고 파일째 저장소에 둔다(오프라인·CDN 장애에도 차트가 뜨고, 버전이
조용히 바뀌지 않는다).

| 파일 | 출처 | 버전 | 라이선스 |
| -- | -- | -- | -- |
| `lightweight-charts.standalone.production.js` | TradingView Lightweight Charts™ (<https://github.com/tradingview/lightweight-charts>) — `~/AlphaBlock/dashboard/static/`에 벤더링된 같은 파일을 그대로 복사(OBS-10) | v5.2.0 | Apache License 2.0 (<https://www.apache.org/licenses/LICENSE-2.0>) · Copyright (c) 2026 TradingView, Inc. |

- 파일 머리의 `@license` 주석(버전·저작권·라이선스)은 지우지 않는다.
- sha256: `c0992580867c4912cc9385b3c2728315bcc1a76c7f1087dca908430fccdf31d7`
  (`tests/test_chart_frontend.py`가 버전 문자열을 확인한다).
- 버전을 올릴 때는 이 표와 테스트를 함께 고친다. v5의 멀티패인(`addSeries(…, paneIndex)`)과
  `ISeriesPrimitive`(`attachPrimitive`)에 기대므로 v4 이하로 내리면 안 된다.
