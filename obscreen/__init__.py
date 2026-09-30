"""Obscreen — 오더블록 기반 국내주식 스크리닝 웹서비스(OBS-4 패키지 구조).

- `obscreen.data` — 시세·수급 제공자(pykrx·FinanceDataReader)와 SQLite 스키마
- `obscreen.detect` — 오더블록 탐지·지표(리모델링 4에서 교체될 자리)
- `obscreen.screening` — 스크리닝 실행·근접도·추천·수급 저장
- `obscreen.web` — Flask 앱 팩토리와 Blueprint
- `obscreen.config` — 설정(pydantic-settings, `.env`)
- `obscreen.cli` — `obscreen collect` · `obscreen serve`
"""
