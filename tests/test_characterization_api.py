"""Flask 주요 API 응답 모양 고정(OBS-3 §5).

`/api/screening/*`·`/api/chart-data/*`·`/health`를 테스트 클라이언트로 부른다. 응답의
**모양**(키·타입)을 스냅샷으로 고정하고, 차트 API의 오더블록 값은 값까지 고정한다(탐지기
교체 이슈가 화면에 무엇이 바뀌는지 숫자로 보게 하려고).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import app_production
import screening_core
from tests.support import SCREEN_DATE, SCREEN_DATE_COMPACT, FakeFdr, assert_matches_snapshot


def _shape(value: Any) -> Any:
    """값을 지우고 타입만 남긴다. 리스트는 첫 원소의 모양 + 길이 대신 '비었나'만 본다."""
    if isinstance(value, dict):
        return {k: _shape(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_shape(value[0])] if value else []
    if value is None:
        return "null"
    return type(value).__name__


def _get(client: Any, url: str) -> tuple[int, Any]:
    res = client.get(url)
    # 레거시 응답은 NaN을 그대로 싣는다(엄격한 JSON이 아니다) — 파이썬 json은 읽는다.
    return res.status_code, json.loads(res.get_data(as_text=True))


@pytest.fixture
def screened_client(fake_fdr: FakeFdr, fresh_db: Path) -> Any:
    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    return app_production.app.test_client()


def test_health() -> None:
    status, body = _get(app_production.app.test_client(), "/health")
    assert (status, body) == (200, {"status": "ok"})


def test_screening_endpoints_shape(screened_client: Any) -> None:
    responses: dict[str, Any] = {}
    for name, url in {
        "dates": "/api/screening/dates",
        "by_date_daily": f"/api/screening/{SCREEN_DATE}",
        "by_date_compact_weekly": f"/api/screening/{SCREEN_DATE_COMPACT}?timeframe=weekly",
        "by_date_missing": "/api/screening/2026-01-05",
        "recommended": f"/api/screening/recommended?date={SCREEN_DATE}",
        "stock": f"/api/stock/228670?date={SCREEN_DATE}",
        "stock_missing": f"/api/stock/005930?date={SCREEN_DATE}",
    }.items():
        status, body = _get(screened_client, url)
        responses[name] = {"status": status, "shape": _shape(body)}
    assert_matches_snapshot("api_screening_shapes", responses)

    # 값 수준에서 걸어 둘 최소한: 날짜 형식 변환 · 저장 행 수 일치.
    _, dates = _get(screened_client, "/api/screening/dates")
    assert dates["dates"][0]["date"] == SCREEN_DATE
    _, daily = _get(screened_client, f"/api/screening/{SCREEN_DATE}")
    _, weekly = _get(screened_client, f"/api/screening/{SCREEN_DATE}?timeframe=weekly")
    assert dates["dates"][0]["count"] == daily["stats"]["total"] + weekly["stats"]["total"]


def test_screening_today_is_a_no_result_shape(screened_client: Any) -> None:
    """`today`는 벽시계 날짜를 쓴다 — 고정 입력 날짜가 아니므로 '결과 없음' 모양이 나온다."""
    status, body = _get(screened_client, "/api/screening/today")
    assert status == 200
    assert body["success"] is False
    assert set(body) == {"success", "message"}


@pytest.mark.parametrize(
    ("route", "ticker"),
    [
        ("chart-data", "228670"),
        ("chart-data", "005930"),
        ("chart-data-weekly", "035420"),
        ("chart-data-weekly", "360750"),
    ],
)
def test_chart_data(route: str, ticker: str, fake_fdr: FakeFdr) -> None:
    client = app_production.app.test_client()
    status, body = _get(client, f"/api/{route}/{ticker}?date={SCREEN_DATE_COMPACT}")
    assert status == 200
    assert body["success"] is True
    data = body["data"]
    summary = {
        "shape": _shape(body),
        "candles": {
            "count": len(data["candles"]),
            "first": data["candles"][0],
            "last": data["candles"][-1],
        },
        "rsi": {"count": len(data["rsi"]), "last": data["rsi"][-1]},
        "rsi_ema": {"count": len(data["rsi_ema"]), "last": data["rsi_ema"][-1]},
        "orderblocks": data["orderblocks"],
    }
    assert_matches_snapshot(f"api_{route}_{ticker}", summary)


def test_chart_data_no_data(fake_fdr: FakeFdr) -> None:
    client = app_production.app.test_client()
    for route in ("chart-data", "chart-data-weekly"):
        status, body = _get(client, f"/api/{route}/999990?date={SCREEN_DATE}")
        assert (status, body) == (200, {"success": False, "message": "No data"})
