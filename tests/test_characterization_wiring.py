"""운영 경로가 탐지기·스크리너에 **넘기는 설정값** 고정(OBS-3 PM 변경요청).

출력 스냅샷만으로는 필터를 **푸는** 변화(`max_atr_mult 2.0 → 2.5`, `proximity_percent
3.0 → 3.5`)가 고정 입력 14종목에서 결과를 안 바꿔 통과한다. 그런데 리모델링 3(OBS-4)이
바로 이 숫자들을 설정 파일로 옮긴다 — 옮기다 값이 바뀌면 이 테스트가 잡아야 한다.

생성자를 감싼 스파이가 **실제로 넘어간 인자**(기본값을 채운 최종값)를 기록한다. 레거시
코드는 건드리지 않고, 감싼 클래스는 진짜 클래스를 상속하므로 동작도 그대로다.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import app_production
import screening_core
import stock_screener
from realtime_detector import RealtimeOrderBlockDetector
from stock_screener import StockScreener
from tests.conftest import query_rows
from tests.support import SCREEN_DATE_COMPACT, FakeFdr, assert_matches_snapshot


def _spy(real: type, calls: list[dict[str, Any]]) -> type:
    """`real`을 상속하고 생성자 인자(기본값 포함)를 `calls`에 적는 클래스."""
    real_init: Callable[..., None] = real.__init__  # type: ignore[misc]
    signature = inspect.signature(real_init)

    def __init__(self: Any, *args: Any, **kwargs: Any) -> None:
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        calls.append({k: v for k, v in bound.arguments.items() if k != "self"})
        real_init(self, *args, **kwargs)

    return type(f"Spy{real.__name__}", (real,), {"__init__": __init__})


def _distinct(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted({json.dumps(c, sort_keys=True) for c in calls})
    return [json.loads(k) for k in keys]


def test_run_and_save_screening_wiring(
    fake_fdr: FakeFdr, fresh_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screeners: list[dict[str, Any]] = []
    detectors: list[dict[str, Any]] = []
    monkeypatch.setattr(screening_core, "StockScreener", _spy(StockScreener, screeners))
    monkeypatch.setattr(
        stock_screener, "RealtimeOrderBlockDetector", _spy(RealtimeOrderBlockDetector, detectors)
    )

    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)

    # 만든 순서: 일봉 주식 → 일봉 ETF → 주봉. 값은 스냅샷과 별개로 명시적으로도 건다.
    assert [s["proximity_percent"] for s in screeners] == [3.0, 1.0, 5.0]
    assert {s["max_atr_mult"] for s in screeners} == {2.0}
    assert {s["swing_length"] for s in screeners} == {10}
    assert {s["ob_end_method"] for s in screeners} == {"Wick"}
    assert {s["combine_obs"] for s in screeners} == {True}
    assert detectors, "스크리너가 탐지기를 하나도 안 만들었다 — 스파이가 안 걸렸다"
    assert _distinct(detectors) == [
        {
            "swing_length": 10,
            "max_atr_mult": 2.0,
            "max_order_blocks": 30,
            "ob_end_method": "Wick",
            "combine_obs": True,
        }
    ]
    assert_matches_snapshot(
        "wiring_run_and_save_screening",
        {"screeners": screeners, "detectors_distinct": _distinct(detectors)},
    )


@pytest.mark.parametrize(
    ("route", "ticker"), [("chart-data", "228670"), ("chart-data-weekly", "035420")]
)
def test_chart_api_wiring(
    route: str, ticker: str, fake_fdr: FakeFdr, monkeypatch: pytest.MonkeyPatch
) -> None:
    detectors: list[dict[str, Any]] = []
    monkeypatch.setattr(
        app_production,
        "RealtimeOrderBlockDetector",
        _spy(RealtimeOrderBlockDetector, detectors),
    )
    res = app_production.app.test_client().get(f"/api/{route}/{ticker}?date={SCREEN_DATE_COMPACT}")
    assert res.status_code == 200
    assert detectors == [
        {
            "swing_length": 10,
            "max_atr_mult": 2.0,
            "max_order_blocks": 30,
            "ob_end_method": "Wick",
            "combine_obs": True,
        }
    ]
    assert_matches_snapshot(f"wiring_api_{route}", detectors)


def test_spy_records_defaults_and_overrides() -> None:
    """스파이 자체 검산 — 기본값이 채워지고, 넘긴 값이 기록되며, 동작은 진짜와 같다."""
    calls: list[dict[str, Any]] = []
    spy_cls = _spy(RealtimeOrderBlockDetector, calls)
    det = spy_cls(max_atr_mult=2.5)
    assert isinstance(det, RealtimeOrderBlockDetector)
    assert calls == [
        {
            "swing_length": 10,
            "max_atr_mult": 2.5,
            "max_order_blocks": 30,
            "ob_end_method": "Wick",
            "combine_obs": False,
        }
    ]


def test_operational_proximity_by_value(
    fake_fdr: FakeFdr, fresh_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(선택 항목) 근접도를 **값으로** 건다 — 모든 종목에 「현재가 3.2% 아래」 지지 존 하나.

    일봉 주식(3%)·ETF(1%)는 못 잡고 주봉(5%)만 잡아야 한다. 3.0 → 3.5로 풀리면 일봉 주식이
    잡히고, 5.0 → 3.0으로 조이면 주봉이 빠진다.
    """
    from orderblock_info import OrderBlockInfo

    def fake_latest(self: Any) -> tuple[list[Any], list[Any]]:
        price = float(self._last_close)
        top = price * (1 - 0.032)
        return [OrderBlockInfo(top, top * 0.98, 0, "Bull", None)], []

    real_detect: Callable[..., None] = RealtimeOrderBlockDetector.detect_order_blocks_realtime

    def remember_close(self: Any, df: Any) -> None:
        self._last_close = df["Close"].iloc[-1]
        real_detect(self, df)

    monkeypatch.setattr(RealtimeOrderBlockDetector, "detect_order_blocks_realtime", remember_close)
    monkeypatch.setattr(RealtimeOrderBlockDetector, "get_latest_orderblocks", fake_latest)

    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    rows = query_rows(fresh_db, "SELECT market, timeframe, distance_percent FROM screening_results")
    daily = [r for r in rows if r["timeframe"] == "daily"]
    weekly = [r for r in rows if r["timeframe"] == "weekly"]
    assert daily == []
    # 시세가 있는 종목(주식 11 · ETF 3) 전부 주봉에서 3.2%로 잡힌다.
    assert len(weekly) == 14
    assert {round(r["distance_percent"], 6) for r in weekly} == {3.2}
