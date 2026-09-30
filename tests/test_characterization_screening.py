"""현행 스크리닝 결과 고정(OBS-3 §3).

`stock_screener.StockScreener`의 종목 선정·근접도 계산과 `screening_core.
run_and_save_screening`의 DB 저장(추천 플래그 포함)을 고정 입력으로 못 박는다.
병렬 처리(ThreadPoolExecutor)라 결과 순서는 보장되지 않으므로 정렬해서 비교한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import screening_core
from data_provider import KoreanStockDataProvider
from orderblock_info import OrderBlockInfo
from stock_screener import StockScreener
from tests.conftest import query_rows
from tests.support import (
    FIXTURE_TICKERS,
    SCREEN_DATE,
    SCREEN_DATE_COMPACT,
    FakeFdr,
    assert_matches_snapshot,
)

# ---------------------------------------------------------------- classify_position


def _ob(top: float, bottom: float, ob_type: str, breaker: bool = False) -> OrderBlockInfo:
    ob = OrderBlockInfo(top, bottom, 0, ob_type, pd.Timestamp("2026-01-02"))
    ob.breaker = breaker
    return ob


@pytest.mark.parametrize(
    ("price", "bull", "bear", "expected"),
    [
        # 지지 존 내부(경계 포함)
        (100.0, [(105, 95)], [], ("내부-지지", "Bull", 0)),
        (105.0, [(105, 95)], [], ("내부-지지", "Bull", 0)),
        # 지지 존 위 1% 이내 — 거리는 **현재가** 대비 %
        (100.0, [(99.5, 90)], [], ("근접-지지", "Bull", 0.5)),
        # 정확히 기준선 위(<=)
        (100.0, [(99.0, 90)], [], ("근접-지지", "Bull", 1.0)),
        # 기준선 밖
        (100.0, [(98.9, 90)], [], (None, None, None)),
        # 저항 존 내부 · 근접
        (100.0, [], [(105, 95)], ("내부-저항", "Bear", 0)),
        (100.0, [], [(120, 100.5)], ("근접-저항", "Bear", 0.5)),
        # 무효화된 존은 무시
        (100.0, [(105, 95, True)], [], (None, None, None)),
        # 지지를 먼저 본다 — 둘 다 맞으면 지지가 이긴다
        (100.0, [(99.5, 90)], [(105, 95)], ("근접-지지", "Bull", 0.5)),
        # 같은 종류 안에서는 목록 순서가 이긴다(가까운 것이 아니라 먼저 온 것)
        (100.0, [(99.2, 90), (99.9, 95)], [], ("근접-지지", "Bull", 0.8)),
    ],
)
def test_classify_position(
    price: float,
    bull: list[tuple[Any, ...]],
    bear: list[tuple[Any, ...]],
    expected: tuple[Any, Any, Any],
) -> None:
    screener = StockScreener(proximity_percent=1.0)
    result = screener.classify_position(
        price,
        [_ob(b[0], b[1], "Bull", *b[2:]) for b in bull],
        [_ob(b[0], b[1], "Bear", *b[2:]) for b in bear],
    )
    status, ob_type, distance = expected
    assert result["status"] == status
    assert result["ob_type"] == ob_type
    if distance is None:
        assert result["distance_percent"] is None
    else:
        assert result["distance_percent"] == pytest.approx(distance)


# ---------------------------------------------------------------- 종목별 근접도


def test_check_proximity_snapshot(fake_fdr: FakeFdr) -> None:
    """실제 운영 기준(일봉 주식 3% · ETF 1% · 주봉 5%)과 넓은 기준(50%)의 결과.

    넓은 기준은 None이 아닌 결과를 늘려 RSI·거래대금·거리 계산까지 숫자로 고정한다.
    """
    data: dict[str, Any] = {}
    for ticker in FIXTURE_TICKERS:
        data[ticker] = {
            f"daily_{p}": StockScreener(proximity_percent=p).check_proximity(
                ticker, 500, SCREEN_DATE
            )
            for p in (1.0, 3.0, 50.0)
        } | {
            f"weekly_{p}": StockScreener(proximity_percent=p).check_proximity_weekly(
                ticker, 500, SCREEN_DATE
            )
            for p in (5.0, 50.0)
        }
    assert_matches_snapshot("check_proximity", data)


def test_check_proximity_missing_data_returns_none(fake_fdr: FakeFdr) -> None:
    screener = StockScreener(proximity_percent=3.0)
    assert screener.check_proximity("999990", 500, SCREEN_DATE) is None
    assert screener.check_proximity_weekly("999990", 500, SCREEN_DATE) is None


# ---------------------------------------------------------------- 종목 선정


def test_universe_selection(fake_fdr: FakeFdr) -> None:
    """시가총액 상위 N · 거래량 상위 ETF(레버리지·인버스·거래 0 제외)."""
    kospi = KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSPI", 3)
    assert kospi["Code"].tolist() == ["005930", "000660", "035420"]
    kosdaq = KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSDAQ", 400)
    assert kosdaq["Code"].tolist() == ["247540", "086520", "196170", "041510", "228670", "999990"]
    etf = KoreanStockDataProvider.get_top_etfs_by_volume(300, exclude_leverage=True)
    assert etf["Symbol"].tolist() == ["360750", "069500", "229200"]
    etf_all = KoreanStockDataProvider.get_top_etfs_by_volume(300, exclude_leverage=False)
    assert etf_all["Symbol"].tolist() == [
        "252670",
        "122630",
        "114800",
        "360750",
        "069500",
        "229200",
    ]


def _sorted_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: (r["Market"], r["Code"]))


def test_screen_markets_snapshot(fake_fdr: FakeFdr) -> None:
    stocks = StockScreener(proximity_percent=3.0).screen_multiple_markets(
        markets=["KOSPI", "KOSDAQ"], top_n=400, days=500, end_date=SCREEN_DATE
    )
    etf = StockScreener(proximity_percent=1.0).screen_etf(top_n=300, days=500, end_date=SCREEN_DATE)
    weekly = [
        r
        for m in ("KOSPI", "KOSDAQ", "ETF")
        for r in StockScreener(proximity_percent=5.0).screen_market_weekly(
            market=m, top_n=400, weeks=500, end_date=SCREEN_DATE
        )
    ]
    data = {
        "daily_stocks": _sorted_records(stocks.to_dict("records")),
        "daily_etf": _sorted_records(etf),
        "weekly": _sorted_records(weekly),
    }
    assert data["daily_stocks"] or data["weekly"], "고정 입력에서 아무 종목도 안 걸리면 무의미"
    assert_matches_snapshot("screen_markets", data)


# ---------------------------------------------------------------- DB 저장


_SCREEN_COLUMNS = (
    "scan_date, market, code, name, current_price, change_percent, rsi, trading_value, "
    "zone_type, zone_position, ob_top, ob_bottom, distance_percent, is_recommended, timeframe"
)


def test_run_and_save_screening_snapshot(fake_fdr: FakeFdr, fresh_db: Path) -> None:
    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    rows = query_rows(
        fresh_db,
        f"SELECT {_SCREEN_COLUMNS} FROM screening_results ORDER BY timeframe, market, code",
    )
    assert rows
    assert_matches_snapshot("run_and_save_screening", rows)

    # 같은 날짜로 다시 돌리면 덮어쓴다(중복 없음).
    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    again = query_rows(
        fresh_db,
        f"SELECT {_SCREEN_COLUMNS} FROM screening_results ORDER BY timeframe, market, code",
    )
    assert again == rows


def test_run_and_save_screening_skips_weekend(fake_fdr: FakeFdr, fresh_db: Path) -> None:
    """주말(2026-09-12 토)은 시세를 조회하지도 저장하지도 않는다."""
    screening_core.run_and_save_screening("20260912")
    assert fake_fdr.calls == []
    assert query_rows(fresh_db, "SELECT * FROM screening_results") == []


def test_run_and_save_screening_does_not_skip_weekday_holiday(
    fake_fdr: FakeFdr, fresh_db: Path
) -> None:
    """⚠️ 현행 동작(버그 후보): 평일 공휴일(2026-09-25 추석)을 **건너뛰지 않는다.**

    `if KR_HOLIDAYS and today in KR_HOLIDAYS` — `holidays.SouthKorea()`는 연도를 지연
    생성해 막 만든 객체의 `bool()`이 거짓이라 `in` 검사까지 가지 않는다. 수정은 이 이슈
    범위 밖이라(스냅샷은 지금 동작 그대로) 고정만 해 둔다. 고치면 이 테스트를 뒤집는다.
    """
    screening_core.run_and_save_screening("20260925")
    assert fake_fdr.calls != []
    assert query_rows(fresh_db, "SELECT COUNT(*) AS n FROM screening_results")[0]["n"] > 0


def _record(
    code: str, ob_type: str, status: str, rsi: float, top: float, bottom: float
) -> dict[str, Any]:
    return {
        "Market": "KOSPI",
        "Code": code,
        "Name": f"종목{code}",
        "Current_Price": 100.0,
        "Change_Percent": 1.0,
        "RSI": rsi,
        "trading_value": 1e9,
        "Status": status,
        "OB_Type": ob_type,
        "OB_Top": top,
        "OB_Bottom": bottom,
        "Distance_Percent": 0.0,
    }


def test_recommendation_flags(monkeypatch: pytest.MonkeyPatch, fresh_db: Path) -> None:
    """추천 규칙: bit1 = RSI<30 · 지지 · 존 폭<10% / bit2 = 일봉·주봉 모두 지지(일봉 행에만)."""
    daily = [
        _record("A00001", "Bull", "내부-지지", 25.0, 105.0, 100.0),  # bit1
        _record("A00002", "Bull", "근접-지지", 25.0, 115.0, 100.0),  # 폭 15% → 아님
        _record("A00003", "Bull", "근접-지지", 35.0, 105.0, 100.0),  # RSI 35 → bit2만
        _record("A00004", "Bear", "내부-저항", 20.0, 105.0, 100.0),  # 저항 → 아님
        _record("A00005", "Bull", "내부-지지", 29.9, 109.9, 100.0),  # bit1 + bit2
        _record("A00006", "Bull", "내부-지지", 30.0, 105.0, 100.0),  # RSI == 30 → 아님(<)
        _record("A00007", "Bull", "내부-지지", 25.0, 110.0, 100.0),  # 폭 == 10% → 아님(<)
    ]
    weekly = [
        _record("A00003", "Bull", "근접-지지", 50.0, 105.0, 100.0),
        _record("A00004", "Bull", "근접-지지", 50.0, 105.0, 100.0),  # 일봉이 저항이라 bit2 아님
        _record("A00005", "Bull", "내부-지지", 50.0, 105.0, 100.0),
    ]

    def fake_multi(self: Any, *a: Any, **k: Any) -> pd.DataFrame:
        return pd.DataFrame(daily)

    def fake_etf(self: Any, *a: Any, **k: Any) -> list[dict[str, Any]]:
        return []

    def fake_weekly(self: Any, market: str = "KOSPI", **k: Any) -> list[dict[str, Any]]:
        return [dict(r) for r in weekly] if market == "KOSPI" else []

    monkeypatch.setattr(StockScreener, "screen_multiple_markets", fake_multi)
    monkeypatch.setattr(StockScreener, "screen_etf", fake_etf)
    monkeypatch.setattr(StockScreener, "screen_market_weekly", fake_weekly)

    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    rows = query_rows(
        fresh_db,
        "SELECT code, timeframe, zone_type, zone_position, is_recommended "
        "FROM screening_results ORDER BY timeframe, code",
    )
    got = {(r["code"], r["timeframe"]): r["is_recommended"] for r in rows}
    assert got == {
        ("A00001", "daily"): 1,
        ("A00002", "daily"): 0,
        ("A00003", "daily"): 2,
        ("A00004", "daily"): 0,
        ("A00005", "daily"): 3,
        ("A00006", "daily"): 0,
        ("A00007", "daily"): 0,
        ("A00003", "weekly"): 0,
        ("A00004", "weekly"): 0,
        ("A00005", "weekly"): 0,
    }
    positions = {(r["code"], r["timeframe"]): (r["zone_type"], r["zone_position"]) for r in rows}
    assert positions[("A00004", "daily")] == ("저항", "내부")
    assert positions[("A00002", "daily")] == ("지지", "근접")


@pytest.mark.parametrize("weekly", [False, True])
def test_zone_count_low_keeps_latest_three(
    weekly: bool, fake_fdr: FakeFdr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """「Zone Count: Low」 — 일봉은 최신 3개 존만 본다(4번째가 맞아도 무시). 주봉은 자르지 않는다.

    ⚠️ 일봉·주봉이 다르다는 것도 현행 동작 그대로 고정한다(주봉 경로엔 `[:3]`이 없다).
    """
    import realtime_detector

    far = [_ob(10.0, 9.0, "Bull") for _ in range(3)]
    df = KoreanStockDataProvider.get_price_data("228670", 500, SCREEN_DATE)
    price = float(df["Close"].iloc[-1])
    hit = _ob(price * 1.01, price * 0.99, "Bull")  # 현재가를 품은 4번째 존

    monkeypatch.setattr(
        realtime_detector.RealtimeOrderBlockDetector,  # stock_screener가 쓰는 바로 그 클래스
        "get_latest_orderblocks",
        lambda self: ([*far, hit], []),
    )
    screener = StockScreener(proximity_percent=3.0)
    if weekly:
        result = screener.check_proximity_weekly("228670", 500, SCREEN_DATE)
        assert result is not None and result["status"] == "내부-지지"
    else:
        assert screener.check_proximity("228670", 500, SCREEN_DATE) is None
