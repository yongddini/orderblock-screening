"""OBS-12 과거 재현 측정 — 미래 차단 · 운영 판정과 같은 결과 · 보유 기간 수익 규칙.

외부 API를 부르지 않는다 — 고정 입력(`tests/fixtures/ohlcv`)을 이력으로 쓴다.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

import screening_core
from obscreen.config import get_settings
from obscreen.research import holding
from obscreen.research.replay import (
    ScreenerConfig,
    StockTask,
    daily_window,
    replay_stock,
    select_universe,
    weekly_window,
)
from tests.conftest import query_rows
from tests.support import FIXTURES_DIR, FakeFdr, load_ohlcv

#: 고정 입력에서 플래그가 실제로 켜지는 날 — 06-08은 RSI과매도(1), 06-25는 이중지지+RSI(3)가 있다.
FLAG_DATES = ("2026-06-08", "2026-06-25")


def _listing() -> pd.DataFrame:
    df = pd.read_csv(FIXTURES_DIR / "listing_krx.csv", dtype={"Code": str})
    seg = {"KOSPI": "KOSPI", "KOSDAQ": "KOSDAQ", "KOSDAQ GLOBAL": "KOSDAQ"}
    df = df[df["Market"].isin(seg)].copy()
    df["MarketGroup"] = df["Market"].map(seg)
    return df


def _histories() -> dict[str, pd.DataFrame]:
    out = {}
    for code in _listing()["Code"]:
        hist = load_ohlcv(code)
        if not hist.empty:
            out[code] = hist
    return out


def _config() -> ScreenerConfig:
    return ScreenerConfig.from_settings(get_settings())


def _replay_all(
    histories: dict[str, pd.DataFrame], dates: list[pd.Timestamp]
) -> pd.DataFrame:
    listing = _listing().set_index("Code")
    rows = []
    for code, hist in histories.items():
        ds = [d for d in dates if d in hist.index]
        rows.extend(
            replay_stock(
                StockTask(
                    code,
                    str(listing.loc[code, "Name"]),
                    str(listing.loc[code, "MarketGroup"]),
                    hist,
                    ds,
                    _config(),
                )
            )
        )
    return pd.DataFrame(rows).sort_values(["date", "code"]).reset_index(drop=True)


# ---------------------------------------------------------------- 창이 운영과 같다


@pytest.mark.parametrize("end", ["2026-06-08", "2026-09-14"])
def test_windows_match_provider(fake_fdr: FakeFdr, end: str) -> None:
    """잘라 낸 창 = 운영 `get_price_data`·`get_price_data_weekly`가 그날 돌려줬을 것."""
    from data_provider import KoreanStockDataProvider

    hist = load_ohlcv("005930")
    ts = pd.Timestamp(end)
    pd.testing.assert_frame_equal(
        daily_window(hist, ts), KoreanStockDataProvider.get_price_data("005930", 500, end)
    )
    pd.testing.assert_frame_equal(
        weekly_window(hist, ts), KoreanStockDataProvider.get_price_data_weekly("005930", 500, end)
    )


# ---------------------------------------------------------------- ② 운영 판정과 같다


@pytest.mark.parametrize("date", FLAG_DATES)
def test_replay_matches_run_and_save_screening(
    date: str, fake_fdr: FakeFdr, fresh_db: Path
) -> None:
    """같은 입력이면 재현 결과의 `is_recommended`가 운영 저장값과 같다(종목 · 일봉 행 전부)."""
    compact = date.replace("-", "")
    screening_core.run_and_save_screening(compact)
    db = query_rows(
        fresh_db,
        "SELECT code, rsi, ob_top, ob_bottom, zone_type, is_recommended FROM screening_results "
        "WHERE scan_date = ? AND timeframe = 'daily' AND market IN ('KOSPI', 'KOSDAQ')",
        (compact,),
    )
    replay = _replay_all(_histories(), [pd.Timestamp(date)])
    got = {r["code"]: r for r in replay.to_dict("records")}
    want = {r["code"]: r for r in db}
    assert set(got) == set(want)
    for code, row in want.items():
        assert got[code]["is_recommended"] == row["is_recommended"], code
        assert got[code]["rsi"] == pytest.approx(row["rsi"])
        assert got[code]["ob_bottom"] == pytest.approx(row["ob_bottom"])
        assert (got[code]["ob_type"] == "Bull") == (row["zone_type"] == "지지")
    # 의미 있는 대조인지 — 플래그가 켜진 행이 실제로 있어야 한다.
    assert any(r["is_recommended"] > 0 for r in db)


def test_flag_dates_cover_both_bits(fake_fdr: FakeFdr) -> None:
    """대조 날짜에 RSI과매도 비트와 이중지지+RSI(3)가 모두 나온다 — 대조가 비어 있지 않다."""
    replay = _replay_all(_histories(), [pd.Timestamp(d) for d in FLAG_DATES])
    assert (replay["is_recommended"] == 3).any()
    assert (replay["is_recommended"] == 1).any()


# ---------------------------------------------------------------- ① 미래 차단


def test_truncating_future_does_not_change_past_signals() -> None:
    """뒤쪽 데이터를 잘라내도 과거 신호(주봉 RSI 포함)가 안 바뀐다."""
    cut = pd.Timestamp("2026-06-25")
    full = _histories()
    truncated = {c: h.loc[:cut] for c, h in full.items()}
    dates = [d for d in load_ohlcv("005930").index if pd.Timestamp("2026-05-20") <= d <= cut]
    a = _replay_all(full, dates)
    b = _replay_all(truncated, dates)
    assert len(a) > 0
    pd.testing.assert_frame_equal(a, b)
    assert a["weekly_rsi"].notna().any()


def test_weekly_window_ends_at_signal_day() -> None:
    """주봉 마지막 봉은 신호일이 속한 주의 그날까지만(미완성 주봉) — 운영과 같다."""
    hist = load_ohlcv("005930")
    wed = pd.Timestamp("2026-06-24")
    weekly = weekly_window(hist, wed)
    assert weekly is not None
    assert weekly.index[-1] == pd.Timestamp("2026-06-26")  # W-FRI 라벨
    assert weekly["Close"].iloc[-1] == hist.loc[wed, "Close"]


# ---------------------------------------------------------------- 보유 기간 수익


def _toy_hist(n: int = 10) -> pd.DataFrame:
    idx = pd.bdate_range("2026-01-05", periods=n)
    close = [100.0 + i for i in range(n)]
    return pd.DataFrame(
        {
            "Open": [c - 0.5 for c in close],
            "High": [c + 1 for c in close],
            "Low": [c - 2 for c in close],
            "Close": close,
            "Volume": [1000] * n,
        },
        index=idx,
    )


def test_forward_returns_use_only_later_bars() -> None:
    hist = _toy_hist()
    ft = holding.forward_table(hist, horizons=(1, 3), cost=0.0)
    t = hist.index[2]  # Close 102
    assert ft.loc[t, "ret_close_1"] == pytest.approx(103 / 102 - 1)
    assert ft.loc[t, "ret_open_3"] == pytest.approx(105 / 102.5 - 1)  # Open[t+1] = 102.5
    assert ft.loc[t, "mdd_close_3"] == pytest.approx(101 / 102 - 1)  # min Low[t+1..t+3] = 101
    # 신호일보다 앞의 봉을 바꿔도 결과가 같다.
    changed = hist.copy()
    changed.iloc[:2] = changed.iloc[:2] * 7
    ft2 = holding.forward_table(changed, horizons=(1, 3), cost=0.0)
    pd.testing.assert_series_equal(ft.loc[t], ft2.loc[t])


def test_unfinished_holding_period_is_excluded() -> None:
    hist = _toy_hist(10)
    ft = holding.forward_table(hist, horizons=(1, 3), cost=0.0)
    last = hist.index[-1]
    assert math.isnan(ft.loc[last, "ret_close_1"])
    assert math.isnan(ft.loc[hist.index[-3], "ret_close_3"])
    assert not math.isnan(ft.loc[hist.index[-4], "ret_close_3"])
    # 잘린 이력의 결과는 전체 이력의 결과와 같다(끝난 칸만).
    short = holding.forward_table(hist.iloc[:7], horizons=(1, 3), cost=0.0)
    pd.testing.assert_series_equal(
        short["ret_close_3"].dropna(), ft["ret_close_3"].loc[short["ret_close_3"].dropna().index]
    )


def test_cost_is_subtracted() -> None:
    hist = _toy_hist()
    a = holding.forward_table(hist, horizons=(1,), cost=0.0)
    b = holding.forward_table(hist, horizons=(1,), cost=0.0023)
    assert (a["ret_close_1"] - b["ret_close_1"]).dropna().round(12).eq(0.0023).all()


def test_stop_returns_exit_on_close_below_zone() -> None:
    hist = _toy_hist()
    hist.loc[hist.index[4], "Close"] = 90.0  # 신호(2) 뒤 두 번째 날 종가가 존 아래
    t = hist.index[2]
    sr = holding.stop_returns(hist, t, stop_price=95.0, entry="close", horizons=(1, 5))
    assert sr[1] == pytest.approx(103 / 102 - 1)  # 1일 안에는 이탈 없음
    assert sr[5] == pytest.approx(90 / 102 - 1)  # 이탈한 날 종가에 판다
    assert math.isnan(holding.stop_returns(hist, hist.index[-1], 95.0, "close", (1,))[1])


def test_first_day_mask_marks_run_starts() -> None:
    cal = list(pd.bdate_range("2026-01-05", periods=6))
    sig = pd.DataFrame(
        {
            "code": ["A", "A", "A", "B", "A"],
            "date": [cal[0], cal[1], cal[2], cal[1], cal[4]],
        }
    )
    mask = pd.Series([True, True, True, True, True])
    first = holding.first_day_mask(sig, mask, cal)
    assert first.tolist() == [True, False, False, True, True]


def test_select_universe_ranks_by_close_times_shares() -> None:
    idx = pd.DatetimeIndex(["2026-01-05", "2026-01-06"])
    closes = pd.DataFrame(
        {"A": [10.0, 10.0], "B": [5.0, 30.0], "C": [100.0, float("nan")], "D": [1.0, 1.0]},
        index=idx,
    )
    shares = {"A": 10, "B": 10, "C": 1, "D": 1000}
    markets = {"A": "KOSPI", "B": "KOSPI", "C": "KOSPI", "D": "KOSDAQ"}
    member = select_universe(closes, shares, markets, top_n=1)
    # 1/5: A 100 > C 100(동률은 열 순서) > B 50 · 1/6: B 300 > A 100, C는 그날 시세 없음
    assert member.loc["2026-01-05"].to_dict() == {"A": True, "B": False, "C": False, "D": True}
    assert member.loc["2026-01-06"].to_dict() == {"A": False, "B": True, "C": False, "D": True}


# ---------------------------------------------------------------- ③④ 리포트


def test_report_and_csv_end_to_end(tmp_path: Path) -> None:
    """작은 고정 입력으로 끝까지 돌려 리포트 머리(엔진·커밋·기간·비용·대상)와 표본 경고를 본다."""
    histories = _histories()
    listing = _listing()
    listing = listing[listing["Code"].isin(histories)].reset_index(drop=True)
    indexes = {"KS11": load_ohlcv("005930"), "KQ11": load_ohlcv("247540")}
    calendar = [
        d
        for d in indexes["KS11"].index
        if pd.Timestamp("2026-06-01") <= d <= pd.Timestamp("2026-06-30")
    ]
    config = _config()
    signals, member = holding.run_replay(histories, listing, calendar, 400, config, jobs=1)
    markets = dict(zip(listing["Code"], listing["MarketGroup"], strict=True))
    panel = holding.build_panel(
        signals, member, histories, indexes, markets, calendar, cost=0.0023
    )
    meta = holding.RunMeta(
        commit="abc1234",
        start=calendar[0],
        end=calendar[-1],
        data_end=pd.Timestamp("2026-09-18"),
        cost_bps=23.0,
        top_n=400,
        n_candidates_pool=len(listing),
        generated_at="2026-10-14 00:00 KST",
        config=config,
    )
    md = holding.render_report(panel, meta)
    assert "탐지기 = AlphaBlock `de825d27` 복사본" in md
    assert "생성 커밋 = `abc1234`" in md
    assert "비용 = 왕복 23bp" in md
    assert "생존 편향" in md
    assert holding.SAMPLE_WARNING in md  # 한 달치 고정 입력이라 표본은 반드시 적다
    out = tmp_path / "signals.csv"
    n = holding.write_signals_csv(panel, meta, out)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# 탐지기 = AlphaBlock")
    assert n == int(holding._dual_rsi(40.0)(panel.signals).sum())
    # 주인공 = 운영 저장값 3
    main = holding.MAIN.predicate(panel.signals)
    assert (panel.signals.loc[main, "is_recommended"] == 3).all()
    assert ((panel.signals["is_recommended"] == 3) == main).all()
