"""현행 탐지기·지표 출력 고정(OBS-3 §2).

`realtime_detector.RealtimeOrderBlockDetector`와 `indicators`가 고정 입력에서 내는 값을
스냅샷으로 못 박는다. 입력 경로는 스크리너가 실제로 쓰는 것과 같다 — 일봉은
`get_price_data(500일)` 후 거래정지(Volume=0) 제거, 주봉은 `get_price_data_weekly(500주)`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import indicators
from data_provider import KoreanStockDataProvider
from realtime_detector import RealtimeOrderBlockDetector
from tests.support import (
    FIXTURE_TICKERS,
    OHLCV_DIR,
    SCREEN_DATE,
    FakeFdr,
    assert_matches_snapshot,
    load_ohlcv,
)


def _ob_dict(ob: Any) -> dict[str, Any]:
    return {
        "type": ob.ob_type,
        "top": ob.top,
        "bottom": ob.bottom,
        "start_time": ob.start_time,
        "breaker": ob.breaker,
        "break_time": ob.break_time,
        "ob_volume": ob.ob_volume,
        "ob_low_volume": ob.ob_low_volume,
        "ob_high_volume": ob.ob_high_volume,
        "bb_volume": ob.bb_volume,
        "combined": ob.combined,
    }


def _daily_input(ticker: str) -> pd.DataFrame:
    df = KoreanStockDataProvider.get_price_data(ticker, 500, SCREEN_DATE)
    return df[df["Volume"] > 0].copy()


def _weekly_input(ticker: str) -> pd.DataFrame | None:
    result: pd.DataFrame | None = KoreanStockDataProvider.get_price_data_weekly(
        ticker, 500, SCREEN_DATE
    )
    return result


def _detect(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {"bars": len(df), "first": df.index[0], "last": df.index[-1]}
    for method in ("Wick", "Close"):
        det = RealtimeOrderBlockDetector(swing_length=10, max_atr_mult=2.0, ob_end_method=method)
        det.detect_order_blocks_realtime(df)
        out[f"raw_{method}"] = {
            "bull": [_ob_dict(ob) for ob in det.latest_bull_obs],
            "bear": [_ob_dict(ob) for ob in det.latest_bear_obs],
        }
    for combine in (True, False):
        det = RealtimeOrderBlockDetector(
            swing_length=10, max_atr_mult=2.0, ob_end_method="Wick", combine_obs=combine
        )
        det.detect_order_blocks_realtime(df)
        bull, bear = det.get_latest_orderblocks()
        out[f"latest_combine_{combine}"] = {
            "bull": [_ob_dict(ob) for ob in bull],
            "bear": [_ob_dict(ob) for ob in bear],
        }
    return out


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_detector_daily_snapshot(ticker: str, fake_fdr: FakeFdr) -> None:
    assert_matches_snapshot(f"detector_daily_{ticker}", _detect(_daily_input(ticker)))


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_detector_weekly_snapshot(ticker: str, fake_fdr: FakeFdr) -> None:
    df = _weekly_input(ticker)
    assert df is not None
    assert_matches_snapshot(f"detector_weekly_{ticker}", _detect(df))


@pytest.mark.parametrize("ticker", FIXTURE_TICKERS)
def test_convert_to_weekly_snapshot(ticker: str) -> None:
    """일봉 → 주봉(W-FRI, 거래정지 제외) 변환 결과를 고정한다."""
    weekly = KoreanStockDataProvider.convert_to_weekly(load_ohlcv(ticker))
    assert weekly is not None
    records = [
        {"week": idx, **{c: row[c] for c in ("Open", "High", "Low", "Close", "Volume")}}
        for idx, row in weekly.iterrows()
    ]
    assert_matches_snapshot(f"weekly_bars_{ticker}", records)


def test_fixture_files_present() -> None:
    """고정 입력이 조용히 빠지면 파라미터 테스트가 0개로 줄어든다 — 개수를 못 박는다."""
    assert len(FIXTURE_TICKERS) == 14
    assert len(list(OHLCV_DIR.glob("*.csv"))) == 14


def _tail(series: pd.Series, n: int = 5) -> list[Any]:
    return [[idx, val] for idx, val in series.tail(n).items()]


def test_indicators_snapshot(fake_fdr: FakeFdr) -> None:
    df = _daily_input("005930")
    highs, lows = indicators.find_swings(df, 10)
    upper, mid, lower = indicators.calculate_bollinger_bands(df)
    data = {
        "atr10_tail": _tail(indicators.calculate_atr(df, 10)),
        "swing_highs": highs,
        "swing_lows": lows,
        "ema20_tail": _tail(indicators.calculate_ema(df["Close"], 20)),
        "sma20_tail": _tail(indicators.calculate_sma(df["Close"], 20)),
        "rsi14_tail": _tail(indicators.calculate_rsi(df, 14)),
        "volume_sma20_tail": _tail(indicators.calculate_volume_sma(df, 20)),
        "high_volume_bars": [
            i for i in range(len(df)) if indicators.is_high_volume(df, i, multiplier=1.5)
        ],
        "bollinger_tail": {"upper": _tail(upper), "mid": _tail(mid), "lower": _tail(lower)},
    }
    assert_matches_snapshot("indicators_005930_daily", data)


# ---------------------------------------------------------------- 경계(합성 입력)
#
# 실데이터에서는 「값이 정확히 같다」가 드물어 부등호 돌연변이(`<` ↔ `<=`)가 스냅샷을
# 빠져나간다. 손으로 만든 입력으로 경계를 직접 건다.

_SYN_CLOSES = [100, 102, 104, 106, 108, 110, 108, 106, 104, 102, 100, 98, 96, 98, 100]
_SYN_CLOSES += [102, 104, 106, 108, 110, 112, 114, 116, 118, 120, 121, 122, 123, 124, 125]


def _synthetic(extra: list[dict[str, float]] | None = None) -> pd.DataFrame:
    rows = [
        {"Open": c - 0.5, "High": c + 1.0, "Low": c - 1.0, "Close": float(c), "Volume": 1000.0}
        for c in _SYN_CLOSES
    ]
    rows += extra or []
    return pd.DataFrame(rows, index=pd.bdate_range("2026-01-01", periods=len(rows)))


def _bull(df: pd.DataFrame, **kwargs: Any) -> list[Any]:
    det = RealtimeOrderBlockDetector(swing_length=3, **kwargs)
    det.detect_order_blocks_realtime(df)
    return list(det.latest_bull_obs)


def test_synthetic_bull_ob_baseline() -> None:
    """스윙 고점(110) 돌파 봉(종가 112)에서 그 사이 최저점 봉(95~97)이 강세 존이 된다."""
    (ob,) = _bull(_synthetic())
    assert (ob.top, ob.bottom, ob.breaker) == (97.0, 95.0, False)
    assert ob.start_time == pd.Timestamp("2026-01-19")
    assert (ob.ob_volume, ob.ob_low_volume, ob.ob_high_volume) == (3000.0, 1000.0, 2000.0)


@pytest.mark.parametrize(
    ("low", "method", "broken"),
    [
        (95.0, "Wick", False),  # 꼬리가 하단에 **닿기만** 하면 살아 있다(엄격 부등호)
        (94.99, "Wick", True),
        (94.0, "Close", False),  # Close 방식은 꼬리를 안 본다
    ],
)
def test_bull_invalidation_boundary(low: float, method: str, broken: bool) -> None:
    bar = {"Open": 100.0, "High": 101.0, "Low": low, "Close": 100.0, "Volume": 1000.0}
    (ob,) = _bull(_synthetic([bar]), ob_end_method=method)
    assert ob.breaker is broken
    assert (ob.break_time is not None) is broken


def test_bull_close_invalidation_boundary() -> None:
    keep = {"Open": 95.0, "High": 101.0, "Low": 90.0, "Close": 100.0, "Volume": 1000.0}
    kill = {"Open": 94.99, "High": 101.0, "Low": 90.0, "Close": 100.0, "Volume": 1000.0}
    (ob_keep,) = _bull(_synthetic([keep]), ob_end_method="Close")
    (ob_kill,) = _bull(_synthetic([kill]), ob_end_method="Close")
    assert ob_keep.breaker is False
    assert ob_kill.breaker is True


def test_atr_filter_boundary() -> None:
    """존 크기(2.0) == ATR(3.0) × 배수(2/3)면 **만든다**(`<=`), 배수를 조금만 줄이면 안 만든다."""
    assert len(_bull(_synthetic(), max_atr_mult=2 / 3)) == 1
    assert _bull(_synthetic(), max_atr_mult=2 / 3 - 1e-9) == []


def test_broken_bull_ob_removed_when_price_reclaims_top() -> None:
    """무효화된 강세 존은 이후 고가가 상단을 넘으면 목록에서 빠진다."""
    # 무효화 봉의 고가를 상단(97) 아래로 둬야 그 봉에서 바로 지워지지 않는다.
    brk = {"Open": 96.0, "High": 96.0, "Low": 90.0, "Close": 95.5, "Volume": 1000.0}
    stay = {"Open": 96.0, "High": 97.0, "Low": 93.0, "Close": 96.0, "Volume": 1000.0}
    gone = {"Open": 96.0, "High": 97.01, "Low": 93.0, "Close": 96.0, "Volume": 1000.0}
    remaining = _bull(_synthetic([brk, stay]))
    assert len(remaining) == 1 and remaining[0].breaker is True
    assert _bull(_synthetic([brk, gone])) == []
