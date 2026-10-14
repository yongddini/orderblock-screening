"""탐지기 출력 고정(OBS-3 §2 → OBS-5에서 AlphaBlock 탐지기로 갱신).

`obscreen.detect.adapter.AlphaBlockOrderBlockDetector`(AlphaBlock 복사본 + 국장 어댑터)가 고정
입력에서 내는 존을 스냅샷으로 못 박는다. 입력 경로는 스크리너가 실제로 쓰는 것과 같다 — 일봉은
`get_price_data(500일)` 후 거래정지(Volume=0) 제거, 주봉은 `get_price_data_weekly(500주)`.

OBS-5 이전 스냅샷(옛 `realtime_detector`)과의 차이는 `docs/obs-5-detector-change.md` 표에 있다.
AlphaBlock 원본과 같은 존이 나오는지는 `tests/test_alphablock_parity.py`가 따로 건다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from data_provider import KoreanStockDataProvider
from obscreen.detect.adapter import AlphaBlockOrderBlockDetector
from obscreen.detect.alphablock.models import OrderBlock, OrderBlockDirection
from obscreen.detect.alphablock.order_blocks import _atr
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
    variants: dict[str, dict[str, Any]] = {
        "default": {},  # 운영 경로(AlphaBlock 기본값)
        "close": {"ob_end_method": "Close"},
        "combine": {"combine_obs": True},
    }
    for name, kwargs in variants.items():
        det = AlphaBlockOrderBlockDetector(**kwargs)
        result = det.detect_order_blocks_realtime(df)
        bull, bear = det.get_latest_orderblocks()
        out[f"latest_{name}"] = {
            "bull": [_ob_dict(ob) for ob in bull],
            "bear": [_ob_dict(ob) for ob in bear],
        }
        out[f"archive_{name}"] = {
            "zones": len(result.order_blocks),
            "bull": sum(ob.direction is OrderBlockDirection.BULLISH for ob in result.order_blocks),
            "broken": sum(ob.breaker for ob in result.order_blocks),
            "swept": sum(ob.swept_time is not None for ob in result.order_blocks),
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


# ---------------------------------------------------------------- 경계(합성 입력)
#
# 실데이터에서는 「값이 정확히 같다」가 드물어 부등호 돌연변이(`<` ↔ `<=`)가 스냅샷을
# 빠져나간다. 손으로 만든 입력으로 어댑터를 거친 경계를 직접 건다.

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
    det = AlphaBlockOrderBlockDetector(swing_length=3, **kwargs)
    det.detect_order_blocks_realtime(df)
    return det.get_latest_orderblocks()[0]


def test_synthetic_bull_ob_baseline() -> None:
    """스윙 고점(110) 돌파 봉(종가 112)에서 그 사이 최저점 봉(95~97)이 강세 존이 된다.

    옛 탐지기(OBS-3)도 같은 존을 냈다 — 정의가 같은 FluxCharts 계열이라 단순한 경우엔 일치한다.
    """
    (ob,) = _bull(_synthetic())
    assert (ob.top, ob.bottom, ob.breaker) == (97.0, 95.0, False)
    assert ob.start_time == pd.Timestamp("2026-01-19")  # 원래 인덱스 값으로 되돌아온다
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
    bar = {"Open": 100.0, "High": 101.0, "Low": low, "Close": 100.0, "Volume": 1234.0}
    (ob,) = _bull(_synthetic([bar]), ob_end_method=method)
    assert ob.breaker is broken
    assert (ob.break_time is not None) is broken
    if broken:  # 무효화 봉의 날짜·거래량이 원래 인덱스에서 온다
        assert ob.break_time == pd.Timestamp("2026-02-12")
        assert ob.bb_volume == 1234.0


def test_bull_close_invalidation_boundary() -> None:
    keep = {"Open": 95.0, "High": 101.0, "Low": 90.0, "Close": 100.0, "Volume": 1000.0}
    kill = {"Open": 94.99, "High": 101.0, "Low": 90.0, "Close": 100.0, "Volume": 1000.0}
    (ob_keep,) = _bull(_synthetic([keep]), ob_end_method="Close")
    (ob_kill,) = _bull(_synthetic([kill]), ob_end_method="Close")
    assert ob_keep.breaker is False
    assert ob_kill.breaker is True


def test_atr_filter_boundary() -> None:
    """존 크기(2.0) == 확정 봉 ATR(Wilder 10) × 배수면 **만든다**(`<=`), 조금만 줄이면 안 만든다."""
    df = _synthetic()
    atr = _atr(df["High"].tolist(), df["Low"].tolist(), df["Close"].tolist(), 10)
    confirm = list(df.index).index(pd.Timestamp("2026-01-29"))  # 종가 112 > 스윙 고점 110
    atr_t = atr[confirm]
    assert atr_t is not None
    assert len(_bull(df, max_atr_mult=2.0 / atr_t)) == 1
    assert _bull(df, max_atr_mult=2.0 / atr_t - 1e-9) == []


def test_broken_bull_ob_removed_when_price_reclaims_top() -> None:
    """무효화된 강세 존은 이후 고가가 상단을 **넘으면** 화면에서 빠진다(소멸 · swept)."""
    brk = {"Open": 96.0, "High": 96.0, "Low": 90.0, "Close": 95.5, "Volume": 1000.0}
    stay = {"Open": 96.0, "High": 97.0, "Low": 93.0, "Close": 96.0, "Volume": 1000.0}
    gone = {"Open": 96.0, "High": 97.01, "Low": 93.0, "Close": 96.0, "Volume": 1000.0}
    remaining = _bull(_synthetic([brk, stay]))
    assert len(remaining) == 1 and remaining[0].breaker is True
    assert _bull(_synthetic([brk, gone])) == []


def test_zone_count_low_returns_latest_three_per_direction(fake_fdr: FakeFdr) -> None:
    """「Zone Count: Low」 — 일봉·주봉 모두 방향별로 **아직 소멸 안 한 존 중 최신 확정 3개**.

    OBS-5 전에는 일봉만 `[:3]`로 잘랐고 주봉은 자르지 않았다(tests/README.md 버그 후보 6).
    이제 개수 규칙은 AlphaBlock `zone_count` 하나라 일봉·주봉이 같다.
    """
    seen_more_than_three = False
    for ticker in FIXTURE_TICKERS:
        for df in (_daily_input(ticker), _weekly_input(ticker)):
            assert df is not None
            det = AlphaBlockOrderBlockDetector()
            result = det.detect_order_blocks_realtime(df)
            last = int(pd.Timestamp(df.index[-1]).value // 1_000_000)  # ns → ms
            for direction, got in zip(
                (OrderBlockDirection.BULLISH, OrderBlockDirection.BEARISH),
                det.get_latest_orderblocks(),
                strict=True,
            ):
                alive: list[OrderBlock] = sorted(
                    (
                        ob
                        for ob in result.order_blocks
                        if ob.direction is direction and ob.alive_at(last)
                    ),
                    key=lambda ob: ob.confirmed_time,
                    reverse=True,
                )
                seen_more_than_three |= len(alive) > 3
                assert [(ob.top, ob.bottom) for ob in got] == [
                    (ob.top, ob.bottom) for ob in alive[:3]
                ]
    assert seen_more_than_three, "살아 있는 존이 3개를 넘는 입력이 없어 자르기를 못 건다"
