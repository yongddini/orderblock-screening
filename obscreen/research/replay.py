"""과거 재현(walk-forward) — 지금 탐지기로 과거 각 영업일의 스크리닝을 다시 만든다(OBS-12).

규칙을 따로 다시 짜지 않는다. 판정은 운영과 **같은 함수**를 부른다:

- 근접도·RSI·존 = `StockScreener.evaluate_daily_frame` / `evaluate_weekly_frame`
  (운영 `check_proximity` / `check_proximity_weekly`가 조회 뒤 부르는 바로 그 코드),
- 추천 플래그 = `obscreen.screening.core.rsi_oversold_flag` · `dual_support_codes`
  (운영 `run_and_save_screening`이 저장할 때 부르는 바로 그 함수).

다른 것은 **시세를 어디서 가져오느냐** 하나다. 운영은 그날
`fdr.DataReader(ticker, end−750일, end)`를 부르고, 이 모듈은 한 번 받아 둔 전체 이력에서
**같은 창**을 잘라 낸다(`daily_window` · `weekly_window` —
`KoreanStockDataProvider.get_price_data`·`get_price_data_weekly`의 기간 계산과 꼬리 자르기를
그대로 옮겼다). 그래서 그날보다 뒤의 데이터는 판정에 들어갈 수 없다
(`tests/test_obs12_replay.py`가 「뒤쪽을 잘라내도 과거 신호가 안 바뀐다」를 건다).

⚠️ 알려진 차이(리포트 머리에도 적는다):

- **대상 종목** — 운영은 그날의 종목 목록·시총으로 상위 N을 뽑는다. 과거 목록·상장주식수는 KRX
  로그인 없이 받을 수 없어서, **오늘 상장된 종목**을 후보로 두고 「그날 종가 × 오늘 상장주식수」로
  그날 시총 순위를 근사한다(`select_universe`). 그 사이 상장폐지된 종목은 빠진다(생존 편향 —
  결과가 좋아 보이는 쪽으로 기운다). 시장 구분도 오늘 기준이다.
- **주봉은 일봉이 「지지」일 때만 계산한다** — 이중지지 비트는 일봉이 지지(Bull)여야만 켜지므로
  (`dual_support_codes`) 일봉이 지지가 아닌 날의 주봉은 어떤 일봉 플래그도 바꾸지 못한다. 계산량만
  줄인다(운영 저장 결과와 플래그가 같음을 테스트가 건다).
- **ETF는 뺀다**(이슈 범위 밖). 운영의 이중지지 조인에 ETF 행이 섞여도 종목 코드가 달라 주식
  플래그는 바뀌지 않는다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from obscreen.config import Settings
from obscreen.data.provider import KoreanStockDataProvider
from obscreen.screening.core import (
    FLAG_RSI_OVERSOLD,
    dual_support_codes,
    rsi_oversold_flag,
)
from obscreen.screening.screener import StockScreener

#: 운영 `run_and_save_screening`이 넘기는 창 길이 — 일봉 500봉 · 주봉 500주.
DAILY_DAYS = 500
WEEKLY_WEEKS = 500

#: 비교용 RSI 기준(사용자 결정 #4). 30이 운영값(주인공)이다.
DAILY_RSI_THRESHOLDS: tuple[float, ...] = (30.0, 35.0, 40.0)


def daily_window(hist: pd.DataFrame, end: pd.Timestamp, days: int = DAILY_DAYS) -> pd.DataFrame:
    """운영 `get_price_data(ticker, days, end)`가 그날 돌려줬을 창(없으면 빈 표).

    `start = end − int(days × 1.5)일`부터 `end`까지(양 끝 포함) 받고, 길면 뒤에서 `days`개.
    """
    start = end - timedelta(days=int(days * 1.5))
    window = hist.loc[start:end]
    if len(window) > days:
        window = window.tail(days)
    return window


def weekly_window(hist: pd.DataFrame, end: pd.Timestamp, weeks: int = WEEKLY_WEEKS) -> Any:
    """운영 `get_price_data_weekly(ticker, weeks, end)`가 그날 돌려줬을 주봉(없으면 None).

    마지막 주봉은 `end`가 속한 주의 **그날까지만**(미완성 주봉) — 운영과 같다.
    """
    daily = daily_window(hist, end, days=weeks * 5)
    if len(daily) == 0:
        return None
    weekly = KoreanStockDataProvider.convert_to_weekly(daily)
    if weekly is None or len(weekly) == 0:
        return None
    if len(weekly) > weeks:
        weekly = weekly.tail(weeks)
    return weekly


@dataclass(frozen=True)
class ScreenerConfig:
    """운영과 같은 스크리너 설정값(근접도·탐지기). `from_settings`로 운영 설정에서 만든다."""

    proximity_daily: float
    proximity_weekly: float
    swing_length: int
    max_atr_mult: float
    ob_end_method: str
    combine_obs: bool

    @classmethod
    def from_settings(cls, settings: Settings) -> ScreenerConfig:
        return cls(
            proximity_daily=settings.proximity_stock_daily,
            proximity_weekly=settings.proximity_weekly,
            swing_length=settings.swing_length,
            max_atr_mult=settings.max_atr_mult,
            ob_end_method=settings.ob_end_method,
            combine_obs=settings.combine_obs,
        )

    def screener(self, weekly: bool) -> StockScreener:
        return StockScreener(
            proximity_percent=self.proximity_weekly if weekly else self.proximity_daily,
            swing_length=self.swing_length,
            max_atr_mult=self.max_atr_mult,
            ob_end_method=self.ob_end_method,
            combine_obs=self.combine_obs,
        )


def _record(market: str, code: str, name: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """스크리너 결과 → 운영 `screen_market`·`screen_market_weekly`가 만드는 행 모양."""
    return {
        "Market": market,
        "Code": code,
        "Name": name,
        "Current_Price": result["current_price"],
        "Change_Percent": result["change_percent"],
        "RSI": result["rsi"],
        "trading_value": result["trading_value"],
        "Status": result["status"],
        "OB_Type": result["ob_type"],
        "OB_Top": result["ob_top"],
        "OB_Bottom": result["ob_bottom"],
        "Distance_Percent": result["distance_percent"],
    }


@dataclass
class StockTask:
    """한 종목의 재현 작업 — 이력 전체와, 그 종목이 대상 종목이었던 날짜들."""

    code: str
    name: str
    market: str
    hist: pd.DataFrame
    dates: Sequence[pd.Timestamp]
    config: ScreenerConfig
    rsi_thresholds: tuple[float, ...] = field(default=DAILY_RSI_THRESHOLDS)


def replay_stock(task: StockTask) -> list[dict[str, Any]]:
    """한 종목을 날짜마다 다시 판정한다. 일봉 결과가 있는(존 내부/근접) 날만 행을 낸다.

    운영과 같이 판정 중 예외는 「결과 없음」으로 본다(`check_proximity`·`screen_market_weekly`가
    예외를 삼킨다).
    """
    daily_screener = task.config.screener(weekly=False)
    weekly_screener = task.config.screener(weekly=True)
    rows: list[dict[str, Any]] = []
    for date in task.dates:
        try:
            daily = daily_screener.evaluate_daily_frame(daily_window(task.hist, date))
        except Exception:
            daily = None
        if not daily:
            continue
        daily_row = _record(task.market, task.code, task.name, daily)
        weekly_rows: list[dict[str, Any]] = []
        if daily_row["OB_Type"] == "Bull":
            try:
                weekly = weekly_screener.evaluate_weekly_frame(weekly_window(task.hist, date))
            except Exception:
                weekly = None
            if weekly:
                weekly_rows.append(_record(task.market, task.code, task.name, weekly))
        dual = task.code in dual_support_codes([daily_row], weekly_rows)
        weekly_row = weekly_rows[0] if weekly_rows else None
        row: dict[str, Any] = {
            "date": date,
            "code": task.code,
            "name": task.name,
            "market": task.market,
            "close": float(daily_row["Current_Price"]),
            "rsi": float(daily_row["RSI"]),
            "status": daily_row["Status"],
            "ob_type": daily_row["OB_Type"],
            "ob_top": float(daily_row["OB_Top"]),
            "ob_bottom": float(daily_row["OB_Bottom"]),
            "distance_percent": float(daily_row["Distance_Percent"]),
            "weekly_ob_type": weekly_row["OB_Type"] if weekly_row else None,
            "weekly_rsi": float(weekly_row["RSI"]) if weekly_row else float("nan"),
            "dual_support": dual,
        }
        for threshold in task.rsi_thresholds:
            row[f"rsi_flag_{threshold:g}"] = rsi_oversold_flag(daily_row, threshold) > 0
        # 운영 저장값과 같은 정의 — 비트1(RSI<30) | 비트2(이중지지).
        row["is_recommended"] = rsi_oversold_flag(daily_row) | (2 if dual else 0)
        assert (row["is_recommended"] & FLAG_RSI_OVERSOLD) == int(row["rsi_flag_30"])
        rows.append(row)
    return rows


def select_universe(
    closes: pd.DataFrame,
    shares: Mapping[str, float],
    markets: Mapping[str, str],
    top_n: int,
) -> pd.DataFrame:
    """날짜 × 종목 표(참이면 그날 대상) — 시장별 「그날 종가 × 오늘 상장주식수」 상위 `top_n`.

    Args:
        closes: 날짜 인덱스 × 종목 열의 종가(그날 봉이 없으면 NaN — 대상에서 빠진다).
        shares: 종목별 상장주식수(오늘 목록 값).
        markets: 종목별 시장(`KOSPI`·`KOSDAQ`).
        top_n: 시장별 상위 개수(운영 `SCREENING_TOP_N`).
    """
    share_row = pd.Series({c: float(shares.get(c, float("nan"))) for c in closes.columns})
    caps = closes.mul(share_row, axis=1)
    member = pd.DataFrame(False, index=closes.index, columns=closes.columns)
    for market in sorted(set(markets.values())):
        cols = [c for c in closes.columns if markets.get(c) == market]
        if not cols:
            continue
        ranks = caps[cols].rank(axis=1, ascending=False, method="first")
        member[cols] = ranks.le(top_n) & caps[cols].notna()
    return member


def membership_dates(member: pd.DataFrame) -> dict[str, list[pd.Timestamp]]:
    """`select_universe` 결과 → 종목별 대상 날짜 목록."""
    return {str(c): list(member.index[member[c].to_numpy()]) for c in member.columns}


def build_close_panel(histories: Mapping[str, pd.DataFrame], dates: Iterable[Any]) -> pd.DataFrame:
    """종목별 이력 → 날짜(`dates`) × 종목 종가 표. 그날 봉이 없거나 거래량 0이면 NaN."""
    index = pd.DatetimeIndex(list(dates))
    data: dict[str, pd.Series] = {}
    for code, hist in histories.items():
        traded = hist[hist["Volume"] > 0]["Close"]
        data[code] = traded.reindex(index)
    return pd.DataFrame(data, index=index)
