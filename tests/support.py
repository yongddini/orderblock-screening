"""회귀(특성) 테스트 공용 도구 — 가짜 시세 제공자와 스냅샷 비교(OBS-3).

- **가짜 제공자**: 레거시 코드는 `data_provider.fdr`(FinanceDataReader)와 `pykrx.stock`을
  직접 부른다. 테스트는 그 두 이름만 `tests/fixtures/`의 CSV를 읽는 가짜로 갈아끼운다.
  그래서 `KoreanStockDataProvider`의 기간 계산·꼬리 자르기·주봉 변환 같은 **진짜 코드는
  그대로 돈다**(경계만 가짜다).
- **스냅샷**: 현행 동작의 출력을 `tests/snapshots/*.json`에 고정한다. 기본 실행은 비교만
  하고, 스냅샷이 없거나 다르면 실패한다. 의도한 변경일 때만 `UPDATE_SNAPSHOTS=1`로
  다시 쓴다(`tests/README.md`).
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
OHLCV_DIR = FIXTURES_DIR / "ohlcv"
INVESTOR_DIR = FIXTURES_DIR / "investor"
SNAPSHOT_DIR = TESTS_DIR / "snapshots"

#: 스크리닝 기준일(월요일 · 공휴일 아님). 고정 입력은 2026-09-18까지 있다.
SCREEN_DATE = "2026-09-14"
SCREEN_DATE_COMPACT = "20260914"

#: 고정 입력이 있는 종목(코스피·코스닥 대형/소형 + ETF).
FIXTURE_TICKERS: tuple[str, ...] = tuple(sorted(p.stem for p in OHLCV_DIR.glob("*.csv")))

#: 스냅샷 비교의 부동소수 허용 오차 — 플랫폼 간 끝자리 잡음만 흡수한다.
REL_TOL = 1e-9
ABS_TOL = 1e-9

_INVESTOR_FILE_KEY = {"외국인": "foreign", "기관합계": "institution"}


def load_ohlcv(ticker: str) -> pd.DataFrame:
    """고정 일봉 OHLCV(FinanceDataReader.DataReader와 같은 모양: DatetimeIndex)."""
    path = OHLCV_DIR / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, index_col="Date", parse_dates=["Date"])


class FakeFdr:
    """`FinanceDataReader` 대역. 네트워크를 쓰지 않는다."""

    calls: list[tuple[str, str]]

    def __init__(self) -> None:
        self.calls = []

    def DataReader(self, ticker: str, start: Any = None, end: Any = None) -> pd.DataFrame:  # noqa: N802
        self.calls.append(("DataReader", ticker))
        df = load_ohlcv(ticker)
        if df.empty:
            return df
        lo = pd.Timestamp(start) if start is not None else df.index[0]
        hi = pd.Timestamp(end) if end is not None else df.index[-1]
        return df.loc[lo:hi].copy()

    def StockListing(self, market: str) -> pd.DataFrame:  # noqa: N802
        self.calls.append(("StockListing", market))
        if market == "KRX":
            return pd.read_csv(FIXTURES_DIR / "listing_krx.csv", dtype={"Code": str})
        if market == "ETF/KR":
            return pd.read_csv(FIXTURES_DIR / "listing_etf.csv", dtype={"Symbol": str})
        raise AssertionError(f"고정 입력에 없는 목록: {market}")


class FakePykrxStock:
    """`pykrx.stock` 대역 — 수급 저장 로직이 부르는 두 함수만 흉내 낸다."""

    def get_market_net_purchases_of_equities(
        self, fromdate: str, todate: str, market: str, investor: str
    ) -> pd.DataFrame:
        key = _INVESTOR_FILE_KEY[investor]
        return pd.read_csv(
            INVESTOR_DIR / f"net_{market}_{key}.csv", index_col="티커", dtype={"티커": str}
        )

    def get_market_ohlcv(self, date: str, market: str = "KOSPI") -> pd.DataFrame:
        return pd.read_csv(
            INVESTOR_DIR / f"ohlcv_{market}.csv", index_col="티커", dtype={"티커": str}
        )

    def get_market_ticker_name(self, ticker: str) -> str:
        return f"이름조회-{ticker}"


def to_jsonable(value: Any) -> Any:
    """numpy·pandas·datetime 값을 JSON에 담을 수 있는 파이썬 값으로 바꾼다."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is None or isinstance(value, (bool, str)):
        return value
    if hasattr(value, "item"):  # numpy 스칼라
        return to_jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return value
    if isinstance(value, int):
        return value
    raise TypeError(f"스냅샷에 담을 수 없는 값: {type(value)!r}")


def _diff(expected: Any, actual: Any, path: str, out: list[str]) -> None:
    if len(out) >= 20:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in sorted(set(expected) | set(actual)):
            if key not in expected:
                out.append(f"{path}.{key}: 새로 생김")
            elif key not in actual:
                out.append(f"{path}.{key}: 사라짐")
            else:
                _diff(expected[key], actual[key], f"{path}.{key}", out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: 길이 {len(expected)} → {len(actual)}")
        for i, (e, a) in enumerate(zip(expected, actual, strict=False)):
            _diff(e, a, f"{path}[{i}]", out)
        return
    both_numbers = (
        isinstance(expected, (int, float))
        and isinstance(actual, (int, float))
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    )
    if both_numbers:
        if not math.isclose(expected, actual, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            out.append(f"{path}: {expected!r} → {actual!r}")
        return
    if expected != actual:
        out.append(f"{path}: {expected!r} → {actual!r}")


def assert_matches_snapshot(name: str, data: Any) -> None:
    """`tests/snapshots/<name>.json`과 비교한다. `UPDATE_SNAPSHOTS=1`이면 다시 쓴다."""
    actual = to_jsonable(data)
    path = SNAPSHOT_DIR / f"{name}.json"
    if os.environ.get("UPDATE_SNAPSHOTS") == "1":
        SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(actual, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    if not path.exists():
        raise AssertionError(
            f"스냅샷이 없다: {path.relative_to(TESTS_DIR.parent)} — 의도한 것이면 "
            "`UPDATE_SNAPSHOTS=1 uv run pytest`로 만든다(tests/README.md)."
        )
    expected = json.loads(path.read_text(encoding="utf-8"))
    problems: list[str] = []
    _diff(expected, actual, name, problems)
    if problems:
        raise AssertionError(
            f"현행 동작 스냅샷과 다르다({path.name}). 의도한 변경이면 "
            "`UPDATE_SNAPSHOTS=1`로 갱신하고 PR에 이유를 적는다:\n  " + "\n  ".join(problems)
        )
