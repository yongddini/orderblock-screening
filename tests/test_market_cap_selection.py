"""시총 상위 N 선정 수정(OBS-7).

2026-09-30 실측에서 `fdr.StockListing("KRX")`의 `Marcap`이 전 종목 NaN이었다. 그 입력에서
옛 코드는 `sort_values("Marcap")`이 정렬을 못 해 「시총 상위 N」이 **목록 순서 앞 N개**가
됐고, `Market == "KOSDAQ"` 정확 일치라 `KOSDAQ GLOBAL`이 늘 빠졌다. 고정 목록
(`fixtures/listing_krx.csv`)은 실측처럼 `Marcap`을 전부 비우고 **이름순**으로 적어 두었으므로
옛 코드로 되돌리면 이 파일의 테스트가 실제로 깨진다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import collect_data
import data_provider
import screening_core
from data_provider import (
    MAX_UNRANKED_RATIO,
    KoreanStockDataProvider,
    MarketCapUnavailableError,
    check_market_cap_coverage,
)
from tests.conftest import query_rows
from tests.support import FIXTURES_DIR, SCREEN_DATE_COMPACT, FakeFdr


def _listing() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "listing_krx.csv", dtype={"Code": str})


def _use_listing(monkeypatch: pytest.MonkeyPatch, fake: FakeFdr, df: pd.DataFrame) -> None:
    """가짜 FDR의 KRX 목록만 `df`로 바꾼다(ETF 목록·시세는 그대로)."""
    original = fake.StockListing

    def listing(market: str) -> pd.DataFrame:
        if market == "KRX":
            fake.calls.append(("StockListing", market))
            return df.copy()
        return original(market)

    monkeypatch.setattr(fake, "StockListing", listing)


# ---------------------------------------------------------------- 고정 입력의 전제


def test_fixture_reproduces_the_production_listing_shape() -> None:
    """고정 목록이 실측 모양을 따른다 — 그래야 버그가 재현되면 테스트가 깨진다."""
    df = _listing()
    assert df["Marcap"].isna().all()
    assert df["Close"].notna().all() and df["Stocks"].notna().all()
    assert "KOSDAQ GLOBAL" in set(df["Market"])
    assert "KONEX" in set(df["Market"])
    # 목록 순서가 시총 순서가 아니어야 「정렬 안 됨」이 드러난다.
    cap = df["Close"] * df["Stocks"]
    assert df.index.tolist() != cap.sort_values(ascending=False).index.tolist()


# ---------------------------------------------------------------- §1·§2 선정


def test_samsung_is_kospi_number_one_when_marcap_is_all_nan(fake_fdr: FakeFdr) -> None:
    kospi = KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSPI", 400)
    assert kospi["Code"].tolist() == [
        "005930",  # 삼성전자 — 이름순이면 여섯째
        "000660",
        "035420",
        "051910",
        "003490",
        "012800",
    ]
    cap = kospi["MarketCapSort"].tolist()
    assert cap == sorted(cap, reverse=True)


def test_kosdaq_includes_global_segment_and_excludes_konex(fake_fdr: FakeFdr) -> None:
    kosdaq = KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSDAQ", 400)
    codes = kosdaq["Code"].tolist()
    assert codes == ["247540", "086520", "196170", "041510", "228670", "999990"]
    # 에코프로비엠·알테오젠은 KOSDAQ GLOBAL — 정확 일치 필터였다면 빠진다.
    assert {"247540", "196170"} <= set(codes)
    assert "999980" not in codes  # KONEX


def test_marcap_is_used_when_the_source_fills_it(
    fake_fdr: FakeFdr, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Marcap`이 채워져 오면 그 값이 이긴다(`Close × Stocks`와 순위가 달라도)."""
    df = _listing()
    marcap = {"012800": 9e20, "005930": 1e12}  # 일부러 뒤집는다
    df["Marcap"] = df["Code"].map(marcap)
    _use_listing(monkeypatch, fake_fdr, df)

    kospi = KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSPI", 3)
    # 012800(Marcap 9e20) 1위, 나머지는 Close × Stocks. 005930은 Marcap 1e12라 밀려난다.
    assert kospi["Code"].tolist() == ["012800", "000660", "035420"]


def test_top_n_cuts_after_ranking(fake_fdr: FakeFdr) -> None:
    assert KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSPI", 2)["Code"].tolist() == [
        "005930",
        "000660",
    ]


# ---------------------------------------------------------------- §5 방어


def test_guard_stops_when_close_and_stocks_are_missing(
    fake_fdr: FakeFdr, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    df = _listing()
    df["Close"] = float("nan")
    _use_listing(monkeypatch, fake_fdr, df)

    with pytest.raises(MarketCapUnavailableError, match="KOSPI"):
        KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSPI", 400)
    assert "시가총액 검사 실패" in capsys.readouterr().err


def test_guard_stops_on_empty_listing(fake_fdr: FakeFdr, monkeypatch: pytest.MonkeyPatch) -> None:
    _use_listing(monkeypatch, fake_fdr, _listing().iloc[0:0])
    with pytest.raises(MarketCapUnavailableError, match="비어"):
        KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSDAQ", 400)


def test_guard_threshold_boundary() -> None:
    """미상 비율이 기준 **이하**면 통과, 넘으면 멈춘다."""
    n = 100
    at_limit = int(n * MAX_UNRANKED_RATIO)
    ok = pd.DataFrame({"MarketCapSort": [1.0] * (n - at_limit) + [float("nan")] * at_limit})
    check_market_cap_coverage(ok, "TEST")  # 5/100 = 기준과 같음 → 통과
    over = pd.DataFrame(
        {"MarketCapSort": [1.0] * (n - at_limit - 1) + [float("nan")] * (at_limit + 1)}
    )
    with pytest.raises(MarketCapUnavailableError):
        check_market_cap_coverage(over, "TEST")


def test_few_unknown_caps_go_last(fake_fdr: FakeFdr, monkeypatch: pytest.MonkeyPatch) -> None:
    """기준 안의 미상 종목은 멈추지 않고 순위 맨 뒤로 간다(시총 큰 종목을 밀어내지 않는다)."""
    df = _listing()
    filler = pd.DataFrame(
        {
            "Code": [f"8{i:05d}" for i in range(40)],
            "Name": [f"채움{i}" for i in range(40)],
            "Market": ["KOSPI"] * 40,
            "Marcap": [float("nan")] * 40,
            "Close": [10.0] * 40,
            "Stocks": [10.0] * 40,
        }
    )
    df = pd.concat([df, filler], ignore_index=True)
    df.loc[df["Code"] == "003490", "Stocks"] = float("nan")  # 1/46 미상
    _use_listing(monkeypatch, fake_fdr, df)

    kospi = KoreanStockDataProvider.get_top_stocks_by_market_cap("KOSPI", 400)
    assert kospi["Code"].tolist()[:5] == ["005930", "000660", "035420", "051910", "012800"]
    assert kospi["Code"].tolist()[-1] == "003490"


def test_screening_stops_before_deleting_existing_results(
    fake_fdr: FakeFdr, fresh_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """소스 장애 날에 전날 돌려 둔 같은 날짜 결과를 지우고 빈손으로 끝나지 않는다."""
    screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    before = query_rows(fresh_db, "SELECT * FROM screening_results")
    assert before

    df = _listing()
    df["Stocks"] = float("nan")
    _use_listing(monkeypatch, fake_fdr, df)
    with pytest.raises(MarketCapUnavailableError):
        screening_core.run_and_save_screening(SCREEN_DATE_COMPACT)
    assert query_rows(fresh_db, "SELECT * FROM screening_results") == before


def test_collect_all_keeps_investor_data_and_exits_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[str] = []

    def broken_screening(target_date: str | None = None) -> None:
        raise MarketCapUnavailableError("KOSPI 시가총액 미상")

    monkeypatch.setattr(collect_data, "run_and_save_screening", broken_screening)
    monkeypatch.setattr(
        collect_data, "run_and_save_investor_data", lambda target_date=None: ran.append("investor")
    )
    monkeypatch.setattr(sys, "argv", ["collect_data.py", "--all"])

    with pytest.raises(SystemExit) as exc:
        collect_data.main()
    assert exc.value.code == 1
    assert ran == ["investor"]


def test_data_provider_module_constants() -> None:
    assert data_provider.MARKET_SEGMENTS["KOSDAQ"] == ("KOSDAQ", "KOSDAQ GLOBAL")
    assert "KONEX" not in data_provider.MARKET_SEGMENTS["KOSDAQ"]
