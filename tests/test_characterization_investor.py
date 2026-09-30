"""수급(외국인·기관) 저장 로직 왕복 테스트(OBS-3 §4).

`screening_core.run_and_save_investor_data`에 pykrx 모양의 고정 입력을 먹여 임시 SQLite에
저장하고, 테이블과 API(`/api/investor-trading`·`/api/investor-dates`)로 다시 읽는다.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import app_production
import screening_core
from tests.conftest import query_rows
from tests.support import SCREEN_DATE_COMPACT, FakePykrxStock, assert_matches_snapshot

_COLUMNS = (
    "scan_date, investor_type, trade_type, rank, code, name, market, current_price, "
    "change_percent, buy_amount, sell_amount, net_amount, buy_volume, sell_volume, net_volume"
)


def _rows(db: Path) -> list[dict[str, object]]:
    return query_rows(
        db,
        f"SELECT {_COLUMNS} FROM investor_trading ORDER BY investor_type, trade_type, rank",
    )


def test_investor_roundtrip(fake_pykrx: FakePykrxStock, fresh_db: Path) -> None:
    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    rows = _rows(fresh_db)
    assert_matches_snapshot("investor_trading_rows", rows)

    by_key = {(r["investor_type"], r["trade_type"], r["rank"]): r for r in rows}
    # 순매수 상위 = 순매수거래대금 내림차순, 순매도 상위 = 오름차순(두 시장 합쳐서).
    # 종목명이 빈 행(777770)도 저장된다 — 티커로 이름을 조회해 채운다(OBS-9). 옛 코드는 그
    # 행을 조용히 빠뜨려 4위가 비어 있었다(OBS-3이 그 구멍을 고정해 뒀던 것을 뒤집었다).
    foreign_buy = [r for r in rows if r["investor_type"] == "foreign" and r["trade_type"] == "buy"]
    assert [(r["rank"], r["code"]) for r in foreign_buy] == [
        (1, "005930"),
        (2, "035420"),
        (3, "228670"),
        (4, "777770"),
        (5, "247540"),
        (6, "000660"),
    ]
    assert by_key[("foreign", "buy", 4)]["name"] == "이름조회-777770"
    # 순위는 1부터 빈틈없이 이어진다(네 분류 전부).
    for inv in ("foreign", "institution"):
        for side in ("buy", "sell"):
            ranks = [r["rank"] for r in rows if (r["investor_type"], r["trade_type"]) == (inv, side)]
            assert ranks == list(range(1, len(ranks) + 1))
    assert by_key[("foreign", "sell", 1)]["code"] == "000660"
    # 시세가 없는 종목은 현재가·등락률 0으로 저장된다.
    inst = {r["code"]: r for r in rows if r["investor_type"] == "institution"}
    assert inst["888880"]["current_price"] == 0
    assert inst["888880"]["change_percent"] == 0
    assert inst["086520"]["current_price"] == 100500
    # 같은 날짜로 다시 저장하면 그 날짜 행을 지우고 다시 쓴다(중복 없음).
    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    assert _rows(fresh_db) == rows


def test_investor_api_reads_back(fake_pykrx: FakePykrxStock, fresh_db: Path) -> None:
    client = app_production.app.test_client()
    # `investor_trading` 테이블은 `init_db()`가 아니라 첫 수집 때 만들어진다. 수집 전 조회는
    # 「데이터 없음」(404)이지 서버 오류(500 no such table)가 아니다(OBS-9 — OBS-3이 500을
    # 고정해 뒀던 것을 뒤집었다).
    assert not query_rows(
        fresh_db, "SELECT name FROM sqlite_master WHERE name = 'investor_trading'"
    )
    for url in ("/api/investor-trading", "/api/investor-trading?date=20260101"):
        empty = client.get(url)
        assert empty.status_code == 404
        assert empty.get_json() == {"success": False, "error": "No data available"}
    assert client.get("/api/investor-dates").get_json() == {"success": True, "dates": []}

    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)

    dates = client.get("/api/investor-dates").get_json()
    assert dates == {"success": True, "dates": [SCREEN_DATE_COMPACT]}

    res = client.get("/api/investor-trading?investor_type=institution&trade_type=sell&limit=2")
    body = res.get_json()
    assert res.status_code == 200
    assert body["date"] == SCREEN_DATE_COMPACT
    assert body["count"] == 2
    assert [d["rank"] for d in body["data"]] == [1, 2]
    assert set(body["data"][0]) == {
        "rank",
        "code",
        "name",
        "market",
        "current_price",
        "change_percent",
        "buy_amount",
        "sell_amount",
        "net_amount",
        "buy_volume",
        "sell_volume",
        "net_volume",
    }
    assert_matches_snapshot("api_investor_trading_institution_sell", body)


class _NameLookup:
    def __init__(self, result: object) -> None:
        self.result = result

    def get_market_ticker_name(self, ticker: str) -> object:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize(
    ("raw", "lookup", "expected"),
    [
        ("삼성전자", "무시됨", "삼성전자"),
        ("  삼성전자 ", "무시됨", "삼성전자"),
        (float("nan"), "조회된이름", "조회된이름"),  # pandas 빈 칸 = NaN (옛 버그 경로)
        (None, "조회된이름", "조회된이름"),
        ("", "조회된이름", "조회된이름"),
        ("   ", "조회된이름", "조회된이름"),
        (float("nan"), RuntimeError("pykrx down"), "123450"),
        (float("nan"), "", "123450"),
        (float("nan"), pd.DataFrame(), "123450"),  # pykrx는 모르는 티커에 빈 DataFrame을 낸다
    ],
)
def test_resolve_investor_name(raw: object, lookup: object, expected: str) -> None:
    assert screening_core._resolve_investor_name(_NameLookup(lookup), "123450", raw) == expected
