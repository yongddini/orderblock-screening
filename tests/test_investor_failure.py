"""수급 수집 실패가 조용히 묻히지 않는다(OBS-11).

옛 코드는 (1) 수집 실패를 `print`만 하고 정상 반환해 `collect`가 「완료」·종료 코드 0으로 끝났고,
(2) 가져오기 전에 그날 행을 지웠으며, (3) 화면은 가장 최근에 **남아 있는** 날짜를 오류 없이
보여 줬다. 그래서 두 달 동안 수집이 죽어 있는 것을 아무도 몰랐다. 외부 API는 부르지 않는다.
"""

from __future__ import annotations

import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

import app_production
import collect_data
import screening_core
from obscreen.screening.core import InvestorDataError
from obscreen.web import investor_api
from tests.conftest import query_rows
from tests.support import SCREEN_DATE_COMPACT, FakePykrxStock

_SECRET = "pw-should-never-be-printed"


class _LoginFailedStock(FakePykrxStock):
    """KRX 로그인이 안 됐을 때의 pykrx — 예외 없이 열 없는 빈 표를 준다."""

    def get_market_net_purchases_of_equities(
        self, fromdate: str, todate: str, market: str, investor: str
    ) -> pd.DataFrame:
        return pd.DataFrame()


class _RaisingStock(FakePykrxStock):
    def get_market_net_purchases_of_equities(
        self, fromdate: str, todate: str, market: str, investor: str
    ) -> pd.DataFrame:
        raise KeyError("순매수거래대금")


def _install(monkeypatch: pytest.MonkeyPatch, stock: object, *, session: object = None) -> None:
    """`from pykrx import stock`과 pykrx 로그인 상태(`_auth_session`)를 가짜로 바꾼다."""
    module = types.ModuleType("pykrx")
    module.stock = stock  # type: ignore[attr-defined]
    auth = types.ModuleType("pykrx.website.comm.auth")
    auth._auth_session = session  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pykrx", module)
    monkeypatch.setitem(sys.modules, "pykrx.website", types.ModuleType("pykrx.website"))
    monkeypatch.setitem(sys.modules, "pykrx.website.comm", types.ModuleType("pykrx.website.comm"))
    monkeypatch.setitem(sys.modules, "pykrx.website.comm.auth", auth)
    monkeypatch.setenv("KRX_ID", "someone")
    monkeypatch.setenv("KRX_PW", _SECRET)


def _rows(db: Path) -> list[dict[str, object]]:
    return query_rows(
        db,
        "SELECT scan_date, investor_type, trade_type, rank, code, net_amount "
        "FROM investor_trading ORDER BY investor_type, trade_type, rank",
    )


@pytest.mark.parametrize("broken", [_LoginFailedStock, _RaisingStock])
def test_fetch_failure_raises_and_keeps_existing_rows(
    broken: type[FakePykrxStock],
    fake_pykrx: FakePykrxStock,
    fresh_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    before = _rows(fresh_db)
    assert before
    capsys.readouterr()  # 위 성공 수집의 출력은 버린다

    _install(monkeypatch, broken())
    with pytest.raises(InvestorDataError):
        screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)

    # 옛 코드는 가져오기 전에 그날 행을 지워 여기서 빈 표가 됐다.
    assert _rows(fresh_db) == before

    captured = capsys.readouterr()
    login_lines = [ln for ln in captured.err.splitlines() if "[KRX 로그인 실패]" in ln]
    assert len(login_lines) == 1  # 로그에서 한 줄로 알아본다
    assert SCREEN_DATE_COMPACT in login_lines[0]
    assert _SECRET not in captured.out + captured.err  # 비밀 값은 찍지 않는다
    assert "someone" not in captured.out + captured.err
    assert "DB 저장 완료" not in captured.out


def test_login_line_names_missing_env_without_values(
    fresh_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _install(monkeypatch, _LoginFailedStock())
    monkeypatch.delenv("KRX_PW")
    with pytest.raises(InvestorDataError):
        screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    err = capsys.readouterr().err
    assert "[KRX 로그인 실패]" in err
    assert "KRX_PW 없음" in err


def test_no_login_line_when_session_is_alive(
    fresh_db: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # 로그인은 됐는데 응답이 빈 경우(휴장일 등) — 로그인 탓으로 돌리지 않되 실패는 올린다.
    _install(monkeypatch, _LoginFailedStock(), session=object())
    with pytest.raises(InvestorDataError):
        screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    err = capsys.readouterr().err
    assert "[KRX 로그인 실패]" not in err
    assert "[수급 수집 실패]" in err


def test_zero_saved_rows_rolls_back(
    fake_pykrx: FakePykrxStock, fresh_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    before = _rows(fresh_db)

    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("모든 행 저장 실패")

    monkeypatch.setattr(screening_core, "_resolve_investor_name", boom)
    with pytest.raises(InvestorDataError):
        screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    assert _rows(fresh_db) == before


@pytest.mark.parametrize("mode", ["--all", "--investor", None])
def test_collect_exits_nonzero_and_does_not_say_done(
    mode: str | None,
    fresh_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """진짜 수집 함수 + 실패하는 가짜 pykrx — `collect`가 종료 코드 1이고 「완료」를 안 찍는다."""
    _install(monkeypatch, _LoginFailedStock())
    ran: list[str] = []
    monkeypatch.setattr(
        collect_data, "run_and_save_screening", lambda target_date=None: ran.append("screening")
    )
    argv = [SCREEN_DATE_COMPACT] if mode is None else [mode, SCREEN_DATE_COMPACT]

    with pytest.raises(SystemExit) as exc:
        collect_data.main(argv)
    assert exc.value.code == 1

    captured = capsys.readouterr()
    assert "완료" not in captured.out  # 「✅ 전체 수집 완료」·「수집 완료」 둘 다 없어야 한다
    assert "수급 수집 실패" in captured.err
    if mode != "--investor":
        assert ran == ["screening"]  # 스크리닝은 성공했어도 종료 코드는 1


def test_collect_success_still_exits_zero(
    fake_pykrx: FakePykrxStock,
    fresh_db: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(collect_data, "run_and_save_screening", lambda target_date=None: None)
    collect_data.main(["--all", SCREEN_DATE_COMPACT])  # SystemExit 없음 = 종료 코드 0
    assert "✅ 전체 수집 완료!" in capsys.readouterr().out


def test_cli_propagates_exit_code(fresh_db: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from obscreen import cli

    _install(monkeypatch, _LoginFailedStock())
    with pytest.raises(SystemExit) as exc:
        cli.main(["collect", "--investor", SCREEN_DATE_COMPACT])
    assert exc.value.code == 1


# --- 화면·API 경고 ---------------------------------------------------------------------


def _pin_today(monkeypatch: pytest.MonkeyPatch, today: date) -> None:
    monkeypatch.setattr(investor_api, "_today", lambda: today)


def test_api_is_fresh_the_day_after_collection(
    fake_pykrx: FakePykrxStock, fresh_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)  # 2026-09-14(월)
    client = app_production.app.test_client()
    # 같은 날 낮(아직 오늘 치 수집 전)과 다음 날 — 둘 다 경고가 없어야 한다.
    for today in (date(2026, 9, 14), date(2026, 9, 15)):
        _pin_today(monkeypatch, today)
        fresh = client.get("/api/investor-trading").get_json()["freshness"]
        assert fresh["stale"] is False
        assert fresh["warning"] is None
        assert fresh["latest_date"] == SCREEN_DATE_COMPACT
        assert fresh["missed_business_days"] == 0


def test_api_warns_when_data_is_old(
    fake_pykrx: FakePykrxStock, fresh_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    screening_core.run_and_save_investor_data(SCREEN_DATE_COMPACT)
    _pin_today(monkeypatch, date(2026, 9, 30))  # 수집이 2주 넘게 멈춘 상황
    client = app_production.app.test_client()

    body = client.get("/api/investor-trading").get_json()
    assert body["date"] == SCREEN_DATE_COMPACT  # 옛 데이터를 보여 주되
    fresh = body["freshness"]
    assert fresh["stale"] is True  # 조용히 보여 주지는 않는다
    assert fresh["latest_date"] == SCREEN_DATE_COMPACT
    assert fresh["expected_date"] == "20260929"
    assert fresh["today"] == "20260930"
    assert fresh["days_old"] == 16
    # 9/15~9/29의 평일 11일 중 추석 연휴(9/24·9/25)를 뺀 9일.
    assert fresh["missed_business_days"] == 9
    assert fresh["warning"].startswith("16일 전 데이터 — 수집 실패 중일 수 있음")

    # 날짜를 콕 집어 조회해도, 날짜 목록 API에서도 같은 근거가 실린다.
    dated = client.get(f"/api/investor-trading?date={SCREEN_DATE_COMPACT}").get_json()
    assert dated["freshness"] == fresh
    assert client.get("/api/investor-dates").get_json()["freshness"] == fresh


def test_weekend_and_holiday_do_not_count_as_missed(monkeypatch: pytest.MonkeyPatch) -> None:
    # 금요일(9/18) 데이터를 월요일(9/21)에 보는 것은 정상이다.
    assert investor_api._freshness("20260918", date(2026, 9, 21))["stale"] is False
    # 추석 연휴 직전(9/23 수) 데이터를 연휴 뒤 월요일(9/28)에 보는 것도 정상이다.
    assert investor_api._freshness("20260923", date(2026, 9, 28))["stale"] is False
    assert investor_api._freshness("20260923", date(2026, 9, 29))["stale"] is True
    # 데이터가 아예 없거나 날짜가 깨졌으면 경고를 지어내지 않는다.
    assert investor_api._freshness(None, date(2026, 9, 28))["stale"] is False
    assert investor_api._freshness("깨진날짜", date(2026, 9, 28))["stale"] is False


def test_investor_page_shows_server_warning() -> None:
    html = app_production.app.test_client().get("/investor").get_data(as_text=True)
    assert 'id="stale-warning"' in html
    assert 'role="alert"' in html
    # 화면은 서버가 계산한 경고 문구를 그대로 띄운다(판단 로직을 두 벌로 두지 않는다).
    assert "freshness.stale && freshness.warning" in html
    assert "staleBox.style.display = 'block'" in html
