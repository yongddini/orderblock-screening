"""외국인·기관 수급 화면과 API — `/investor` · `/api/investor-*`(OBS-4: 그대로 옮김)."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta
from typing import Any

from flask import Blueprint, jsonify, render_template, request
from flask.typing import ResponseReturnValue

from obscreen.config import KST
from obscreen.web import deps

bp = Blueprint("investor_api", __name__)


@bp.route("/investor")
def investor_page() -> ResponseReturnValue:
    """Foreign/Institution investor trading page"""
    return render_template("investor.html")


def _investor_table_exists(conn: sqlite3.Connection) -> bool:
    """`investor_trading`은 init_db()가 아니라 첫 수급 수집 때 만들어진다 — 수집 전 조회는
    「데이터 없음」이지 서버 오류가 아니다(OBS-9)."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'investor_trading'"
    ).fetchone()
    return row is not None


def _today() -> date:
    """KST 기준 오늘 — 테스트가 바꿔 끼운다."""
    return datetime.now(KST).date()


def _is_business_day(day: date) -> bool:
    if day.weekday() >= 5:
        return False
    try:
        import holidays
    except ImportError:  # pragma: no cover — 의존성에 있다
        return True
    # `bool(holidays 객체)`로 거르지 말 것 — 연도를 지연 생성해 막 만든 객체는 비어 있다(OBS-9).
    return day not in holidays.country_holidays("KR")


def _expected_date(today: date) -> date:
    """수급 데이터가 **적어도** 있어야 하는 날 = 오늘보다 앞선 가장 최근 영업일.

    수집은 장 마감 뒤(20:05 KST)에 돌므로 오늘 치는 낮에는 없는 게 정상이다. 그래서 오늘이
    아니라 직전 영업일을 기준으로 삼는다(낮 동안 거짓 경고를 띄우지 않는다).
    """
    day = today - timedelta(days=1)
    for _ in range(30):
        if _is_business_day(day):
            return day
        day -= timedelta(days=1)
    return day


def _freshness(latest: str | None, today: date) -> dict[str, Any]:
    """수급 데이터가 얼마나 오래됐는지(OBS-11) — 화면 경고의 판단 근거.

    `latest`는 DB에 있는 가장 최근 기준일(YYYYMMDD). 수집이 계속 실패하면 화면은 옛 날짜를
    오류 없이 보여 주므로, 그 사실을 응답에 실어 화면이 경고를 띄우게 한다.
    """
    expected = _expected_date(today)
    info: dict[str, Any] = {
        "latest_date": latest,
        "expected_date": expected.strftime("%Y%m%d"),
        "today": today.strftime("%Y%m%d"),
        "days_old": None,
        "missed_business_days": None,
        "stale": False,
        "warning": None,
    }
    if latest is None:
        return info
    try:
        latest_day = datetime.strptime(latest, "%Y%m%d").date()
    except ValueError:
        return info
    missed = 0
    day = latest_day + timedelta(days=1)
    while day <= expected:
        if _is_business_day(day):
            missed += 1
        day += timedelta(days=1)
    info["days_old"] = (today - latest_day).days
    info["missed_business_days"] = missed
    if missed > 0:
        info["stale"] = True
        info["warning"] = (
            f"{info['days_old']}일 전 데이터 — 수집 실패 중일 수 있음 "
            f"(최근 영업일 {expected.isoformat()} · 빠진 영업일 {missed}일)"
        )
    return info


def _latest_scan_date(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT scan_date FROM investor_trading ORDER BY scan_date DESC LIMIT 1"
    ).fetchone()
    return str(row[0]) if row else None


@bp.route("/api/investor-trading")
def get_investor_trading() -> ResponseReturnValue:
    """Get investor trading data API

    Query parameters:
        date (str): Date (YYYYMMDD) - optional
        investor_type (str): foreign | institution
        trade_type (str): buy | sell
        limit (int): Number of results (default 100)
    """
    date = request.args.get("date")
    investor_type = request.args.get("investor_type", "foreign")
    trade_type = request.args.get("trade_type", "buy")
    limit = int(request.args.get("limit", 100))

    conn = sqlite3.connect(deps.db_path())
    conn.row_factory = sqlite3.Row

    try:
        if not _investor_table_exists(conn):
            return jsonify({"success": False, "error": "No data available"}), 404

        # If no date specified, get latest date
        latest = _latest_scan_date(conn)
        if not date:
            date = latest

        if not date:
            return jsonify({"success": False, "error": "No data available"}), 404

        # Query data
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT rank, code, name, market, current_price, change_percent,
                   buy_amount, sell_amount, net_amount,
                   buy_volume, sell_volume, net_volume
            FROM investor_trading
            WHERE scan_date = ? AND investor_type = ? AND trade_type = ?
            ORDER BY rank LIMIT ?
        """,
            (date, investor_type, trade_type, limit),
        )

        data = [dict(row) for row in cursor.fetchall()]

        return jsonify(
            {
                "success": True,
                "date": date,
                "investor_type": investor_type,
                "trade_type": trade_type,
                "count": len(data),
                "data": data,
                # 가장 최근 수집일이 직전 영업일보다 오래됐으면 stale=True(OBS-11).
                "freshness": _freshness(latest, _today()),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()


@bp.route("/api/investor-dates")
def get_investor_dates() -> ResponseReturnValue:
    """Get available investor trading dates"""
    conn = sqlite3.connect(deps.db_path())
    conn.row_factory = sqlite3.Row

    try:
        if not _investor_table_exists(conn):
            return jsonify({"success": True, "dates": []})
        cursor = conn.cursor()
        cursor.execute(
            "SELECT DISTINCT scan_date FROM investor_trading ORDER BY scan_date DESC LIMIT 30"
        )
        dates = [row["scan_date"] for row in cursor.fetchall()]
        return jsonify(
            {
                "success": True,
                "dates": dates,
                "freshness": _freshness(dates[0] if dates else None, _today()),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()
