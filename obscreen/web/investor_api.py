"""외국인·기관 수급 화면과 API — `/investor` · `/api/investor-*`(OBS-4: 그대로 옮김)."""

from __future__ import annotations

import sqlite3

from flask import Blueprint, jsonify, render_template, request
from flask.typing import ResponseReturnValue

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
        if not date:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT scan_date FROM investor_trading ORDER BY scan_date DESC LIMIT 1"
            )
            row = cursor.fetchone()
            date = row["scan_date"] if row else None

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
        return jsonify({"success": True, "dates": dates})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
    finally:
        conn.close()
