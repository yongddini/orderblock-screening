"""스크리닝 결과 API — `/api/screening/*` · `/api/stock/*` · `/api/market-indices`.

OBS-4: `app_production.py`에서 본문을 그대로 옮겼다. DB 경로는 `deps.db_path()`가 조합 루트
(`obscreen.web.app.DB_PATH`)에서 호출 시점에 읽는다.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pandas as pd
from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

from obscreen.config import KST
from obscreen.web import deps

bp = Blueprint("screening_api", __name__)


@bp.route("/api/screening/dates")
def get_available_dates() -> ResponseReturnValue:
    """Available screening dates"""
    conn = sqlite3.connect(deps.db_path())
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT scan_date, COUNT(*) as count
        FROM screening_results
        GROUP BY scan_date
        ORDER BY scan_date DESC
        LIMIT 30
    """)
    # Convert YYYYMMDD integer to YYYY-MM-DD string
    dates = []
    for row in cursor.fetchall():
        date_str = row[0]  # YYYYMMDD string
        # YYYYMMDD -> YYYY-MM-DD
        formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        dates.append({"date": formatted_date, "count": row[1]})
    conn.close()

    return jsonify({"success": True, "dates": dates})


@bp.route("/api/market-indices")
def get_market_indices() -> ResponseReturnValue:
    """코스피/코스닥 실시간 지수 정보

    Returns:
        JSON: {
            success: bool,
            kospi: {value, change, change_percent, chart},
            kosdaq: {value, change, change_percent, chart}
        }
    """
    try:
        from datetime import timedelta

        from pykrx import stock

        today = datetime.now(KST).date()

        # 최근 30일 데이터 (차트용)
        start_date = (today - timedelta(days=30)).strftime("%Y%m%d")
        end_date = today.strftime("%Y%m%d")

        # 코스피 지수 (1001)
        kospi_df = stock.get_index_ohlcv(start_date, end_date, "1001")

        # 코스닥 지수 (2001)
        kosdaq_df = stock.get_index_ohlcv(start_date, end_date, "2001")

        if kospi_df is None or len(kospi_df) == 0 or kosdaq_df is None or len(kosdaq_df) == 0:
            return jsonify({"success": False, "error": "No market data available"}), 404

        # 최신 데이터 (오늘)
        kospi_latest = kospi_df.iloc[-1]
        kospi_prev = kospi_df.iloc[-2] if len(kospi_df) > 1 else kospi_latest

        kosdaq_latest = kosdaq_df.iloc[-1]
        kosdaq_prev = kosdaq_df.iloc[-2] if len(kosdaq_df) > 1 else kosdaq_latest

        # 변동률 계산
        kospi_change = kospi_latest["종가"] - kospi_prev["종가"]
        kospi_change_percent = (kospi_change / kospi_prev["종가"]) * 100

        kosdaq_change = kosdaq_latest["종가"] - kosdaq_prev["종가"]
        kosdaq_change_percent = (kosdaq_change / kosdaq_prev["종가"]) * 100

        # 최근 7일 차트 데이터
        kospi_chart = kospi_df.tail(7)["종가"].tolist()
        kosdaq_chart = kosdaq_df.tail(7)["종가"].tolist()

        return jsonify(
            {
                "success": True,
                "kospi": {
                    "value": float(kospi_latest["종가"]),
                    "change": float(kospi_change),
                    "change_percent": float(kospi_change_percent),
                    "chart": kospi_chart,
                },
                "kosdaq": {
                    "value": float(kosdaq_latest["종가"]),
                    "change": float(kosdaq_change),
                    "change_percent": float(kosdaq_change_percent),
                    "chart": kosdaq_chart,
                },
            }
        )

    except Exception as e:
        print(f"Market indices error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route("/api/screening/recommended")
def get_recommended_stocks() -> ResponseReturnValue:
    """Recommended stocks for selected date"""
    date_param = request.args.get("date")

    if date_param:
        # YYYY-MM-DD -> YYYYMMDD
        date_str = date_param.replace("-", "") if "-" in date_param else date_param
    else:
        # If no parameter, use today
        today = datetime.now(KST).date()
        date_str = today.strftime("%Y%m%d")

    conn = sqlite3.connect(deps.db_path())
    df = pd.read_sql_query(
        """
        SELECT * FROM screening_results 
        WHERE scan_date = ? AND is_recommended > 0 AND timeframe = 'daily'
        ORDER BY 
            rsi ASC,
            CASE market 
                WHEN 'KOSPI' THEN 1 
                WHEN 'KOSDAQ' THEN 2 
                WHEN 'ETF' THEN 3 
                ELSE 4 
            END,
            trading_value DESC
    """,
        conn,
        params=(date_str,),
    )
    conn.close()

    if df.empty:
        return jsonify({"success": False, "message": "No recommended stocks"})

    return jsonify({"success": True, "count": len(df), "results": df.to_dict("records")})


@bp.route("/api/screening/today")
def get_today_screening() -> ResponseReturnValue:
    today = datetime.now(KST).date()
    today_str = today.strftime("%Y%m%d")
    return get_screening_by_date(today_str)


@bp.route("/api/screening/<date>")
def get_screening_by_date(date: str) -> ResponseReturnValue:
    """Screening results for specific date"""
    # Handle YYYYMMDD or YYYY-MM-DD format
    # YYYY-MM-DD -> YYYYMMDD
    date_str = date.replace("-", "") if "-" in date else date

    # timeframe parameter (daily or weekly)
    timeframe = request.args.get("timeframe", "daily")

    conn = sqlite3.connect(deps.db_path())
    df = pd.read_sql_query(
        """
        SELECT * FROM screening_results 
        WHERE scan_date = ? AND timeframe = ?
        ORDER BY 
            CASE market 
                WHEN 'KOSPI' THEN 1 
                WHEN 'KOSDAQ' THEN 2 
                WHEN 'ETF' THEN 3 
                ELSE 4 
            END,
            trading_value DESC
    """,
        conn,
        params=(date_str, timeframe),
    )
    conn.close()

    if df.empty:
        return jsonify({"success": False, "message": f"No screening results for {date}"})

    stats = {
        "total": len(df),
        "zone_type": df["zone_type"].value_counts().to_dict(),
        "zone_position": df["zone_position"].value_counts().to_dict(),
        "markets": df["market"].value_counts().to_dict(),
    }

    return jsonify(
        {"success": True, "scan_date": date, "stats": stats, "results": df.to_dict("records")}
    )


@bp.route("/api/stock/<ticker>")
def get_stock_info(ticker: str) -> ResponseReturnValue:
    # Get date parameter (use today if not provided)
    date_param = request.args.get("date")

    if date_param:
        # YYYY-MM-DD -> YYYYMMDD
        date_str = date_param.replace("-", "") if "-" in date_param else date_param
    else:
        # If no parameter, use today
        today = datetime.now(KST).date()
        date_str = today.strftime("%Y%m%d")

    conn = sqlite3.connect(deps.db_path())
    df = pd.read_sql_query(
        """
        SELECT * FROM screening_results 
        WHERE scan_date = ? AND code = ?
    """,
        conn,
        params=(date_str, ticker),
    )
    conn.close()

    if not df.empty:
        stock = df.iloc[0].to_dict()
        return jsonify({"success": True, "stock": stock})
    else:
        return jsonify({"success": False, "message": "Stock not found"})
