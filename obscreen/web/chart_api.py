"""차트 데이터 API — `/api/chart-data/*`(Lightweight Charts, OBS-10).

OBS-4: 본문을 그대로 옮겼다. 탐지기 클래스는 `deps.detector_class()`가 조합 루트
(`obscreen.web.app.RealtimeOrderBlockDetector`)에서 호출 시점에 읽는다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from flask import Blueprint, jsonify, request
from flask.typing import ResponseReturnValue

from obscreen.config import get_settings
from obscreen.data.provider import KoreanStockDataProvider
from obscreen.web import deps

bp = Blueprint("chart_api", __name__)


@bp.route("/api/chart-data/<ticker>")
def get_chart_data(ticker: str) -> ResponseReturnValue:
    """Daily chart data (for Lightweight Charts)"""
    date_param = request.args.get("date")
    end_date = None

    if date_param:
        if "-" in date_param:
            end_date = date_param
        else:
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"

    try:
        df = KoreanStockDataProvider.get_price_data(ticker, days=500, end_date=end_date)

        if df is None or len(df) < 50:
            return jsonify({"success": False, "message": "No data"})

        df = df[df["Volume"] > 0].copy()

        if len(df) < 50:
            return jsonify({"success": False, "message": "Insufficient data"})

        # Calculate RSI
        delta = df["Close"].diff()

        def rma(series: pd.Series, length: int) -> pd.Series:
            alpha = 1.0 / length
            result = series.copy()
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
            return result

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = rma(gain, 14)
        avg_loss = rma(loss, 14)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # RSI EMA
        rsi_ema = rsi.ewm(span=14, adjust=False).mean()

        # Orderblock detection
        settings = get_settings()
        detector = deps.detector_class()(
            swing_length=settings.swing_length,
            max_atr_mult=settings.max_atr_mult,
            ob_end_method=settings.ob_end_method,
            combine_obs=settings.combine_obs,
            max_order_blocks=settings.chart_max_order_blocks,
        )

        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()

        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]

        # Lightweight Charts format
        candles = []
        for date, row in df.iterrows():
            candles.append(
                {
                    "time": int(date.timestamp()),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )

        rsi_data = []
        for date, value in rsi.items():
            if pd.notna(value):
                rsi_data.append({"time": int(date.timestamp()), "value": float(value)})

        rsi_ema_data = []
        for date, value in rsi_ema.items():
            if pd.notna(value):
                rsi_ema_data.append({"time": int(date.timestamp()), "value": float(value)})

        orderblocks: dict[str, list[dict[str, Any]]] = {"bull": [], "bear": []}

        for ob in bull_obs:
            orderblocks["bull"].append(
                {
                    "top": float(ob.top),
                    "bottom": float(ob.bottom),
                    "start_time": int(ob.start_time.timestamp()) if ob.start_time else None,
                    "break_time": int(ob.break_time.timestamp()) if ob.break_time else None,
                    "breaker": ob.breaker,
                    "combined": getattr(ob, "combined", False),
                }
            )

        for ob in bear_obs:
            orderblocks["bear"].append(
                {
                    "top": float(ob.top),
                    "bottom": float(ob.bottom),
                    "start_time": int(ob.start_time.timestamp()) if ob.start_time else None,
                    "break_time": int(ob.break_time.timestamp()) if ob.break_time else None,
                    "breaker": ob.breaker,
                    "combined": getattr(ob, "combined", False),
                }
            )

        return jsonify(
            {
                "success": True,
                "data": {
                    "candles": candles,
                    "rsi": rsi_data,
                    "rsi_ema": rsi_ema_data,
                    "orderblocks": orderblocks,
                },
            }
        )

    except Exception as e:
        print(f"Chart data error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@bp.route("/api/chart-data-weekly/<ticker>")
def get_chart_data_weekly(ticker: str) -> ResponseReturnValue:
    """Weekly chart data (for Lightweight Charts)"""
    date_param = request.args.get("date")
    end_date = None

    if date_param:
        if "-" in date_param:
            end_date = date_param
        else:
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"

    try:
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks=500, end_date=end_date)

        if df is None or len(df) < 50:
            return jsonify({"success": False, "message": "No data"})

        df = df[df["Volume"] > 0].copy()

        if len(df) < 50:
            return jsonify({"success": False, "message": "Insufficient data"})

        # Calculate RSI
        delta = df["Close"].diff()

        def rma(series: pd.Series, length: int) -> pd.Series:
            alpha = 1.0 / length
            result = series.copy()
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i - 1]
            return result

        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = rma(gain, 14)
        avg_loss = rma(loss, 14)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # RSI EMA
        rsi_ema = rsi.ewm(span=14, adjust=False).mean()

        # Orderblock detection
        settings = get_settings()
        detector = deps.detector_class()(
            swing_length=settings.swing_length,
            max_atr_mult=settings.max_atr_mult,
            ob_end_method=settings.ob_end_method,
            combine_obs=settings.combine_obs,
            max_order_blocks=settings.chart_max_order_blocks,
        )

        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()

        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]

        # Lightweight Charts format
        candles = []
        for date, row in df.iterrows():
            candles.append(
                {
                    "time": int(date.timestamp()),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )

        rsi_data = []
        for date, value in rsi.items():
            if pd.notna(value):
                rsi_data.append({"time": int(date.timestamp()), "value": float(value)})

        rsi_ema_data = []
        for date, value in rsi_ema.items():
            if pd.notna(value):
                rsi_ema_data.append({"time": int(date.timestamp()), "value": float(value)})

        orderblocks: dict[str, list[dict[str, Any]]] = {"bull": [], "bear": []}

        for ob in bull_obs:
            orderblocks["bull"].append(
                {
                    "top": float(ob.top),
                    "bottom": float(ob.bottom),
                    "start_time": int(ob.start_time.timestamp()) if ob.start_time else None,
                    "break_time": int(ob.break_time.timestamp()) if ob.break_time else None,
                    "breaker": ob.breaker,
                    "combined": getattr(ob, "combined", False),
                }
            )

        for ob in bear_obs:
            orderblocks["bear"].append(
                {
                    "top": float(ob.top),
                    "bottom": float(ob.bottom),
                    "start_time": int(ob.start_time.timestamp()) if ob.start_time else None,
                    "break_time": int(ob.break_time.timestamp()) if ob.break_time else None,
                    "breaker": ob.breaker,
                    "combined": getattr(ob, "combined", False),
                }
            )

        return jsonify(
            {
                "success": True,
                "data": {
                    "candles": candles,
                    "rsi": rsi_data,
                    "rsi_ema": rsi_ema_data,
                    "orderblocks": orderblocks,
                },
            }
        )

    except Exception as e:
        print(f"Weekly chart data error: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})
