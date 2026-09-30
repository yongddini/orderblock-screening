"""실험·구버전 라우트 — 화면이 쓰지 않는 것들을 한곳에 격리했다(OBS-4).

- plotly 차트(`/api/chart/*`, `create_chart_html*`) — 화면은 OBS-10 이후 `/api/chart-data/*`만 쓴다.
- `/chart-test` · `/ob-comparison` — 템플릿이 저장소에 없어 호출하면 500이다.
- `/api/compare-ob-methods/*` · `/api/test-chart/*` — 실험용 JSON.

설정 `OBSCREEN_EXPERIMENTAL_ROUTES`(기본 켬 = 지금과 같음)로 등록 여부를 정한다. 지울지 끌지는
사용자 확인 사항이다(OBS-4 §3). 본문은 손대지 않고 옮겼다.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from flask import Blueprint, jsonify, render_template, request
from flask.typing import ResponseReturnValue

from obscreen.data.provider import KoreanStockDataProvider
from obscreen.web import deps

bp = Blueprint("experimental", __name__)


def create_chart_html(ticker: str, days: int = 500, end_date: str | None = None) -> str | None:
    """Create chart

    Args:
        ticker: Stock code
        days: Number of days to display
        end_date: Chart end date (YYYY-MM-DD). None for today
    """
    try:
        from plotly.subplots import make_subplots

        df = KoreanStockDataProvider.get_price_data(ticker, days)

        if df is None or len(df) < 50:
            return None

        # Remove trading halt periods (volume=0)
        df = df[df["Volume"] > 0].copy()

        # Filter by end_date if specified
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            except Exception:
                pass  # Use full data if date parsing fails

        if len(df) < 50:
            return None

        # Calculate RSI (14-day) - Pine Script method (RMA)
        delta = df["Close"].diff()

        # RMA (Wilder's Smoothing) calculation
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

        # RSI EMA(14)
        rsi_ema = rsi.ewm(span=14, adjust=False).mean()

        detector = deps.detector_class()(
            swing_length=10,
            max_atr_mult=2.0,
            ob_end_method="Wick",
            combine_obs=True,
            max_order_blocks=30,
        )

        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()

        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]

        date_to_idx = {date: idx for idx, date in enumerate(df.index)}

        # 2 subplots: price + RSI
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
            subplot_titles=("", "RSI (14)"),
        )

        # Candlestick hover text
        hover_texts = [
            f"Date: {date.strftime('%Y-%m-%d')}<br>"
            f"Open: {row['Open']:,.0f}<br>"
            f"High: {row['High']:,.0f}<br>"
            f"Low: {row['Low']:,.0f}<br>"
            f"Close: {row['Close']:,.0f}"
            for date, row in df.iterrows()
        ]

        fig.add_trace(
            go.Candlestick(
                x=list(range(len(df))),
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                text=hover_texts,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

        # Bull order blocks (Pine Script style transparency)
        for ob in bull_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue

            start_idx = date_to_idx[ob.start_time]
            end_idx = (
                date_to_idx[ob.break_time]
                if ob.break_time and ob.break_time in df.index
                else len(df) - 1
            )

            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = "#757575"
            elif getattr(ob, "combined", False):
                alpha = 0.73
                color = "#26a69a"
            else:
                alpha = 0.47
                color = "#26a69a"

            fig.add_shape(
                type="rect",
                x0=start_idx,
                x1=end_idx,
                y0=ob.bottom,
                y1=ob.top,
                fillcolor=color,
                opacity=1 - alpha,
                line=dict(color=color, width=1),
                row=1,
                col=1,
            )

        # Bear order blocks (Pine Script style transparency)
        for ob in bear_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue

            start_idx = date_to_idx[ob.start_time]
            end_idx = (
                date_to_idx[ob.break_time]
                if ob.break_time and ob.break_time in df.index
                else len(df) - 1
            )

            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = "#757575"
            elif getattr(ob, "combined", False):
                alpha = 0.73
                color = "#ef5350"
            else:
                alpha = 0.47
                color = "#ef5350"

            fig.add_shape(
                type="rect",
                x0=start_idx,
                x1=end_idx,
                y0=ob.bottom,
                y1=ob.top,
                fillcolor=color,
                opacity=1 - alpha,
                line=dict(color=color, width=1),
                row=1,
                col=1,
            )

        # RSI trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi,
                name="RSI",
                line=dict(color="#7E57C2", width=2),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # RSI EMA(14) trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi_ema,
                name="RSI EMA",
                line=dict(color="#FFEB3B", width=2),
                hovertemplate="RSI EMA: %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)

        tick_step = max(1, len(df) // 10)
        tickvals = list(range(0, len(df), tick_step))
        ticktext = [df.index[i].strftime("%Y-%m-%d") for i in tickvals]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45, row=2, col=1)

        fig.update_layout(
            height=1000,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#2d2d2d",
            font=dict(color="#e0e0e0"),
            dragmode="pan",
            # TradingView style interaction
            xaxis=dict(
                fixedrange=False,
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#888888",
                spikethickness=1,
                spikedash="dot",
            ),
            yaxis=dict(fixedrange=False, scaleanchor=None, showticklabels=True),
            yaxis2=dict(fixedrange=False, showticklabels=True),
        )

        # Price chart y-axis (right side, actual price)
        fig.update_xaxes(gridcolor="#3a3a3a", fixedrange=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(
            gridcolor="#3a3a3a",
            side="right",
            tickformat=",",
            hoverformat=",",
            separatethousands=True,
            fixedrange=False,
            automargin=True,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=1,
            col=1,
        )

        # RSI chart y-axis (right side)
        fig.update_xaxes(gridcolor="#3a3a3a", fixedrange=False, zeroline=False, row=2, col=1)
        fig.update_yaxes(
            gridcolor="#3a3a3a",
            range=[0, 100],
            side="right",
            fixedrange=False,
            automargin=True,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=2,
            col=1,
        )

        chart_json: str = fig.to_json()
        return chart_json

    except Exception as e:
        print(f"Chart creation error: {e}")
        return None


def create_chart_html_weekly(
    ticker: str, weeks: int = 500, end_date: str | None = None
) -> str | None:
    """Create weekly chart

    Args:
        ticker: Stock code
        weeks: Number of weeks to display
        end_date: Chart end date (YYYY-MM-DD)
    """
    try:
        from plotly.subplots import make_subplots

        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks, end_date)

        if df is None or len(df) < 50:
            return None

        # Remove volume 0
        df = df[df["Volume"] > 0].copy()

        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            except Exception:
                pass

        if len(df) < 50:
            return None

        # Calculate RSI (RMA method)
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
        rsi_ema = rsi.ewm(span=14, adjust=False).mean()

        # Weekly OB detection
        detector = deps.detector_class()(
            swing_length=10,
            max_atr_mult=2.0,
            ob_end_method="Wick",
            combine_obs=True,
            max_order_blocks=30,
        )

        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()

        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]

        date_to_idx = {date: idx for idx, date in enumerate(df.index)}

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
        )

        # Weekly candlestick hover text
        hover_texts = [
            f"Date: {date.strftime('%Y-%m-%d')}<br>"
            f"Open: {row['Open']:,.0f}<br>"
            f"High: {row['High']:,.0f}<br>"
            f"Low: {row['Low']:,.0f}<br>"
            f"Close: {row['Close']:,.0f}"
            for date, row in df.iterrows()
        ]

        # Candlestick (same colors as daily)
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(df))),
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Weekly",
                increasing_line_color="#26a69a",
                decreasing_line_color="#ef5350",
                text=hover_texts,
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )

        # Bull order blocks (same colors and transparency as daily)
        for ob in bull_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue

            start_idx = date_to_idx[ob.start_time]
            end_idx = (
                date_to_idx[ob.break_time]
                if ob.break_time and ob.break_time in df.index
                else len(df) - 1
            )

            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = "#757575"
            elif getattr(ob, "combined", False):
                alpha = 0.73
                color = "#26a69a"
            else:
                alpha = 0.47
                color = "#26a69a"

            fig.add_shape(
                type="rect",
                x0=start_idx,
                x1=end_idx,
                y0=ob.bottom,
                y1=ob.top,
                fillcolor=color,
                opacity=1 - alpha,
                line=dict(color=color, width=1),
                row=1,
                col=1,
            )

        # Bear order blocks (same colors and transparency as daily)
        for ob in bear_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue

            start_idx = date_to_idx[ob.start_time]
            end_idx = (
                date_to_idx[ob.break_time]
                if ob.break_time and ob.break_time in df.index
                else len(df) - 1
            )

            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = "#757575"
            elif getattr(ob, "combined", False):
                alpha = 0.73
                color = "#ef5350"
            else:
                alpha = 0.47
                color = "#ef5350"

            fig.add_shape(
                type="rect",
                x0=start_idx,
                x1=end_idx,
                y0=ob.bottom,
                y1=ob.top,
                fillcolor=color,
                opacity=1 - alpha,
                line=dict(color=color, width=1),
                row=1,
                col=1,
            )

        # RSI trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi,
                name="RSI",
                line=dict(color="#7E57C2", width=2),
                hovertemplate="RSI: %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi_ema,
                name="RSI EMA",
                line=dict(color="#FFEB3B", width=2),
                hovertemplate="RSI EMA: %{y:.1f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)

        tick_step = max(1, len(df) // 10)
        tickvals = list(range(0, len(df), tick_step))
        ticktext = [df.index[i].strftime("%Y-%m-%d") for i in tickvals]

        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45, row=2, col=1)

        fig.update_layout(
            height=1000,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            hovermode="x unified",
            plot_bgcolor="#1e1e1e",
            paper_bgcolor="#2d2d2d",
            font=dict(color="#e0e0e0"),
            dragmode="pan",
            # TradingView style interaction
            xaxis=dict(
                fixedrange=False,
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode="across",
                spikesnap="cursor",
                spikecolor="#888888",
                spikethickness=1,
                spikedash="dot",
            ),
            yaxis=dict(fixedrange=False, scaleanchor=None, showticklabels=True),
            yaxis2=dict(fixedrange=False, showticklabels=True),
        )

        # Price chart y-axis (right side, actual price)
        fig.update_xaxes(gridcolor="#3a3a3a", fixedrange=False, zeroline=False, row=1, col=1)
        fig.update_yaxes(
            gridcolor="#3a3a3a",
            side="right",
            tickformat=",",
            hoverformat=",",
            separatethousands=True,
            fixedrange=False,
            automargin=True,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=1,
            col=1,
        )

        # RSI chart y-axis
        fig.update_xaxes(gridcolor="#3a3a3a", fixedrange=False, zeroline=False, row=2, col=1)
        fig.update_yaxes(
            gridcolor="#3a3a3a",
            range=[0, 100],
            side="right",
            fixedrange=False,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=2,
            col=1,
        )

        chart_json: str = fig.to_json()
        return chart_json

    except Exception as e:
        print(f"Weekly chart creation error: {e}")
        return None


@bp.route("/api/chart/<ticker>")
def get_chart(ticker: str) -> ResponseReturnValue:
    # Get date parameter
    date_param = request.args.get("date")
    end_date = None

    if date_param:
        # Convert to YYYY-MM-DD format
        if "-" in date_param:
            end_date = date_param
        else:
            # YYYYMMDD -> YYYY-MM-DD
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"

    chart_json = create_chart_html(ticker, end_date=end_date)

    if chart_json:
        return jsonify({"success": True, "chart": chart_json})
    else:
        return jsonify({"success": False, "message": "Cannot create chart"})


@bp.route("/api/chart-weekly/<ticker>")
def get_chart_weekly(ticker: str) -> ResponseReturnValue:
    """Weekly chart data API"""
    date_param = request.args.get("date")
    end_date = None

    if date_param:
        if "-" in date_param:
            end_date = date_param
        else:
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"

    chart_json = create_chart_html_weekly(ticker, end_date=end_date)

    if chart_json:
        return jsonify({"success": True, "chart": chart_json})
    else:
        return jsonify({"success": False, "message": "Cannot create weekly chart"})


@bp.route("/chart-test")
def chart_test() -> ResponseReturnValue:
    """Chart test page"""
    return render_template("chart_test.html")


@bp.route("/ob-comparison")
def ob_comparison() -> ResponseReturnValue:
    """Order block method comparison page"""
    return render_template("ob_comparison.html")


@bp.route("/api/compare-ob-methods/<ticker>")
def compare_ob_methods(ticker: str) -> ResponseReturnValue:
    """Compare two OB generation methods"""
    weeks = int(request.args.get("weeks", 500))

    try:
        # Weekly data
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks)

        if df is None or len(df) == 0:
            return jsonify({"success": False, "message": "No data"})

        # Find swings
        swing_length = 10
        swing_lows = []

        for i in range(swing_length, len(df) - swing_length):
            current_low = df["Low"].iloc[i]
            left_lows = df["Low"].iloc[i - swing_length : i]
            right_lows = df["Low"].iloc[i + 1 : i + swing_length + 1]

            if all(current_low <= left_lows) and all(current_low <= right_lows):
                swing_lows.append({"index": i, "low": current_low})

        # Method 1: Current (candle before swing)
        current_obs = []
        for swing in swing_lows:
            ob_idx = swing["index"] - 1
            if ob_idx >= 0:
                ob_candle = df.iloc[ob_idx]
                current_obs.append(
                    {
                        "ob_idx": ob_idx,
                        "swing_idx": swing["index"],
                        "start_date": df.index[ob_idx].strftime("%Y-%m-%d"),
                        "top": float(ob_candle["High"]),
                        "bottom": float(ob_candle["Low"]),
                        "invalidated": False,
                    }
                )

        # Invalidation check (current method)
        for ob in current_obs:
            for i in range(ob["ob_idx"] + 1, len(df)):
                if df.iloc[i]["Low"] < ob["bottom"]:
                    ob["invalidated"] = True
                    ob["end_date"] = df.index[i].strftime("%Y-%m-%d")
                    break

        # Method 2: TradingView (lowest candle in range)
        tradingview_obs = []
        for i, swing in enumerate(swing_lows):
            swing_idx = swing["index"]

            # Search range: from swing to next swing or end
            search_end = swing_lows[i + 1]["index"] if i + 1 < len(swing_lows) else len(df)

            # Find lowest candle in range
            min_low = float("inf")
            ob_idx = swing_idx

            for j in range(swing_idx, search_end):
                if df.iloc[j]["Low"] < min_low:
                    min_low = df.iloc[j]["Low"]
                    ob_idx = j

            ob_candle = df.iloc[ob_idx]
            tradingview_obs.append(
                {
                    "ob_idx": ob_idx,
                    "swing_idx": swing_idx,
                    "start_date": df.index[ob_idx].strftime("%Y-%m-%d"),
                    "top": float(ob_candle["High"]),
                    "bottom": float(ob_candle["Low"]),
                    "invalidated": False,
                }
            )

        # Invalidation check (TradingView method)
        for ob in tradingview_obs:
            for i in range(ob["ob_idx"] + 1, len(df)):
                if df.iloc[i]["Low"] < ob["bottom"]:
                    ob["invalidated"] = True
                    ob["end_date"] = df.index[i].strftime("%Y-%m-%d")
                    break

        return jsonify(
            {
                "success": True,
                "ticker": ticker,
                "weekly_data": {
                    "dates": df.index.strftime("%Y-%m-%d").tolist(),
                    "open": df["Open"].tolist(),
                    "high": df["High"].tolist(),
                    "low": df["Low"].tolist(),
                    "close": df["Close"].tolist(),
                },
                "current_method": {
                    "name": "Current (Before Swing)",
                    "order_blocks": current_obs[-20:],  # Last 20
                },
                "tradingview_method": {
                    "name": "TradingView (Range Lowest)",
                    "order_blocks": tradingview_obs[-20:],  # Last 20
                },
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@bp.route("/api/test-chart/daily/<ticker>")
def test_chart_daily(ticker: str) -> ResponseReturnValue:
    """Daily chart data"""
    end_date = request.args.get("end_date")
    days = int(request.args.get("days", 500))

    try:
        df = KoreanStockDataProvider.get_price_data(ticker, days, end_date)

        if df is None or len(df) == 0:
            return jsonify({"success": False, "message": "No data"})

        # Calculate period
        period = f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}"
        years = (df.index[-1] - df.index[0]).days / 365.25

        return jsonify(
            {
                "success": True,
                "candle_count": len(df),
                "period": period,
                "years": f"{years:.1f}",
                "data": {
                    "dates": df.index.strftime("%Y-%m-%d").tolist(),
                    "open": df["Open"].tolist(),
                    "high": df["High"].tolist(),
                    "low": df["Low"].tolist(),
                    "close": df["Close"].tolist(),
                    "volume": df["Volume"].tolist(),
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})


@bp.route("/api/test-chart/weekly/<ticker>")
def test_chart_weekly(ticker: str) -> ResponseReturnValue:
    """Weekly chart data"""
    end_date = request.args.get("end_date")
    weeks = int(request.args.get("weeks", 500))

    try:
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks, end_date)

        if df is None or len(df) == 0:
            return jsonify({"success": False, "message": "No data"})

        # Calculate period
        period = f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}"
        years = (df.index[-1] - df.index[0]).days / 365.25

        return jsonify(
            {
                "success": True,
                "candle_count": len(df),
                "period": period,
                "years": f"{years:.1f}",
                "data": {
                    "dates": df.index.strftime("%Y-%m-%d").tolist(),
                    "open": df["Open"].tolist(),
                    "high": df["High"].tolist(),
                    "low": df["Low"].tolist(),
                    "close": df["Close"].tolist(),
                    "volume": df["Volume"].tolist(),
                },
            }
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)})
