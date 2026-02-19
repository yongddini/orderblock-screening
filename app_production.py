# -*- coding: utf-8 -*-
"""
Production web server app.py
Gunicorn + Flask
"""

from flask import Flask, render_template, jsonify, request
import sqlite3
from datetime import datetime
import pandas as pd
from stock_screener import StockScreener
from data_provider import KoreanStockDataProvider
from realtime_detector import RealtimeOrderBlockDetector
import plotly.graph_objects as go
import os
from zoneinfo import ZoneInfo

# Import screening core logic
from screening_core import run_and_save_screening, run_and_save_investor_data

# Korea timezone
KST = ZoneInfo('Asia/Seoul')

KST = ZoneInfo('Asia/Seoul')

try:
    import holidays
    KR_HOLIDAYS = holidays.SouthKorea()
except ImportError:
    print("Warning: holidays library not found. Holiday check disabled.")
    print("Install: pip install holidays --break-system-packages")
    KR_HOLIDAYS = None

app = Flask(__name__)

# Database path
DB_PATH = os.environ.get('DB_PATH', 'orderblock_screening.db')

def init_db():
    """Initialize database tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS screening_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date VARCHAR(8) NOT NULL,
            market TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            current_price REAL NOT NULL,
            change_percent REAL,
            rsi REAL,
            trading_value REAL,
            zone_type TEXT NOT NULL,
            zone_position TEXT NOT NULL,
            ob_top REAL NOT NULL,
            ob_bottom REAL NOT NULL,
            distance_percent REAL,
            is_recommended INTEGER DEFAULT 0,
            timeframe TEXT DEFAULT 'daily',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(scan_date, code, timeframe)
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_date ON screening_results(scan_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_type ON screening_results(zone_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_zone_position ON screening_results(zone_position)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON screening_results(timeframe)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_scan_date_timeframe ON screening_results(scan_date, timeframe)')
    
    conn.commit()
    conn.close()
    print("Database initialized")

def run_and_save_screening(target_date=None):
    """Run screening and save results
    
    Args:
        target_date (str): Date to screen (YYYYMMDD format). None for today
    """
    if target_date:
        # Parse YYYYMMDD format
        today = datetime.strptime(target_date, '%Y%m%d').date()
        today_str = target_date
    else:
        today = datetime.now(KST).date()
        today_str = today.strftime('%Y%m%d')
    
    
    # Weekend check (Saturday=5, Sunday=6)
    if today.weekday() >= 5:
        print(f"Weekend {today}, skipping screening.")
        return
    
    # Holiday check (using holidays library)
    if KR_HOLIDAYS and today in KR_HOLIDAYS:
        holiday_name = KR_HOLIDAYS.get(today)
        print(f"Holiday {today} ({holiday_name}), skipping screening.")
        return
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM screening_results WHERE scan_date = ?', (today_str,))
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"Deleting {count} existing results for {today}...")
        cursor.execute('DELETE FROM screening_results WHERE scan_date = ?', (today_str,))
        conn.commit()
    conn.close()
    
    print(f"Starting screening for {today}...")
    
    # Daily screener (1% proximity)
    screener_daily = StockScreener(
        proximity_percent=1.0,
        swing_length=10,
        max_atr_mult=2.0,
        ob_end_method="Wick",
        combine_obs=True
    )
    
    # Weekly screener (5% proximity)
    screener_weekly = StockScreener(
        proximity_percent=5.0,
        swing_length=10,
        max_atr_mult=2.0,
        ob_end_method="Wick",
        combine_obs=True
    )
    
    # Calculate end_date (YYYY-MM-DD format)
    end_date_str = today.strftime('%Y-%m-%d')
    
    # ========== Daily Screening ==========
    print("\n" + "="*50)
    print(f"Daily screening started (Date: {end_date_str}, Proximity: 1%)")
    print("="*50)
    
    # KOSPI, KOSDAQ daily
    stock_results_daily = screener_daily.screen_multiple_markets(
        markets=['KOSPI', 'KOSDAQ'],
        top_n=int(os.environ.get('SCREENING_TOP_N', '400')),
        days=500,
        end_date=end_date_str
    )
    
    # ETF daily
    etf_results_daily = screener_daily.screen_etf(
        top_n=int(os.environ.get('SCREENING_ETF_N', '300')),
        days=500,
        end_date=end_date_str
    )
    
    # Combine daily results
    if isinstance(stock_results_daily, pd.DataFrame):
        results_daily = stock_results_daily.to_dict('records')
    else:
        results_daily = stock_results_daily
    
    if isinstance(etf_results_daily, list):
        results_daily.extend(etf_results_daily)
    elif isinstance(etf_results_daily, pd.DataFrame):
        results_daily.extend(etf_results_daily.to_dict('records'))
    
    # ========== Weekly Screening ==========
    print("\n" + "="*50)
    print(f"Weekly screening started (Date: {end_date_str}, Proximity: 5%)")
    print("="*50)
    
    # KOSPI, KOSDAQ weekly
    results_weekly = []
    for market in ['KOSPI', 'KOSDAQ']:
        weekly_result = screener_weekly.screen_market_weekly(
            market=market,
            top_n=int(os.environ.get('SCREENING_TOP_N', '400')),
            weeks=500,
            end_date=end_date_str
        )
        results_weekly.extend(weekly_result)
    
    # ETF weekly
    etf_weekly = screener_weekly.screen_market_weekly(
        market='ETF',
        top_n=int(os.environ.get('SCREENING_ETF_N', '300')),
        weeks=500,
        end_date=end_date_str
    )
    results_weekly.extend(etf_weekly)
    
    # Combine all results
    all_results = []
    
    # Add timeframe to daily data
    for r in results_daily:
        r['timeframe'] = 'daily'
        all_results.append(r)
    
    # Add timeframe to weekly data
    for r in results_weekly:
        r['timeframe'] = 'weekly'
        all_results.append(r)
    
    if not all_results or len(all_results) == 0:
        print("No screening results")
        return
    
    results_df = pd.DataFrame(all_results)
    
    conn = sqlite3.connect(DB_PATH)
    
    # Delete existing data for the date (maintain ID continuity)
    conn.execute('DELETE FROM screening_results WHERE scan_date = ?', (today_str,))
    print(f"Deleted existing data for {today_str}")
    
    for _, row in results_df.iterrows():
        try:
            # Calculate zone_type, zone_position
            zone_type = 'Support' if row['OB_Type'] == 'Bull' else 'Resistance'
            zone_position = 'Inside' if 'Inside' in row['Status'] else 'Near'
            
            # Check recommendation criteria (existing criteria)
            is_recommended = 0
            if (row['RSI'] < 30 and 
                zone_type == 'Support' and 
                row['OB_Top'] > 0 and row['OB_Bottom'] > 0):
                
                # Calculate OB range percentage
                ob_range_percent = ((row['OB_Top'] - row['OB_Bottom']) / row['OB_Bottom']) * 100
                
                if ob_range_percent < 10:
                    is_recommended = 1  # Existing criteria (RSI<30 + Support + Narrow OB)
            
            conn.execute('''
                INSERT INTO screening_results 
                (scan_date, market, code, name, current_price, change_percent, rsi, trading_value, 
                 zone_type, zone_position, ob_top, ob_bottom, distance_percent, is_recommended, timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                today_str, row['Market'], row['Code'], row['Name'],
                row['Current_Price'], row['Change_Percent'], row['RSI'], row['trading_value'],
                zone_type, zone_position,
                row['OB_Top'], row['OB_Bottom'],
                row['Distance_Percent'], is_recommended, row.get('timeframe', 'daily')
            ))
        except Exception as e:
            print(f"Error saving {row['Code']}: {e}")
    
    conn.commit()
    
    # Find stocks with both daily+weekly support and update recommendations
    cursor = conn.cursor()
    cursor.execute('''
        SELECT d.code, d.is_recommended
        FROM screening_results d
        JOIN screening_results w ON d.code = w.code AND d.scan_date = w.scan_date
        WHERE d.scan_date = ?
        AND d.timeframe = 'daily' AND w.timeframe = 'weekly'
        AND d.zone_type = 'Support' AND w.zone_type = 'Support'
    ''', (today_str,))
    
    dual_support_stocks = cursor.fetchall()
    
    # Update daily+weekly support stocks as recommended (add 2 to existing value)
    if dual_support_stocks:
        for code, existing_flag in dual_support_stocks:
            new_flag = existing_flag | 2  # Bitwise OR (existing + 2)
            cursor.execute('''
                UPDATE screening_results 
                SET is_recommended = ? 
                WHERE scan_date = ? AND code = ? AND timeframe = 'daily'
            ''', (new_flag, today_str, code))
        conn.commit()
        print(f"{len(dual_support_stocks)} stocks have both daily+weekly support (recommendation added)")
    
    # Print recommended stock count
    cursor.execute('SELECT COUNT(*) FROM screening_results WHERE scan_date = ? AND is_recommended > 0 AND timeframe = "daily"', (today_str,))
    recommended_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"Saved {len(results_df)} stocks")
    print(f"Recommended stocks: {recommended_count}")


def create_chart_html(ticker, days=500, end_date=None):
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
        df = df[df['Volume'] > 0].copy()
        
        # Filter by end_date if specified
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            except:
                pass  # Use full data if date parsing fails
        
        if len(df) < 50:
            return None
        
        # Calculate RSI (14-day) - Pine Script method (RMA)
        delta = df['Close'].diff()
        
        # RMA (Wilder's Smoothing) calculation
        def rma(series, length):
            alpha = 1.0 / length
            result = series.copy()
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = rma(gain, 14)
        avg_loss = rma(loss, 14)
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        # RSI EMA(14)
        rsi_ema = rsi.ewm(span=14, adjust=False).mean()
        
        detector = RealtimeOrderBlockDetector(
            swing_length=10,
            max_atr_mult=2.0,
            ob_end_method="Wick",
            combine_obs=True,
            max_order_blocks=30
        )
        
        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()
        
        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]
        
        date_to_idx = {date: idx for idx, date in enumerate(df.index)}
        
        # 2 subplots: price + RSI
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.75, 0.25],
            subplot_titles=('', 'RSI (14)')
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
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                text=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Bull order blocks (Pine Script style transparency)
        for ob in bull_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue
            
            start_idx = date_to_idx[ob.start_time]
            end_idx = date_to_idx[ob.break_time] if ob.break_time and ob.break_time in df.index else len(df) - 1
            
            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = '#757575'
            elif getattr(ob, 'combined', False):
                alpha = 0.73
                color = '#26a69a'
            else:
                alpha = 0.47
                color = '#26a69a'
            
            fig.add_shape(
                type="rect",
                x0=start_idx, x1=end_idx,
                y0=ob.bottom, y1=ob.top,
                fillcolor=color,
                opacity=1-alpha,
                line=dict(color=color, width=1),
                row=1, col=1
            )
        
        # Bear order blocks (Pine Script style transparency)
        for ob in bear_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue
            
            start_idx = date_to_idx[ob.start_time]
            end_idx = date_to_idx[ob.break_time] if ob.break_time and ob.break_time in df.index else len(df) - 1
            
            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = '#757575'
            elif getattr(ob, 'combined', False):
                alpha = 0.73
                color = '#ef5350'
            else:
                alpha = 0.47
                color = '#ef5350'
            
            fig.add_shape(
                type="rect",
                x0=start_idx, x1=end_idx,
                y0=ob.bottom, y1=ob.top,
                fillcolor=color,
                opacity=1-alpha,
                line=dict(color=color, width=1),
                row=1, col=1
            )
        
        # RSI trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi,
                name='RSI',
                line=dict(color='#7E57C2', width=2),
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # RSI EMA(14) trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi_ema,
                name='RSI EMA',
                line=dict(color='#FFEB3B', width=2),
                hovertemplate='RSI EMA: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # RSI overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)
        
        tick_step = max(1, len(df) // 10)
        tickvals = list(range(0, len(df), tick_step))
        ticktext = [df.index[i].strftime('%Y-%m-%d') for i in tickvals]
        
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45, row=2, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(color='#e0e0e0'),
            dragmode='pan',
            # TradingView style interaction
            xaxis=dict(
                fixedrange=False,
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='#888888',
                spikethickness=1,
                spikedash='dot'
            ),
            yaxis=dict(
                fixedrange=False,
                scaleanchor=None,
                showticklabels=True
            ),
            yaxis2=dict(
                fixedrange=False,
                showticklabels=True
            )
        )
        
        # Price chart y-axis (right side, actual price)
        fig.update_xaxes(
            gridcolor='#3a3a3a',
            fixedrange=False,
            zeroline=False,
            row=1, col=1
        )
        fig.update_yaxes(
            gridcolor='#3a3a3a', 
            side='right',
            tickformat=',',
            hoverformat=',',
            separatethousands=True,
            fixedrange=False,
            automargin=True,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=1, col=1
        )
        
        # RSI chart y-axis (right side)
        fig.update_xaxes(
            gridcolor='#3a3a3a',
            fixedrange=False,
            zeroline=False,
            row=2, col=1
        )
        fig.update_yaxes(
            gridcolor='#3a3a3a', 
            range=[0, 100],
            side='right',
            fixedrange=False,
            automargin=True,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=2, col=1
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Chart creation error: {e}")
        return None


def create_chart_html_weekly(ticker, weeks=500, end_date=None):
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
        df = df[df['Volume'] > 0].copy()
        
        if end_date:
            try:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]
            except:
                pass
        
        if len(df) < 50:
            return None
        
        # Calculate RSI (RMA method)
        delta = df['Close'].diff()
        
        def rma(series, length):
            alpha = 1.0 / length
            result = series.copy()
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
            return result
        
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = rma(gain, 14)
        avg_loss = rma(loss, 14)
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi_ema = rsi.ewm(span=14, adjust=False).mean()
        
        # Weekly OB detection
        detector = RealtimeOrderBlockDetector(
            swing_length=10,
            max_atr_mult=2.0,
            ob_end_method="Wick",
            combine_obs=True,
            max_order_blocks=30
        )
        
        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()
        
        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]
        
        date_to_idx = {date: idx for idx, date in enumerate(df.index)}
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
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
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Weekly',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350',
                text=hover_texts,
                hoverinfo='text'
            ),
            row=1, col=1
        )
        
        # Bull order blocks (same colors and transparency as daily)
        for ob in bull_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue
            
            start_idx = date_to_idx[ob.start_time]
            end_idx = date_to_idx[ob.break_time] if ob.break_time and ob.break_time in df.index else len(df) - 1
            
            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = '#757575'
            elif getattr(ob, 'combined', False):
                alpha = 0.73
                color = '#26a69a'
            else:
                alpha = 0.47
                color = '#26a69a'
            
            fig.add_shape(
                type="rect",
                x0=start_idx, x1=end_idx,
                y0=ob.bottom, y1=ob.top,
                fillcolor=color,
                opacity=1-alpha,
                line=dict(color=color, width=1),
                row=1, col=1
            )
        
        # Bear order blocks (same colors and transparency as daily)
        for ob in bear_obs:
            if not ob.start_time or ob.start_time not in df.index:
                continue
            
            start_idx = date_to_idx[ob.start_time]
            end_idx = date_to_idx[ob.break_time] if ob.break_time and ob.break_time in df.index else len(df) - 1
            
            # Pine Script transparency: combined=0.73, normal=0.47, breaker=0.9
            if ob.breaker:
                alpha = 0.9
                color = '#757575'
            elif getattr(ob, 'combined', False):
                alpha = 0.73
                color = '#ef5350'
            else:
                alpha = 0.47
                color = '#ef5350'
            
            fig.add_shape(
                type="rect",
                x0=start_idx, x1=end_idx,
                y0=ob.bottom, y1=ob.top,
                fillcolor=color,
                opacity=1-alpha,
                line=dict(color=color, width=1),
                row=1, col=1
            )
        
        # RSI trace
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi,
                name='RSI',
                line=dict(color='#7E57C2', width=2),
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(len(df))),
                y=rsi_ema,
                name='RSI EMA',
                line=dict(color='#FFEB3B', width=2),
                hovertemplate='RSI EMA: %{y:.1f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        fig.add_hline(y=50, line_dash="solid", line_color="gray", opacity=0.3, row=2, col=1)
        
        tick_step = max(1, len(df) // 10)
        tickvals = list(range(0, len(df), tick_step))
        ticktext = [df.index[i].strftime('%Y-%m-%d') for i in tickvals]
        
        fig.update_xaxes(tickvals=tickvals, ticktext=ticktext, tickangle=-45, row=2, col=1)
        
        fig.update_layout(
            height=1000,
            showlegend=False,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            plot_bgcolor='#1e1e1e',
            paper_bgcolor='#2d2d2d',
            font=dict(color='#e0e0e0'),
            dragmode='pan',
            # TradingView style interaction
            xaxis=dict(
                fixedrange=False,
                rangeslider=dict(visible=False),
                showspikes=True,
                spikemode='across',
                spikesnap='cursor',
                spikecolor='#888888',
                spikethickness=1,
                spikedash='dot'
            ),
            yaxis=dict(
                fixedrange=False,
                scaleanchor=None,
                showticklabels=True
            ),
            yaxis2=dict(
                fixedrange=False,
                showticklabels=True
            )
        )
        
        # Price chart y-axis (right side, actual price)
        fig.update_xaxes(
            gridcolor='#3a3a3a',
            fixedrange=False,
            zeroline=False,
            row=1, col=1
        )
        fig.update_yaxes(
            gridcolor='#3a3a3a', 
            side='right',
            tickformat=',',
            hoverformat=',',
            separatethousands=True,
            fixedrange=False,
            automargin=True,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=1, col=1
        )
        
        # RSI chart y-axis
        fig.update_xaxes(
            gridcolor='#3a3a3a',
            fixedrange=False,
            zeroline=False,
            row=2, col=1
        )
        fig.update_yaxes(
            gridcolor='#3a3a3a',
            range=[0, 100],
            side='right',
            fixedrange=False,
            tickwidth=2,
            ticklen=10,
            zeroline=False,
            row=2, col=1
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Weekly chart creation error: {e}")
        return None


# ==================== Foreign/Institution Investor Trading API ====================

@app.route('/investor')
def investor_page():
    """Foreign/Institution investor trading page"""
    return render_template('investor.html')


@app.route('/api/investor-trading')
def get_investor_trading():
    """Get investor trading data API
    
    Query parameters:
        date (str): Date (YYYYMMDD) - optional
        investor_type (str): foreign | institution
        trade_type (str): buy | sell
        limit (int): Number of results (default 100)
    """
    date = request.args.get('date')
    investor_type = request.args.get('investor_type', 'foreign')
    trade_type = request.args.get('trade_type', 'buy')
    limit = int(request.args.get('limit', 100))
    
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        # If no date specified, get latest date
        if not date:
            cursor = conn.cursor()
            cursor.execute('SELECT DISTINCT scan_date FROM investor_trading ORDER BY scan_date DESC LIMIT 1')
            row = cursor.fetchone()
            date = row['scan_date'] if row else None
        
        if not date:
            return jsonify({'success': False, 'error': 'No data available'}), 404
        
        # Query data
        cursor = conn.cursor()
        cursor.execute('''
            SELECT rank, code, name, market, current_price, change_percent,
                   buy_amount, sell_amount, net_amount,
                   buy_volume, sell_volume, net_volume
            FROM investor_trading
            WHERE scan_date = ? AND investor_type = ? AND trade_type = ?
            ORDER BY rank LIMIT ?
        ''', (date, investor_type, trade_type, limit))
        
        data = [dict(row) for row in cursor.fetchall()]
        
        return jsonify({
            'success': True,
            'date': date,
            'investor_type': investor_type,
            'trade_type': trade_type,
            'count': len(data),
            'data': data
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/api/investor-dates')
def get_investor_dates():
    """Get available investor trading dates"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT scan_date FROM investor_trading ORDER BY scan_date DESC LIMIT 30')
        dates = [row['scan_date'] for row in cursor.fetchall()]
        return jsonify({'success': True, 'dates': dates})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        conn.close()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/screening/dates')
def get_available_dates():
    """Available screening dates"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT DISTINCT scan_date, COUNT(*) as count
        FROM screening_results
        GROUP BY scan_date
        ORDER BY scan_date DESC
        LIMIT 30
    ''')
    # Convert YYYYMMDD integer to YYYY-MM-DD string
    dates = []
    for row in cursor.fetchall():
        date_str = row[0]  # YYYYMMDD string
        # YYYYMMDD -> YYYY-MM-DD
        formatted_date = f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}"
        dates.append({'date': formatted_date, 'count': row[1]})
    conn.close()
    
    return jsonify({
        'success': True,
        'dates': dates
    })


# ==================== Market Indices API ====================

@app.route('/api/market-indices')
def get_market_indices():
    """코스피/코스닥 실시간 지수 정보
    
    Returns:
        JSON: {
            success: bool,
            kospi: {value, change, change_percent, chart},
            kosdaq: {value, change, change_percent, chart}
        }
    """
    try:
        from pykrx import stock
        from datetime import timedelta
        
        today = datetime.now(KST).date()
        
        # 최근 30일 데이터 (차트용)
        start_date = (today - timedelta(days=30)).strftime('%Y%m%d')
        end_date = today.strftime('%Y%m%d')
        
        # 코스피 지수 (1001)
        kospi_df = stock.get_index_ohlcv(start_date, end_date, "1001")
        
        # 코스닥 지수 (2001)
        kosdaq_df = stock.get_index_ohlcv(start_date, end_date, "2001")
        
        if kospi_df is None or len(kospi_df) == 0 or kosdaq_df is None or len(kosdaq_df) == 0:
            return jsonify({
                'success': False,
                'error': 'No market data available'
            }), 404
        
        # 최신 데이터 (오늘)
        kospi_latest = kospi_df.iloc[-1]
        kospi_prev = kospi_df.iloc[-2] if len(kospi_df) > 1 else kospi_latest
        
        kosdaq_latest = kosdaq_df.iloc[-1]
        kosdaq_prev = kosdaq_df.iloc[-2] if len(kosdaq_df) > 1 else kosdaq_latest
        
        # 변동률 계산
        kospi_change = kospi_latest['종가'] - kospi_prev['종가']
        kospi_change_percent = (kospi_change / kospi_prev['종가']) * 100
        
        kosdaq_change = kosdaq_latest['종가'] - kosdaq_prev['종가']
        kosdaq_change_percent = (kosdaq_change / kosdaq_prev['종가']) * 100
        
        # 최근 7일 차트 데이터
        kospi_chart = kospi_df.tail(7)['종가'].tolist()
        kosdaq_chart = kosdaq_df.tail(7)['종가'].tolist()
        
        return jsonify({
            'success': True,
            'kospi': {
                'value': float(kospi_latest['종가']),
                'change': float(kospi_change),
                'change_percent': float(kospi_change_percent),
                'chart': kospi_chart
            },
            'kosdaq': {
                'value': float(kosdaq_latest['종가']),
                'change': float(kosdaq_change),
                'change_percent': float(kosdaq_change_percent),
                'chart': kosdaq_chart
            }
        })
        
    except Exception as e:
        print(f"Market indices error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/screening/recommended')
def get_recommended_stocks():
    """Recommended stocks for selected date"""
    date_param = request.args.get('date')
    
    if date_param:
        # YYYY-MM-DD -> YYYYMMDD
        if '-' in date_param:
            date_str = date_param.replace('-', '')
        else:
            date_str = date_param
    else:
        # If no parameter, use today
        today = datetime.now(KST).date()
        date_str = today.strftime('%Y%m%d')
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
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
    ''', conn, params=(date_str,))
    conn.close()
    
    if df.empty:
        return jsonify({
            'success': False,
            'message': 'No recommended stocks'
        })
    
    return jsonify({
        'success': True,
        'count': len(df),
        'results': df.to_dict('records')
    })

@app.route('/api/screening/today')
def get_today_screening():
    today = datetime.now(KST).date()
    today_str = today.strftime('%Y%m%d')
    return get_screening_by_date(today_str)

@app.route('/api/screening/<date>')
def get_screening_by_date(date):
    """Screening results for specific date"""
    # Handle YYYYMMDD or YYYY-MM-DD format
    if '-' in date:
        # YYYY-MM-DD -> YYYYMMDD
        date_str = date.replace('-', '')
    else:
        date_str = date
    
    # timeframe parameter (daily or weekly)
    timeframe = request.args.get('timeframe', 'daily')
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
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
    ''', conn, params=(date_str, timeframe))
    conn.close()
    
    if df.empty:
        return jsonify({
            'success': False,
            'message': f'No screening results for {date}'
        })
    
    stats = {
        'total': len(df),
        'zone_type': df['zone_type'].value_counts().to_dict(),
        'zone_position': df['zone_position'].value_counts().to_dict(),
        'markets': df['market'].value_counts().to_dict(),
    }
    
    return jsonify({
        'success': True,
        'scan_date': date,
        'stats': stats,
        'results': df.to_dict('records')
    })

@app.route('/api/chart/<ticker>')
def get_chart(ticker):
    # Get date parameter
    date_param = request.args.get('date')
    end_date = None
    
    if date_param:
        # Convert to YYYY-MM-DD format
        if '-' in date_param:
            end_date = date_param
        else:
            # YYYYMMDD -> YYYY-MM-DD
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"
    
    chart_json = create_chart_html(ticker, end_date=end_date)
    
    if chart_json:
        return jsonify({'success': True, 'chart': chart_json})
    else:
        return jsonify({'success': False, 'message': 'Cannot create chart'})


@app.route('/api/chart-weekly/<ticker>')
def get_chart_weekly(ticker):
    """Weekly chart data API"""
    date_param = request.args.get('date')
    end_date = None
    
    if date_param:
        if '-' in date_param:
            end_date = date_param
        else:
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"
    
    chart_json = create_chart_html_weekly(ticker, end_date=end_date)
    
    if chart_json:
        return jsonify({'success': True, 'chart': chart_json})
    else:
        return jsonify({'success': False, 'message': 'Cannot create weekly chart'})

@app.route('/api/stock/<ticker>')
def get_stock_info(ticker):
    # Get date parameter (use today if not provided)
    date_param = request.args.get('date')
    
    if date_param:
        # YYYY-MM-DD -> YYYYMMDD
        if '-' in date_param:
            date_str = date_param.replace('-', '')
        else:
            date_str = date_param
    else:
        # If no parameter, use today
        today = datetime.now(KST).date()
        date_str = today.strftime('%Y%m%d')
    
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM screening_results 
        WHERE scan_date = ? AND code = ?
    ''', conn, params=(date_str, ticker))
    conn.close()
    
    if not df.empty:
        stock = df.iloc[0].to_dict()
        return jsonify({'success': True, 'stock': stock})
    else:
        return jsonify({'success': False, 'message': 'Stock not found'})


@app.route('/chart-test')
def chart_test():
    """Chart test page"""
    return render_template('chart_test.html')


@app.route('/ob-comparison')
def ob_comparison():
    """Order block method comparison page"""
    return render_template('ob_comparison.html')


@app.route('/api/compare-ob-methods/<ticker>')
def compare_ob_methods(ticker):
    """Compare two OB generation methods"""
    weeks = int(request.args.get('weeks', 500))
    
    try:
        # Weekly data
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks)
        
        if df is None or len(df) == 0:
            return jsonify({'success': False, 'message': 'No data'})
        
        # Find swings
        swing_length = 10
        swing_lows = []
        
        for i in range(swing_length, len(df) - swing_length):
            current_low = df['Low'].iloc[i]
            left_lows = df['Low'].iloc[i-swing_length:i]
            right_lows = df['Low'].iloc[i+1:i+swing_length+1]
            
            if all(current_low <= left_lows) and all(current_low <= right_lows):
                swing_lows.append({'index': i, 'low': current_low})
        
        # Method 1: Current (candle before swing)
        current_obs = []
        for swing in swing_lows:
            ob_idx = swing['index'] - 1
            if ob_idx >= 0:
                ob_candle = df.iloc[ob_idx]
                current_obs.append({
                    'ob_idx': ob_idx,
                    'swing_idx': swing['index'],
                    'start_date': df.index[ob_idx].strftime('%Y-%m-%d'),
                    'top': float(ob_candle['High']),
                    'bottom': float(ob_candle['Low']),
                    'invalidated': False
                })
        
        # Invalidation check (current method)
        for ob in current_obs:
            for i in range(ob['ob_idx'] + 1, len(df)):
                if df.iloc[i]['Low'] < ob['bottom']:
                    ob['invalidated'] = True
                    ob['end_date'] = df.index[i].strftime('%Y-%m-%d')
                    break
        
        # Method 2: TradingView (lowest candle in range)
        tradingview_obs = []
        for i, swing in enumerate(swing_lows):
            swing_idx = swing['index']
            
            # Search range: from swing to next swing or end
            search_end = swing_lows[i+1]['index'] if i+1 < len(swing_lows) else len(df)
            
            # Find lowest candle in range
            min_low = float('inf')
            ob_idx = swing_idx
            
            for j in range(swing_idx, search_end):
                if df.iloc[j]['Low'] < min_low:
                    min_low = df.iloc[j]['Low']
                    ob_idx = j
            
            ob_candle = df.iloc[ob_idx]
            tradingview_obs.append({
                'ob_idx': ob_idx,
                'swing_idx': swing_idx,
                'start_date': df.index[ob_idx].strftime('%Y-%m-%d'),
                'top': float(ob_candle['High']),
                'bottom': float(ob_candle['Low']),
                'invalidated': False
            })
        
        # Invalidation check (TradingView method)
        for ob in tradingview_obs:
            for i in range(ob['ob_idx'] + 1, len(df)):
                if df.iloc[i]['Low'] < ob['bottom']:
                    ob['invalidated'] = True
                    ob['end_date'] = df.index[i].strftime('%Y-%m-%d')
                    break
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'weekly_data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'close': df['Close'].tolist()
            },
            'current_method': {
                'name': 'Current (Before Swing)',
                'order_blocks': current_obs[-20:]  # Last 20
            },
            'tradingview_method': {
                'name': 'TradingView (Range Lowest)',
                'order_blocks': tradingview_obs[-20:]  # Last 20
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/test-chart/daily/<ticker>')
def test_chart_daily(ticker):
    """Daily chart data"""
    end_date = request.args.get('end_date')
    days = int(request.args.get('days', 500))
    
    try:
        df = KoreanStockDataProvider.get_price_data(ticker, days, end_date)
        
        if df is None or len(df) == 0:
            return jsonify({'success': False, 'message': 'No data'})
        
        # Calculate period
        period = f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}"
        years = (df.index[-1] - df.index[0]).days / 365.25
        
        return jsonify({
            'success': True,
            'candle_count': len(df),
            'period': period,
            'years': f'{years:.1f}',
            'data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'close': df['Close'].tolist(),
                'volume': df['Volume'].tolist()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/test-chart/weekly/<ticker>')
def test_chart_weekly(ticker):
    """Weekly chart data"""
    end_date = request.args.get('end_date')
    weeks = int(request.args.get('weeks', 500))
    
    try:
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks, end_date)
        
        if df is None or len(df) == 0:
            return jsonify({'success': False, 'message': 'No data'})
        
        # Calculate period
        period = f"{df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}"
        years = (df.index[-1] - df.index[0]).days / 365.25
        
        return jsonify({
            'success': True,
            'candle_count': len(df),
            'period': period,
            'years': f'{years:.1f}',
            'data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'open': df['Open'].tolist(),
                'high': df['High'].tolist(),
                'low': df['Low'].tolist(),
                'close': df['Close'].tolist(),
                'volume': df['Volume'].tolist()
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})




@app.route('/api/chart-data/<ticker>')
def get_chart_data(ticker):
    """Daily chart data (for Lightweight Charts)"""
    date_param = request.args.get('date')
    end_date = None
    
    if date_param:
        if '-' in date_param:
            end_date = date_param
        else:
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"
    
    try:
        df = KoreanStockDataProvider.get_price_data(ticker, days=500, end_date=end_date)
        
        if df is None or len(df) < 50:
            return jsonify({'success': False, 'message': 'No data'})
        
        df = df[df['Volume'] > 0].copy()
        
        if len(df) < 50:
            return jsonify({'success': False, 'message': 'Insufficient data'})
        
        # Calculate RSI
        delta = df['Close'].diff()
        
        def rma(series, length):
            alpha = 1.0 / length
            result = series.copy()
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
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
        detector = RealtimeOrderBlockDetector(
            swing_length=10,
            max_atr_mult=2.0,
            ob_end_method="Wick",
            combine_obs=True,
            max_order_blocks=30
        )
        
        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()
        
        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]
        
        # Lightweight Charts format
        candles = []
        for date, row in df.iterrows():
            candles.append({
                'time': int(date.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
            })
        
        rsi_data = []
        for date, value in rsi.items():
            if pd.notna(value):
                rsi_data.append({
                    'time': int(date.timestamp()),
                    'value': float(value)
                })
        
        rsi_ema_data = []
        for date, value in rsi_ema.items():
            if pd.notna(value):
                rsi_ema_data.append({
                    'time': int(date.timestamp()),
                    'value': float(value)
                })
        
        orderblocks = {'bull': [], 'bear': []}
        
        for ob in bull_obs:
            orderblocks['bull'].append({
                'top': float(ob.top),
                'bottom': float(ob.bottom),
                'start_time': int(ob.start_time.timestamp()) if ob.start_time else None,
                'break_time': int(ob.break_time.timestamp()) if ob.break_time else None,
                'breaker': ob.breaker,
                'combined': getattr(ob, 'combined', False)
            })
        
        for ob in bear_obs:
            orderblocks['bear'].append({
                'top': float(ob.top),
                'bottom': float(ob.bottom),
                'start_time': int(ob.start_time.timestamp()) if ob.start_time else None,
                'break_time': int(ob.break_time.timestamp()) if ob.break_time else None,
                'breaker': ob.breaker,
                'combined': getattr(ob, 'combined', False)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'candles': candles,
                'rsi': rsi_data,
                'rsi_ema': rsi_ema_data,
                'orderblocks': orderblocks
            }
        })
        
    except Exception as e:
        print(f"Chart data error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/chart-data-weekly/<ticker>')
def get_chart_data_weekly(ticker):
    """Weekly chart data (for Lightweight Charts)"""
    date_param = request.args.get('date')
    end_date = None
    
    if date_param:
        if '-' in date_param:
            end_date = date_param
        else:
            end_date = f"{date_param[0:4]}-{date_param[4:6]}-{date_param[6:8]}"
    
    try:
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks=500, end_date=end_date)
        
        if df is None or len(df) < 50:
            return jsonify({'success': False, 'message': 'No data'})
        
        df = df[df['Volume'] > 0].copy()
        
        if len(df) < 50:
            return jsonify({'success': False, 'message': 'Insufficient data'})
        
        # Calculate RSI
        delta = df['Close'].diff()
        
        def rma(series, length):
            alpha = 1.0 / length
            result = series.copy()
            result.iloc[0] = series.iloc[0]
            for i in range(1, len(series)):
                result.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * result.iloc[i-1]
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
        detector = RealtimeOrderBlockDetector(
            swing_length=10,
            max_atr_mult=2.0,
            ob_end_method="Wick",
            combine_obs=True,
            max_order_blocks=30
        )
        
        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()
        
        bull_obs = bull_obs[:3]
        bear_obs = bear_obs[:3]
        
        # Lightweight Charts format
        candles = []
        for date, row in df.iterrows():
            candles.append({
                'time': int(date.timestamp()),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
            })
        
        rsi_data = []
        for date, value in rsi.items():
            if pd.notna(value):
                rsi_data.append({
                    'time': int(date.timestamp()),
                    'value': float(value)
                })
        
        rsi_ema_data = []
        for date, value in rsi_ema.items():
            if pd.notna(value):
                rsi_ema_data.append({
                    'time': int(date.timestamp()),
                    'value': float(value)
                })
        
        orderblocks = {'bull': [], 'bear': []}
        
        for ob in bull_obs:
            orderblocks['bull'].append({
                'top': float(ob.top),
                'bottom': float(ob.bottom),
                'start_time': int(ob.start_time.timestamp()) if ob.start_time else None,
                'break_time': int(ob.break_time.timestamp()) if ob.break_time else None,
                'breaker': ob.breaker,
                'combined': getattr(ob, 'combined', False)
            })
        
        for ob in bear_obs:
            orderblocks['bear'].append({
                'top': float(ob.top),
                'bottom': float(ob.bottom),
                'start_time': int(ob.start_time.timestamp()) if ob.start_time else None,
                'break_time': int(ob.break_time.timestamp()) if ob.break_time else None,
                'breaker': ob.breaker,
                'combined': getattr(ob, 'combined', False)
            })
        
        return jsonify({
            'success': True,
            'data': {
                'candles': candles,
                'rsi': rsi_data,
                'rsi_ema': rsi_ema_data,
                'orderblocks': orderblocks
            }
        })
        
    except Exception as e:
        print(f"Weekly chart data error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'message': str(e)})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok'})

# Initialize
init_db()

if __name__ == '__main__':
    print("Application starting")
    
    # Run screening only in development mode
    if os.environ.get('FLASK_ENV') == 'development':
        run_and_save_screening()
    
    # Gunicorn runs in production
    app.run(debug=True, host='0.0.0.0', port=5000)