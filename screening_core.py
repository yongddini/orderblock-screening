#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
스크리닝 핵심 로직
"""

import sqlite3
from datetime import datetime, timedelta
import pandas as pd
from stock_screener import StockScreener
from zoneinfo import ZoneInfo
import os
import sys

KST = ZoneInfo('Asia/Seoul')
DB_PATH = os.environ.get('DB_PATH', 'orderblock_screening.db')


def run_and_save_screening(target_date=None):
    """오더블록 스크리닝 실행 및 저장"""
    if target_date:
        today = datetime.strptime(target_date, '%Y%m%d').date()
        today_str = target_date
    else:
        today = datetime.now(KST).date()
        today_str = today.strftime('%Y%m%d')
    
    if today.weekday() >= 5:
        print(f"⏸️  {today} 주말이므로 스크리닝을 건너뜁니다.")
        return
    
    try:
        import holidays
        KR_HOLIDAYS = holidays.SouthKorea()
        if KR_HOLIDAYS and today in KR_HOLIDAYS:
            holiday_name = KR_HOLIDAYS.get(today)
            print(f"🎉 {today} 공휴일({holiday_name})이므로 스크리닝을 건너뜁니다.")
            return
    except ImportError:
        pass
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM screening_results WHERE scan_date = ?', (today_str,))
    count = cursor.fetchone()[0]
    
    if count > 0:
        print(f"🗑️  {today} 기존 결과 {count}개 삭제...")
        cursor.execute('DELETE FROM screening_results WHERE scan_date = ?', (today_str,))
        conn.commit()
    conn.close()
    
    print(f"🔍 {today} 스크리닝 시작...")
    
    screener_daily = StockScreener(
        proximity_percent=1.0,
        swing_length=10,
        max_atr_mult=2.0,
        ob_end_method="Wick",
        combine_obs=True
    )
    
    screener_weekly = StockScreener(
        proximity_percent=5.0,
        swing_length=10,
        max_atr_mult=2.0,
        ob_end_method="Wick",
        combine_obs=True
    )
    
    end_date_str = today.strftime('%Y-%m-%d')
    
    print("\n" + "="*50)
    print(f"일봉 스크리닝 시작 (기준일: {end_date_str}, 근접도: 1%)")
    print("="*50)
    
    stock_results_daily = screener_daily.screen_multiple_markets(
        markets=['KOSPI', 'KOSDAQ'],
        top_n=int(os.environ.get('SCREENING_TOP_N', '400')),
        days=500,
        end_date=end_date_str
    )
    
    etf_results_daily = screener_daily.screen_etf(
        top_n=int(os.environ.get('SCREENING_ETF_N', '300')),
        days=500,
        end_date=end_date_str
    )
    
    if isinstance(stock_results_daily, pd.DataFrame):
        results_daily = stock_results_daily.to_dict('records')
    else:
        results_daily = stock_results_daily
    
    if isinstance(etf_results_daily, list):
        results_daily.extend(etf_results_daily)
    elif isinstance(etf_results_daily, pd.DataFrame):
        results_daily.extend(etf_results_daily.to_dict('records'))
    
    print("\n" + "="*50)
    print(f"주봉 스크리닝 시작 (기준일: {end_date_str}, 근접도: 5%)")
    print("="*50)
    
    results_weekly = []
    for market in ['KOSPI', 'KOSDAQ']:
        weekly_result = screener_weekly.screen_market_weekly(
            market=market,
            top_n=int(os.environ.get('SCREENING_TOP_N', '400')),
            weeks=500,
            end_date=end_date_str
        )
        results_weekly.extend(weekly_result)
    
    etf_weekly = screener_weekly.screen_market_weekly(
        market='ETF',
        top_n=int(os.environ.get('SCREENING_ETF_N', '300')),
        weeks=500,
        end_date=end_date_str
    )
    results_weekly.extend(etf_weekly)
    
    all_results = []
    for r in results_daily:
        r['timeframe'] = 'daily'
        all_results.append(r)
    
    for r in results_weekly:
        r['timeframe'] = 'weekly'
        all_results.append(r)
    
    if not all_results or len(all_results) == 0:
        print("❌ 스크리닝 결과 없음")
        return
    
    results_df = pd.DataFrame(all_results)
    
    conn = sqlite3.connect(DB_PATH)
    conn.execute('DELETE FROM screening_results WHERE scan_date = ?', (today_str,))
    print(f"🗑️  기존 {today_str} 데이터 삭제 완료")
    
    for _, row in results_df.iterrows():
        try:
            zone_type = '지지' if row['OB_Type'] == 'Bull' else '저항'
            zone_position = '내부' if '내부' in row['Status'] else '근접'
            
            is_recommended = 0
            if (row['RSI'] < 30 and 
                zone_type == '지지' and 
                row['OB_Top'] > 0 and row['OB_Bottom'] > 0):
                
                ob_range_percent = ((row['OB_Top'] - row['OB_Bottom']) / row['OB_Bottom']) * 100
                
                if ob_range_percent < 10:
                    is_recommended = 1
            
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
            print(f"❌ {row['Code']} 저장 오류: {e}")
    
    conn.commit()
    
    cursor = conn.cursor()
    cursor.execute('''
        SELECT d.code, d.is_recommended
        FROM screening_results d
        JOIN screening_results w ON d.code = w.code AND d.scan_date = w.scan_date
        WHERE d.scan_date = ?
        AND d.timeframe = 'daily' AND w.timeframe = 'weekly'
        AND d.zone_type = '지지' AND w.zone_type = '지지'
    ''', (today_str,))
    
    dual_support_stocks = cursor.fetchall()
    
    if dual_support_stocks:
        for code, existing_flag in dual_support_stocks:
            new_flag = existing_flag | 2
            cursor.execute('''
                UPDATE screening_results 
                SET is_recommended = ? 
                WHERE scan_date = ? AND code = ? AND timeframe = 'daily'
            ''', (new_flag, today_str, code))
        conn.commit()
        print(f"🔥 {len(dual_support_stocks)}개 종목이 일봉+주봉 모두 지지 (추천 추가)")
    
    cursor.execute('SELECT COUNT(*) FROM screening_results WHERE scan_date = ? AND is_recommended > 0 AND timeframe = "daily"', (today_str,))
    recommended_count = cursor.fetchone()[0]
    
    conn.close()
    
    print(f"✅ {len(results_df)}개 종목 저장 완료")
    print(f"⭐ {recommended_count}개 추천 종목")


def run_and_save_investor_data(target_date=None):
    """외국인/기관 매매 데이터 수집 및 저장"""
    try:
        from pykrx import stock
    except ImportError as e:
        print(f"\n❌ pykrx import 실패: {e}")
        print(f"설치: {sys.executable} -m pip install pykrx")
        return
    
    if target_date:
        date_str = target_date
    else:
        date = datetime.now(KST).date()
        
        for i in range(7):
            check_date = date - timedelta(days=i)
            date_str = check_date.strftime('%Y%m%d')
            
            if check_date.weekday() >= 5:
                continue
            
            try:
                test = stock.get_market_net_purchases_of_equities(
                    fromdate=date_str,
                    todate=date_str,
                    market="KOSPI",
                    investor="외국인"
                )
                if not test.empty:
                    print(f"✅ 최근 영업일: {check_date.strftime('%Y-%m-%d')}")
                    break
            except:
                continue
        else:
            date_str = (datetime.now(KST).date() - timedelta(days=1)).strftime('%Y%m%d')
    
    print(f"\n📅 수집 날짜: {date_str}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS investor_trading (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date TEXT NOT NULL,
            investor_type TEXT NOT NULL,
            trade_type TEXT NOT NULL,
            rank INTEGER NOT NULL,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            market TEXT NOT NULL,
            current_price REAL,
            change_percent REAL,
            buy_amount INTEGER,
            sell_amount INTEGER,
            net_amount INTEGER,
            buy_volume INTEGER DEFAULT 0,
            sell_volume INTEGER DEFAULT 0,
            net_volume INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_investor_trading 
        ON investor_trading(scan_date, investor_type, trade_type, rank)
    ''')
    
    cursor.execute('DELETE FROM investor_trading WHERE scan_date = ?', (date_str,))
    conn.commit()
    conn.close()
    
    try:
        print("\n🌍 외국인/기관 매매동향 수집 중...")
        
        results_data = {}
        
        # 외국인, 기관합계 데이터 수집
        investor_configs = [
            ('외국인', 'foreign'),
            ('기관합계', 'institution')  # ✅ '기관' → '기관합계'
        ]
        
        for investor_name, investor_type in investor_configs:
            print(f"\n  📊 {investor_name} 데이터 수집 중...")
            
            kospi_df = stock.get_market_net_purchases_of_equities(
                fromdate=date_str,
                todate=date_str,
                market="KOSPI",
                investor=investor_name
            )
            
            kosdaq_df = stock.get_market_net_purchases_of_equities(
                fromdate=date_str,
                todate=date_str,
                market="KOSDAQ",
                investor=investor_name
            )
            
            kospi_df['시장'] = 'KOSPI'
            kosdaq_df['시장'] = 'KOSDAQ'
            
            all_df = pd.concat([kospi_df, kosdaq_df])
            
            buy_top = all_df.nlargest(100, '순매수거래대금')
            sell_top = all_df.nsmallest(100, '순매수거래대금')
            
            results_data[f'{investor_type}_buy'] = buy_top
            results_data[f'{investor_type}_sell'] = sell_top
            
            print(f"    ✅ {investor_name} 순매수 상위 {len(buy_top)}개")
            print(f"    ✅ {investor_name} 순매도 상위 {len(sell_top)}개")
        
        conn = sqlite3.connect(DB_PATH)
        saved_count = 0
        
        categories = [
            ('foreign', 'buy', results_data['foreign_buy']),
            ('foreign', 'sell', results_data['foreign_sell']),
            ('institution', 'buy', results_data['institution_buy']),
            ('institution', 'sell', results_data['institution_sell'])
        ]
        
        for investor_type, trade_type, df in categories:
            if df is None or df.empty:
                continue
            
            print(f"\n💰 {investor_type}/{trade_type} 현재가 조회 중...")
            
            for rank, (ticker, row) in enumerate(df.iterrows(), 1):
                try:
                    name = row.get('종목명', stock.get_market_ticker_name(ticker))
                    
                    try:
                        price_df = stock.get_market_ohlcv_by_date(
                            (datetime.now(KST).date() - timedelta(days=7)).strftime('%Y%m%d'),
                            datetime.now(KST).date().strftime('%Y%m%d'),
                            ticker
                        )
                        
                        if len(price_df) > 0:
                            latest = price_df.iloc[-1]
                            prev = price_df.iloc[-2] if len(price_df) > 1 else latest
                            current = latest['종가']
                            change = ((current - prev['종가']) / prev['종가'] * 100) if prev['종가'] > 0 else 0
                        else:
                            current = 0
                            change = 0
                    except:
                        current = 0
                        change = 0
                    
                    buy_amount = int(row.get('매수거래대금', 0))
                    sell_amount = int(row.get('매도거래대금', 0))
                    net_amount = int(row.get('순매수거래대금', 0))
                    
                    # 거래량 추가
                    buy_volume = int(row.get('매수거래량', 0))
                    sell_volume = int(row.get('매도거래량', 0))
                    net_volume = int(row.get('순매수거래량', 0))
                    
                    conn.execute('''
                        INSERT INTO investor_trading 
                        (scan_date, investor_type, trade_type, rank, code, name, market,
                         current_price, change_percent, buy_amount, sell_amount, net_amount,
                         buy_volume, sell_volume, net_volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        date_str, investor_type, trade_type, rank, ticker, name,
                        row.get('시장', 'KOSPI'), current, change,
                        buy_amount, sell_amount, net_amount,
                        buy_volume, sell_volume, net_volume
                    ))
                    
                    saved_count += 1
                    
                except Exception as e:
                    print(f"  ⚠️ {ticker} 저장 실패: {e}")
                    continue
        
        conn.commit()
        conn.close()
        
        print(f"\n✅ 총 {saved_count}개 종목 DB 저장 완료")
        
    except Exception as e:
        print(f"❌ 데이터 수집 실패: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    run_and_save_screening()