"""
데이터 제공 모듈
한국 주식 데이터를 가져오는 기능
"""

import pandas as pd
import FinanceDataReader as fdr
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class KoreanStockDataProvider:
    """한국 주식 데이터 제공 클래스"""
    
    @staticmethod
    def get_market_stocks(market='KOSPI'):
        """
        시장별 전체 종목 리스트 가져오기
        
        Args:
            market (str): 'KOSPI' 또는 'KOSDAQ'
        
        Returns:
            pd.DataFrame: 종목 정보 데이터프레임
        """
        df_krx = fdr.StockListing('KRX')
        df_market = df_krx[df_krx['Market'] == market].copy()
        return df_market
    
    @staticmethod
    def get_top_stocks_by_market_cap(market='KOSPI', top_n=400):
        """
        시가총액 기준 상위 종목 가져오기
        
        Args:
            market (str): 'KOSPI' 또는 'KOSDAQ'
            top_n (int): 상위 몇 개 종목
        
        Returns:
            pd.DataFrame: 시가총액 순 정렬된 종목 정보
        """
        print(f"Fetching {market} stocks...")
        
        df_market = KoreanStockDataProvider.get_market_stocks(market)
        
        # 시가총액 기준 정렬 및 상위 N개 선택
        df_market = df_market.sort_values('Marcap', ascending=False).head(top_n)
        
        return df_market
    
    @staticmethod
    def get_price_data(ticker, days=200, end_date=None):
        """
        종목의 가격 데이터 가져오기
        
        Note: FinanceDataReader는 실제 거래일 데이터만 반환하므로
        주말과 공휴일은 자동으로 제외됩니다.
        
        Args:
            ticker (str): 종목 코드
            days (int): 과거 몇 일간의 데이터 (여유있게 가져옴)
            end_date (datetime or str): 종료 날짜. None이면 오늘
        
        Returns:
            pd.DataFrame: OHLCV 데이터프레임 (거래일만 포함)
        """
        # end_date 처리
        if end_date is None:
            end_dt = datetime.now()
        elif isinstance(end_date, str):
            # YYYY-MM-DD 형식
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        else:
            end_dt = end_date
        
        # 주말/공휴일을 고려해 여유있게 가져오기
        start_date = end_dt - timedelta(days=int(days * 1.5))
        
        try:
            df = fdr.DataReader(ticker, start_date, end_dt)
            
            if df is None or len(df) == 0:
                return None
            
            # 최근 N개 거래일만 반환
            if len(df) > days:
                df = df.tail(days)
            
            return df
            
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    @staticmethod
    def get_stock_info(ticker):
        """
        종목 상세 정보 가져오기
        
        Args:
            ticker (str): 종목 코드
        
        Returns:
            dict: 종목 정보
        """
        df_krx = fdr.StockListing('KRX')
        stock_info = df_krx[df_krx['Code'] == ticker]
        
        if len(stock_info) > 0:
            return stock_info.iloc[0].to_dict()
        return None
    
    @staticmethod
    def search_stocks_by_name(name):
        """
        종목명으로 검색
        
        Args:
            name (str): 검색할 종목명
        
        Returns:
            pd.DataFrame: 검색 결과
        """
        df_krx = fdr.StockListing('KRX')
        result = df_krx[df_krx['Name'].str.contains(name, na=False)]
        return result
    
    @staticmethod
    def get_multiple_stocks_data(tickers, days=200):
        """
        여러 종목의 데이터를 한번에 가져오기
        
        Args:
            tickers (list): 종목 코드 리스트
            days (int): 과거 몇 일간의 데이터
        
        Returns:
            dict: {ticker: DataFrame} 형태
        """
        result = {}
        
        for ticker in tickers:
            df = KoreanStockDataProvider.get_price_data(ticker, days)
            if df is not None and len(df) > 0:
                result[ticker] = df
        
        return result
    
    @staticmethod
    def get_top_etfs_by_volume(top_n=300, exclude_leverage=True):
        """
        거래량 기준 상위 ETF 가져오기
        
        Args:
            top_n (int): 상위 몇 개 ETF
            exclude_leverage (bool): 레버리지/인버스 제외 여부
        
        Returns:
            pd.DataFrame: 거래량 순 정렬된 ETF 정보
        """
        print(f"Fetching top {top_n} ETFs by volume...")
        
        try:
            # ETF 목록 가져오기
            df_etf = fdr.StockListing('ETF/KR')
            
            if df_etf is None or len(df_etf) == 0:
                print("ETF 목록을 가져올 수 없습니다.")
                return pd.DataFrame()
            
            # 레버리지/인버스 제외
            if exclude_leverage:
                exclude_keywords = ['레버리지', '인버스', '곱버스', '2X', '3X', '-1X', '-2X', 
                                   'SHORT', 'INVERSE', 'BEAR', '곱']
                mask = ~df_etf['Name'].str.contains('|'.join(exclude_keywords), case=False, na=False)
                df_etf = df_etf[mask]
            
            # 'Volume' 컬럼이 있으면 사용, 없으면 최근 거래량 조회
            if 'Volume' in df_etf.columns:
                df_etf = df_etf[df_etf['Volume'] > 0]
                df_etf = df_etf.sort_values('Volume', ascending=False).head(top_n)
                print(f"✅ 상위 {len(df_etf)}개 ETF 선택 완료 (Volume 컬럼 사용)")
                return df_etf
            
            # Volume 컬럼이 없으면 거래량 데이터 조회
            from datetime import datetime, timedelta
            import time
            
            volumes = []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            print(f"총 {len(df_etf)}개 ETF 거래량 조회 중...")
            
            for idx, row in df_etf.iterrows():
                try:
                    ticker = row['Code']
                    df_price = fdr.DataReader(ticker, start_date, end_date)
                    
                    if df_price is not None and len(df_price) > 0:
                        avg_volume = df_price['Volume'].mean()
                        volumes.append(avg_volume)
                    else:
                        volumes.append(0)
                    
                    # Rate limiting
                    time.sleep(0.05)
                    
                    if (idx + 1) % 50 == 0:
                        print(f"  진행: {idx + 1}/{len(df_etf)}")
                        
                except Exception as e:
                    volumes.append(0)
            
            df_etf['AvgVolume'] = volumes
            
            # 거래량 기준 정렬 및 상위 N개 선택
            df_etf = df_etf[df_etf['AvgVolume'] > 0]
            df_etf = df_etf.sort_values('AvgVolume', ascending=False).head(top_n)
            
            print(f"✅ 상위 {len(df_etf)}개 ETF 선택 완료")
            
            return df_etf
            
        except Exception as e:
            print(f"ETF 목록 가져오기 실패: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def convert_to_weekly(df_daily):
        """
        일봉 데이터를 주봉으로 변환 (차트 표시용)
        거래정지 일자는 제외하고 실제 거래된 날만 집계
        
        Args:
            df_daily: 일봉 DataFrame (DatetimeIndex)
        
        Returns:
            pd.DataFrame: 주봉 데이터
        """
        if df_daily is None or len(df_daily) == 0:
            return None
        
        # 거래정지 기간 제거 (Volume=0인 날)
        df_daily = df_daily[df_daily['Volume'] > 0].copy()
        
        if len(df_daily) == 0:
            return None
        
        # 주봉으로 리샘플링 (금요일 기준)
        # 각 주에서 실제 거래된 날만 집계
        df_weekly = df_daily.resample('W-FRI').agg({
            'Open': 'first',   # 주의 첫 거래일 시가
            'High': 'max',     # 주간 최고가
            'Low': 'min',      # 주간 최저가
            'Close': 'last',   # 주의 마지막 거래일 종가
            'Volume': 'sum'    # 주간 총 거래량
        }).dropna()
        
        # 거래량이 0인 주 제거 (전체 주가 거래정지인 경우)
        df_weekly = df_weekly[df_weekly['Volume'] > 0].copy()
        
        return df_weekly
    
    @staticmethod
    def get_price_data_weekly(ticker, weeks=500, end_date=None):
        """
        종목의 주봉 데이터 가져오기 (차트 표시용)
        
        Args:
            ticker (str): 종목 코드
            weeks (int): 주 수 (기본 500주)
            end_date (str): 종료 날짜 (YYYY-MM-DD)
        
        Returns:
            pd.DataFrame: 주봉 데이터
        """
        days = weeks * 5
        df_daily = KoreanStockDataProvider.get_price_data(ticker, days, end_date)
        
        if df_daily is None or len(df_daily) == 0:
            return None
        
        # convert_to_weekly에서 거래정지 기간 필터링 처리
        df_weekly = KoreanStockDataProvider.convert_to_weekly(df_daily)
        
        if df_weekly is None or len(df_weekly) == 0:
            return None
        
        if len(df_weekly) > weeks:
            df_weekly = df_weekly.tail(weeks)
        
        return df_weekly