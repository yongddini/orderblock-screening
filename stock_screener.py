"""
종목 스크리닝 모듈
오더블록에 근접한 종목을 찾는 핵심 기능 (Realtime 방식)
"""

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from realtime_detector import RealtimeOrderBlockDetector
from data_provider import KoreanStockDataProvider


class StockScreener:
    """오더블록 기반 종목 스크리너"""
    
    def __init__(self, proximity_percent=1.0, swing_length=10, 
                 max_atr_mult=2.0, ob_end_method="Wick", combine_obs=True, max_workers=4):
        """
        Args:
            proximity_percent (float): 오더블록 근접도 기준 (%, 기본 1%)
            swing_length (int): 스윙 감지 길이
            max_atr_mult (float): 최대 ATR 배수
            ob_end_method (str): 오더블록 무효화 방식
            combine_obs (bool): 오더블록 합치기 (기본 True)
            max_workers (int): 병렬 처리 워커 수
        """
        self.proximity_percent = proximity_percent
        self.swing_length = swing_length
        self.max_atr_mult = max_atr_mult
        self.ob_end_method = ob_end_method
        self.combine_obs = combine_obs
        self.max_workers = max_workers
    
    def classify_position(self, current_price, bull_obs, bear_obs):
        """
        현재가가 오더블록과의 관계 분류
        
        Args:
            current_price (float): 현재가
            bull_obs (list): 강세(지지) 오더블록 리스트
            bear_obs (list): 약세(저항) 오더블록 리스트
        
        Returns:
            dict: {
                'status': '내부-지지' | '근접-지지' | '내부-저항' | '근접-저항' | None,
                'ob_type': 'Bull' | 'Bear' | None,
                'ob_top': float,
                'ob_bottom': float,
                'distance_percent': float
            }
        """
        # 활성 오더블록만
        active_bull = [ob for ob in bull_obs if not ob.breaker]
        active_bear = [ob for ob in bear_obs if not ob.breaker]
        
        # 1. 지지선(강세 오더블록) 확인
        for ob in active_bull:
            # 내부: 오더블록 안에 있음
            if ob.bottom <= current_price <= ob.top:
                return {
                    'status': '내부-지지',
                    'ob_type': 'Bull',
                    'ob_top': ob.top,
                    'ob_bottom': ob.bottom,
                    'distance_percent': 0
                }
            
            # 근접: 상단 1% 이내 (위쪽)
            distance_from_top = ((current_price - ob.top) / current_price) * 100
            if 0 < distance_from_top <= self.proximity_percent:
                return {
                    'status': '근접-지지',
                    'ob_type': 'Bull',
                    'ob_top': ob.top,
                    'ob_bottom': ob.bottom,
                    'distance_percent': distance_from_top
                }
        
        # 2. 저항선(약세 오더블록) 확인
        for ob in active_bear:
            # 내부: 오더블록 안에 있음
            if ob.bottom <= current_price <= ob.top:
                return {
                    'status': '내부-저항',
                    'ob_type': 'Bear',
                    'ob_top': ob.top,
                    'ob_bottom': ob.bottom,
                    'distance_percent': 0
                }
            
            # 근접: 하단 1% 이내 (아래쪽)
            distance_from_bottom = ((ob.bottom - current_price) / current_price) * 100
            if 0 < distance_from_bottom <= self.proximity_percent:
                return {
                    'status': '근접-저항',
                    'ob_type': 'Bear',
                    'ob_top': ob.top,
                    'ob_bottom': ob.bottom,
                    'distance_percent': distance_from_bottom
                }
        
        # 해당 없음
        return {
            'status': None,
            'ob_type': None,
            'ob_top': None,
            'ob_bottom': None,
            'distance_percent': None
        }
    
    def check_proximity(self, ticker, days=500, end_date=None):
        """
        특정 종목이 오더블록에 근접했는지 확인 (새 분류 방식)
        
        Args:
            ticker (str): 종목 코드
            days (int): 분석할 과거 데이터 일수
            end_date (str): 종료 날짜 (YYYY-MM-DD). None이면 오늘
        
        Returns:
            dict or None: 분류 결과, 해당 없으면 None
        """
        try:
            # 가격 데이터 가져오기
            df = KoreanStockDataProvider.get_price_data(ticker, days, end_date)
            
            if df is None or len(df) < 50:
                return None
            
            # 거래정지 기간 제거 (거래량이 0인 날)
            df = df[df['Volume'] > 0].copy()
            
            if len(df) < 50:
                return None
            
            # 오더블록 감지 (실시간 방식, Combine 적용)
            detector = RealtimeOrderBlockDetector(
                swing_length=self.swing_length,
                max_atr_mult=self.max_atr_mult,
                ob_end_method=self.ob_end_method,
                combine_obs=self.combine_obs
            )
            history = detector.detect_order_blocks_realtime(df)
            bull_obs, bear_obs = detector.get_latest_orderblocks()
            
            # Zone Count: Low (최신 3개씩)
            bull_obs = bull_obs[:3]
            bear_obs = bear_obs[:3]
            
            # 현재가
            current_price = df['Close'].iloc[-1]
            
            # 금일 등락율 계산
            prev_close = df['Close'].iloc[-2] if len(df) >= 2 else current_price
            change_percent = ((current_price - prev_close) / prev_close * 100) if prev_close != 0 else 0
            
            # RSI 계산 (14일) - RMA 방식
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            
            # RMA (Wilder's Smoothing) = EWM with alpha=1/length
            alpha = 1.0 / 14
            avg_gain = gain.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = loss.ewm(alpha=alpha, adjust=False).mean()
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]
            
            # 거래대금 계산 (당일)
            current_volume = df['Volume'].iloc[-1]
            trading_value = current_volume * current_price  # 거래대금 = 거래량 × 현재가
            
            # 위치 분류
            position = self.classify_position(current_price, bull_obs, bear_obs)
            
            if position['status']:
                return {
                    'current_price': current_price,
                    'change_percent': change_percent,
                    'rsi': current_rsi,
                    'trading_value': trading_value,  # 거래대금 추가
                    'status': position['status'],
                    'ob_type': position['ob_type'],
                    'ob_top': position['ob_top'],
                    'ob_bottom': position['ob_bottom'],
                    'distance_percent': position['distance_percent']
                }
            
            return None
            
        except Exception as e:
            print(f"{ticker} 처리 중 오류: {str(e)}")
            return None
    
    def _is_near_orderblock(self, current_price, ob):
        """
        현재가가 오더블록에 근접했는지 판단
        
        Args:
            current_price (float): 현재가
            ob (OrderBlockInfo): 오더블록 정보
        
        Returns:
            bool: 근접 여부
        """
        distance_to_top = ob.get_distance_to_top(current_price)
        distance_to_bottom = ob.get_distance_to_bottom(current_price)
        
        # 오더블록 구간 내에 있거나
        if ob.is_price_in_zone(current_price):
            return True
        
        # 상단/하단에 근접한 경우
        if (-self.proximity_percent <= distance_to_bottom <= self.proximity_percent or
            -self.proximity_percent <= distance_to_top <= self.proximity_percent):
            return True
        
        return False
    
    def screen_market(self, market='KOSPI', top_n=400, days=200, end_date=None):
        """
        특정 시장의 상위 종목들을 스크리닝
        
        Args:
            market (str): 'KOSPI' 또는 'KOSDAQ'
            top_n (int): 상위 몇 개 종목
            days (int): 분석할 과거 데이터 일수
            end_date (str): 종료 날짜 (YYYY-MM-DD). None이면 오늘
        
        Returns:
            list: 오더블록에 근접한 종목들의 정보
        """
        results = []
        
        # 시가총액 상위 종목 가져오기
        df_stocks = KoreanStockDataProvider.get_top_stocks_by_market_cap(market, top_n)
        
        print(f"\n{market} {len(df_stocks)}개 종목 스크리닝 중...")
        
        # 개별 종목 처리 함수
        def process_stock(ticker, name):
            try:
                result = self.check_proximity(ticker, days, end_date)
                if result:
                    return {
                        'Market': market,
                        'Code': ticker,
                        'Name': name,
                        'Current_Price': result['current_price'],
                        'Change_Percent': result['change_percent'],
                        'RSI': result['rsi'],
                        'trading_value': result['trading_value'],  # 거래대금 사용
                        'Status': result['status'],
                        'OB_Type': result['ob_type'],
                        'OB_Top': result['ob_top'],
                        'OB_Bottom': result['ob_bottom'],
                        'Distance_Percent': result['distance_percent']
                    }
            except Exception as e:
                print(f"\n⚠️  {ticker} ({name}) 오류: {e}")
            return None

        # 10개씩 동시 처리 + 프로그레스 바
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {
                executor.submit(process_stock, row['Code'], row['Name']): row['Code']
                for _, row in df_stocks.iterrows()
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=market, unit="종목"):
                result = future.result()
                if result:
                    results.append(result)
        
        print(f"오더블록 근처 종목 {len(results)}개 발견")
        
        return results
    
    def screen_multiple_markets(self, markets=['KOSPI', 'KOSDAQ'], 
                                top_n=400, days=200, end_date=None):
        """
        여러 시장을 동시에 스크리닝
        
        Args:
            markets (list): 시장 리스트
            top_n (int): 각 시장에서 상위 몇 개 종목
            days (int): 분석할 과거 데이터 일수
            end_date (str): 종료 날짜 (YYYY-MM-DD). None이면 오늘
        
        Returns:
            pd.DataFrame: 스크리닝 결과
        """
        all_results = []
        
        for market in markets:
            market_results = self.screen_market(market, top_n, days, end_date)
            all_results.extend(market_results)
        
        if all_results:
            return pd.DataFrame(all_results)
        else:
            return pd.DataFrame()
    
    def filter_by_orderblock_type(self, results_df, ob_type='Bullish'):
        """
        특정 타입의 오더블록만 필터링
        
        Args:
            results_df (pd.DataFrame): 스크리닝 결과
            ob_type (str): 'Bullish' 또는 'Bearish'
        
        Returns:
            pd.DataFrame: 필터링된 결과
        """
        filtered = []
        
        for idx, row in results_df.iterrows():
            for ob in row['Order_Blocks']:
                if ob['type'] == ob_type:
                    filtered.append({
                        'Market': row['Market'],
                        'Code': row['Code'],
                        'Name': row['Name'],
                        'Current_Price': row['Current_Price'],
                        'trading_value': row['trading_value'],
                        'OB_Type': ob['type'],
                        'OB_Top': ob['top'],
                        'OB_Bottom': ob['bottom'],
                        'In_Zone': ob['in_zone'],
                        'Distance_to_Top': ob['distance_to_top'],
                        'Distance_to_Bottom': ob['distance_to_bottom']
                    })
        
        return pd.DataFrame(filtered)
    
    def filter_in_zone(self, results_df):
        """
        현재가가 오더블록 구간 내에 있는 종목만 필터링
        
        Args:
            results_df (pd.DataFrame): 스크리닝 결과
        
        Returns:
            pd.DataFrame: 필터링된 결과
        """
        filtered = []
        
        for idx, row in results_df.iterrows():
            for ob in row['Order_Blocks']:
                if ob['in_zone']:
                    filtered.append({
                        'Market': row['Market'],
                        'Code': row['Code'],
                        'Name': row['Name'],
                        'Current_Price': row['Current_Price'],
                        'trading_value': row['trading_value'],
                        'OB_Type': ob['type'],
                        'OB_Top': ob['top'],
                        'OB_Bottom': ob['bottom'],
                        'In_Zone': 'YES',
                        'Distance_to_Top': ob['distance_to_top'],
                        'Distance_to_Bottom': ob['distance_to_bottom']
                    })
        
        return pd.DataFrame(filtered)
    
    def get_buy_signals(self, results_df):
        """
        매수 시그널 (강세 OB 구간 내)
        
        Args:
            results_df (pd.DataFrame): 스크리닝 결과
        
        Returns:
            pd.DataFrame: 매수 시그널 종목
        """
        filtered = []
        
        for idx, row in results_df.iterrows():
            for ob in row['Order_Blocks']:
                if ob['type'] == 'Bullish' and ob['in_zone']:
                    filtered.append({
                        'Market': row['Market'],
                        'Code': row['Code'],
                        'Name': row['Name'],
                        'Current_Price': row['Current_Price'],
                        'Support_Level': ob['bottom'],
                        'Target_Level': ob['top'],
                        'Distance_to_Support': ob['distance_to_bottom']
                    })
        
        return pd.DataFrame(filtered)
    
    def get_sell_signals(self, results_df):
        """
        매도 시그널 (약세 OB 구간 내)
        
        Args:
            results_df (pd.DataFrame): 스크리닝 결과
        
        Returns:
            pd.DataFrame: 매도 시그널 종목
        """
        filtered = []
        
        for idx, row in results_df.iterrows():
            for ob in row['Order_Blocks']:
                if ob['type'] == 'Bearish' and ob['in_zone']:
                    filtered.append({
                        'Market': row['Market'],
                        'Code': row['Code'],
                        'Name': row['Name'],
                        'Current_Price': row['Current_Price'],
                        'Resistance_Level': ob['top'],
                        'Stop_Level': ob['bottom'],
                        'Distance_to_Resistance': ob['distance_to_top']
                    })
        
        return pd.DataFrame(filtered)
    
    def screen_etf(self, top_n=300, days=200, end_date=None):
        """
        거래량 상위 ETF 스크리닝
        
        Args:
            top_n (int): 상위 몇 개 ETF
            days (int): 분석할 과거 데이터 일수
            end_date (str): 종료 날짜 (YYYY-MM-DD). None이면 오늘
        
        Returns:
            list: 오더블록에 근접한 ETF 정보
        """
        results = []
        
        # 거래량 상위 ETF 가져오기
        df_etfs = KoreanStockDataProvider.get_top_etfs_by_volume(top_n, exclude_leverage=True)
        
        if df_etfs.empty:
            print("❌ ETF 목록을 가져올 수 없습니다.")
            return results
        
        # 컬럼명 확인 및 표준화
        print(f"ETF DataFrame 컬럼: {df_etfs.columns.tolist()}")
        
        # Symbol/Code 컬럼 통일
        if 'Symbol' in df_etfs.columns and 'Code' not in df_etfs.columns:
            df_etfs['Code'] = df_etfs['Symbol']
        elif 'Ticker' in df_etfs.columns and 'Code' not in df_etfs.columns:
            df_etfs['Code'] = df_etfs['Ticker']
        
        print(f"\nETF {len(df_etfs)}개 종목 스크리닝 중...")
        
        # 개별 ETF 처리 함수
        def process_etf(ticker, name):
            try:
                result = self.check_proximity(ticker, days, end_date)
                if result:
                    return {
                        'Market': 'ETF',
                        'Code': ticker,
                        'Name': name,
                        'Current_Price': result['current_price'],
                        'Change_Percent': result['change_percent'],
                        'RSI': result['rsi'],
                        'trading_value': result['trading_value'],  # 거래대금 사용
                        'Status': result['status'],
                        'OB_Type': result['ob_type'],
                        'OB_Top': result['ob_top'],
                        'OB_Bottom': result['ob_bottom'],
                        'Distance_Percent': result['distance_percent']
                    }
            except Exception as e:
                pass
            return None
        
        # 병렬 처리
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_etf, row['Code'], row['Name']): row['Code'] 
                for _, row in df_etfs.iterrows()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="ETF 스크리닝"):
                result = future.result()
                if result:
                    results.append(result)
        
        print(f"✅ ETF 스크리닝 완료: {len(results)}개 발견")
        return results
    
    def check_proximity_weekly(self, ticker, weeks=500, end_date=None):
        """
        주봉 기준 오더블록 근접도 체크
        
        Args:
            ticker (str): 종목 코드
            weeks (int): 분석할 주 수 (기본 500주 = ~10년)
            end_date (str): 종료 날짜 (YYYY-MM-DD). None이면 오늘
        
        Returns:
            dict or None: 근접한 경우 상세 정보, 아니면 None
        """
        # 주봉 데이터 가져오기
        df = KoreanStockDataProvider.get_price_data_weekly(ticker, weeks, end_date)
        
        if df is None or len(df) < 50:
            return None
        
        # 주봉 OB 감지
        detector = RealtimeOrderBlockDetector(
            swing_length=self.swing_length,
            max_atr_mult=self.max_atr_mult,
            ob_end_method=self.ob_end_method,
            combine_obs=self.combine_obs
        )
        
        detector.detect_order_blocks_realtime(df)
        bull_obs, bear_obs = detector.get_latest_orderblocks()
        
        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
        change_percent = ((current_price - prev_price) / prev_price) * 100
        
        # RSI 계산 (주봉 RMA)
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
        current_rsi = rsi.iloc[-1] if len(rsi) > 0 else 50
        
        # 거래대금 계산 (주봉 마지막 주)
        current_volume = df['Volume'].iloc[-1]
        trading_value = current_volume * current_price
        
        # 오더블록 근접도 판단
        position_info = self.classify_position(current_price, bull_obs, bear_obs)
        
        if position_info['status']:
            return {
                'current_price': current_price,
                'change_percent': change_percent,
                'rsi': current_rsi,
                'trading_value': trading_value,  # 거래대금 추가
                'status': position_info['status'],
                'ob_type': position_info['ob_type'],
                'ob_top': position_info['ob_top'],
                'ob_bottom': position_info['ob_bottom'],
                'distance_percent': position_info['distance_percent']
            }
        
        return None
    
    def screen_market_weekly(self, market='KOSPI', top_n=400, weeks=500, end_date=None):
        """
        주봉 기준 시장 스크리닝
        
        Args:
            market (str): 'KOSPI', 'KOSDAQ', 'ETF'
            top_n (int): 상위 종목 수
            weeks (int): 분석할 주 수 (기본 500주 = ~10년)
            end_date (str): 종료 날짜 (YYYY-MM-DD). None이면 오늘
        
        Returns:
            list: 스크리닝 결과
        """
        results = []
        
        if market == 'ETF':
            df_stocks = KoreanStockDataProvider.get_top_etfs_by_volume(top_n, exclude_leverage=True)
            if df_stocks.empty:
                return results
            
            # 컬럼명 표준화
            if 'Symbol' in df_stocks.columns and 'Code' not in df_stocks.columns:
                df_stocks['Code'] = df_stocks['Symbol']
            elif 'Ticker' in df_stocks.columns and 'Code' not in df_stocks.columns:
                df_stocks['Code'] = df_stocks['Ticker']
        else:
            df_stocks = KoreanStockDataProvider.get_top_stocks_by_market_cap(market, top_n)
        
        print(f"\n{market} 주봉 {len(df_stocks)}개 종목 스크리닝 중...")
        
        def process_stock(ticker, name):
            try:
                result = self.check_proximity_weekly(ticker, weeks, end_date)
                if result:
                    return {
                        'Market': market,
                        'Code': ticker,
                        'Name': name,
                        'Current_Price': result['current_price'],
                        'Change_Percent': result['change_percent'],
                        'RSI': result['rsi'],
                        'trading_value': result['trading_value'],  # 거래대금 사용
                        'Status': result['status'],
                        'OB_Type': result['ob_type'],
                        'OB_Top': result['ob_top'],
                        'OB_Bottom': result['ob_bottom'],
                        'Distance_Percent': result['distance_percent']
                    }
            except Exception as e:
                pass
            return None
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(process_stock, row['Code'], row['Name']): row['Code']
                for _, row in df_stocks.iterrows()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"{market} 주봉"):
                result = future.result()
                if result:
                    results.append(result)
        
        print(f"✅ {market} 주봉 스크리닝 완료: {len(results)}개 발견")
        return results