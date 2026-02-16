"""
기술적 지표 계산 모듈
ATR, 스윙 하이/로우 등의 계산 함수들
"""

import pandas as pd
import numpy as np


def calculate_atr(df, period=10):
    """
    Average True Range (ATR) 계산
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        period (int): ATR 계산 기간
    
    Returns:
        pd.Series: ATR 값
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range 계산
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    # 세 가지 중 최대값
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 이동평균
    atr = tr.rolling(window=period).mean()
    
    return atr


def find_swing_highs(df, length):
    """
    스윙 하이(고점) 찾기
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        length (int): 스윙 감지 길이
    
    Returns:
        list: 스윙 하이 정보 딕셔너리 리스트
    """
    swing_highs = []
    
    for i in range(length, len(df)):
        # 윈도우 내 최고가
        window_high = df['High'].iloc[i-length:i+1].max()
        
        # 스윙 하이 조건: 중심점이 윈도우 내에서 최고가
        if (df['High'].iloc[i-length] > window_high and 
            df['High'].iloc[i-length] == df['High'].iloc[i-length:i].max()):
            
            swing_highs.append({
                'index': i - length,
                'price': df['High'].iloc[i-length],
                'volume': df['Volume'].iloc[i-length],
                'crossed': False
            })
    
    return swing_highs


def find_swing_lows(df, length):
    """
    스윙 로우(저점) 찾기
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        length (int): 스윙 감지 길이
    
    Returns:
        list: 스윙 로우 정보 딕셔너리 리스트
    """
    swing_lows = []
    
    for i in range(length, len(df)):
        # 윈도우 내 최저가
        window_low = df['Low'].iloc[i-length:i+1].min()
        
        # 스윙 로우 조건: 중심점이 윈도우 내에서 최저가
        if (df['Low'].iloc[i-length] < window_low and 
            df['Low'].iloc[i-length] == df['Low'].iloc[i-length:i].min()):
            
            swing_lows.append({
                'index': i - length,
                'price': df['Low'].iloc[i-length],
                'volume': df['Volume'].iloc[i-length],
                'crossed': False
            })
    
    return swing_lows


def find_swings(df, length):
    """
    스윙 하이와 스윙 로우를 모두 찾기
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        length (int): 스윙 감지 길이
    
    Returns:
        tuple: (swing_highs, swing_lows)
    """
    swing_highs = find_swing_highs(df, length)
    swing_lows = find_swing_lows(df, length)
    
    return swing_highs, swing_lows


def calculate_ema(series, period):
    """
    Exponential Moving Average (EMA) 계산
    
    Args:
        series (pd.Series): 가격 시리즈
        period (int): EMA 기간
    
    Returns:
        pd.Series: EMA 값
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series, period):
    """
    Simple Moving Average (SMA) 계산
    
    Args:
        series (pd.Series): 가격 시리즈
        period (int): SMA 기간
    
    Returns:
        pd.Series: SMA 값
    """
    return series.rolling(window=period).mean()


def calculate_rsi(df, period=14):
    """
    Relative Strength Index (RSI) 계산
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        period (int): RSI 기간
    
    Returns:
        pd.Series: RSI 값
    """
    close = df['Close']
    delta = close.diff()
    
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_volume_sma(df, period=20):
    """
    거래량 이동평균 계산
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        period (int): 이동평균 기간
    
    Returns:
        pd.Series: 거래량 이동평균
    """
    return df['Volume'].rolling(window=period).mean()


def is_high_volume(df, idx, multiplier=1.5):
    """
    특정 캔들이 고거래량인지 확인
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        idx (int): 확인할 인덱스
        multiplier (float): 평균 대비 배수
    
    Returns:
        bool: 고거래량 여부
    """
    volume_sma = calculate_volume_sma(df, 20)
    
    if idx < len(volume_sma) and pd.notna(volume_sma.iloc[idx]):
        return df['Volume'].iloc[idx] > volume_sma.iloc[idx] * multiplier
    
    return False


def calculate_bollinger_bands(df, period=20, std_dev=2):
    """
    볼린저 밴드 계산
    
    Args:
        df (pd.DataFrame): OHLCV 데이터프레임
        period (int): 이동평균 기간
        std_dev (float): 표준편차 배수
    
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    close = df['Close']
    sma = calculate_sma(close, period)
    std = close.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    return upper_band, sma, lower_band