"""
실시간 시뮬레이션 오더블록 감지기
Look-ahead Bias 제거 버전
트레이딩뷰처럼 캔들 하나씩 처리
"""

import pandas as pd
from orderblock_info import OrderBlockInfo


class RealtimeOrderBlockDetector:
    """
    실시간 시뮬레이션 오더블록 감지기
    과거 시점에서 미래 정보를 사용하지 않음
    """
    
    def __init__(self, swing_length=10, max_atr_mult=2.0, 
                 max_order_blocks=30, ob_end_method="Wick", 
                 combine_obs=False):
        """
        초기화
        
        Args:
            swing_length (int): 스윙 감지 길이
            max_atr_mult (float): ATR 배수 필터
            max_order_blocks (int): 최대 보관 오더블록 수
            ob_end_method (str): 오더블록 무효화 방법 ("Wick" or "Close")
            combine_obs (bool): 인접한 오더블록 합치기 활성화 (IoU 방식)
        """
        self.swing_length = swing_length
        self.max_atr_mult = max_atr_mult
        self.max_order_blocks = max_order_blocks
        self.ob_end_method = ob_end_method
        self.combine_obs = combine_obs
        
        # 최신 오더블록만 저장 (history 제거)
        self.latest_bull_obs = []
        self.latest_bear_obs = []
    
    def combine_order_blocks(self, order_blocks):
        """
        인접한 오더블록을 합치기 (Pine Script와 동일한 로직)
        IoU (Intersection over Union) 방식 사용
        
        Args:
            order_blocks: 오더블록 리스트
        
        Returns:
            합쳐진 오더블록 리스트
        """
        if not self.combine_obs or len(order_blocks) <= 1:
            return order_blocks
        
        # Pine Script의 combineOBsFunc 재현
        # 더 이상 합칠 게 없을 때까지 반복
        
        result = [ob for ob in order_blocks]  # 복사
        
        last_combinations = 999
        while last_combinations > 0:
            last_combinations = 0
            new_result = []
            disabled_indices = set()
            
            for i in range(len(result)):
                if i in disabled_indices:
                    continue
                    
                ob1 = result[i]
                combined = False
                
                for j in range(len(result)):
                    if i == j or j in disabled_indices:
                        continue
                    
                    ob2 = result[j]
                    
                    # 같은 타입만 합침 (Bull끼리, Bear끼리)
                    if ob1.ob_type != ob2.ob_type:
                        continue
                    
                    # 두 오더블록이 겹치는지 확인 (IoU)
                    if self._do_obs_touch(ob1, ob2):
                        # 두 OB를 합쳐서 새로운 OB 생성
                        new_ob = self._merge_order_blocks(ob1, ob2)
                        new_result.append(new_ob)
                        
                        # 원본 두 개는 비활성화
                        disabled_indices.add(i)
                        disabled_indices.add(j)
                        
                        last_combinations += 1
                        combined = True  # 병합되었음을 표시
                        break
                
                if not combined:
                    new_result.append(ob1)
            
            result = new_result
        
        return result
    
    def _area_of_ob(self, ob):
        """오더블록의 면적 계산 (Pine Script areaOfOB)"""
        # 시간 범위
        x1 = ob.start_time.timestamp() if ob.start_time else 0
        x2 = ob.break_time.timestamp() if ob.break_time else x1 + 86400  # 하루 추가
        
        # 가격 범위
        y1 = ob.top
        y2 = ob.bottom
        
        # 면적 = 가로 × 세로
        width = abs(x2 - x1)
        height = abs(y1 - y2)
        
        return width * height
    
    def _do_obs_touch(self, ob1, ob2):
        """
        두 오더블록이 겹치는지 확인 (Pine Script doOBsTouch)
        IoU (Intersection over Union) 사용
        """
        # OB1 좌표
        x1_start = ob1.start_time.timestamp() if ob1.start_time else 0
        x1_end = ob1.break_time.timestamp() if ob1.break_time else x1_start + 86400
        y1_top = ob1.top
        y1_bottom = ob1.bottom
        
        # OB2 좌표
        x2_start = ob2.start_time.timestamp() if ob2.start_time else 0
        x2_end = ob2.break_time.timestamp() if ob2.break_time else x2_start + 86400
        y2_top = ob2.top
        y2_bottom = ob2.bottom
        
        # 교집합 영역 계산
        x_overlap = max(0, min(x1_end, x2_end) - max(x1_start, x2_start))
        y_overlap = max(0, min(y1_top, y2_top) - max(y1_bottom, y2_bottom))
        intersection_area = x_overlap * y_overlap
        
        # 합집합 영역 계산
        area1 = self._area_of_ob(ob1)
        area2 = self._area_of_ob(ob2)
        union_area = area1 + area2 - intersection_area
        
        # 겹침 비율 (IoU)
        if union_area > 0:
            overlap_percentage = (intersection_area / union_area) * 100.0
        else:
            overlap_percentage = 0
        
        # overlapThresholdPercentage = 0 (Pine Script 기본값)
        threshold = 0
        
        return overlap_percentage > threshold
    
    def _merge_order_blocks(self, ob1, ob2):
        """두 오더블록을 합치기 (Pine Script 로직)"""
        from orderblock_info import OrderBlockInfo
        import copy
        
        # 새로운 합쳐진 오더블록 생성
        merged = copy.deepcopy(ob1)
        
        # 상단/하단: 두 OB를 포함하는 범위
        merged.top = max(ob1.top, ob2.top)
        merged.bottom = min(ob1.bottom, ob2.bottom)
        
        # 거래량: 합산
        merged.ob_volume = ob1.ob_volume + ob2.ob_volume
        
        # low/high 거래량도 합산
        if hasattr(merged, 'ob_low_volume'):
            merged.ob_low_volume = getattr(ob1, 'ob_low_volume', 0) + getattr(ob2, 'ob_low_volume', 0)
        if hasattr(merged, 'ob_high_volume'):
            merged.ob_high_volume = getattr(ob1, 'ob_high_volume', 0) + getattr(ob2, 'ob_high_volume', 0)
        
        # 시작 시간: 더 이른 것
        if ob1.start_time and ob2.start_time:
            merged.start_time = min(ob1.start_time, ob2.start_time)
        
        # 무효화 시간: 더 늦은 것
        if ob1.breaker or ob2.breaker:
            merged.breaker = True
            if ob1.break_time and ob2.break_time:
                merged.break_time = max(ob1.break_time, ob2.break_time)
            elif ob1.break_time:
                merged.break_time = ob1.break_time
            elif ob2.break_time:
                merged.break_time = ob2.break_time
        
        # Combined 플래그 설정 비활성화
        merged.combined = False
        
        return merged
        
    def detect_order_blocks_realtime(self, df):
        """
        실시간 시뮬레이션 오더블록 감지
        각 시점에서 당시에 알 수 있었던 정보만 사용
        
        Args:
            df: OHLCV 데이터프레임 (날짜 인덱스)
        """
        df = df.copy()
        
        # 날짜 인덱스를 보존
        original_index = df.index.copy()
        df.reset_index(drop=True, inplace=True)
        
        bullish_order_blocks = []
        bearish_order_blocks = []
        
        # 스윙 추적
        swing_highs = []
        swing_lows = []
        
        # ATR 전체 한번만 계산
        atr = self._calculate_atr(df, period=10)
        
        # numpy array로 한번만 추출 (iloc 반복 호출 방지)
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        volumes = df['Volume'].values
        atr_values = atr.values
        
        # 스윙 중복 체크용 set
        swing_high_indices = set()
        swing_low_indices = set()
        
        # 캔들 하나씩 처리 (실시간 시뮬레이션)
        for i in range(self.swing_length + 1, len(df)):
            # ATR 값 직접 참조
            current_atr = atr_values[i] if pd.notna(atr_values[i]) else 0
            
            # 현재 캔들 정보
            current_close = closes[i]
            current_high = highs[i]
            current_low = lows[i]
            current_open = opens[i]
            current_volume = volumes[i]
            current_time = original_index[i]
            
            # 스윙 감지
            self._update_swings(highs, lows, volumes, swing_highs, swing_lows, swing_high_indices, swing_low_indices, i, len(df))
            
            # 기존 오더블록 업데이트
            self._update_existing_orderblocks(
                bullish_order_blocks, bearish_order_blocks,
                current_low, current_high, current_open, current_close,
                current_volume, current_time
            )
            
            # 새로운 오더블록 생성
            self._create_new_orderblocks(
                highs, lows, volumes, i, current_close, current_atr,
                swing_highs, swing_lows,
                bullish_order_blocks, bearish_order_blocks,
                original_index
            )
        
        # 최신 오더블록 저장
        self.latest_bull_obs = bullish_order_blocks
        self.latest_bear_obs = bearish_order_blocks
    
    def get_latest_orderblocks(self):
        """가장 최근 시점의 오더블록 (Combine 적용 + 포함관계 제거)"""
        bull_obs = self.latest_bull_obs
        bear_obs = self.latest_bear_obs
        
        # Combine 기능 적용
        if self.combine_obs:
            bull_obs = self.combine_order_blocks(bull_obs)
            bear_obs = self.combine_order_blocks(bear_obs)
        
        # 포함관계 제거: 작은 OB가 큰 OB에 완전히 포함되면 큰 OB 제거
        bull_obs = self._remove_containing_obs(bull_obs)
        bear_obs = self._remove_containing_obs(bear_obs)
        
        return bull_obs, bear_obs
    
    def _remove_containing_obs(self, obs_list):
        """
        큰 OB가 작은 OB를 완전히 포함하면 큰 OB를 제거
        (더 정확한 작은 OB만 남김)
        
        조건: ob2의 범위가 ob1 범위 안에 있고, 완전히 동일하지 않으면 ob1 제거
        
        Args:
            obs_list: 오더블록 리스트
        
        Returns:
            포함관계가 제거된 오더블록 리스트
        """
        if len(obs_list) <= 1:
            return obs_list
        
        filtered = []
        
        for i, ob1 in enumerate(obs_list):
            is_container = False
            
            # ob1이 다른 OB를 완전히 포함하는지 확인
            for j, ob2 in enumerate(obs_list):
                if i == j:
                    continue
                
                # ob2가 ob1 안에 완전히 포함되고, 둘이 완전히 같지 않으면
                # ob1은 컨테이너 -> 제거 대상
                if (ob2.bottom >= ob1.bottom and ob2.top <= ob1.top):
                    # 완전히 같은 경우는 제외
                    if not (ob2.bottom == ob1.bottom and ob2.top == ob1.top):
                        is_container = True
                        break
            
            # 다른 OB를 포함하지 않으면 유지
            if not is_container:
                filtered.append(ob1)
        
        return filtered
    
    
    def _calculate_atr(self, df, period=10):
        """ATR 계산 (Average True Range)"""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    # 스윙 업데이트
    def _update_swings(self, highs, lows, volumes, swing_highs, swing_lows, swing_high_indices, swing_low_indices, current_idx, total_len):
        """
        스윙 업데이트 (현재 시점까지만)
        중요: 미래 데이터 사용 안함
        """
        length = self.swing_length
        
        # 스윙 하이 체크 (length 봉 전)
        swing_idx = current_idx - length
        if swing_idx >= length and swing_idx < total_len - length:
            window_start = max(0, swing_idx - length)
            window_end = min(total_len, swing_idx + length + 1)
            
            window_high = highs[window_start:window_end].max()
            
            if highs[swing_idx] >= window_high:
                if swing_idx not in swing_high_indices:
                    swing_high_indices.add(swing_idx)
                    swing_highs.append({
                        'index': swing_idx,
                        'price': highs[swing_idx],
                        'volume': volumes[swing_idx],
                        'crossed': False
                    })
        
        # 스윙 로우 체크
        if swing_idx >= length and swing_idx < total_len - length:
            window_start = max(0, swing_idx - length)
            window_end = min(total_len, swing_idx + length + 1)
            
            window_low = lows[window_start:window_end].min()
            
            if lows[swing_idx] <= window_low:
                if swing_idx not in swing_low_indices:
                    swing_low_indices.add(swing_idx)
                    swing_lows.append({
                        'index': swing_idx,
                        'price': lows[swing_idx],
                        'volume': volumes[swing_idx],
                        'crossed': False
                    })
    
    # 기존 오더블록 상태 업데이트
    def _update_existing_orderblocks(self, bull_obs, bear_obs,
                                     current_low, current_high,
                                     current_open, current_close,
                                     current_volume, current_time):
        """기존 오더블록 상태 업데이트"""
        
        # 강세 오더블록 업데이트
        for ob in bull_obs[:]:
            if not ob.breaker:
                if self.ob_end_method == "Wick":
                    if current_low < ob.bottom:
                        ob.breaker = True
                        ob.break_time = current_time
                        ob.bb_volume = current_volume
                else:  # Close
                    if min(current_open, current_close) < ob.bottom:
                        ob.breaker = True
                        ob.break_time = current_time
                        ob.bb_volume = current_volume
            else:
                if current_high > ob.top:
                    bull_obs.remove(ob)
        
        # 약세 오더블록 업데이트
        for ob in bear_obs[:]:
            if not ob.breaker:
                if self.ob_end_method == "Wick":
                    if current_high > ob.top:
                        ob.breaker = True
                        ob.break_time = current_time
                        ob.bb_volume = current_volume
                else:  # Close
                    if max(current_open, current_close) > ob.top:
                        ob.breaker = True
                        ob.break_time = current_time
                        ob.bb_volume = current_volume
            else:
                if current_low < ob.bottom:
                    bear_obs.remove(ob)
    
    # 새로운 오더블록 생성
    def _create_new_orderblocks(self, highs, lows, volumes, current_idx, current_close, current_atr,
                                swing_highs, swing_lows, bull_obs, bear_obs, original_index):
        """새로운 오더블록 생성"""
        
        # 강세 오더블록 (스윙 하이 돌파 시)
        for swing in swing_highs:
            if swing['crossed']:
                continue
            
            if current_idx > swing['index'] and current_close > swing['price']:
                swing['crossed'] = True
                
                # 오더블록 구간 찾기
                start_idx = swing['index'] + 1
                if start_idx >= current_idx:
                    continue
                
                # 스윙 하이 이후 ~ 현재까지의 최저점
                segment_low = lows[start_idx:current_idx+1]
                box_bottom = segment_low.min()
                box_bottom_idx = start_idx + segment_low.argmin()
                
                box_top = highs[box_bottom_idx]
                
                # 거래량
                vol_idx = current_idx
                if vol_idx >= 3:
                    ob_volume = volumes[vol_idx] + volumes[vol_idx-1] + volumes[vol_idx-2]
                    ob_low_volume = volumes[vol_idx-2]
                    ob_high_volume = volumes[vol_idx] + volumes[vol_idx-1]
                else:
                    ob_volume = volumes[vol_idx]
                    ob_low_volume = 0
                    ob_high_volume = ob_volume
                
                # ATR 필터
                ob_size = abs(box_top - box_bottom)
                if current_atr > 0 and ob_size <= current_atr * self.max_atr_mult:
                    start_date = original_index[box_bottom_idx] if box_bottom_idx < len(original_index) else original_index[-1]
                    
                    new_ob = OrderBlockInfo(
                        box_top, box_bottom, ob_volume, "Bull",
                        start_date
                    )
                    new_ob.ob_low_volume = ob_low_volume
                    new_ob.ob_high_volume = ob_high_volume
                    
                    bull_obs.insert(0, new_ob)
                    
                    if len(bull_obs) > self.max_order_blocks:
                        bull_obs.pop()
        
        # 약세 오더블록 (스윙 로우 돌파 시)
        for swing in swing_lows:
            if swing['crossed']:
                continue
            
            if current_idx > swing['index'] and current_close < swing['price']:
                swing['crossed'] = True
                
                start_idx = swing['index'] + 1
                if start_idx >= current_idx:
                    continue
                
                segment_high = highs[start_idx:current_idx+1]
                box_top = segment_high.max()
                box_top_idx = start_idx + segment_high.argmax()
                
                box_bottom = lows[box_top_idx]
                
                vol_idx = current_idx
                if vol_idx >= 3:
                    ob_volume = volumes[vol_idx] + volumes[vol_idx-1] + volumes[vol_idx-2]
                    ob_low_volume = volumes[vol_idx] + volumes[vol_idx-1]
                    ob_high_volume = volumes[vol_idx-2]
                else:
                    ob_volume = volumes[vol_idx]
                    ob_low_volume = ob_volume
                    ob_high_volume = 0
                
                ob_size = abs(box_top - box_bottom)
                if current_atr > 0 and ob_size <= current_atr * self.max_atr_mult:
                    start_date = original_index[box_top_idx] if box_top_idx < len(original_index) else original_index[-1]
                    
                    new_ob = OrderBlockInfo(
                        box_top, box_bottom, ob_volume, "Bear",
                        start_date
                    )
                    new_ob.ob_low_volume = ob_low_volume
                    new_ob.ob_high_volume = ob_high_volume
                    
                    bear_obs.insert(0, new_ob)
                    
                    if len(bear_obs) > self.max_order_blocks:
                        bear_obs.pop()
    
    # 오더블록 복사 (스냅샷용)