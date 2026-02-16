"""
Order Block 데이터 구조 모듈
오더블록 정보를 저장하는 클래스 정의
"""

class OrderBlockInfo:
    """Order Block 정보 저장 클래스"""
    
    def __init__(self, top, bottom, ob_volume, ob_type, start_time):
        """
        Args:
            top (float): 오더블록 상단 가격
            bottom (float): 오더블록 하단 가격
            ob_volume (float): 오더블록 형성 시 거래량
            ob_type (str): 오더블록 타입 ("Bull" 또는 "Bear")
            start_time: 오더블록 시작 시간
        """
        self.top = top
        self.bottom = bottom
        self.ob_volume = ob_volume
        self.ob_type = ob_type
        self.start_time = start_time
        
        # 추가 정보
        self.bb_volume = 0  # Breaker 발생 시 거래량
        self.ob_low_volume = 0  # 매수/매도 압력이 낮은 쪽 거래량
        self.ob_high_volume = 0  # 매수/매도 압력이 높은 쪽 거래량
        self.breaker = False  # 오더블록이 무효화되었는지 여부
        self.break_time = None  # 무효화된 시간
        self.disabled = False  # 결합 등으로 비활성화되었는지 여부
        self.combined = False  # 여러 오더블록이 결합되었는지 여부
    
    def __repr__(self):
        """오더블록 정보를 문자열로 표현"""
        status = "Broken" if self.breaker else "Active"
        return f"OrderBlock({self.ob_type}, {self.bottom:.0f}-{self.top:.0f}, {status})"
    
    def get_size(self):
        """오더블록의 크기(가격 범위) 반환"""
        return abs(self.top - self.bottom)
    
    def is_price_in_zone(self, price):
        """주어진 가격이 오더블록 구간 내에 있는지 확인"""
        return self.bottom <= price <= self.top
    
    def get_distance_to_top(self, current_price):
        """현재가에서 오더블록 상단까지의 거리 (%) 계산"""
        return ((current_price - self.top) / self.top) * 100
    
    def get_distance_to_bottom(self, current_price):
        """현재가에서 오더블록 하단까지의 거리 (%) 계산"""
        return ((current_price - self.bottom) / self.bottom) * 100
    
    def get_volume_balance(self):
        """거래량 밸런스 (%) 계산"""
        if self.ob_high_volume == 0:
            return 0
        return int((min(self.ob_high_volume, self.ob_low_volume) / 
                   max(self.ob_high_volume, self.ob_low_volume)) * 100.0)
    
    def to_dict(self):
        """딕셔너리로 변환"""
        return {
            'top': self.top,
            'bottom': self.bottom,
            'ob_volume': self.ob_volume,
            'ob_type': self.ob_type,
            'start_time': self.start_time,
            'bb_volume': self.bb_volume,
            'ob_low_volume': self.ob_low_volume,
            'ob_high_volume': self.ob_high_volume,
            'breaker': self.breaker,
            'break_time': self.break_time,
            'size': self.get_size(),
            'volume_balance': self.get_volume_balance()
        }