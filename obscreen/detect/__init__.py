"""오더블록 탐지·지표.

리모델링 4(OBS-5)가 탐지기를 AlphaBlock 것으로 바꾼다. 스크리너·차트 API가 탐지기에 기대는
면은 `OrderBlockDetector` 프로토콜 하나다 — 새 탐지기는 이 모양만 맞추면 된다.
"""

from obscreen.detect.base import OrderBlockDetector

__all__ = ["OrderBlockDetector"]
