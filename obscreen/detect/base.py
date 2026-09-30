"""탐지기 인터페이스 — 리모델링 4(OBS-5)에서 탐지기를 갈아끼울 자리(OBS-4).

스크리너(`obscreen.screening.screener`)와 차트 API(`obscreen.web.chart_api`)는 탐지기에
딱 두 가지를 기대한다: 시세 전체를 한 번 훑는 것과, 마지막 시점에 살아 있는 강세·약세 존을
돌려주는 것. 지금 구현은 `obscreen.detect.realtime.RealtimeOrderBlockDetector`다.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd

from obscreen.detect.orderblock import OrderBlockInfo


class OrderBlockDetector(Protocol):
    """오더블록 탐지기가 지켜야 할 모양."""

    def detect_order_blocks_realtime(self, df: pd.DataFrame) -> Any:
        """OHLCV 전체(날짜 인덱스)를 시간 순서대로 훑어 존 상태를 만든다(룩어헤드 없음)."""
        ...

    def get_latest_orderblocks(self) -> tuple[list[OrderBlockInfo], list[OrderBlockInfo]]:
        """마지막 봉 시점의 (강세 존, 약세 존)."""
        ...
