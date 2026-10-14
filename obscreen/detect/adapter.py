"""AlphaBlock 탐지기 ↔ 국장 데이터 어댑터(OBS-5).

탐지 로직은 `obscreen.detect.alphablock`(AlphaBlock 복사본)이 전부 한다. 이 모듈은 경계만 맞춘다.

- **입력**: 스크리너·차트의 시세는 날짜 인덱스(`DatetimeIndex`, KRX 영업일만) + `Open`·`High`·
  `Low`·`Close`·`Volume` 열이다. AlphaBlock은 `open_time`(ms) + 소문자 열을 받는다
  (`to_alphablock_frame`). 탐지기는 **봉 순서**만 보므로 국장의 휴장일·장 시간은 아무 가정도
  필요 없다. AlphaBlock의 크립토 전용 부분(`closed` 미확정봉 열, 1분봉 서브스텝·진입 시그널)은
  이 경로가 쓰지 않는다 — 우리 입력엔 `closed` 열이 없고, 시그널(`signals`)은 읽지 않는다.
- **출력**: 마지막 봉 시점의 렌더 뷰(`OrderBlockResult.rendered_order_blocks` — 트레이딩뷰가
  지금 그리는 박스와 같은 집합: 아직 소멸하지 않은 존 중 방향별 최신 `zone_count`개)를
  스크리너·차트가 쓰는 `OrderBlockInfo`로 바꾼다. 시각은 입력의 **원래 인덱스 값**으로
  되돌린다(같은 봉 = 같은 라벨).

탐지 파라미터의 기본값은 AlphaBlock `OrderBlockParams()` 그대로다(`DEFAULT_PARAMS` — 스윙 10 ·
ATR(Wilder 10) 배수 3.5 · 꼬리 무효화 · 병합 안 함 · 존 3개). 국장에 맞춘 조정은 별도 이슈다.
"""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd

from obscreen.detect.alphablock.models import (
    OrderBlock,
    OrderBlockDirection,
    OrderBlockParams,
    OrderBlockResult,
)
from obscreen.detect.alphablock.order_blocks import OrderBlockDetector
from obscreen.detect.orderblock import OrderBlockInfo

#: AlphaBlock 탐지 파라미터 기본값 — 스크리너·설정 기본값의 출처(숫자를 한 곳에만 적는다).
DEFAULT_PARAMS = OrderBlockParams()

ZoneCount = Literal["high", "medium", "low", "one"]

_INVALIDATION: dict[str, Literal["wick", "close"]] = {"Wick": "wick", "Close": "close"}
_OB_TYPE = {OrderBlockDirection.BULLISH: "Bull", OrderBlockDirection.BEARISH: "Bear"}


def to_alphablock_frame(df: pd.DataFrame) -> pd.DataFrame:
    """날짜 인덱스 OHLCV → AlphaBlock 입력(`open_time` ms + 소문자 열).

    시간대 없는 날짜는 UTC 자정으로 본다(봉의 순서·구분만 쓰이므로 어떤 기준이든 결과는 같다).
    """
    index = pd.DatetimeIndex(df.index)
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    return pd.DataFrame(
        {
            "open_time": index.as_unit("ms").asi8,
            "open": df["Open"].to_numpy(dtype=float),
            "high": df["High"].to_numpy(dtype=float),
            "low": df["Low"].to_numpy(dtype=float),
            "close": df["Close"].to_numpy(dtype=float),
            "volume": df["Volume"].to_numpy(dtype=float),
        }
    )


class AlphaBlockOrderBlockDetector:
    """`OrderBlockDetector` 프로토콜(`obscreen.detect.base`)을 AlphaBlock 탐지기로 구현한다."""

    def __init__(
        self,
        swing_length: int = DEFAULT_PARAMS.swing_length,
        max_atr_mult: float = DEFAULT_PARAMS.max_atr_mult,
        ob_end_method: str = "Wick",
        combine_obs: bool = DEFAULT_PARAMS.combine_obs,
        zone_count: ZoneCount = DEFAULT_PARAMS.zone_count,
    ) -> None:
        """
        Args:
            swing_length: 스윙 감지 길이(AlphaBlock `swing_length`).
            max_atr_mult: 존 크기 상한 = ATR × 이 배수(`max_atr_mult`).
            ob_end_method: 무효화 기준 `"Wick"`(꼬리) · `"Close"`(몸통) — `zone_invalidation`.
            combine_obs: 겹치는 존을 하나로 접을지(`combine_obs`).
            zone_count: 방향별로 돌려줄 최신 존 개수 `high`(10)·`medium`(5)·`low`(3)·`one`(1).
        """
        if ob_end_method not in _INVALIDATION:
            raise ValueError(f"ob_end_method는 'Wick' 또는 'Close'여야 한다: {ob_end_method!r}")
        self.params = OrderBlockParams(
            swing_length=swing_length,
            max_atr_mult=max_atr_mult,
            zone_invalidation=_INVALIDATION[ob_end_method],
            combine_obs=combine_obs,
            zone_count=zone_count,
        )
        self.result: OrderBlockResult | None = None
        self._labels: dict[int, Any] = {}
        self._volumes: dict[int, float] = {}

    def detect_order_blocks_realtime(self, df: pd.DataFrame) -> OrderBlockResult:
        """시세 전체를 시간 순서대로 훑는다(AlphaBlock `OrderBlockDetector.run`, 룩어헤드 없음)."""
        frame = to_alphablock_frame(df)
        times = frame["open_time"].tolist()
        self._labels = dict(zip(times, df.index, strict=True))
        self._volumes = dict(zip(times, frame["volume"].tolist(), strict=True))
        self.result = OrderBlockDetector(self.params).run(frame)
        return self.result

    def get_latest_orderblocks(self) -> tuple[list[OrderBlockInfo], list[OrderBlockInfo]]:
        """마지막 봉 시점의 (강세 존, 약세 존) — 방향별 최신 확정순, 무효화된 존 포함."""
        if self.result is None:
            return [], []
        bull: list[OrderBlockInfo] = []
        bear: list[OrderBlockInfo] = []
        for ob in self.result.rendered_order_blocks:
            info = self._to_info(ob)
            (bull if ob.direction is OrderBlockDirection.BULLISH else bear).append(info)
        return bull, bear

    def _to_info(self, ob: OrderBlock) -> OrderBlockInfo:
        info = OrderBlockInfo(
            ob.top, ob.bottom, ob.ob_volume, _OB_TYPE[ob.direction], self._labels[ob.start_time]
        )
        info.ob_low_volume = ob.ob_low_volume
        info.ob_high_volume = ob.ob_high_volume
        info.breaker = ob.breaker
        if ob.break_time is not None:
            info.break_time = self._labels[ob.break_time]
            info.bb_volume = self._volumes[ob.break_time]
        info.combined = ob.combined
        return info
