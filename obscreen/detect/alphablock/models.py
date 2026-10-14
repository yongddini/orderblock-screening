# ── 복사 이식(vendoring, OBS-5) ─────────────────────────────────────────────────
# 출처: github.com/yongddini/AlphaBlock  strategy/models.py
# 커밋: de825d2762c94e2b177db4608af8682f2a7fe842
# 바꾼 것: 없음(원본 그대로).
# 이 줄 아래를 손으로 고치지 말 것 — tests/test_alphablock_parity.py가 원본과 대조한다.
# 갱신 방법은 obscreen/detect/alphablock/__init__.py 설명을 따른다.
# ── 이하 원본 그대로 ──────────────────────────────────────────────────────────────
"""오더블록 탐지·시그널 출력 모델과 파라미터.

Fluxchart "Volumized Order Blocks" (`strategy/reference/`) 로직에 대응하는
불변 값 객체들이다. 파라미터는 원본 인디케이터 입력값과 1:1로 대응한다.
"""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_ZONE_COUNT_LIMITS: dict[str, int] = {"high": 10, "medium": 5, "low": 3, "one": 1}

#: **차트 표시선** EMA 세트(사용자 트레이딩뷰 설정, `indicators.DEFAULT_EMA_LENGTHS`와 동일).
#: 익절 판정이 아니라 오직 대시보드 오버레이용이다(WAN-66). 익절 목표선은 아래
#: `DEFAULT_TP_EMA_LENGTHS`를 쓴다.
DEFAULT_CONFLUENCE_EMA_LENGTHS: tuple[int, ...] = (20, 60, 120, 240, 365)

#: **익절 판정선** EMA 세트(WAN-66). 사용자 확정 규칙: 익절 목표선은 EMA 60 + VWMA 100
#: 두 개뿐이다. EMA 20/120/240/365는 차트에 그리기만 하는 선이지 익절 판정에 쓰지 않는다.
#: (배경) WAN-23 명세가 "차트 표시선"과 "익절 목표선"을 한 배열로 뒤섞어 적어, 코드가
#: 표시선 5개 전부를 익절 후보로 써 왔다 — 가장 빠른 EMA 20에서 사실상 항상 조기 익절.
DEFAULT_TP_EMA_LENGTHS: tuple[int, ...] = (60,)


class OrderBlockDirection(StrEnum):
    """오더블록 방향. 원본의 `obType` ("Bull"/"Bear")에 대응."""

    BULLISH = "bull"
    BEARISH = "bear"


class OrderBlockParams(BaseModel):
    """오더블록 탐지 파라미터. 원본 인디케이터 설정값과 대응한다."""

    model_config = ConfigDict(frozen=True)

    swing_length: int = Field(default=10, ge=3)
    zone_invalidation: Literal["wick", "close"] = "wick"
    zone_count: Literal["high", "medium", "low", "one"] = "low"
    combine_obs: bool = False
    """겹치는 오더블록을 하나로 접을지. **WAN-149가 기본값을 `True`→`False`로 옮겼다**
    (사용자 결정 2026-07-21) — 원본 존 단위로 탐지·진입·손절한다.

    ⚠️ **「데이터가 분리를 골랐다」로 읽지 말 것** — WAN-134 부검의 공식 판정은 (c)
    「병합은 손절률에 중립」이었고 채택 권고가 없었다. 분리 팔이 4칸 중 3칸에서 이기지만
    **거래가 5~8% 늘어** 두 팔이 아예 다른 거래를 하고, 그 차이는 「병합이 존폭의
    대리변수」라 존폭 효과일 수 있다(WAN-142 소관). 배경은 docs/decisions/wan149.md.

    `True` 옵트인 경로(`_generate_merged_signals`)는 **존치**한다 — 옛 수치를 결론에 박아
    둔 리포트는 `harness.LEGACY_COMBINE_OBS`로 이 값을 명시 고정한다.
    """
    merged_break_time_rule: Literal["distal", "pine_max"] = "distal"
    """병합 존의 무효화 시각(`break_time`)을 구성 존들에서 어떻게 접을지 (WAN-400 §2).

    `combine_obs=True`일 때만 읽는다(분리 경로는 병합 존을 만들지 않는다).

    * `"distal"` — **현행 기본값**. 합집합 경계를 정의하는 존(강세 = `bottom` 최저,
      약세 = `top` 최고)의 `break_time`. 병합 존의 손절선이 곧 그 존의 경계이므로,
      그 경계가 뚫리면 병합 존도 무효화된다는 **매매 규칙**의 논리다.
    * `"pine_max"` — 원본 `combineOBsFunc`(`.pine:391-392`) 충실:
      `math.max(nz(A.breakTime), nz(B.breakTime))` 뒤 `== 0 ? na`. `nz(na)=0`이라
      **구성 존 중 하나라도 죽었으면 병합 존도 죽고**(그중 가장 늦은 시각으로),
      **전부 살아 있을 때만** 살아 있다. 🚨 「늦게 죽는 쪽까지 살아 있다」가 아니다.

    ⚠️ **기본값 전환은 재-베이스라인 = 사용자 결정이고 개발자 임의 선택 금지**
    (WAN-400 §2 ★). `break_time`은 `obs_touch`의 오른쪽 변이라 **겹침 판정 결과까지**
    달라진다. 크기는 `backtest/reports/wan400_merge_parity_census_summary.md`.
    """
    merged_tap_state: Literal["cluster", "zone", "zone_all"] = "cluster"
    """병합 경로의 탭 상태(직전 봉 포함 여부·재탭 카운터)를 무엇으로 키잉할지 (WAN-400 §3).

    `combine_obs=True`일 때만 읽는다. **원본에는 대응물이 없다** — 원본은 탭·재탭을
    세지 않고 박스만 그린다(탭 카운팅은 WAN-56/81/82 확장). 그래서 「원본과 동일하게」가
    정의되지 않고, **우리가 설계하는 자리**다.

    * `"cluster"` — **현행 기본값**. 키가 그 시점 **구성 존 인덱스 집합**이라,
      클러스터 구성이 바뀌면(존 하나가 소멸해 쪼개지거나, 새 존이 합류하거나) 키가
      달라져 「직전 봉에 안에 있었다」 기록이 **리셋**된다 → 가격이 계속 박스 안에
      있었는데도 새 탭으로 셀 수 있고 재탭 번호도 1로 되돌아간다. 같은 구성이 나중에
      다시 나타나면 **낡은 값이 되살아난다**.
    * `"zone"` — **사용자 결정 규칙 (2026-09-01)**: *"새롭게 존이 병합하는 경우는 새로
      탭을 셀 필요가 없지. 단순히 존만 넓어지는거야"* · *"기회를 안주는게 맞지"*.
      상태를 **아카이브 존 인덱스마다** 들고, 그룹의 「직전 봉에 안에 있었다」를
      **구성 존 중 하나라도 그랬는가**(`any`)로 정의한다 — 병합된 것은 **하나의 박스**
      이고 커져도 같은 박스이므로, **가격이 안 움직였으면 새 진입 기회가 아니다**.
      재탭 카운터도 구성 존들의 최댓값에서 이어 센다(병합·분리로 1로 안 돌아간다).
      🚨 **WAN-82를 되돌리는 게 아니다** — 가격이 밖으로 **나갔다 다시 들어오면** 탭이
      나고 그때 `member_indices - entered`가 미진입 구성 존에 기회를 준다. 이 규칙은
      「가격이 안 움직인 동안에는 기회를 만들지 않는다」까지다.
    * `"zone_all"` — **측정 대조 전용**(WAN-400 §0). `any` 대신 `all`로 접는다 =
      개발자 초안이었고 **위 사용자 결정으로 대체됐다**. 두 규칙의 차이(= 「구성 변경 +
      가격은 계속 안에」로 세던 탭)의 크기를 재려고 존치한다 — 새 측정에 쓰지 말 것.

    ⚠️ **기본값 전환은 재-베이스라인이다** — `harness.LEGACY_COMBINE_OBS`로 병합을
    고정한 옛 리포트(WAN-134/149/388 계열)의 탭 집합이 통째로 움직인다.
    """
    max_atr_mult: float = Field(default=3.5, gt=0)
    atr_length: int = Field(default=10, ge=1)
    max_order_blocks: int = Field(default=30, ge=1)
    """(WAN-47) 탐지 아카이브에는 더 이상 상한을 적용하지 않는다(전체 생애 보존).
    원본의 표시 개수 캡은 렌더 뷰의 `zone_limit`으로만 남는다. 하위호환용으로 유지."""
    max_distance_to_last_bar: int = Field(default=1750, ge=1)
    """(WAN-47) 렌더 뷰의 **최근성 필터**: 마지막 봉에서 이 봉 수 이내에 확정된 존만
    "현재 그림"(`rendered_order_blocks`)에 그린다. 원본에서는 탐지 스캔 상한이었으나,
    탐지/렌더 분리 후 아카이브(`order_blocks`)는 전체 히스토리를 스캔한다."""

    @model_validator(mode="after")
    def _reject_merge_knobs_without_merging(self) -> OrderBlockParams:
        """병합 노브를 **분리 경로에** 주면 거부한다 (WAN-400).

        `merged_*`는 `combine_obs=True`일 때만 읽히므로, 분리 경로에 주면 **아무 일도
        안 하면서 라벨만 붙는다** — 「pine 규칙으로 쟀다」고 믿으면서 현행 분리 엔진을
        돌리게 된다(WAN-91/95/112/123/159가 반복해 겪은 조용한 실패). 기본값끼리는
        당연히 통과하므로 채택 경로(`combine_obs=False`)는 영향받지 않는다.
        """
        if self.combine_obs:
            return self
        offenders = [
            name
            for name, value, default in (
                ("merged_break_time_rule", self.merged_break_time_rule, "distal"),
                ("merged_tap_state", self.merged_tap_state, "cluster"),
            )
            if value != default
        ]
        if offenders:
            raise ValueError(
                f"{', '.join(offenders)}는 병합 경로 전용입니다 — combine_obs=True가 아니면 "
                "읽히지 않습니다(라벨만 붙는 실행 방지)."
            )
        return self

    @property
    def zone_limit(self) -> int:
        """`zone_count` 문자열을 방향별 렌더/채택 개수로 변환."""
        return _ZONE_COUNT_LIMITS[self.zone_count]


class OrderBlock(BaseModel):
    """탐지된 오더블록 하나. 원본 `orderBlockInfo`에 대응.

    이 값 객체는 존의 **전체 생애주기**를 담는다 (WAN-47). 원본 인디케이터는
    깨진 뒤 되쓸린 존을 `bullishOrderBlocksList.remove(i)`(`.pine:267`)로 **데이터
    리스트에서** 뺀다(WAN-400 §1 정정 — `box.delete()`는 그와 별개인 렌더 정리다).
    백테스트 신호원으로 쓰려면 생애 기록이 소실되면 안 되므로(생존자 편향) 탐지기는
    존을 지우지 않고
    `break_time`(무효화)·`swept_time`(소멸)·`tapped_times`(재진입)로 상태 전이만
    기록한다. "지금 차트에 그릴 박스"는 `OrderBlockResult.active_at()` 렌더링
    뷰가 이 아카이브에서 파생한다.
    """

    model_config = ConfigDict(frozen=True)

    direction: OrderBlockDirection
    top: float
    bottom: float
    start_time: int
    """존이 시작되는(박스 왼쪽 변) 봉의 `open_time`(ms). 원본 `startTime`."""
    confirmed_time: int
    """오더블록이 실제로 확정(탐지)된 봉의 `open_time`(ms). 존의 생성 시각."""
    ob_volume: float
    ob_low_volume: float
    ob_high_volume: float
    breaker: bool = False
    break_time: int | None = None
    """존이 반대편으로 돌파되어 breaker로 무효화된 봉의 `open_time`(ms). 없으면 None."""
    swept_time: int | None = None
    """breaker 존이 되쓸려 완전 소멸(원본은 `box.delete()`)한 봉의 `open_time`(ms).

    없으면 아직 차트에 존재. 현재까지 살아있는(active·breaker) 존은 None.
    무효화(`break_time`) 이후에만 발생하므로 `swept_time`이 있으면 `break_time`도 있다.
    """
    tapped_times: tuple[int, ...] = ()
    """확정 이후 가격이 존 범위(`bottom`~`top`)에 재진입(tap)한 봉들의 `open_time`(ms).

    바깥→안 전이 시각만 기록한다(존 안에 머무는 연속 봉은 첫 진입만)."""
    combined: bool = False
    """`combine_obs`로 다른 존과 병합되어 생성된 존인지 여부."""
    num_component_obs: int = 1
    """이 존이 병합한 구성 존 개수(단일 존이면 1). 측정용(WAN-134) — 진입·손절·익절
    로직에는 쓰이지 않고, 병합 강도를 사후 분석에서 세기 위한 주석이다."""
    displacement_atr: float | None = None
    """확정(돌파) 시점의 **변위 강도** = 스윙 초과폭 ÷ ATR (WAN-254 · 측정 전용).

    불리시는 `(종가 − 돌파된 스윙고) ÷ ATR`, 베어리시는 `(돌파된 스윙저 − 종가) ÷ ATR`로,
    둘 다 확정 봉 `t`가 **종가로 닫힌 뒤** 판정되므로(`closes[t] > top.price`) 룩어헤드가
    없다(ATR도 확정 봉 기준, 탐지기 자신이 존 생성에 쓰는 그 값). 진입·손절·익절 로직에는
    쓰이지 않고, 「제일 센 변위 OB가 되돌아오지 않는가」(생존편향)를 사후 분석에서 재기 위한
    주석이다. `None`이면 미측정(옛 픽스처·역직렬화 호환)."""

    def alive_at(self, time_ms: int) -> bool:
        """`time_ms` 시점에 이 존이 차트에 존재(생존)하는지.

        확정 이후(`confirmed_time <= time_ms`)이고 아직 소멸하지 않았으면
        (`swept_time`이 없거나 `time_ms` 이후이면) True. breaker(무효화)는 소멸이
        아니므로, 되쓸리기 전까지는 여전히 생존으로 본다(원본이 breaker 박스를
        재색칠해 계속 그리는 것과 동일).
        """
        if time_ms < self.confirmed_time:
            return False
        return self.swept_time is None or time_ms < self.swept_time


class SignalExitReason(StrEnum):
    """전략이 계획한 청산의 사유 (WAN-23)."""

    TAKE_PROFIT = "take_profit"
    """진입가 너머 가장 가까운 EMA/VWMA 선에 도달(전량 익절)."""
    STOP_LOSS = "stop_loss"
    """진입 근거 오더블록이 breaker로 무효화(손절)."""


class PlannedExit(BaseModel):
    """전략이 진입 시점에 계획한 명시적 청산 이벤트 (WAN-23).

    익절(선 도달)·손절(오더블록 무효화)이 모두 봉마다 달라지는 **동적** 규칙이라,
    전략이 진입가를 기준으로 청산 봉·참조가·사유를 미리 산출해 시그널에 실어
    보내면 백테스트(`backtest.run_backtest`)가 이를 그대로 소비한다.
    """

    model_config = ConfigDict(frozen=True)

    time: int
    """청산이 발생하는 봉의 `open_time`(ms)."""
    price: float
    """청산 참조가. 익절=도달한 선 가격, 손절=무효화 봉 종가와 오더블록 무효화 경계
    (`ob.bottom`/`ob.top`) 중 진입가에 더 불리한 쪽(WAN-65, 손절이 이익을 내지 않도록
    clamp)."""
    reason: SignalExitReason


class OrderBlockSignal(BaseModel):
    """오더블록 기반 진입 후보 시그널 (AlphaBlock 확장, 원본에는 없음)."""

    model_config = ConfigDict(frozen=True)

    direction: OrderBlockDirection
    trigger_time: int
    """가격이 오더블록 존에 재진입(tap)한 봉의 `open_time`(ms)."""
    price: float
    order_block: OrderBlock
    status: Literal["active", "cancelled"] = "active"
    planned_exit: PlannedExit | None = None
    """컨플루언스 전략이 계획한 청산(WAN-23). 없으면 백테스트의 고정 %TP/SL 경로를 따른다."""
    tap_index: int = 0
    """이 존(개별 존 또는 병합 존, `combine_obs` 기준)의 몇 번째 탭인지(0-based, WAN-81).

    `0`은 존(병합 존 포함) 확정 이후 **첫 탭**. `ConfluenceParams.rsi_gate_mode=
    "first_tap_free"`가 이 값으로 첫 탭은 RSI 게이트를 면제하고 재탭(`>=1`)부터
    적용한다. 병합 존은 구성 존이 개별적으로 각자 첫 탭을 셀 수 있으므로(WAN-82
    흡수), 같은 병합 클러스터 안에서도 신규로 편입된 존은 다시 `0`을 받을 수 있다."""
    zone_key: frozenset[int] | None = None
    """이 시그널이 속한 존(병합 존 포함)의 안정적 식별자(탐지 아카이브 인덱스 집합,
    WAN-83). 병합 존은 `merged_ob`(값 객체) 인스턴스가 `dirty` 재계산마다 새로
    만들어져 `order_block` 객체 동일성으로는 "같은 병합 클러스터"를 추적할 수
    없다(재계산은 무관한 클러스터도 함께 새 인스턴스를 만든다) — 반면 이 필드가 담는
    구성 인덱스 집합(`_MergedGroup.member_indices`)은 멤버십이 그대로인 한 항상
    동일하다. 같은 존의 여러 탭(`tap_index` 0, 1, 2…)을 이 값으로 그룹핑할 수
    있다. 진입·손절·익절 로직에는 전혀 쓰이지 않는 순수 진단용 메타데이터다."""


def _merge_max_optional(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def _ob_area(ob: OrderBlock, now: int) -> float:
    end = ob.break_time if ob.break_time is not None else now
    return (end - ob.start_time) * (ob.top - ob.bottom)


def obs_touch(a: OrderBlock, b: OrderBlock, now: int) -> bool:
    """두 존이 병합 대상으로 겹치는지(IoU 교집합/합집합 * 100 > 0). 원본 `doOBsTouch`.

    시간축(존의 왼쪽 변~오른쪽 변)과 가격축(bottom~top)의 교집합 면적을 합집합
    면적으로 나눈 IoU가 양수면 병합한다. `now`는 아직 무효화되지 않은 존의 오른쪽
    변(면적 계산용) 센티넬이다. 방향(obType)은 이 함수에서 검사하지 않으므로 호출부가
    같은 방향끼리만 넘겨야 한다(`combine_order_blocks`·`_generate_merged_signals`).
    """
    # 가격축 교집합을 먼저 본다 — 가격대가 다른 대다수 쌍은 여기서 즉시 걸러져
    # 시간축·면적 계산을 건너뛴다(WAN-56 성능: 매 봉 O(존²) 병합의 핫패스).
    intersection_price = min(a.top, b.top) - max(a.bottom, b.bottom)
    if intersection_price <= 0.0:
        return False
    a_end = a.break_time if a.break_time is not None else now
    b_end = b.break_time if b.break_time is not None else now
    intersection_time = min(a_end, b_end) - max(a.start_time, b.start_time)
    if intersection_time <= 0:
        return False
    intersection = intersection_time * intersection_price
    union = _ob_area(a, now) + _ob_area(b, now) - intersection
    if union <= 0:
        return False
    return (intersection / union) * 100.0 > 0


def combine_order_blocks(obs: list[OrderBlock], now: int) -> list[OrderBlock]:
    """겹치는(IoU 교집합>0) 동일 방향 존을 병합. 원본 `combineOBsFunc`에 대응.

    렌더링 뷰(`select_active`)와 백테스트 시그널(`_generate_merged_signals`,
    WAN-56)이 공유한다 — 탐지 아카이브(`order_blocks`) 자체는 원본 단위로 남긴다.
    `now`는 아직 무효화되지 않은 존의 오른쪽 변(면적 계산용) 센티넬이다.
    """

    def touch(a: OrderBlock, b: OrderBlock) -> bool:
        return obs_touch(a, b, now)

    items = list(obs)
    merged = True
    while merged:
        merged = False
        for i in range(len(items)):
            for j in range(len(items)):
                if i == j:
                    continue
                a, b = items[i], items[j]
                if a.direction != b.direction:
                    continue
                if touch(a, b):
                    new_ob = OrderBlock(
                        direction=a.direction,
                        top=max(a.top, b.top),
                        bottom=min(a.bottom, b.bottom),
                        start_time=min(a.start_time, b.start_time),
                        confirmed_time=min(a.confirmed_time, b.confirmed_time),
                        ob_volume=a.ob_volume + b.ob_volume,
                        ob_low_volume=a.ob_low_volume + b.ob_low_volume,
                        ob_high_volume=a.ob_high_volume + b.ob_high_volume,
                        breaker=a.breaker or b.breaker,
                        break_time=_merge_max_optional(a.break_time, b.break_time),
                        swept_time=_merge_max_optional(a.swept_time, b.swept_time),
                        tapped_times=tuple(sorted(set(a.tapped_times) | set(b.tapped_times))),
                        combined=True,
                    )
                    remove_indices = {i, j}
                    items = [x for k, x in enumerate(items) if k not in remove_indices]
                    items.append(new_ob)
                    merged = True
                    break
            if merged:
                break
    return items


def select_active(
    order_blocks: list[OrderBlock],
    time_ms: int,
    *,
    limit: int | None = None,
    combine: bool = False,
) -> list[OrderBlock]:
    """`time_ms` 시점에 차트에 그려질 존(렌더링 뷰)을 아카이브에서 파생한다 (WAN-47).

    각 방향별로 그 시점에 생존(`alive_at`)한 존을 최신 확정순으로 정렬해 `limit`개만
    채택하고, `combine`이면 겹치는 존을 병합한다. 트레이딩뷰 원본이 그 시점에
    그렸을 박스 집합과 동일한 패리티를 유지한다.
    """
    result: list[OrderBlock] = []
    for direction in (OrderBlockDirection.BULLISH, OrderBlockDirection.BEARISH):
        alive = [ob for ob in order_blocks if ob.direction is direction and ob.alive_at(time_ms)]
        alive.sort(key=lambda ob: ob.confirmed_time, reverse=True)
        if limit is not None:
            alive = alive[:limit]
        # 각 존을 `time_ms` 시점 상태로 클리핑한다 — 그 시점에 아직 일어나지 않은
        # 무효화/탭은 렌더에 반영되지 않아야 한다(미래 정보 유출 방지, 트레이딩뷰가
        # 그 시점에 그렸을 그림과 정확히 일치).
        clipped = [_clip_to_time(ob, time_ms) for ob in alive]
        if combine:
            clipped = combine_order_blocks(clipped, time_ms + 1)
        result.extend(clipped)
    return result


def _clip_to_time(ob: OrderBlock, time_ms: int) -> OrderBlock:
    """존의 생애주기 필드를 `time_ms` 시점까지로 자른 뷰 사본을 만든다.

    `time_ms` 이후에야 일어나는 무효화(`break_time`)는 아직 breaker가 아닌 것으로,
    아직 없던 탭은 제외해 반영한다. `alive_at`이 참인 존만 넘어오므로 소멸
    (`swept_time`)은 항상 미래(또는 없음)이라 뷰에서는 항상 None이다.
    """
    broke = ob.break_time is not None and ob.break_time <= time_ms
    tapped = tuple(t for t in ob.tapped_times if t <= time_ms)
    if broke and ob.swept_time is None and tapped == ob.tapped_times:
        return ob  # 이미 시점 상태와 동일 — 불필요한 복사 회피.
    return ob.model_copy(
        update={
            "breaker": broke,
            "break_time": ob.break_time if broke else None,
            "swept_time": None,
            "tapped_times": tapped,
        }
    )


class OrderBlockResult(BaseModel):
    """`OrderBlockDetector.run()`의 반환값 (WAN-47).

    `order_blocks`는 **생성된 모든 존의 전체 생애주기 아카이브**다(트리밍·병합·삭제
    없음). 트레이딩뷰와 동일한 "지금 그릴 박스"는 `rendered_order_blocks`(또는 임의
    시점 `active_at()`)로 파생한다. 백테스트 신호(`signals`)는 아카이브 전체를 소비해
    생존자 편향 없이 산출된다.
    """

    model_config = ConfigDict(frozen=True)

    order_blocks: list[OrderBlock]
    """생성된 모든 존의 전체 아카이브(생애주기 필드 포함, 병합 전)."""
    signals: list[OrderBlockSignal]
    """존(또는 병합 존, `combine_obs` 기준)당 **첫 탭 1회**만 담은 시그널(`tap_index=0`
    고정). `ConfluenceParams.retap_mode="once"`가 그대로 소비한다."""
    retap_signals: list[OrderBlockSignal] = Field(default_factory=list)
    """존이 무효화되기 전까지의 **모든** 탭(재탭 포함, `tap_index` 부착)을 담은 시그널
    (WAN-81). `combine_obs=True`면 병합 존 경계·병합 상태 기준으로 생성해 재탭
    경로도 병합을 반영한다(WAN-56 패리티, WAN-81 갭B). `retap_mode="every_tap"`이
    소비한다."""
    rendered_order_blocks: list[OrderBlock] = Field(default_factory=list)
    """마지막 봉 시점의 렌더링 뷰(트레이딩뷰 패리티: 방향별 `zone_limit`개, 병합 적용)."""

    def active_at(
        self, time_ms: int, *, limit: int | None = None, combine: bool = False
    ) -> list[OrderBlock]:
        """`time_ms` 시점에 그려질 존을 아카이브에서 파생한다(`select_active` 위임)."""
        return select_active(self.order_blocks, time_ms, limit=limit, combine=combine)


RsiGateMode = Literal["extreme", "neutral", "none", "first_tap_free", "unconditional"]

#: 이격 밴드의 표본 시점(`DeviationFilterParams.band_bar`). 채택 기본값은
#: `"intrabar_live"`(WAN-132) — 뜻과 함께 그 필드의 독스트링 참고.
BandBar = Literal["tap", "prev_closed", "intrabar_live", "intrabar_causal"]

#: 미체결 지정가를 「존이 깨졌다」로 취소하는 **시점**(`ConfluenceParams.invalidation_cancel`).
#: 채택 기본값은 `"bar_close"`(WAN-365 = 인과) — 뜻과 근거는 그 필드의 독스트링 참고.
InvalidationCancel = Literal["bar_open", "bar_close"]

#: 지정가를 걸 **존 내 기준선**(`ConfluenceParams.zone_limit_ref`). 채택 기본값은
#: `"proximal"`(존 근단, WAN-95) — 뜻은 그 필드의 독스트링 참고. 이름을 붙여 둔 것은 측정
#: 모듈이 이 축을 옵트인 인자로 나를 때 세 값이 한 곳에서만 정의되게 하기 위해서다(WAN-378).
ZoneLimitRef = Literal["proximal", "mid", "distal"]


def rsi_gate_passes(
    rsi: float,
    *,
    is_long: bool,
    mode: RsiGateMode = "extreme",
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    rsi_neutral_band: tuple[float, float] = (40.0, 60.0),
) -> bool:
    """RSI 게이트 판정(WAN-73). A안(`strategy.confluence`)과 B안
    (`backtest.substep.simulate_zone_limit_trade`)이 공유하는 순수 함수다.

    `mode="extreme"`(구 엔진 기본): 롱은 `rsi <= rsi_oversold`, 숏은 `rsi >=
    rsi_overbought`(과매도/과매수 극단). `"neutral"`: 방향과 무관하게 `rsi_neutral_band`
    안이면 통과(존이 조용히 지켜지는 국면). `"first_tap_free"`(WAN-81): 첫 탭 면제는
    `tap_index`를 아는 호출부가 이 함수를 호출하기 전에 직접 처리하므로, 이 함수에
    도달했다는 것은 이미 재탭(첫 탭이 아님)이라는 뜻이다 — `extreme`과 동일한 극단
    규칙을 적용한다.

    ⚠️ **`"none"`과 `"unconditional"`은 동의어가 아니다** — 둘 다 여기서는 True를
    반환하지만 **워밍업 처리가 다르고, 그 차이는 호출부에 있다**:

    * `"none"`: 게이트 **판정**만 없다. 호출부는 여전히 `rsi is not None`(워밍업 통과)을
      요구하므로 **RSI 워밍업 구간의 탭은 진입하지 못한다**. 그래서 `"none"`은
      `"first_tap_free"`의 깨끗한 상위집합이 아니다(WAN-114 `L0r`이 이 비대칭을 안고
      측정됐고, 그 CSV가 이 의미에 고정돼 있다).
    * `"unconditional"`(WAN-123 채택 기본값): 게이트가 **아예 없다**. 호출부가 RSI를
      읽지도 않으므로 **워밍업(NaN)이어도 지정가 터치 즉시 진입**한다 — 즉 첫 탭·재탭을
      가리지 않는 `"first_tap_free"`의 진짜 상위집합이다.

    ⚠️ "호출부가 처리한다"는 계약은 **두 경로 모두**의 책임이다: A안은
    `ConfluenceStrategy._evaluate_entry`, B안은 `backtest.zone_limit_backtest.
    build_zone_limit_candidates`가 `signal.tap_index == 0`을 보고
    `simulate_zone_limit_trade(first_tap_free=...)`로 넘긴다. B안이 이 계약을 지키지
    않아 채택 경로에서 첫 탭 면제가 통째로 빠져 있었다(WAN-100) — 이 함수는 모드만
    보고는 첫 탭인지 알 수 없으므로 누락이 조용히 지나갔다. 새 진입 경로를 붙일 때
    `tap_index` 배선과 `"unconditional"` 면제를 빠뜨리지 말 것.
    """
    if mode in ("none", "unconditional"):
        return True
    if mode == "neutral":
        low, high = rsi_neutral_band
        return low <= rsi <= high
    return rsi <= rsi_oversold if is_long else rsi >= rsi_overbought


class DeviationFilterParams(BaseModel):
    """SMA/종가 대비 이격 진입 필터 (WAN-75).

    `ConfluenceParams.deviation_filter`가 `None`이면 꺼짐(기본, 현행 동작 보존).
    켜지면 기준선(anchor) 대비 폭(width)만큼 떨어진 밴드를 계산해 오더블록 진입가를
    재산정하거나 진입을 기각한다(`deviation_entry_price` 참고). 롱은 하단선
    (`anchor - width`), 숏은 상단선(`anchor + width`)을 대칭 적용한다.
    """

    model_config = ConfigDict(frozen=True)

    anchor: Literal["close", "sma"] = "close"
    """밴드 기준선. `close`: 그 봉 종가. `sma`: `sma_length` 단순이동평균."""
    sma_length: int = Field(default=20, ge=1)
    """`anchor="sma"`일 때의 SMA 길이. `width_kind="stdev"`(볼린저)의 표준편차 창도
    동일 길이를 공유한다(볼린저 밴드 표준 정의)."""
    width_kind: Literal["pct", "stdev", "atr"] = "pct"
    """밴드 폭 산정 방식. `pct`: 기준선의 고정 비율. `stdev`: `sma_length`창 모표준편차의
    배수(볼린저). `atr`: `atr_length` ATR의 배수."""
    width_value: float = Field(gt=0)
    """`width_kind`별 배수/비율. `pct`는 소수(0.02=2%), `stdev`·`atr`는 배수."""
    atr_length: int = Field(default=14, ge=1)
    """`width_kind="atr"`일 때의 ATR 길이."""
    band_bar: BandBar = "intrabar_live"
    """밴드의 **20번째 표본에 무엇을 넣나** (WAN-115 · WAN-119 · WAN-120 · **WAN-132**).

    `tap`(WAN-132 이전 기본값): 탭 봉 **최종 종가**. **B안(지정가)에서는 룩어헤드다** —
    그 봉의 종가를 알아야 나오는 SMA20/σ에 그 봉 **내부**에서 체결되기 때문이다
    (A안은 탭 봉 종가에 진입하므로 룩어헤드가 아니다).
    `prev_closed`(WAN-115 교정): **직전 확정봉** 종가. 탭 봉이 열리기 전에 확정돼 있으므로
    지정가를 미리 걸 수 있다 = 룩어헤드 없음. 대신 현재 봉을 통째로 버려 **보수적**이고,
    워밍업이 한 봉 늘어난다(`pos=0`은 판정 불가).
    `intrabar_live`(WAN-119 · **WAN-132 채택 기본값**): **그 순간의 현재가**. 트레이딩뷰가
    형성 중인 봉에 밴드를 그리는 방식이자 사용자가 실제로 보고 지정가를 거는 값이다 —
    룩어헤드 없이 재현 가능하다. 밴드가 봉내에 계속 움직이므로 **지정가도 매 서브스텝 재산정**된다
    (`backtest.substep`의 `LiveLimitProvider`). B안(지정가) 전용 성질이며, A안(탭 봉
    종가 진입)에서는 현재가 = 탭 봉 종가라 `tap`과 **정확히 같은 값**이다
    (`ConfluenceStrategy.deviation_band_at` 참고).
    `intrabar_causal`(WAN-120): **직전 서브스텝(1분봉) 종가**. `intrabar_live`가 남긴
    **잔여 1분 룩어헤드를 0으로 만든** 변형이다 — `intrabar_live`는 밴드를 close(t)로
    계산하는데 체결 판정(`low(t) <= 지정가`)도 **같은 1분봉 안**이라, 주문 가격을 그 분의
    결과를 조금 알고 고르는 셈이다. 이 모드는 그 분이 **시작될 때 이미 알던 가격**만
    쓴다(= 분당 주문을 재호가하는 봇이 실제로 할 수 있는 것). `intrabar_live`와 마찬가지로
    B안 전용이며, A안은 이 값을 표현할 수 없어 **거부한다**
    (`ConfluenceStrategy.deviation_band_at`).

    📌 **WAN-132(사용자 결정)가 기본값을 `tap` → `intrabar_live`로 옮겼다.** 최적화가
    아니라 **정확성**이다 — `tap`은 그 봉이 어떻게 끝날지 알아야 나오는 가격이라 실거래에서
    재현할 수 없고, `intrabar_live`는 사용자가 화면에서 보고 지정가를 거는 바로 그 값이다.
    ⚠️ **「수익이 올라서 바꿨다」로 읽지 말 것**(15m OOS +0.50%p지만 1h OOS는 −1.62%p다) ·
    ⚠️ **「엣지 없음」(WAN-84/88/111/114/124)은 그대로**다. 근거는
    [`docs/decisions/wan132.md`](../docs/decisions/wan132.md).

    ⚠️ 그 이전 수치를 결론 문장에 박아 둔 리포트는 `harness.LEGACY_BAND_BAR`(= `"tap"`)로
    **명시 고정**해 당시 엔진의 기록으로 보존한다 — 고정하지 않으면 기본값을 따라 조용히
    새 밴드로 다시 돌아 본문과 어긋난다(WAN-112·WAN-123과 같은 패턴)."""

    select_only: bool = False
    """밴드를 **선별(기각)에만** 쓰고 진입가는 재산정하지 않는다 (WAN-131, 옵트인 · 측정 전용).

    기본값 `False`(현행)는 밴드가 진입가를 **재산정**한다(`deviation_entry_price`의 규칙 2:
    밴드가 존 안에 있으면 밴드 값에서 진입). `True`면 밴드 판정은 그대로 하되(규칙 3 =
    밴드가 존 전체보다 불리하면 **셋업 기각** — 선별) 진입가는 **존 근단**(`zone_limit_price`)을
    쓴다 — 볼린저를 **필터로만** 쓰고 가격은 안 준다.

    이 필드가 가르는 것: 볼린저의 기여가 「좋은 존을 고르는 것(선별)」인가 「그냥 더 싸게
    사는 것(가격)」인가. WAN-96/114/115/120이 전부 가격 쪽을 가리켰지만 직접 가른 표가
    없었다(WAN-124 §7-2가 후속으로 남김). `A`(볼린저 off) → `B`(`select_only=True`) →
    `C`(현행, `select_only=False`) 사다리에서 선별 = B−A · 가격 = C−B로 분해한다.

    ⚠️ **기본값(`False`)에서는 아무것도 바뀌지 않는다** — 켜야만 규칙 2가 존 근단으로
    갈리고, 규칙 1(밴드가 존보다 관대)·규칙 3(기각)은 두 값에서 **완전히 같다**(규칙 1은
    이미 존 근단, 규칙 3은 두 값 모두 기각). 즉 `B`와 `C`는 **같은 셋업 집합**을 갖고
    (선별이 동일) 규칙 2 셋업의 **진입가만** 다르다 — 그게 `C−B`가 순수 가격 효과인 근거다.
    회귀 테스트가 라벨이 아니라 동작으로 고정한다."""


def deviation_entry_price(direction_sign: int, ob: OrderBlock, band: float) -> float | None:
    """이격 필터 밴드에 따른 진입가 재산정(WAN-75). A안·B안이 공유하는 순수 함수다.

    존의 근단(proximal, 먼저 닿는 경계)·원단(distal, 무효화 경계)을 밴드와 비교해
    3가지 규칙 중 하나를 적용한다(롱 기준, 숏은 부호 대칭):

    1. 밴드가 근단보다 진입 방향으로 더 유리한 쪽(= 존 전체가 밴드보다 유리한 쪽)에
       있으면 근단에서 그대로 진입(필터가 관대해 영향 없음).
    2. 밴드가 존 안(근단~원단)에 있으면 밴드 값에서 진입(손절은 원단 그대로라
       손익비가 달라진다).
    3. 밴드가 원단보다 더 불리한 쪽에 있으면(존 전체가 밴드에 못 미침) 진입하지
       않는다(`None`).
    """
    proximal = ob.top if direction_sign > 0 else ob.bottom
    distal = ob.bottom if direction_sign > 0 else ob.top
    signed_band = direction_sign * band
    signed_proximal = direction_sign * proximal
    signed_distal = direction_sign * distal
    if signed_band > signed_proximal:
        return proximal
    if signed_band >= signed_distal:
        return band
    return None


class ConfluenceParams(BaseModel):
    """오더블록 + RSI 진입 / EMA·VWMA 선 익절 / 오더블록 무효화 손절 규칙 (WAN-23).

    사용자의 실제 매매 방식에 맞춘 규칙이다. **EMA·VWMA는 진입 판정에 쓰지 않고**
    오직 익절 목표선으로만 쓴다. 기본값은 사용자 트레이딩뷰 설정과 일치한다
    (`indicators.py` 기본값 참고). 필드는 `config.Settings`에서
    `ALPHABLOCK_CONFLUENCE__*` 환경변수로 덮어쓸 수 있다.

    ## 진입 (오더블록 탭 + RSI, 필수 조건)

    기준 신호는 활성(비-breaker) 오더블록 탭(tap). 탭 봉의 RSI가

    * **롱**(강세 오더블록): RSI ≤ `rsi_oversold`(과매도)면 진입.
    * **숏**(약세 오더블록): RSI ≥ `rsi_overbought`(과매수)면 진입.

    RSI가 워밍업 중(NaN)이면 진입하지 않는다. EMA·VWMA는 진입에 영향이 없다.

    ## 익절 (EMA/VWMA 중 진입가 너머 가장 가까운 선 도달)

    대상 선은 `tp_ema_lengths` EMA들과 `tp_vwma_length` VWMA. **기본은 EMA 60 +
    VWMA 100 두 개뿐이다**(WAN-66 사용자 확정 규칙). 진입 이후 매 봉, 진입가 **너머**
    (롱은 위·숏은 아래)에 있는 대상 선들 중 **가장 가까운 선**에 고가(롱)/저가(숏)가
    도달하면 그 선 가격에서 **전량 익절**한다. 선은 봉마다 움직이므로 매 봉
    재평가한다. 진입가 너머에 대상 선이 하나도 없으면 익절 목표가 없어 손절/청산에만
    의존한다.

    **차트 표시선과 익절 목표선은 다르다**(WAN-66). 대시보드는 `display_ema_lengths`
    (기본 EMA 20/60/120/240/365)를 그리고, 익절 판정은 이 필드가 아니라
    `tp_ema_lengths`(기본 EMA 60)만 본다. 두 역할을 한 필드가 겸하면 EMA 20에서
    조기 익절하던 버그가 재발하므로 반드시 분리한다.

    ## 손절 (오더블록 무효화)

    진입 근거였던 오더블록이 breaker로 무효화되면(존 반대편 돌파 — 무효화 기준
    Wick/Close는 `OrderBlockParams.zone_invalidation`을 재사용) 그 봉에서 손절한다.
    체결가는 그 봉의 종가와 오더블록 무효화 경계 중 진입가에 더 불리한 쪽이다
    (WAN-65) — wick 무효화 봉이 반전해 종가가 유리하게 마감해도 손절이 이익을
    내지 않는다.

    ## 우선순위

    같은 봉에서 손절·익절이 동시에 충족되면 `stop_before_take_profit`(기본 True)이면
    **손절을 우선**한다(보수적). 한 오더블록당 진입은 첫 탭 1회로 제한된다.
    """

    model_config = ConfigDict(frozen=True)

    # --- 진입: RSI (필수 조건) ---
    rsi_length: int = Field(default=14, ge=1)
    rsi_overbought: float = Field(default=70.0, gt=0, lt=100)
    rsi_oversold: float = Field(default=30.0, gt=0, lt=100)

    # --- 익절: EMA/VWMA 목표선 ---
    use_line_take_profit: bool = False
    """익절(선 도달) 규칙 on/off. **기본 `False`(WAN-81)** — 익절은 `take_profit_mode=
    "fixed_r"`(고정 1.5R)만 쓰고 EMA·VWMA는 익절 판정에서 완전히 뺀다. `True`로
    켜면(구 엔진) `take_profit_mode="line"`과 함께 선 도달 익절을 쓴다."""
    tp_ema_lengths: tuple[int, ...] = DEFAULT_TP_EMA_LENGTHS
    """**익절 판정**에 쓸 EMA 길이들(WAN-66, 기본 EMA 60뿐). 비우면 EMA 목표선 없음.
    차트에 그리기만 하는 선은 여기가 아니라 `display_ema_lengths`에 둔다."""
    tp_vwma_length: int | None = Field(default=100, ge=1)
    """익절 목표로 쓸 VWMA 길이. None이면 VWMA 목표선 없음."""

    # --- 차트 표시선 (익절 판정과 무관, WAN-66) ---
    display_ema_lengths: tuple[int, ...] = DEFAULT_CONFLUENCE_EMA_LENGTHS
    """**대시보드 차트에만** 그리는 EMA 길이들(기본 20/60/120/240/365). 익절 판정에는
    쓰지 않는다 — 판정선은 `tp_ema_lengths`. 두 필드를 분리해 표시선이 익절 후보로
    새어 들어가는 WAN-66 버그의 재발을 막는다."""

    # --- 손절: 오더블록 무효화 ---
    use_order_block_stop: bool = True
    """손절(오더블록 무효화) 규칙 on/off."""

    # --- 우선순위 ---
    stop_before_take_profit: bool = True
    """동일 봉에서 손절·익절 동시 충족 시 손절 우선(보수적)."""

    # --- 진입 방식 (WAN-41 / WAN-215) ---
    entry_mode: Literal["zone_limit"] = "zone_limit"
    """진입 방식 = **지정가(B안, `zone_limit`)** 단독이다. 활성 오더블록 존 근단
    (proximal)에 지정가를 걸어 두고 가격이 닿는 순간 체결한다(`backtest.substep`가
    1분봉 서브스텝으로 시뮬레이션). 사용자의 실제 매매(존에 지정가를 걸어두고 닿는
    순간 체결)를 재현한다.

    ⚠️ 옛 `close`(A안, 탭 봉 종가 시장가 진입)는 사용자가 하지 않는 매매였고 그 손익표는
    전부 무효라, **A안 경로 자체가 WAN-208/WAN-215로 코드에서 제거됐다**. 이 필드는
    이제 단일값이지만, `zone_limit` 진입점(`run_zone_limit_backtest`)이 다른 값을
    `ValueError`로 거부하는 스위치 성격은 그대로다(WAN-95).
    """
    rsi_mode: Literal["closed_bar", "realtime"] = "realtime"
    """RSI 판정 기준. **기본 `realtime`(B안, WAN-95 채택)**: 체결 순간의 실시간(봉내)
    RSI로 판정한다(`strategy.realtime_rsi`). 라이브·백테스트가 동일 상태 머신을 공유한다.

    지정가는 봉 중간에 체결되므로 확정봉 RSI로 판정하면 **체결 시점과 판정 시점이
    어긋난다** — `entry_mode="zone_limit"`과 한 세트로 묶어야 한다(WAN-41 B안 설계,
    WAN-70/WAN-84 검정이 쓴 조합).

    `closed_bar`: 확정봉 RSI로 판정하던 옛 A안(종가 진입)용 값. A안 경로는
    WAN-208/WAN-215로 제거됐고 채택 기본값은 `realtime` 단독이라, 이 값은 호환용으로만
    남아 있다.
    """
    zone_limit_ref: ZoneLimitRef = "proximal"
    """`entry_mode=zone_limit`일 때 지정가를 걸 존 내 기준선.

    `proximal`(기본): 존 근단(롱=존 상단, 숏=존 하단) — 가장 먼저 닿는 경계.
    `mid`: 존 중앙. `distal`: 존 원단(무효화 경계에 가장 가까움 — 더 깊은 진입).
    """
    zone_limit_offset_bps: float = 2.0
    """지정가를 존 근단에서 **얼마나 옮겨 걸 것인가**(bp, 1bp = 0.01%). **기본 2.0(WAN-112)**.

    ⚠️ **단위는 bp지 분수가 아니다** — 2bp를 원하면 `0.0002`가 아니라 `2.0`이다
    (아래 `apply_zone_limit_offset`이 `/ 10_000`으로 나눈다). WAN-112 이슈 본문이
    "0.0 → 0.0002"로 적었는데 그건 분수 표기이고, 그 값을 그대로 넣으면 0.0002bp =
    사실상 무효과다. 사용자 결정("offset 2bp")을 이 필드의 단위로 옮기면 `2.0`이다.

    부호 규약: **양수 = 체결이 쉬워지는 방향**(롱은 위로, 숏은 아래로 — 가격이 오는
    방향으로 마중 나간다). 음수는 반대로 더 깊은 진입(체결은 어려워지고 진입가는
    유리해진다). 롱/숏 대칭이다.

    이 값은 **체결 확실성을 사고 진입가를 파는 거래**다(WAN-99). 롱을 존 근단보다 위에
    걸면 가격이 내 레벨을 관통해 지나가므로 체결이 사실상 보장되지만, 손절(오더블록
    무효화 경계)까지 거리가 늘어 **1R이 커지고 고정 1.5R 익절 목표도 그만큼 멀어진다**.
    `fill_penetration_bps`(WAN-96)가 "얼마나 관통해야 체결로 쳐줄 것인가"를 요구 조건으로
    다룬다면, 이쪽은 **같은 요구를 주문 가격으로 미리 만족시키는** 반대편 수단이다.

    ⚠️ **적용 순서**: `deviation_filter`(볼린저)가 진입가를 재산정한 **다음**에 얹는다
    (`build_zone_limit_candidates`). WAN-95의 "볼린저가 이긴다" 규칙을 깨지 않으며,
    볼린저가 "진입 없음"으로 판정한 셋업을 오프셋으로 되살리지도 않는다.

    ⚠️ **기본 2bp는 데이터가 고른 값이 아니라 사용자 판단이다**(WAN-112). 순수 수익
    최대화로는 **0bp가 이긴다**(WAN-104 IS: 0bp +16.22% > 2bp +15.79%)이고, 심볼별로도
    2bp를 원한 건 SOL 하나뿐(BTC·ETH는 0bp 선호)이다. 2bp는 그 수익 약 0.43%p를
    **체결 신뢰성과 맞바꾸는 보험**이다 — 스치듯 닿은 체결(실거래에서 큐 우선순위 때문에
    가장 안 될 가능성이 높은 체결)을 관통 체결로 바꿔 준다. "데이터가 2bp를 골랐다"로
    인용하지 말 것. 자세한 근거·트레이드오프는 `docs/decisions/wan112.md`.
    """
    limit_valid_bars: int | None = Field(default=24, ge=1)
    """미체결 지정가 주문이 유효한 상위TF 봉 수. 경과하면 취소한다(`zone_limit`).

    **기본 `24`(현행 동작 보존)**. `None`(WAN-73): 유효기간을 두지 않고 존이 무효화
    (breaker)될 때까지 지정가를 유지한다 — 실측 사례에서 존 생성 후 6주 뒤에 첫 탭이
    발생하는 등, 사용자는 존이 살아있는 한 계속 지켜보기 때문이다."""
    cancel_limit_on_condition_fail: bool = False
    """지정가에 닿았지만 실시간 RSI 조건 미충족 시 주문을 취소할지 여부.

    기본 False면 조건이 충족될 때까지(또는 만료·무효화까지) 주문을 유지한다.
    """
    invalidation_cancel: InvalidationCancel = "bar_close"
    """미체결 지정가를 「존이 깨졌다」로 취소하는 **시점**. **기본 `"bar_close"`(WAN-365)**.

    * `"bar_close"` — **채택 기본값(인과)**. 존을 깬 상위TF 봉이 **닫힐 때** 비로소 취소한다.
      그 봉 **안에서**의 탭·체결은 살아남고, 결과는 손절 규칙이 낸다.
    * `"bar_open"` — WAN-364까지의 옛 동작(소급 취소). 무효화 봉의 `open_time`부터 취소된
      것으로 본다. 옛 리포트 재현 전용이다(`harness.LEGACY_INVALIDATION_CANCEL`).

    🚨 **옛 기본값(`"bar_open"`)은 봉이 끝나야 알 수 있는 사실을 봉 처음으로 되돌려 썼다** —
    `ob.break_time`이 존을 깬 상위TF 봉의 `open_time`이라(`strategy/order_blocks.py`) 그 봉
    안에서 체결됐을 주문이 소급 취소된다. 그렇게 지워지는 셋업은 무작위가 아니라 **정의상
    손절로 끝났을 것**이다(롱이면 가격이 존 아랫변을 뚫어야 무효화인데 지정가는 그 위에
    있으므로 **체결이 무효화보다 반드시 먼저**다). 인과적으로는 「취소」가 아니라 「체결 후
    손절」이 옳다 — 사용자 지적 *"손절을 해야지 뭘 취소한다는 거냐"*(WAN-364).

    ⚠️ **「취소를 아예 끄는」 값이 아니다** — 봉이 닫힌 뒤에는 두 값 모두 취소한다. 존이
    죽은 걸 안 다음에도 주문을 걸어 두는 것은 인과가 아니라 다른 엔진이다.

    ⚠️ **알고 받는 것**: 전환하면 성과가 나빠진다(WAN-364 §4 — 채택 북 `oos_warm` 거래당
    net R `+0.1985` → `−0.1798`로 부호가 뒤집힌다). 되살아나는 거래의 88.3%가 손절이다.
    그게 맞는 숫자다 — 옛 플러스가 「지워진 손절」 덕이었다는 뜻이다.

    📌 **층이 둘이다**(WAN-364 §2) — 이 값은 시그널 필터(무효화 봉의 탭을 후보로 받는가,
    `build_zone_limit_candidates`)와 시뮬레이터 취소 시각(`invalidation_cutoff`)을 **함께**
    바꾼다. 한 층만 바꾸면 라벨만 붙은 잡종이 된다.

    ⚠️ **라이브(`live/limit_engine.py`)는 이 필드를 읽지 않는다 — 읽을 필요가 없다.** 라이브는
    존 무효화를 **확정봉에서** 알아 한 봉 늦게 취소하므로 **구조적으로 인과**이고, 채택값과
    같은 쪽이다. `"bar_open"`은 실거래에서 **흉내 낼 방법이 없다**(그 봉이 어떻게 끝날지 알아야
    한다) — 백테스트 전용의 옛 기록 재현 값이다.
    """

    # --- 체결 가정 보수화 (WAN-96) ---
    fill_penetration_bps: float = Field(default=0.0, ge=0.0)
    """지정가 체결에 요구하는 **관통 폭**(bp, 1bp = 0.01%). **기본 0.0 = 터치 체결(현행)**.

    서브스텝 시뮬레이터는 가격이 지정가에 **닿기만 하면** 체결로 본다. 실거래에서는
    닿아도 내 주문 앞에 줄이 있어(큐 우선순위) 체결되지 않을 수 있고, 가격이 스치듯
    찍고 되돌아가면 체결 없이 지나간다. 이 값을 올리면 가격이 지정가를 그만큼 **지나쳐야**
    체결로 인정한다(롱: `low <= 지정가×(1−bps/1e4)`, 숏: `high >= 지정가×(1+bps/1e4)`).

    체결가는 여전히 **지정가 그대로**다 — 관통은 "내 주문이 소화될 만큼 물량이 지나갔다"의
    대리 변수일 뿐, 더 유리한 가격을 받는다는 뜻이 아니기 때문이다.
    """
    fill_dropout_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """체결된 셋업을 무작위로 탈락시킬 비율(큐 우선순위 근사). **기본 0.0(현행)**.

    관통 요구(`fill_penetration_bps`)가 "얼마나 지나가야 체결인가"를 가격 경로로 다룬다면,
    이쪽은 체결률 자체를 강제로 깎아 **결론이 체결률에 얼마나 민감한지**를 본다. 탈락한
    셋업은 거래가 되지 않으므로 단일 포지션 슬롯이 비고, 다른 셋업이 그 자리를 채울 수
    있다(실거래에서 미체결이 다음 기회를 여는 것과 같다).
    """
    fill_dropout_seed: int = 0
    """`fill_dropout_rate` 추첨 시드. 같은 시드 → 같은 탈락 집합(재현 가능).

    `fill_dropout_rate=0.0`이면 난수를 아예 뽑지 않으므로, 기본값 실행은 이 필드와
    무관하게 WAN-95 결과와 **비트 단위로 동일**하다.
    """

    # --- 진입 근거 게이트: 존폭 (WAN-158 옵트인 → WAN-159 채택 → WAN-384 폐지) ---
    max_zone_width_atr: float | None = Field(default=None, gt=0)
    """존폭 필터 — 존폭 ÷ ATR가 이 값 **이하**인 셋업만 진입한다. **기본 `None`(꺼짐 =
    존 두께로는 거르지 않는다, WAN-384)**.

    존폭 = `top − bottom`, ATR는 탭 봉 **직전 확정봉**의 `zone_width_atr_length` ATR다
    (룩어헤드 금지 — 탭 봉 자신의 ATR은 그 봉 종가를 알아야 나온다). 측정 모듈
    (`wan117`/`wan133`/`wan142`/`wan152`/`wan154`)의 `zone_width_atr` 특징과 **같은 정의**라
    그 표의 판정을 엔진에서 그대로 재현한다.

    ⚠️ **단위는 ATR 배수지 퍼센트가 아니다** — 1.28은 "존폭이 ATR의 1.28배 이하"라는 뜻이다.
    퍼센트로 착각해 `0.0128` 같은 값을 넣으면 거의 모든 셋업이 걸러진다(WAN-112의
    `2.0` vs `0.0002` 함정과 같은 부류).

    📌 **꺼짐이 채택 기본값이다**(WAN-384 = 사용자 결정 2026-08-27 「끄자」, WAN-159의 `1.28`
    폐지). 근거는 WAN-378 §1b 격자 — **문턱을 조일수록 거래당 net R이 나빠지고**(`끔`
    −0.1207R → `1.28` −0.1309R → `0.80` −0.1651R) 그 방향이 **진입가 3팔 · 구간 4개 전부**
    같다. 고른 이유는 성적이 최선이라서가 아니라(`끔`과 `2.60`은 규약 폭 ±0.005R 안이라
    구분되지 않는다) **자유 파라미터를 하나 없애기 때문**이다.

    🚨 **「끄면 이긴다」가 아니다** — `끔`도 **−0.1207R로 여전히 마이너스**이고(WAN-378의 108팔
    전부 음수) **거래가 56% 늘어난다**(9,515 → 14,843)라 총 손실·노출은 커진다. 이 결정이
    고른 것은 **덜 잃는 쪽**이다. 경고·잃는 것은
    [`docs/decisions/wan384.md`](../docs/decisions/wan384.md).

    📌 **손절폭 가드(0.3%)와 혼동하지 말 것** — 필터를 꺼도 가드는 **여전히 크다**(문턱 끔에서
    가드 끔 −0.2821R → 0.30% −0.1207R = **+0.1614R**). 존폭 축(끔→1.28의 −0.0101R)보다 **16배
    큰 일**을 한다. 가드는 WAN-76/79·WAN-381 소관이고 이 결정이 건드리지 않는다.

    ⚠️ **지정가(B안) 경로 전용이다** — 종가 진입(A안, `entry_mode="close"`)은 이 필드를
    읽지 않으므로 `build_params`/CLI가 그 조합을 **거부한다**. 조용히 무시하면 "필터를
    켜고 돌렸다"고 믿는 표에 라벨만 붙는다(WAN-95의 교훈).

    🚨 **핀의 방향이 두 세대다** — WAN-158 시절(필터 미존재) 수치를 박아 둔 리포트는
    `None`으로 고정하고(`harness.LEGACY_MAX_ZONE_WIDTH_ATR`, 지금은 기본값과 같아 무동작),
    **WAN-159~WAN-383 사이에 필터를 켠 채 결론을 낸 리포트는 `1.28`로 고정한다**
    (`harness.LEGACY_ZONE_WIDTH_FILTER_ON`). 후자를 빠뜨리면 그 리포트가 조용히 필터 꺼진
    엔진으로 다시 돌아 본문과 어긋난다(WAN-132/149/159가 겪은 부류).
    """
    zone_width_atr_length: int = Field(default=14, ge=1)
    """`max_zone_width_atr`가 쓰는 ATR 길이. 기본 `14` — 측정 모듈이 쓴 길이 그대로다.

    `OrderBlockParams.atr_length`(탐지의 `max_atr_mult` 판정용, 기본 10)와 **다른 필드다** —
    한 필드가 두 역할을 겸하면 탐지 기준을 손댔을 때 진입 필터가 조용히 따라 움직인다
    (WAN-66이 표시선과 익절선을 한 필드에 섞어 겪은 그 사고)."""

    # --- 진입 근거 게이트 (WAN-68) ---
    min_rr: float | None = Field(default=None, gt=0)
    """최소 손익비(R:R) 게이트. **기본 `None`(꺼짐, 현행 동작 보존)**.

    진입 시점에 `먹을 거리`(진입가→진입가 너머 가장 가까운 익절선)와 `잃을 거리`
    (진입가→오더블록 무효화 경계)의 비율(R:R = 먹을 거리 / 잃을 거리)이 이 값보다
    작으면 시그널 생성 단계에서 진입을 기각한다. 진입가 너머에 익절선이 하나도
    없으면(먹을 거리 없음) R:R을 0으로 간주해 기각한다.
    """
    long_deviation_gate_ema_length: int = Field(default=240, ge=1)
    """이격도 게이트가 참조할 EMA 길이. `tp_ema_lengths`·`display_ema_lengths`와
    독립된 필드다(WAN-66 교훈 — 판정용 선과 표시용 선을 한 필드로 섞지 않는다)."""
    long_max_deviation: float | None = None
    """롱 이격도 게이트(연속값). **기본 `None`(꺼짐, 현행 동작 보존)**.

    `(종가 − EMA) / 종가`(이격도)가 이 값보다 **더 음수**일 때만 롱 진입을 허용한다
    (장기선에서 충분히 아래로 벌어졌을 때만 진입). 이격도가 이 값 이상이면(덜
    벌어졌으면) 기각한다. EMA가 워밍업 중(NaN)이면 판정 불가로 간주해 기각한다.
    숏에는 적용하지 않는다.
    """
    short_enabled: bool = False
    """숏(약세 오더블록) 진입 허용 여부. **기본 `False`(WAN-87, WAN-86 결정 1 반영 —
    WAN-81의 숏 활성화를 사용자가 데이터에 근거해 번복)**. `False`면 숏 신호는 항상
    미확정(`confirmed=False`) 처리한다.

    WAN-84(PR #64) OOS 롱/숏 분해에서 롱 +3.07%(9셀) vs 숏 −3.82%(7셀)로 숏이 롱
    수익을 상쇄함을 확인했다. WAN-86(`docs/decisions/wan86.md`)에서 사용자가 이를
    근거로 숏을 다시 비활성화하기로 확정했다. 숏 경로 코드 자체는 삭제하지 않았으므로
    `short_enabled=True`로 재검증 여지를 남겨둔다.

    (과거 경위: WAN-69가 한때 롱 온리를 채택했으나 WAN-81이 사용자 확정 규칙으로 숏을
    기본 활성화했고, WAN-87이 다시 롱 온리로 되돌렸다.)"""

    # --- 진입/익절 재현 (WAN-73) ---
    retap_mode: Literal["once", "every_tap"] = "every_tap"
    """존당 재진입 허용 여부. **기본 `every_tap`(WAN-81 메인 엔진)**: 존이 무효화되기
    전까지 매 탭(재진입)마다 진입을 평가한다(RSI 게이트 등 다른 조건은 그대로
    적용). `combine_obs=True`면 병합 존 경계·병합 상태 기준으로 재탭 후보를
    생성한다(WAN-56 패리티 유지, WAN-81 갭B — 과거엔 재탭 경로가 병합을 무시하고
    원본 존으로 돌아갔다). `once`: 존(병합 포함) 확정 후 첫 탭만 진입 후보로
    삼는다(구 엔진 동작). 동시 포지션은 여전히 1개로 제한되므로(백테스트 엔진이
    플랫일 때만 진입) 청산 후에만 다음 탭이 진입으로 이어진다."""
    rsi_gate_mode: RsiGateMode = "unconditional"
    """RSI 게이트 방향. **기본 `unconditional`(WAN-123 재-베이스라인 = WAN-116 결정 B)**:
    RSI 게이트가 **아예 없다** — 첫 탭·재탭을 가리지 않고 워밍업(NaN)이어도 지정가
    터치 즉시 진입한다. 근거는 WAN-114 ablation의 `L0r→L1`(게이트만의 순수 기여):
    12셀 중 8셀 음수, 1h는 6셀 전부 음수, 유일한 일관 플러스는 학습 구간인 15m IS뿐이다.
    게이트는 거래의 13~14%를 쳐내는데 **그 쳐냄이 순손해**였다 = 고르는 방향이 틀렸다.
    자세한 근거·파급은 [`docs/decisions/wan123.md`](../docs/decisions/wan123.md).

    `first_tap_free`(WAN-81~WAN-122 채택 기본값, 옵트인 보존): 존(병합 존 포함) 확정 후
    **첫 탭**(`tap_index=0`)은 RSI 무관하게(워밍업 NaN이어도) 무조건 진입하고,
    **재탭**(`tap_index>=1`)부터 `extreme` 규칙(롱 `RSI<=rsi_oversold`, 숏
    `RSI>=rsi_overbought`)을 적용한다. 미충족 탭은 기각하되 존을 소각하지 않는다 —
    무효화 전까지 다음 탭에서 다시 평가한다. `extreme`(구 엔진 기본): 모든 탭에 항상
    극단 규칙 적용. `neutral`: RSI가 `rsi_neutral_band` 안(방향 무관)이면 진입.

    ⚠️ **`none`은 이 기본값과 다르다** — 게이트 판정만 없앨 뿐 호출부의 워밍업 요구
    (`rsi is not None`)는 남아 워밍업 구간 탭이 막힌다(`rsi_gate_passes` 참고). 게이트를
    끄려던 자리에 `none`을 넣으면 **끈 줄 알면서 워밍업 필터만 남는다** — WAN-112의
    단위 함정(`2.0` vs `0.0002`)과 같은 부류의 조용한 실패다. 회귀 테스트가 막는다."""
    rsi_neutral_band: tuple[float, float] = (40.0, 60.0)
    """`rsi_gate_mode="neutral"`일 때의 RSI 허용 밴드 `(하한, 상한)`."""
    take_profit_mode: Literal["line", "fixed_r"] = "fixed_r"
    """익절 목표 산정 방식. **기본 `fixed_r`(WAN-81 메인 엔진, 고정 1:1.5R)**: 진입가로부터
    진입 근거 오더블록 무효화 경계까지의 거리(위험 R 1개)의 `take_profit_r`배
    지점에 고정 익절선을 둔다(선 재평가 없이 진입 시점에 확정). 볼린저 진입가
    재산정으로 1R이 줄면 익절 목표도 그만큼 가까워진다.

    `line`(구 엔진): `tp_ema_lengths`·`tp_vwma_length` 선 중 진입가 너머 가장
    가까운 선(`use_line_take_profit`로 on/off)."""
    take_profit_r: float = Field(default=1.5, gt=0)
    """`take_profit_mode="fixed_r"`일 때의 목표 손익비(R 배수). 기본 `1.5`(WAN-81)."""

    # --- 진입가 재산정: SMA/종가 대비 이격 필터 (WAN-75) ---
    deviation_filter: DeviationFilterParams | None = Field(
        default_factory=lambda: DeviationFilterParams(
            anchor="sma", sma_length=20, width_kind="stdev", width_value=2.0
        )
    )
    """이격 필터. **기본 볼린저밴드(SMA 20 ± 2σ, WAN-81 메인 엔진)**. 켜져 있으면
    `deviation_entry_price` 규칙에 따라 오더블록 진입가를 밴드 값으로 재산정하거나
    진입을 기각한다(볼린저 하단선이 존 위/겹침/아래에 있을 때 각각 근단 진입/밴드
    진입/진입 없음). `None`으로 두면 꺼짐(구 엔진 동작)."""

    source: str = "close"

    @model_validator(mode="after")
    def _validate(self) -> ConfluenceParams:
        if self.rsi_oversold >= self.rsi_overbought:
            raise ValueError("rsi_oversold는 rsi_overbought보다 작아야 합니다.")
        if any(length < 1 for length in self.tp_ema_lengths):
            raise ValueError("tp_ema_lengths의 모든 길이는 1 이상이어야 합니다.")
        if len(set(self.tp_ema_lengths)) != len(self.tp_ema_lengths):
            raise ValueError("tp_ema_lengths에 중복된 길이가 있습니다.")
        if any(length < 1 for length in self.display_ema_lengths):
            raise ValueError("display_ema_lengths의 모든 길이는 1 이상이어야 합니다.")
        if len(set(self.display_ema_lengths)) != len(self.display_ema_lengths):
            raise ValueError("display_ema_lengths에 중복된 길이가 있습니다.")
        if self.use_line_take_profit and not self.tp_ema_lengths and self.tp_vwma_length is None:
            raise ValueError(
                "use_line_take_profit=True면 tp_ema_lengths 또는 tp_vwma_length 중 "
                "최소 하나의 익절 목표선이 필요합니다."
            )
        low, high = self.rsi_neutral_band
        if not (0.0 < low < high < 100.0):
            raise ValueError("rsi_neutral_band는 0 < 하한 < 상한 < 100을 만족해야 합니다.")
        return self

    def rsi_gate_passes(self, is_long: bool, rsi: float) -> bool:
        """`rsi_gate_mode`에 따른 RSI 게이트 판정(WAN-73). `rsi`는 워밍업 통과(NaN 아님) 값."""
        return rsi_gate_passes(
            rsi,
            is_long=is_long,
            mode=self.rsi_gate_mode,
            rsi_oversold=self.rsi_oversold,
            rsi_overbought=self.rsi_overbought,
            rsi_neutral_band=self.rsi_neutral_band,
        )

    def zone_limit_price(self, order_block: OrderBlock) -> float:
        """`zone_limit_ref`에 따른 지정가(존 내 기준선)를 반환한다 (WAN-41).

        롱(강세 오더블록)은 존 상단이 근단(proximal, 먼저 닿음)·하단이 원단(distal,
        무효화 경계), 숏(약세 오더블록)은 그 반대다. `mid`는 존 중앙.
        """
        top, bottom = order_block.top, order_block.bottom
        if self.zone_limit_ref == "mid":
            return (top + bottom) / 2.0
        is_long = order_block.direction is OrderBlockDirection.BULLISH
        proximal, distal = (top, bottom) if is_long else (bottom, top)
        return proximal if self.zone_limit_ref == "proximal" else distal

    def zone_width_filter_passes(self, order_block: OrderBlock, atr_value: float | None) -> bool:
        """존폭 필터(WAN-158) 판정. 필터가 꺼져 있으면(기본) 항상 True.

        `atr_value`는 **탭 봉 직전 확정봉**의 ATR다 — 호출부가 그 규칙을 지킨다(여기서는
        봉 위치를 알 수 없다. `rsi_gate_passes`가 첫 탭 면제를 호출부 책임으로 두는 것과
        같은 계약이다).

        판정 불가(워밍업 NaN·비양수 ATR)는 **기각**한다. 통과로 두면 워밍업 구간에서만
        필터가 조용히 꺼져 "켰다고 믿는데 안 켜진" 구간이 생긴다(WAN-123 `none` 함정).
        """
        threshold = self.max_zone_width_atr
        if threshold is None:
            return True
        if atr_value is None or math.isnan(atr_value) or atr_value <= 0.0:
            return False
        return (order_block.top - order_block.bottom) / atr_value <= threshold

    def apply_zone_limit_offset(self, price: float, *, is_long: bool) -> float:
        """지정가에 `zone_limit_offset_bps`를 얹는다 (WAN-99).

        양수 오프셋은 체결이 쉬워지는 방향(롱=위, 숏=아래)으로 가격을 민다. 손절 참조가
        (존 원단)는 건드리지 않으므로 1R이 그만큼 늘고, 1R에서 파생되는 고정 R 익절
        목표도 함께 멀어진다 — 그게 이 오프셋이 지불하는 대가다.

        **`zone_limit_price`와 합치지 않고 따로 두는 이유**: 오프셋은 `deviation_filter`
        (볼린저)가 진입가를 재산정한 **뒤에** 얹혀야 하는데, 존 근단 계산은 그 **전에**
        일어난다. 한 메서드로 묶으면 순서가 강제되지 않아 볼린저 재산정 가격에 오프셋이
        빠지거나 두 번 얹힐 수 있다.

        오프셋이 0이면 입력을 그대로 돌려준다 — **0bp로 명시 고정한 과거 리포트**
        (WAN-88/96/103/108/110 등, WAN-112 이전 엔진 기록)가 부동소수점 연산 없이
        비트 단위로 재현된다. 채택 기본값은 WAN-112부터 2bp라 이 지름길을 타지 않는다.
        """
        if self.zone_limit_offset_bps == 0.0:
            return price
        sign = 1.0 if is_long else -1.0
        return price * (1.0 + sign * self.zone_limit_offset_bps / 10_000.0)

    @property
    def tp_vwma_key(self) -> str | None:
        """익절 VWMA 선의 스냅샷 키(`vwma_<길이>`). 미사용이면 None."""
        return None if self.tp_vwma_length is None else f"vwma_{self.tp_vwma_length}"

    @property
    def sorted_tp_ema_lengths(self) -> list[int]:
        """**익절 판정**에 쓰는 EMA 길이들을 오름차순 정렬한 리스트."""
        return sorted(self.tp_ema_lengths)

    @property
    def sorted_display_ema_lengths(self) -> list[int]:
        """**차트 표시**용 EMA 길이들을 오름차순 정렬한 리스트(WAN-66, 익절과 무관)."""
        return sorted(self.display_ema_lengths)
