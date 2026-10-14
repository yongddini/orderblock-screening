# ── 복사 이식(vendoring, OBS-5) ─────────────────────────────────────────────────
# 출처: github.com/yongddini/AlphaBlock  strategy/order_blocks.py
# 커밋: de825d2762c94e2b177db4608af8682f2a7fe842
# 바꾼 것: import 경로 한 줄(`strategy.models` → `obscreen.detect.alphablock.models`).
# 이 줄 아래를 손으로 고치지 말 것 — tests/test_alphablock_parity.py가 원본과 대조한다.
# 갱신 방법은 obscreen/detect/alphablock/__init__.py 설명을 따른다.
# ── 이하 원본 그대로 ──────────────────────────────────────────────────────────────
"""오더블록(Order Block) 탐지 & 시그널 생성.

Fluxchart "Volumized Order Blocks" (TradingView, MPL-2.0)의 탐지 로직을
`strategy/reference/README.md` 명세를 기준으로 pandas OHLCV 입력에 대해
이식한 순수 함수/클래스. 원본은 존 탐지·무효화까지만 하며, 진입 시그널
레이어는 AlphaBlock의 확장이다 (`strategy/reference/README.md` 참고).

입력 DataFrame은 `data.storage.OhlcvStore.load()`가 반환하는 스키마
(`open_time`(ms), `open`, `high`, `low`, `close`, `volume`, 선택적 `closed`)
를 따른다. `closed` 컬럼이 있으면 확정봉(`closed=True`)만 사용한다 — 원본이
`barstate.isconfirmed`에서만 갱신하는 것과 동일한 제약.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import pandas as pd

from obscreen.detect.alphablock.models import (
    OrderBlock,
    OrderBlockDirection,
    OrderBlockParams,
    OrderBlockResult,
    OrderBlockSignal,
    obs_touch,
    select_active,
)

_REQUIRED_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


@dataclass(eq=False)
class _SwingPoint:
    """지연 스윙 지점. 원본 `obSwing`에 대응."""

    index: int
    price: float
    crossed: bool = False


@dataclass(eq=False)
class _RawOrderBlock:
    """탐지 진행 중 상태를 갖는 오더블록. 원본 `orderBlockInfo`에 대응.

    WAN-47: 존은 삭제되지 않고 전체 생애주기를 기록한다. `swept`로 소멸 여부를,
    `swept_time`으로 소멸 시각을, `tapped_times`로 재진입 시각들을 보존한다.
    `_inside`는 tap 전이(바깥→안) 판정을 위한 내부 상태다.
    """

    top: float
    bottom: float
    ob_volume: float
    direction: OrderBlockDirection
    start_time: int
    confirmed_time: int
    ob_low_volume: float
    ob_high_volume: float
    breaker: bool = False
    break_time: int | None = None
    swept: bool = False
    swept_time: int | None = None
    tapped_times: list[int] = field(default_factory=list)
    displacement_atr: float | None = None
    _inside: bool = False

    def to_model(self, *, combined: bool = False) -> OrderBlock:
        return OrderBlock(
            direction=self.direction,
            top=self.top,
            bottom=self.bottom,
            start_time=self.start_time,
            confirmed_time=self.confirmed_time,
            ob_volume=self.ob_volume,
            ob_low_volume=self.ob_low_volume,
            ob_high_volume=self.ob_high_volume,
            breaker=self.breaker,
            break_time=self.break_time,
            swept_time=self.swept_time,
            tapped_times=tuple(self.tapped_times),
            combined=combined,
            displacement_atr=self.displacement_atr,
        )


def _true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    n = len(highs)
    tr = [0.0] * n
    for i in range(n):
        if i == 0:
            tr[i] = highs[i] - lows[i]
        else:
            tr[i] = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
    return tr


def _wilder_rma(values: list[float], length: int) -> list[float | None]:
    """Wilder's RMA (`ta.rma`). 처음 `length-1`개는 `None`(미확정)."""
    n = len(values)
    result: list[float | None] = [None] * n
    if n < length:
        return result
    seed = sum(values[:length]) / length
    result[length - 1] = seed
    prev = seed
    for i in range(length, n):
        prev = (prev * (length - 1) + values[i]) / length
        result[i] = prev
    return result


def _atr(
    highs: list[float], lows: list[float], closes: list[float], length: int
) -> list[float | None]:
    return _wilder_rma(_true_range(highs, lows, closes), length)


def _rolling_max(values: list[float], length: int) -> list[float]:
    result = pd.Series(values).rolling(window=length, min_periods=1).max().tolist()
    return [float(v) for v in result]


def _rolling_min(values: list[float], length: int) -> list[float]:
    result = pd.Series(values).rolling(window=length, min_periods=1).min().tolist()
    return [float(v) for v in result]


def _generate_signals(
    order_blocks: list[OrderBlock],
    times: list[int],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    include_retaps: bool = False,
) -> list[OrderBlockSignal]:
    """활성(비-breaker) 존에 가격이 재진입(tap)하면 진입 후보 시그널 생성.

    원본에는 없는 AlphaBlock 확장(기본 골격). `include_retaps=False`(기본)면 존이
    확정된 이후 **첫 재진입만**(`tap_index=0`) 시그널로 기록한다(`OrderBlockResult.
    signals`가 쓰는 경로). `include_retaps=True`면 존이 무효화되기 전까지의 **모든**
    재진입을 `tap_index`(0-based) 부착해 기록한다(WAN-81, `OrderBlockResult.
    retap_signals`가 쓰는 경로) — 이미 `_invalidate()`가 바깥→안 전이 시각만 기록해 둔
    `ob.tapped_times`를 그대로 순회한다. 두 경로 모두 breaker로 전환된 존은
    `status="cancelled"`로 표시한다.

    WAN-47: `order_blocks`는 **전체 아카이브**(살아남은 존뿐 아니라 깨지고 소멸한
    존까지)를 받는다. 이로써 생존자 편향 없이 모든 존의 탭·손절이 백테스트에
    반영된다. **look-ahead 금지** — 각 존의 신호는 그 존이 확정(`confirmed_time`)된
    이후, 무효화(`break_time`) 이전(무효화 봉 포함)의 자기 시간축에서만 나온다.
    시각 `t`의 신호는 `t` 시점에 이미 확정·미소멸한 존만 근거로 하므로, 데이터를
    미래에서 잘라도 과거 신호는 바뀌지 않는다.
    """
    signals: list[OrderBlockSignal] = []
    n = len(times)
    # WAN-49: times는 정렬된 배열이므로 존별 구간 경계를 이진 탐색으로 O(log n)에 구한다.
    # (이전에는 매 존마다 0부터 선형 스캔 → O(존수 × 봉수)로 3년치에서 수천만 회.)
    # bisect_right(confirmed_time): 확정 시각을 **초과**하는 첫 봉(구버전 `<= ` 루프와 동일).
    # bisect_left(break_time): 무효화 시각 **이상**인 첫 봉(구버전 `< ` 루프와 동일).
    # break_time > confirmed_time 이 항상 성립하므로 break_pos >= start_pos 가 보장된다.
    for archive_idx, ob in enumerate(order_blocks):
        zone_key = frozenset({archive_idx})
        break_pos: int | None = None
        if ob.break_time is not None:
            break_pos = bisect.bisect_left(times, ob.break_time)

        if not include_retaps:
            start_pos = bisect.bisect_right(times, ob.confirmed_time)
            end_pos = n if break_pos is None else min(n, break_pos + 1)
            for i in range(start_pos, end_pos):
                if lows[i] <= ob.top and highs[i] >= ob.bottom:
                    # WAN-47: 상태는 존의 **최종** breaker 여부가 아니라 **이 탭이
                    # 무효화 전인지**로 정한다. 무효화 봉 자체에서의 탭만 cancelled고,
                    # 그 전의 탭은 유효한 진입(나중에 무효화되면 손절)이다. 최종
                    # 상태로 판정하면 결국 깨질 존의 정상 진입까지 모두 배제돼
                    # 생존자 편향이 재발한다.
                    is_break_bar = break_pos is not None and i >= break_pos
                    signals.append(
                        OrderBlockSignal(
                            direction=ob.direction,
                            trigger_time=times[i],
                            price=closes[i],
                            order_block=ob,
                            status="cancelled" if is_break_bar else "active",
                            tap_index=0,
                            zone_key=zone_key,
                        )
                    )
                    break
            continue

        # include_retaps=True(WAN-81): 이미 기록된 모든 탭 전이 시각을 재생한다.
        for tap_index, tap_time in enumerate(ob.tapped_times):
            pos = bisect.bisect_left(times, tap_time)
            if pos >= n or times[pos] != tap_time:
                continue  # 방어적: tapped_times는 항상 times의 원소여야 한다.
            if break_pos is not None and pos > break_pos:
                break  # tapped_times는 오름차순 — 무효화 창을 벗어나면 이후도 모두 벗어남.
            is_break_bar = break_pos is not None and pos >= break_pos
            signals.append(
                OrderBlockSignal(
                    direction=ob.direction,
                    trigger_time=tap_time,
                    price=closes[pos],
                    order_block=ob,
                    status="cancelled" if is_break_bar else "active",
                    tap_index=tap_index,
                    zone_key=zone_key,
                )
            )
    return signals


@dataclass(eq=False)
class _MergedGroup:
    """어떤 봉 시점의 병합 존 하나. 원본 `combineOBsFunc` 결과 박스에 대응(WAN-56).

    같은 방향·겹치는(touch) 활성 존들의 연결 요소(connected component)다. `top`/`bottom`은
    구성 존들의 **합집합** 경계이고, `merged_ob`는 그 병합 존을 백테스트 시그널에 실어
    보낼 값 객체다(단일 존이면 원본 존 그대로 — `combine_obs=False` 경로와 동일 시그널).
    `member_indices`는 아카이브 인덱스 집합으로, "병합 단위당 1회 진입(R1)" 불변식을
    유지하는 데 쓴다.
    """

    direction: OrderBlockDirection
    top: float
    bottom: float
    latest_confirmed: int
    break_time: int | None
    member_indices: frozenset[int]
    merged_ob: OrderBlock


def _pine_max_break_time(obs: list[OrderBlock]) -> int | None:
    """원본 `combineOBsFunc`의 `breakTime` 접기 (`.pine:391-392`, WAN-400 §2).

    원본은 쌍 병합마다 `math.max(nz(A.breakTime), nz(B.breakTime))`을 취하고 그 결과가
    `0`이면 `na`로 되돌린다. `nz(na) = 0`이고 실제 시각은 항상 양수이므로, 이 접기는
    구성원 수·병합 순서와 무관하게 다음과 **정확히 같다**:

    * 구성 존이 **전부 살아 있으면**(`break_time`이 전부 `None`) 병합 존도 살아 있다.
    * 하나라도 죽었으면 병합 존은 **죽은 것들 중 가장 늦은 시각**에 죽는다 —
      🚨 아직 살아 있는 구성 존이 있어도 그렇다(`nz(na)=0`은 최댓값을 못 올린다).
    """
    dead = [ob.break_time for ob in obs if ob.break_time is not None]
    return max(dead) if dead else None


def _make_merged_group(
    direction: OrderBlockDirection,
    members: list[tuple[int, OrderBlock]],
    *,
    break_time_rule: str = "distal",
) -> _MergedGroup:
    """연결 요소(같은 방향, 서로 touch) 존들을 하나의 병합 존으로 접는다.

    병합 존의 무효화(`break_time`)는 기본값(`break_time_rule="distal"`)에서 **원단(distal)
    경계를 정의하는 구성 존**의 무효화와 같다. 강세 병합 존의 distal은
    `bottom = min(구성 bottom)`이고, 그 최저 bottom을 가진 구성 존이 깨지는 순간(가격이 그
    bottom 아래로) 병합 존도 무효화된다(합집합 경계 이탈). 약세는 `top = max(구성 top)`의
    최고 top 존이 distal이다. 따라서 병합 존의 손절선은 구성 존 중 가장 바깥 존의 손절선과
    일치한다(WAN-56 영향 #2: 손절 거리 확장).

    `break_time_rule="pine_max"`(옵트인, WAN-400 §2)는 그 자리에 원본 규칙
    (`_pine_max_break_time`)을 쓴다 — ⚠️ 기본값 전환은 **사용자 결정**이다.
    """
    obs = [ob for _, ob in members]
    indices = frozenset(idx for idx, _ in members)
    if len(obs) == 1:
        # 단일 존: 원본 존을 그대로 실어 `combine_obs=False`와 완전히 동일한 시그널을 낸다.
        ob = obs[0]
        return _MergedGroup(
            direction=direction,
            top=ob.top,
            bottom=ob.bottom,
            latest_confirmed=ob.confirmed_time,
            break_time=ob.break_time,
            member_indices=indices,
            merged_ob=ob,
        )

    is_bullish = direction is OrderBlockDirection.BULLISH
    top = max(ob.top for ob in obs)
    bottom = min(ob.bottom for ob in obs)
    # 원단(distal) 경계를 정의하는 구성 존 → 병합 존의 무효화/소멸 시각을 이 존에서 취한다.
    distal = min(obs, key=lambda ob: ob.bottom) if is_bullish else max(obs, key=lambda ob: ob.top)
    break_time = _pine_max_break_time(obs) if break_time_rule == "pine_max" else distal.break_time
    tapped: set[int] = set()
    for ob in obs:
        tapped.update(ob.tapped_times)
    merged_ob = OrderBlock(
        direction=direction,
        top=top,
        bottom=bottom,
        start_time=min(ob.start_time for ob in obs),
        confirmed_time=min(ob.confirmed_time for ob in obs),
        ob_volume=sum(ob.ob_volume for ob in obs),
        ob_low_volume=sum(ob.ob_low_volume for ob in obs),
        ob_high_volume=sum(ob.ob_high_volume for ob in obs),
        # 원본은 `breaker := A.breaker or B.breaker`(`.pine:399`)라 `pine_max`와 짝이 맞는다.
        breaker=break_time is not None,
        break_time=break_time,
        swept_time=distal.swept_time,
        tapped_times=tuple(sorted(tapped)),
        combined=True,
        num_component_obs=len(obs),
    )
    return _MergedGroup(
        direction=direction,
        top=top,
        bottom=bottom,
        latest_confirmed=max(ob.confirmed_time for ob in obs),
        break_time=break_time,
        member_indices=indices,
        merged_ob=merged_ob,
    )


def _build_merged_groups(
    alive: list[tuple[int, OrderBlock]], now: int, *, break_time_rule: str = "distal"
) -> list[_MergedGroup]:
    """현재 활성 존들을 방향별로 touch 연결 요소(병합 단위)로 묶는다.

    원본 `combineOBsFunc`는 매 봉 병합을 처음부터 다시 계산하므로, 여기서도 그 시점의
    활성 집합만으로 union-find 연결 요소를 구한다. 병합은 touch 관계의 추이 폐포이고,
    연결 요소의 합집합 경계가 원본의 반복 병합 결과와 동일하다(연결된 존들은 가격·시간축
    모두 연속이라 합집합 박스에 새로 닿는 제3의 존은 반드시 어떤 구성 존과도 닿는다).
    """
    groups: list[_MergedGroup] = []
    for direction in (OrderBlockDirection.BULLISH, OrderBlockDirection.BEARISH):
        members = [(idx, ob) for idx, ob in alive if ob.direction is direction]
        m = len(members)
        if m == 0:
            continue
        # bottom 오름차순 스윕: 가격축이 겹치는 후보끼리만 touch를 검사한다(O(존²) 회피).
        # 정렬 후 뒤쪽 존일수록 bottom이 크므로, 열린 존의 top이 현재 bottom 이하가 되면
        # 이후 어떤 존과도 가격이 겹칠 수 없어 후보에서 제거한다(구간 겹침 스윕).
        members.sort(key=lambda p: p[1].bottom)
        parent = list(range(m))

        def find(x: int, parent: list[int] = parent) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        open_idx: list[int] = []
        for k in range(m):
            ob_k = members[k][1]
            bottom_k = ob_k.bottom
            open_idx = [o for o in open_idx if members[o][1].top > bottom_k]
            rk = find(k)
            for o in open_idx:
                # 가격은 이미 겹침(top_o > bottom_k) — touch는 시간축까지 확인한다.
                if find(o) != rk and obs_touch(members[o][1], ob_k, now):
                    parent[find(o)] = rk
            open_idx.append(k)

        components: dict[int, list[tuple[int, OrderBlock]]] = {}
        for i in range(m):
            components.setdefault(find(i), []).append(members[i])
        for comp in components.values():
            groups.append(_make_merged_group(direction, comp, break_time_rule=break_time_rule))
    return groups


def _generate_merged_signals(
    archive: list[OrderBlock],
    times: list[int],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    include_retaps: bool = False,
    break_time_rule: str = "distal",
    tap_state: str = "cluster",
) -> list[OrderBlockSignal]:
    """병합 존(combine_obs) 기준으로 탭 진입 시그널을 **시간 순 재생**으로 생성한다 (WAN-56).

    렌더링에만 적용되던 존 병합을 백테스트 시그널까지 끌어올린다. 원본이 매 봉
    `combineOBsFunc`을 호출하듯, 각 봉 `t`에서 **그 시점까지 확정·미소멸한 존들**의 병합
    상태를 구성하고, 그 병합 존에 대해 탭(재진입, 바깥→안 전이)을 판정한다. 미래에
    생길 존과 미리 합치지 않으므로 look-ahead가 없다(데이터를 미래에서 잘라도 과거
    신호 불변).

    **개별 존 단위 `entered` 추적 (WAN-81 §5, 구 WAN-82 버그 흡수)**: 예전 구현은
    `entered`를 구성 인덱스 집합 전체로만 검사해, 이미 진입한 클러스터에 **새로
    편입된 존**(그 존 자신은 한 번도 탭된 적이 없음)까지 영구적으로 진입 기회를
    잃었다. 이제는 `g.member_indices - entered`(아직 진입하지 않은 구성 존)가
    비어있지 않은 탭만 "새 진입"(`tap_index=0`)으로 세고, 그 순간의 구성 존
    전체를 `entered`에 편입한다. 이미 모든 구성원이 진입을 마친 병합 단위가 다시
    탭되면(재탭) `include_retaps=True`일 때만 `tap_index`를 1부터 증가시키며
    추가 시그널을 낸다(`include_retaps=False`면 스킵 — `OrderBlockResult.signals`가
    쓰는 경로로, "병합 단위당 존별 첫 탭만" 유지).

    탭은 **바깥→안 전이**만 센다(직전 봉의 포함 여부를 추적) — 그렇지 않으면 가격이
    며칠씩 존 안에 머무는 매 봉마다 재탭이 발생해 버린다. 그 상태를 무엇으로 키잉할지가
    `tap_state`다(WAN-400 §3): 기본 `"cluster"`는 구성 존 집합(`member_indices`)이라
    **구성이 바뀌면 리셋**되고, 옵트인 `"zone"`은 아카이브 존 인덱스마다 들고 그룹 값을
    구성 존들에서 접는다(`all`). 크기는 `wan400_merge_parity_census`가 낸다.

    성능(WAN-49): 병합 상태는 활성 집합이 바뀔 때(존 확정·무효화·소멸)만 재계산하고,
    변화 없는 봉에서는 캐시한 병합 존에 탭만 확인한다. 겹치지 않는(단일) 존은
    `combine_obs=False`와 동일한 시그널을 내므로, 병합의 영향은 겹치는 클러스터에 한정된다.
    """
    n = len(times)
    if n == 0 or not archive:
        return []

    # 확정 시각 오름차순으로 활성 집합에 편입(동시각은 아카이브 인덱스로 안정 정렬).
    to_add = sorted(enumerate(archive), key=lambda p: (p[1].confirmed_time, p[0]))
    add_ptr = 0
    alive: list[tuple[int, OrderBlock]] = []
    entered: set[int] = set()
    # `tap_state="cluster"`(기본): 키가 그 시점 구성 존 집합 — 구성이 바뀌면 리셋된다.
    inside_state: dict[frozenset[int], bool] = {}
    retap_counter: dict[frozenset[int], int] = {}
    # `tap_state="zone"`/`"zone_all"`(옵트인, WAN-400 §3): 상태를 아카이브 존 인덱스마다
    # 들고, 그룹의 값은 구성 존들에서 접는다(재탭 카운터는 `max`).
    inside_by_zone: dict[int, bool] = {}
    retap_by_zone: dict[int, int] = {}
    by_zone = tap_state in ("zone", "zone_all")
    # 포함 여부를 접는 방식이 두 규칙을 가른다 — `any`가 사용자 결정(2026-09-01)이고
    # `all`은 그것으로 대체된 개발자 초안이다(크기를 재려고 측정 대조로만 존치).
    fold_inside = all if tap_state == "zone_all" else any
    groups: list[_MergedGroup] = []
    dirty = True
    signals: list[OrderBlockSignal] = []

    for t in range(n):
        now = times[t]
        while add_ptr < len(to_add) and to_add[add_ptr][1].confirmed_time <= now:
            alive.append(to_add[add_ptr])
            add_ptr += 1
            dirty = True
        kept: list[tuple[int, OrderBlock]] = []
        for idx, ob in alive:
            if ob.swept_time is not None and ob.swept_time <= now:
                # 소멸 → 병합 후보 집합에서 제외. WAN-400 §1: 이것이 **원본과 같다** —
                # 원본도 `bullishOrderBlocksList.remove(i)`(`.pine:267`)로 데이터 리스트에서
                # 빼고, `handleOrderBlocksFinal`이 그 리스트에서만 병합 입력을 만든다.
                dirty = True
                continue
            if ob.break_time is not None and ob.break_time == now:
                dirty = True  # 무효화 상태 전이 → 병합 경계(면적/end) 재계산 필요.
            kept.append((idx, ob))
        alive = kept

        if dirty:
            groups = _build_merged_groups(alive, now, break_time_rule=break_time_rule)
            dirty = False

        for g in groups:
            if now <= g.latest_confirmed:
                continue  # 병합 존 형성 봉의 자기-포함 탭 배제(원본 확정 봉 제외와 동일).
            in_window = g.break_time is None or now <= g.break_time
            is_inside = lows[t] <= g.top and highs[t] >= g.bottom
            if by_zone:
                # 🚨 `any`(사용자 결정): 병합된 것은 **하나의 박스**이고 커져도 같은
                # 박스다 — 가격이 안 움직였으면 새 진입 기회가 아니다. 새 존이 합류해도
                # 기존 구성 존이 직전 봉에 안에 있었으면 「계속 안에 있었다」로 본다.
                # WAN-82는 그대로다: 가격이 **나갔다 다시 들어오면** 그때 탭이 나고
                # `member_indices - entered`가 미진입 구성 존에 기회를 준다.
                was_inside = fold_inside(inside_by_zone.get(i, False) for i in g.member_indices)
            else:
                was_inside = inside_state.get(g.member_indices, False)
            if in_window and is_inside and not was_inside:
                new_members = g.member_indices - entered
                tap_index: int | None = None
                if new_members:
                    tap_index = 0
                    entered |= g.member_indices
                    if by_zone:
                        for i in g.member_indices:
                            retap_by_zone[i] = 1
                    else:
                        retap_counter[g.member_indices] = 1
                elif include_retaps:
                    if by_zone:
                        # 구성 존들의 최댓값에서 이어 센다 — 병합·분리로 1로 안 돌아간다.
                        tap_index = max(retap_by_zone.get(i, 1) for i in g.member_indices)
                        for i in g.member_indices:
                            retap_by_zone[i] = tap_index + 1
                    else:
                        tap_index = retap_counter.get(g.member_indices, 1)
                        retap_counter[g.member_indices] = tap_index + 1
                if tap_index is not None:
                    is_break_bar = g.break_time is not None and now >= g.break_time
                    signals.append(
                        OrderBlockSignal(
                            direction=g.direction,
                            trigger_time=now,
                            price=closes[t],
                            order_block=g.merged_ob,
                            status="cancelled" if is_break_bar else "active",
                            tap_index=tap_index,
                            zone_key=g.member_indices,
                        )
                    )
            if by_zone:
                for i in g.member_indices:
                    inside_by_zone[i] = is_inside
            else:
                inside_state[g.member_indices] = is_inside
    return signals


class OrderBlockDetector:
    """오더블록 탐지 & 시그널 생성기.

    사용법::

        detector = OrderBlockDetector(OrderBlockParams())
        result = detector.run(ohlcv_df)
    """

    def __init__(self, params: OrderBlockParams | None = None) -> None:
        self.params = params or OrderBlockParams()

    def run(self, df: pd.DataFrame) -> OrderBlockResult:
        frame = self._prepare(df)
        n = len(frame)
        if n == 0:
            return OrderBlockResult(order_blocks=[], signals=[])

        highs = frame["high"].astype(float).tolist()
        lows = frame["low"].astype(float).tolist()
        closes = frame["close"].astype(float).tolist()
        opens = frame["open"].astype(float).tolist()
        volumes = frame["volume"].astype(float).tolist()
        times = frame["open_time"].astype("int64").tolist()

        params = self.params
        swing_length = params.swing_length
        atr = _atr(highs, lows, closes, params.atr_length)
        upper = _rolling_max(highs, swing_length)
        lower = _rolling_min(lows, swing_length)

        use_wick = params.zone_invalidation == "wick"

        swing_type = 0
        top: _SwingPoint | None = None
        bottom: _SwingPoint | None = None

        # WAN-49: 아카이브(전체 생애 보존)와 활성 리스트(아직 소멸 안 한 존)를 분리한다.
        # `bullish_obs`/`bearish_obs`는 WAN-47 그대로 생성된 모든 존을 담는 아카이브이고,
        # `active_bull`/`active_bear`는 `_invalidate()`의 **순회 대상**이다. 존이 소멸(swept)
        # 하면 활성 리스트에서만 빠지고 아카이브에는 남는다. 아카이브의 내용·순서는
        # WAN-47과 100% 동일하다(순회 집합만 좁혔을 뿐, 존별 상태 전이는 불변).
        bullish_obs: list[_RawOrderBlock] = []
        bearish_obs: list[_RawOrderBlock] = []
        active_bull: list[_RawOrderBlock] = []
        active_bear: list[_RawOrderBlock] = []

        # WAN-47: 탐지는 **전체 히스토리**를 스캔한다. 원본의 max_distance_to_last_bar는
        # "마지막 봉에서 N봉 이내만 탐지"하는 스캔 상한이었지만, 그러면 백테스트 기간의
        # 앞부분에서 당시 유효했던 존이 아카이브에서 통째로 빠진다(생존자 편향의 또 다른
        # 얼굴). 탐지/렌더 분리에 따라 이 상한은 **렌더 최근성 필터**로 옮겨(아래
        # rendered 계산), 아카이브는 생성된 모든 존을 담는다.
        for t in range(n):
            if t >= swing_length:
                lag = t - swing_length
                if highs[lag] > upper[t]:
                    new_swing_type = 0
                elif lows[lag] < lower[t]:
                    new_swing_type = 1
                else:
                    new_swing_type = swing_type
                if new_swing_type == 0 and swing_type != 0:
                    top = _SwingPoint(index=lag, price=highs[lag])
                if new_swing_type == 1 and swing_type != 1:
                    bottom = _SwingPoint(index=lag, price=lows[lag])
                swing_type = new_swing_type

            active_bull = self._invalidate(
                active_bull,
                is_bullish=True,
                use_wick=use_wick,
                t=t,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                times=times,
            )
            top = self._create_bullish(
                top, bullish_obs, active_bull, params, t, highs, lows, closes, volumes, times, atr
            )

            active_bear = self._invalidate(
                active_bear,
                is_bullish=False,
                use_wick=use_wick,
                t=t,
                opens=opens,
                highs=highs,
                lows=lows,
                closes=closes,
                times=times,
            )
            bottom = self._create_bearish(
                bottom,
                bearish_obs,
                active_bear,
                params,
                t,
                highs,
                lows,
                closes,
                volumes,
                times,
                atr,
            )

        # WAN-47: 탐지(archive)와 렌더링(view)을 분리한다. 아카이브는 생성된 모든
        # 존의 전체 생애주기를 담고(트리밍·삭제 없음), 신호는 아카이브 전체에서
        # 생성한다(생존자 편향 제거). "지금 차트에 그릴 박스"는 렌더링 뷰가 파생한다.
        archive = [ob.to_model() for ob in bullish_obs] + [ob.to_model() for ob in bearish_obs]
        # WAN-56: `combine_obs`면 병합 존 기준으로 시그널을 낸다(트레이딩뷰 렌더와 동일한
        # 존 집합을 백테스트도 본다). `combine_obs=False`는 원본 단위 경로를 유지해 비교
        # 가능하게 남긴다. 아카이브(`order_blocks`)는 두 경로 모두 원본 단위로 보존한다.
        # WAN-81: `retap_signals`는 `ConfluenceParams.retap_mode="every_tap"`이 소비하는
        # 전체 탭(재탭 포함) 뷰 — combine_obs 여부에 따라 병합/원본 경로를 그대로 따른다
        # (갭B: 재탭 경로도 병합을 반영해야 한다).
        if params.combine_obs:
            rule, state = params.merged_break_time_rule, params.merged_tap_state
            signals = _generate_merged_signals(
                archive, times, highs, lows, closes, break_time_rule=rule, tap_state=state
            )
            retap_signals = _generate_merged_signals(
                archive,
                times,
                highs,
                lows,
                closes,
                include_retaps=True,
                break_time_rule=rule,
                tap_state=state,
            )
        else:
            signals = _generate_signals(archive, times, highs, lows, closes)
            retap_signals = _generate_signals(
                archive, times, highs, lows, closes, include_retaps=True
            )

        # 렌더 뷰(트레이딩뷰 "현재 그림"): 마지막 봉에서 max_distance_to_last_bar봉 이내에
        # 확정된 존만 대상으로, 방향별 zone_limit개를 병합해 낸다. 데이터가 스캔 상한보다
        # 짧으면(대부분의 테스트·픽스처) 필터는 무효라 기존 동작과 동일하다.
        cutoff_index = max(0, (n - 1) - params.max_distance_to_last_bar + 1)
        cutoff_time = times[cutoff_index]
        recent = [ob for ob in archive if ob.confirmed_time >= cutoff_time]
        rendered = select_active(
            recent, times[-1], limit=params.zone_limit, combine=params.combine_obs
        )

        return OrderBlockResult(
            order_blocks=archive,
            signals=signals,
            retap_signals=retap_signals,
            rendered_order_blocks=rendered,
        )

    @staticmethod
    def _invalidate(
        obs: list[_RawOrderBlock],
        *,
        is_bullish: bool,
        use_wick: bool,
        t: int,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        times: list[int],
    ) -> list[_RawOrderBlock]:
        """활성 존만 순회해 무효화·소멸·탭을 갱신하고, **아직 소멸 안 한 존**을 반환한다.

        WAN-49: `obs`는 활성 리스트(비-swept 존만)다. 이번 봉에 소멸(`swept`)한 존은
        반환 리스트에서 빠지고(다음 봉부터 순회 제외), 아카이브에는 그대로 남는다.
        구버전은 전체 아카이브를 순회하며 `if ob.swept: continue`로 건너뛰었는데,
        소멸 존이 누적되면서 매 봉 순회 길이가 최대 존수까지 커져 O(봉수 × 존수)로
        퇴화했다. 순회 집합만 좁혔을 뿐 존별 상태 전이 로직은 완전히 동일하다.
        """
        still_active: list[_RawOrderBlock] = []
        for ob in obs:
            if not ob.breaker:
                if is_bullish:
                    cmp_value = lows[t] if use_wick else min(opens[t], closes[t])
                    if cmp_value < ob.bottom:
                        ob.breaker = True
                        ob.break_time = times[t]
                else:
                    cmp_value = highs[t] if use_wick else max(opens[t], closes[t])
                    if cmp_value > ob.top:
                        ob.breaker = True
                        ob.break_time = times[t]
            else:
                # WAN-47: 되쓸린 존을 아카이브에서 지우지 않고 소멸 시각만 기록한다.
                # 🚨 WAN-400 §1 정정: 원본은 여기서 `box.delete()`(렌더링)가 아니라
                # `bullishOrderBlocksList.remove(i)`(`.pine:267` / 약세 `:308`)로 **데이터
                # 리스트에서** 뺀다. 그 리스트가 곧 `handleOrderBlocksFinal`이 병합 입력
                # (`allOrderBlocksList`)을 만드는 원천이므로, **원본에서도 소멸 존은 겹침
                # 판정에 참여하지 않는다**. 우리 조건(`high > top` / `low < bottom`)도
                # 원본과 글자 그대로 같다. 아카이브 보존은 생존자 편향을 없애려는 우리
                # 확장이고, 병합 후보 집합은 `_generate_merged_signals`가 소멸 존을 빼서
                # 원본과 맞춘다.
                if is_bullish:
                    if highs[t] > ob.top:
                        ob.swept = True
                        ob.swept_time = times[t]
                else:
                    if lows[t] < ob.bottom:
                        ob.swept = True
                        ob.swept_time = times[t]

            # tap(재진입) 전이 기록: 확정 이후, 존 범위에 바깥→안으로 진입한 시각.
            if not ob.swept:
                inside = lows[t] <= ob.top and highs[t] >= ob.bottom
                if inside and not ob._inside and times[t] > ob.confirmed_time:
                    ob.tapped_times.append(times[t])
                ob._inside = inside
                still_active.append(ob)
        return still_active

    @staticmethod
    def _create_bullish(
        top: _SwingPoint | None,
        bullish_obs: list[_RawOrderBlock],
        active_bull: list[_RawOrderBlock],
        params: OrderBlockParams,
        t: int,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        times: list[int],
        atr: list[float | None],
    ) -> _SwingPoint | None:
        if top is None or top.crossed or closes[t] <= top.price:
            return top
        top.crossed = True

        lo, hi = 1, t - top.index - 1
        sel = min(range(t - hi, t - lo + 1), key=lambda i: lows[i]) if hi >= lo else t - 1
        box_bottom, box_top, box_loc = lows[sel], highs[sel], times[sel]

        ob_volume = volumes[t] + volumes[t - 1] + volumes[t - 2]
        ob_low_volume = volumes[t - 2]
        ob_high_volume = volumes[t] + volumes[t - 1]

        atr_t = atr[t]
        if atr_t is not None and abs(box_top - box_bottom) <= atr_t * params.max_atr_mult:
            # WAN-254(측정 전용): 확정 시점 변위 강도 = 종가가 돌파한 스윙고를 넘은 폭 ÷ ATR.
            # `closes[t] > top.price`가 이미 성립하고 봉 t가 종가로 닫힌 뒤라 룩어헤드 없음.
            displacement_atr = (closes[t] - top.price) / atr_t if atr_t > 0 else None
            new_ob = _RawOrderBlock(
                top=box_top,
                bottom=box_bottom,
                ob_volume=ob_volume,
                direction=OrderBlockDirection.BULLISH,
                start_time=box_loc,
                confirmed_time=times[t],
                ob_low_volume=ob_low_volume,
                ob_high_volume=ob_high_volume,
                displacement_atr=displacement_atr,
            )
            # WAN-47: 아카이브는 개수 캡으로 오래된 존을 버리지 않는다(전체 생애 보존).
            # 표시 개수 제한은 렌더링 뷰(`select_active`)에서만 적용한다.
            bullish_obs.insert(0, new_ob)
            # WAN-49: 새 존은 다음 봉부터 무효화 순회 대상이 된다(활성 리스트에 추가).
            active_bull.append(new_ob)
        return top

    @staticmethod
    def _create_bearish(
        bottom: _SwingPoint | None,
        bearish_obs: list[_RawOrderBlock],
        active_bear: list[_RawOrderBlock],
        params: OrderBlockParams,
        t: int,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        times: list[int],
        atr: list[float | None],
    ) -> _SwingPoint | None:
        if bottom is None or bottom.crossed or closes[t] >= bottom.price:
            return bottom
        bottom.crossed = True

        lo, hi = 1, t - bottom.index - 1
        sel = max(range(t - hi, t - lo + 1), key=lambda i: highs[i]) if hi >= lo else t - 1
        box_top, box_bottom, box_loc = highs[sel], lows[sel], times[sel]

        ob_volume = volumes[t] + volumes[t - 1] + volumes[t - 2]
        ob_low_volume = volumes[t] + volumes[t - 1]
        ob_high_volume = volumes[t - 2]

        atr_t = atr[t]
        if atr_t is not None and abs(box_top - box_bottom) <= atr_t * params.max_atr_mult:
            # WAN-254(측정 전용): 확정 시점 변위 강도 = 종가가 돌파한 스윙저를 넘은 폭 ÷ ATR.
            # `closes[t] < bottom.price`가 이미 성립하고 봉 t가 종가로 닫힌 뒤라 룩어헤드 없음.
            displacement_atr = (bottom.price - closes[t]) / atr_t if atr_t > 0 else None
            new_ob = _RawOrderBlock(
                top=box_top,
                bottom=box_bottom,
                ob_volume=ob_volume,
                direction=OrderBlockDirection.BEARISH,
                start_time=box_loc,
                confirmed_time=times[t],
                ob_low_volume=ob_low_volume,
                ob_high_volume=ob_high_volume,
                displacement_atr=displacement_atr,
            )
            bearish_obs.insert(0, new_ob)
            active_bear.append(new_ob)
        return bottom

    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"OHLCV DataFrame에 필요한 컬럼이 없습니다: {missing}")
        frame = df
        if "closed" in df.columns:
            frame = frame[frame["closed"].astype(bool)]
        return frame.sort_values("open_time").reset_index(drop=True)


def detect_order_blocks(
    df: pd.DataFrame, params: OrderBlockParams | None = None
) -> OrderBlockResult:
    """`OrderBlockDetector(params).run(df)`의 편의 함수."""
    return OrderBlockDetector(params).run(df)


def signals_for_archive(
    archive: list[OrderBlock],
    times: list[int],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    *,
    include_retaps: bool = False,
) -> list[OrderBlockSignal]:
    """원본 단위(비병합) 탭 시그널의 **공개 진입점** (WAN-405).

    `_generate_signals`를 그대로 위임한다 — 다른 탐지기(`strategy.lux_order_blocks`)가
    **같은 시그널 층**을 쓰게 하려고 열어 둔 문이다. 🚨 사본을 만들면 두 탐지기의 탭·취소
    규칙이 조용히 갈라져 손익 차이를 「탐지의 몫」이라 부를 수 없게 된다(WAN-405 §설계).
    """
    return _generate_signals(archive, times, highs, lows, closes, include_retaps=include_retaps)


def merged_signals_for_archive(
    archive: list[OrderBlock],
    df: pd.DataFrame,
    *,
    include_retaps: bool = False,
    break_time_rule: str = "distal",
    tap_state: str = "cluster",
) -> list[OrderBlockSignal]:
    """**측정용** 진입점 (WAN-400 §0): 탐지 아카이브를 직접 주고 병합 시그널만 다시 낸다.

    탐지기가 만든 아카이브를 손봐 **반사실 팔**을 세울 때 쓴다. WAN-400 §0의 A 팔이
    그 예다 — 아카이브의 `swept_time`을 전부 지우면 「소멸 존이 병합 후보에서 안 빠지는」
    엔진이 되고, 그 팔과 기준 팔의 탭 차이가 곧 A의 크기다(엔진 기본값은 안 건드린다).

    ⚠️ `df`는 탐지에 쓴 **그 프레임**이어야 한다 — 여기서도 같은 확정봉 필터·정렬을
    적용하므로, 아카이브의 시각들이 이 프레임의 시간축과 어긋나면 결과가 무의미하다.
    """
    frame = OrderBlockDetector._prepare(df)
    if frame.empty or not archive:
        return []
    return _generate_merged_signals(
        archive,
        frame["open_time"].astype("int64").tolist(),
        frame["high"].astype(float).tolist(),
        frame["low"].astype(float).tolist(),
        frame["close"].astype(float).tolist(),
        include_retaps=include_retaps,
        break_time_rule=break_time_rule,
        tap_state=tap_state,
    )
