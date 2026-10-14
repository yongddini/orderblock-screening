"""AlphaBlock 오더블록 탐지기 복사본(복사 이식 · vendoring, OBS-5).

스크리너의 오더블록 정의를 AlphaBlock 탐지기(FluxCharts 「Volumized Order Blocks」 이식)로
통일한다(사용자 결정 2026-09-30). AlphaBlock은 패키지 모양이 아니고 기본값을 자주 바꾸는 측정
저장소라, git 의존성으로 걸면 스크리닝 화면이 예고 없이 바뀐다 — 그래서 **특정 커밋의 두
파일을 그대로 복사**하고 어긋남은 테스트로 감지한다(`tests/test_alphablock_parity.py`).

- `models.py` · `order_blocks.py` — 원본 `strategy/models.py` · `strategy/order_blocks.py`.
  각 파일 머리의 주석 블록(출처·커밋) 아래는 원본과 **바이트 단위로 같고**, 다른 것은
  `order_blocks.py`의 import 경로 한 줄뿐이다(`VENDORED_FILES`).
- `models.py`에는 스크리너가 안 쓰는 매매 파라미터(`ConfluenceParams` 등)도 들어 있다.
  잘라 내면 「원본 그대로」를 기계로 확인할 수 없어 통째로 둔다 — 스크리너는 탐지 부분
  (`OrderBlockParams`·`OrderBlock`·`OrderBlockResult`)만 쓴다.
- 국장 데이터와의 접점(날짜 인덱스 ↔ `open_time` ms, `OrderBlockInfo` 변환)은 복사본이
  아니라 `obscreen.detect.adapter`에 있다.

갱신(AlphaBlock 쪽 수정을 받아들일 때)은 **의도한 변경**으로만 한다:

1. 두 파일을 새 커밋에서 다시 복사하고 머리 주석의 커밋과 `UPSTREAM_COMMIT`을 바꾼다.
2. `BODY_SHA256`을 새 값으로 바꾼다.
3. `UPDATE_ALPHABLOCK_GOLDEN=1 uv run pytest tests/test_alphablock_parity.py`로 골든을
   원본 코드에서 다시 만들고, 탐지 스냅샷(`UPDATE_SNAPSHOTS=1`)의 변화를 PR에 적는다.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

UPSTREAM_REPO = "github.com/yongddini/AlphaBlock"
UPSTREAM_COMMIT = "de825d2762c94e2b177db4608af8682f2a7fe842"
"""복사해 온 AlphaBlock 커밋(그 시점 `origin/main`)."""

#: 복사본 파일 → (원본 경로, 원본에 가한 치환 목록).
VENDORED_FILES: dict[str, tuple[str, tuple[tuple[str, str], ...]]] = {
    "models.py": ("strategy/models.py", ()),
    "order_blocks.py": (
        "strategy/order_blocks.py",
        (("from strategy.models import (", "from obscreen.detect.alphablock.models import ("),),
    ),
}

#: 머리 주석 블록의 마지막 줄 머리. 이 줄 다음부터가 원본이다.
HEADER_END = "# ── 이하 원본 그대로 "

#: 원본 부분의 sha256 — 업스트림 없이(CI) 복사본을 손으로 고친 것을 잡는다.
BODY_SHA256: dict[str, str] = {
    "models.py": "c02f88dbcbe4d66f74d837d4bc4c6a3c9b475181cb678f7b39242289714121d7",
    "order_blocks.py": "c8a62cb20fd18320c13cf721c3b3e6c4f96f759e56e76443bc66812f6a7dff40",
}

_HERE = Path(__file__).resolve().parent


def vendored_body(name: str) -> str:
    """복사본 `name`에서 머리 주석을 뺀 원본 부분."""
    text = (_HERE / name).read_text(encoding="utf-8")
    head, sep, rest = text.partition(HEADER_END)
    if not sep:
        raise ValueError(f"{name}: 머리 주석의 끝 표시({HEADER_END!r})가 없다")
    return rest.split("\n", 1)[1]


def expected_body(name: str, upstream_text: str) -> str:
    """원본 텍스트에 이 저장소가 가한 치환을 적용한 값 = 복사본 원본 부분이어야 할 값."""
    _, replacements = VENDORED_FILES[name]
    for old, new in replacements:
        if upstream_text.count(old) != 1:
            raise ValueError(f"{name}: 치환 대상이 정확히 한 번 나오지 않는다: {old!r}")
        upstream_text = upstream_text.replace(old, new)
    return upstream_text


def body_sha256(name: str) -> str:
    """복사본 원본 부분의 sha256."""
    return hashlib.sha256(vendored_body(name).encode("utf-8")).hexdigest()
