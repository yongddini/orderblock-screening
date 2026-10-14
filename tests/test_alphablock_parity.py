"""AlphaBlock 동등성 — 복사본이 원본과 같은 코드이고, 같은 입력에서 같은 존을 내는지(OBS-5).

세 겹으로 건다.

1. **복사본 = 원본 코드** — 복사본(`obscreen/detect/alphablock/`)의 원본 부분이 기록된 해시와
   같은지(항상), 그리고 AlphaBlock 저장소의 그 커밋 파일과 바이트 단위로 같은지(로컬 —
   AlphaBlock 저장소가 있을 때만).
2. **같은 입력 → 같은 존** — 고정 입력 14종목 × 일봉/주봉 × (기본값 · 병합 켬)을 **AlphaBlock
   원본 코드**로 돌려 만든 골든(`tests/fixtures/alphablock_golden/`)과, 복사본으로 돌린 존 대장
   (아카이브 전체 + 마지막 봉 렌더 뷰)이 **완전히 같은지**(항상 — CI에서도 돈다).
3. **골든 = 원본 출력** — 골든이 정말 원본 코드의 출력인지 로컬에서 다시 돌려 확인한다(AlphaBlock
   저장소가 있을 때만). `UPDATE_ALPHABLOCK_GOLDEN=1`이면 원본 출력으로 골든을 다시 쓴다.

AlphaBlock 저장소 위치는 `ALPHABLOCK_REPO`(기본 `~/AlphaBlock`). 그 저장소는 **읽기만** 한다 —
`git show <커밋>:<경로>`로 파일을 꺼내 임시 폴더에서 import하므로 작업 트리·브랜치를 건드리지
않고, 작업 트리에 무엇이 체크아웃돼 있든 기록된 커밋과 비교한다.
"""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd
import pytest

from data_provider import KoreanStockDataProvider
from obscreen.detect import alphablock
from obscreen.detect.adapter import AlphaBlockOrderBlockDetector, to_alphablock_frame
from obscreen.detect.alphablock.models import OrderBlockParams
from obscreen.detect.alphablock.order_blocks import OrderBlockDetector
from tests.support import FIXTURE_TICKERS, FIXTURES_DIR, SCREEN_DATE, FakeFdr

GOLDEN_DIR = FIXTURES_DIR / "alphablock_golden"
ALPHABLOCK_REPO = Path(os.environ.get("ALPHABLOCK_REPO", "~/AlphaBlock")).expanduser()

#: 골든을 만드는 파라미터 — 운영 기본값과 병합 켬(차트·스크리너가 설정으로 켤 수 있다).
PARAM_SETS: dict[str, dict[str, Any]] = {"default": {}, "combine": {"combine_obs": True}}
CASES = [
    (tf, ticker, name)
    for tf in ("daily", "weekly")
    for ticker in FIXTURE_TICKERS
    for name in PARAM_SETS
]


def _git_show(path: str) -> str | None:
    """AlphaBlock 저장소의 기록된 커밋에서 파일을 꺼낸다(없으면 None — CI)."""
    if not (ALPHABLOCK_REPO / ".git").exists():
        return None
    proc = subprocess.run(
        ["git", "-C", str(ALPHABLOCK_REPO), "show", f"{alphablock.UPSTREAM_COMMIT}:{path}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


@pytest.fixture(scope="module")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> Iterator[ModuleType]:
    """AlphaBlock 원본 `strategy.order_blocks`(기록된 커밋)를 임시 폴더에서 import한다."""
    sources = {
        name: _git_show(upstream_path)
        for name, (upstream_path, _) in alphablock.VENDORED_FILES.items()
    }
    if any(text is None for text in sources.values()):
        pytest.skip(f"AlphaBlock 저장소 커밋 {alphablock.UPSTREAM_COMMIT[:8]}을 못 읽는다(CI 등)")
    root = tmp_path_factory.mktemp("alphablock_upstream")
    package = root / "strategy"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for name, text in sources.items():
        assert text is not None
        (package / name).write_text(text, encoding="utf-8")
    saved = {k: v for k, v in sys.modules.items() if k == "strategy" or k.startswith("strategy.")}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("strategy.order_blocks")
        assert Path(module.__file__ or "").is_relative_to(root), "다른 `strategy`를 import했다"
        yield module
    finally:
        sys.path.remove(str(root))
        for key in [k for k in sys.modules if k == "strategy" or k.startswith("strategy.")]:
            del sys.modules[key]
        sys.modules.update(saved)


def _input(tf: str, ticker: str) -> pd.DataFrame:
    """스크리너가 쓰는 그 입력(일봉은 거래정지 제거 후, 주봉은 주봉 변환)."""
    if tf == "daily":
        df = KoreanStockDataProvider.get_price_data(ticker, 500, SCREEN_DATE)
        return df[df["Volume"] > 0].copy()
    weekly: pd.DataFrame | None = KoreanStockDataProvider.get_price_data_weekly(
        ticker, 500, SCREEN_DATE
    )
    assert weekly is not None
    return weekly


def _ledger(result: Any) -> dict[str, Any]:
    """존 대장 — 아카이브 전체(생애주기 포함)와 마지막 봉 렌더 뷰."""
    return {
        "archive": [ob.model_dump(mode="json") for ob in result.order_blocks],
        "rendered": [ob.model_dump(mode="json") for ob in result.rendered_order_blocks],
    }


def _golden_path(tf: str, ticker: str, name: str) -> Path:
    return GOLDEN_DIR / f"{tf}_{ticker}_{name}.json"


def _load_golden(tf: str, ticker: str, name: str) -> dict[str, Any]:
    path = _golden_path(tf, ticker, name)
    if not path.exists():
        raise AssertionError(
            f"골든이 없다: {path.name} — `UPDATE_ALPHABLOCK_GOLDEN=1 uv run pytest "
            "tests/test_alphablock_parity.py`로 AlphaBlock 원본에서 만든다"
        )
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


# ── 1. 복사본 = 원본 코드 ────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(alphablock.VENDORED_FILES))
def test_vendored_body_unchanged(name: str) -> None:
    """복사본 원본 부분을 손으로 고치지 않았다(기록된 해시와 같다)."""
    assert alphablock.body_sha256(name) == alphablock.BODY_SHA256[name]
    head = (Path(alphablock.__file__).parent / name).read_text(encoding="utf-8").split("\n", 8)
    assert any(alphablock.UPSTREAM_COMMIT in line for line in head), "머리 주석에 출처 커밋이 없다"


@pytest.mark.parametrize("name", sorted(alphablock.VENDORED_FILES))
def test_vendored_source_matches_upstream(name: str) -> None:
    """복사본 = AlphaBlock 그 커밋의 파일(+ 기록된 import 치환) — 바이트 단위."""
    upstream_path, _ = alphablock.VENDORED_FILES[name]
    text = _git_show(upstream_path)
    if text is None:
        pytest.skip(f"AlphaBlock 저장소 커밋 {alphablock.UPSTREAM_COMMIT[:8]}을 못 읽는다(CI 등)")
    assert alphablock.vendored_body(name) == alphablock.expected_body(name, text)


# ── 2. 같은 입력 → 같은 존 (항상) ────────────────────────────────────────────


@pytest.mark.parametrize(("tf", "ticker", "name"), CASES)
def test_vendored_detector_matches_golden(
    tf: str, ticker: str, name: str, fake_fdr: FakeFdr
) -> None:
    frame = to_alphablock_frame(_input(tf, ticker))
    result = OrderBlockDetector(OrderBlockParams(**PARAM_SETS[name])).run(frame)
    golden = _load_golden(tf, ticker, name)
    assert golden["commit"] == alphablock.UPSTREAM_COMMIT
    assert _ledger(result) == golden["ledger"]


@pytest.mark.parametrize(("tf", "ticker", "name"), CASES)
def test_adapter_returns_alphablock_rendered_zones(
    tf: str, ticker: str, name: str, fake_fdr: FakeFdr
) -> None:
    """스크리너·차트가 받는 존 = AlphaBlock 렌더 뷰(가격·시각·상태가 그대로 옮겨진다)."""
    df = _input(tf, ticker)
    detector = AlphaBlockOrderBlockDetector(**PARAM_SETS[name])
    detector.detect_order_blocks_realtime(df)
    bull, bear = detector.get_latest_orderblocks()

    ms_of = {int(pd.Timestamp(label).value // 1_000_000): label for label in df.index}
    expected = [
        (
            "Bull" if ob["direction"] == "bull" else "Bear",
            ob["top"],
            ob["bottom"],
            ms_of[ob["start_time"]],
            ob["breaker"],
            None if ob["break_time"] is None else ms_of[ob["break_time"]],
            ob["combined"],
        )
        for ob in _load_golden(tf, ticker, name)["ledger"]["rendered"]
    ]
    got = [
        (ob.ob_type, ob.top, ob.bottom, ob.start_time, ob.breaker, ob.break_time, ob.combined)
        for ob in [*bull, *bear]
    ]
    assert got == expected


# ── 3. 골든 = AlphaBlock 원본 출력 (로컬) ────────────────────────────────────


@pytest.mark.parametrize(("tf", "ticker", "name"), CASES)
def test_golden_is_upstream_output(
    tf: str, ticker: str, name: str, fake_fdr: FakeFdr, upstream: ModuleType
) -> None:
    frame = to_alphablock_frame(_input(tf, ticker))
    params = upstream.OrderBlockParams(**PARAM_SETS[name])  # 원본 모델(복사본과 다른 클래스)
    ledger = _ledger(upstream.OrderBlockDetector(params).run(frame))
    if os.environ.get("UPDATE_ALPHABLOCK_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "commit": alphablock.UPSTREAM_COMMIT,
            "input": {"timeframe": tf, "ticker": ticker, "screen_date": SCREEN_DATE},
            "params": PARAM_SETS[name],
            "ledger": ledger,
        }
        _golden_path(tf, ticker, name).write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return
    assert ledger == _load_golden(tf, ticker, name)["ledger"]


def test_golden_files_cover_all_cases() -> None:
    """골든이 조용히 빠지면 동등성 테스트가 줄어든다 — 개수를 못 박는다."""
    assert len(CASES) == 2 * 14 * 2
    assert sorted(p.name for p in GOLDEN_DIR.glob("*.json")) == sorted(
        _golden_path(*case).name for case in CASES
    )
