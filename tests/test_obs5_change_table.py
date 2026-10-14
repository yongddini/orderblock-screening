"""OBS-5 변화 표가 스냅샷 갱신과 일치하는지(완료 기준 3).

표(`docs/obs-5-detector-change.md`)는 옛 스냅샷(OBS-5 직전 커밋)과 지금 스냅샷을 맞대
`tests.tools.obs5_detector_diff`가 만든다. 스냅샷을 다시 갱신하고 표를 안 고치면 여기서 걸린다.
옛 커밋이 없는 얕은 체크아웃(CI)에서는 건너뛴다.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.tools import obs5_detector_diff

TABLE = Path(__file__).resolve().parent.parent / "docs" / "obs-5-detector-change.md"


def test_change_table_matches_snapshots() -> None:
    if not obs5_detector_diff.base_available():
        pytest.skip(f"옛 스냅샷 커밋 {obs5_detector_diff.BASE}이 없다(얕은 체크아웃)")
    assert TABLE.read_text(encoding="utf-8") == obs5_detector_diff.render(), (
        "변화 표가 스냅샷과 다르다 — `uv run python -m tests.tools.obs5_detector_diff "
        "> docs/obs-5-detector-change.md`로 다시 만든다"
    )
