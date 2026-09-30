"""테스트 공통 설정.

레거시 모듈(`app_production` 등)은 import 순간 `DB_PATH` 환경변수를 읽어 SQLite 파일을
만든다(`init_db()`). 테스트가 작업 폴더에 DB를 흘리지 않도록 import 전에 임시 경로를
환경변수로 먼저 박는다. 외부 API(pykrx·FinanceDataReader·yfinance)는 부르지 않는다.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_TMP_DIR = tempfile.mkdtemp(prefix="obscreen-test-")
os.environ["DB_PATH"] = str(Path(_TMP_DIR) / "test.db")

# ruff: noqa: E402 — 위 환경변수 설정이 레거시 모듈 import보다 먼저여야 한다.
import socket
import sqlite3
import types
from collections.abc import Iterator
from typing import Any

import pytest

from tests.support import FakeFdr, FakePykrxStock

_real_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    if self.family in (socket.AF_INET, socket.AF_INET6):
        raise RuntimeError(f"테스트에서 네트워크 호출 금지(OBS-3): {address!r}")
    return _real_connect(self, address)


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """모든 테스트에서 TCP 연결을 막는다 — 외부 API가 새면 즉시 실패한다."""
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)

    def _no_dns(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(f"테스트에서 DNS 조회 금지(OBS-3): {args!r}")

    monkeypatch.setattr(socket, "getaddrinfo", _no_dns)


@pytest.fixture
def fake_fdr(monkeypatch: pytest.MonkeyPatch) -> FakeFdr:
    """`data_provider.fdr`를 고정 CSV를 읽는 가짜로 바꾼다."""
    import data_provider

    fake = FakeFdr()
    monkeypatch.setattr(data_provider, "fdr", fake)
    return fake


@pytest.fixture
def fake_pykrx(monkeypatch: pytest.MonkeyPatch) -> FakePykrxStock:
    """`from pykrx import stock`이 가짜를 받게 한다(함수 안 지연 import까지)."""
    stock = FakePykrxStock()
    module = types.ModuleType("pykrx")
    module.stock = stock  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pykrx", module)
    return stock


@pytest.fixture
def fresh_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """테스트마다 빈 SQLite DB. 레거시 모듈은 import 시점에 `DB_PATH`를 읽어 두므로
    모듈 전역을 직접 바꾼다."""
    import app_production
    import screening_core

    db = tmp_path / "obscreen.db"
    monkeypatch.setattr(app_production, "DB_PATH", str(db))
    monkeypatch.setattr(screening_core, "DB_PATH", str(db))
    app_production.init_db()
    yield db


def query_rows(db: Path, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()
