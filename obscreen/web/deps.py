"""웹 계층의 런타임 의존성 — 조합 루트(`obscreen.web.app`)에서 **호출 시점에** 읽는다.

DB 경로와 탐지기 클래스는 조합 루트 모듈의 전역(`DB_PATH`·`RealtimeOrderBlockDetector`)
하나에만 산다. Blueprint가 import 시점에 값을 복사해 가면 그 전역을 바꿔 끼워도(테스트의
`monkeypatch.setattr(app_production, "DB_PATH", ...)` — `app_production`은 조합 루트의 별칭이다)
라우트가 옛 값을 계속 쓴다. 그래서 여기서 매번 읽는다.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from obscreen.detect.realtime import RealtimeOrderBlockDetector


def _root() -> ModuleType:
    return importlib.import_module("obscreen.web.app")


def db_path() -> str:
    """지금 쓸 SQLite 경로."""
    path: str = _root().DB_PATH
    return path


def detector_class() -> type[RealtimeOrderBlockDetector]:
    """차트 API가 쓸 오더블록 탐지기 클래스."""
    cls: type[RealtimeOrderBlockDetector] = _root().RealtimeOrderBlockDetector
    return cls
