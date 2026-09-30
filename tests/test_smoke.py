"""스모크 테스트 — CI가 실제로 앱을 띄워 보게 한다(OBS-2).

동작을 검증하는 회귀 테스트는 리모델링 2(OBS-3) 소관이다. 여기서는 앱이 import되고
`/health`가 응답하는지만 본다. 네트워크 호출은 없다.
"""

from __future__ import annotations

import os
from pathlib import Path

import app_production


def test_app_imports_and_uses_test_db() -> None:
    assert app_production.app is not None
    assert os.environ["DB_PATH"] == app_production.DB_PATH
    assert Path(app_production.DB_PATH).exists()


def test_health_endpoint() -> None:
    client = app_production.app.test_client()
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}
