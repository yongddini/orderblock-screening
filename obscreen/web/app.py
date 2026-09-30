"""조합 루트 — Flask 앱 팩토리와 웹 계층의 런타임 의존성(OBS-4).

운영 진입점은 그대로 `gunicorn app_production:app`이다 — 루트의 `app_production.py`가 이
모듈의 별칭이다. 라우트는 `obscreen/web/*` Blueprint에 있고, DB 경로·탐지기 클래스는 이 모듈의
전역을 `obscreen.web.deps`가 호출 시점에 읽는다.

import 순간 `init_db()`로 `screening_results` 테이블을 만든다(옮기기 전 `app_production`과 같다).
"""

from __future__ import annotations

from pathlib import Path

from flask import Flask

from obscreen.config import get_settings
from obscreen.data import store
from obscreen.detect.realtime import RealtimeOrderBlockDetector
from obscreen.screening.core import run_and_save_investor_data, run_and_save_screening

__all__ = [
    "DB_PATH",
    "RealtimeOrderBlockDetector",
    "app",
    "create_app",
    "init_db",
    "run_and_save_investor_data",
    "run_and_save_screening",
]

#: 저장소 루트 — `templates/`·`static/`이 여기 있다(패키지로 옮기지 않았다: 서버 경로·URL 불변).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

#: 웹 계층이 쓰는 SQLite 경로. import 시점에 설정(`DB_PATH`)에서 한 번 읽는다.
DB_PATH: str = get_settings().db_path


def init_db() -> None:
    """`DB_PATH`에 스크리닝 결과 테이블을 만든다."""
    store.init_db(DB_PATH)


def create_app(*, experimental_routes: bool | None = None) -> Flask:
    """Flask 앱을 만든다.

    Args:
        experimental_routes: 실험·구버전 라우트 등록 여부. `None`이면 설정
            (`OBSCREEN_EXPERIMENTAL_ROUTES`, 기본 켬)을 따른다.
    """
    from obscreen.web import chart_api, experimental, investor_api, pages, screening_api

    flask_app = Flask(
        __name__,
        template_folder=str(REPO_ROOT / "templates"),
        static_folder=str(REPO_ROOT / "static"),
    )
    flask_app.register_blueprint(pages.bp)
    flask_app.register_blueprint(investor_api.bp)
    flask_app.register_blueprint(screening_api.bp)
    flask_app.register_blueprint(chart_api.bp)
    if experimental_routes is None:
        experimental_routes = get_settings().experimental_routes
    if experimental_routes:
        flask_app.register_blueprint(experimental.bp)
    return flask_app


app = create_app()

init_db()
