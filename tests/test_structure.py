"""패키지 구조 정리(OBS-4)가 지키는 약속.

- 옛 모듈 이름(`app_production` 등)은 `obscreen` 패키지 모듈의 **별칭**이다 — 같은 객체라서
  옛 이름으로 바꿔 끼운 값이 실제 구현에 닿는다(특성 테스트가 그렇게 쓴다).
- 중복이 없다: `run_and_save_screening`은 하나, 서빙 안 되던 루트 `index.html`은 없다.
- 설정 기본값이 옮기기 전 코드에 박혀 있던 숫자와 같다(`test_characterization_wiring.py`가
  넘어가는 값을 따로 건다).
- 서버 경로를 코드·스크립트에 하드코딩하지 않는다.
- 화면이 쓰는 URL은 살아 있고, 지운 실험 라우트(사용자 결정)는 404다.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

import app_production
import screening_core
from obscreen import cli, collect
from obscreen.config import Settings, get_settings
from obscreen.web.app import create_app

REPO_ROOT = Path(__file__).resolve().parent.parent

LEGACY_ALIASES = {
    "app_production": "obscreen.web.app",
    "screening_core": "obscreen.screening.core",
    "stock_screener": "obscreen.screening.screener",
    "realtime_detector": "obscreen.detect.realtime",
    "orderblock_info": "obscreen.detect.orderblock",
    "indicators": "obscreen.detect.indicators",
    "data_provider": "obscreen.data.provider",
    "collect_data": "obscreen.collect",
}

#: 화면(templates/·static/js)이 부르는 URL — 반드시 살아 있어야 한다.
SCREEN_URLS = (
    "/",
    "/investor",
    "/health",
    "/api/screening/dates",
    "/api/screening/recommended",
    "/api/screening/20260914",
    "/api/stock/005930",
    "/api/investor-trading",
    "/api/investor-dates",
)
#: OBS-4에서 지운 실험·구버전 라우트(사용자 결정).
REMOVED_URLS = (
    "/chart-test",
    "/ob-comparison",
    "/api/compare-ob-methods/005930",
    "/api/test-chart/daily/005930",
    "/api/test-chart/weekly/005930",
    "/api/chart/005930",
    "/api/chart-weekly/005930",
)


@pytest.mark.parametrize(("legacy", "target"), sorted(LEGACY_ALIASES.items()))
def test_legacy_name_is_the_same_module(legacy: str, target: str) -> None:
    assert importlib.import_module(legacy) is importlib.import_module(target)


def test_single_run_and_save_screening() -> None:
    assert app_production.run_and_save_screening is screening_core.run_and_save_screening
    assert app_production.run_and_save_screening.__module__ == "obscreen.screening.core"
    assert vars(collect)["run_and_save_screening"] is screening_core.run_and_save_screening


def test_single_index_template() -> None:
    assert not (REPO_ROOT / "index.html").exists()
    assert (REPO_ROOT / "templates" / "index.html").exists()


def test_settings_defaults_match_pre_obs4_literals(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SCREENING_TOP_N",
        "SCREENING_ETF_N",
        "FLASK_ENV",
        "OBSCREEN_PROXIMITY_STOCK_DAILY",
        "OBSCREEN_MAX_ATR_MULT",
    ):
        monkeypatch.delenv(name, raising=False)
    s = Settings(_env_file=None)
    assert (s.screening_top_n, s.screening_etf_n) == (400, 300)
    assert (s.proximity_stock_daily, s.proximity_etf_daily, s.proximity_weekly) == (3.0, 1.0, 5.0)
    assert (s.swing_length, s.max_atr_mult, s.ob_end_method, s.combine_obs) == (
        10,
        2.0,
        "Wick",
        True,
    )
    assert s.chart_max_order_blocks == 30
    assert (s.host, s.port, s.flask_env) == (
        "0.0.0.0",
        5000,
        "production",
    )


def test_settings_read_legacy_and_new_env_names(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCREENING_TOP_N", "123")
    monkeypatch.setenv("OBSCREEN_PROXIMITY_WEEKLY", "4.5")
    s = get_settings()
    assert (s.screening_top_n, s.proximity_weekly) == (123, 4.5)


def test_db_path_comes_from_settings() -> None:
    # conftest가 DB_PATH 환경변수를 박은 뒤 import했다 — 두 모듈이 같은 값을 읽었다.
    assert app_production.DB_PATH == screening_core.DB_PATH == get_settings().db_path


def test_no_hardcoded_server_path_in_code_or_scripts() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    hits = [
        path
        for path in tracked
        if path.endswith((".py", ".sh", ".toml", ".html", ".js", ".cfg", ".ini", ".service"))
        and path != "tests/test_structure.py"
        and "/home/rocky" in (REPO_ROOT / path).read_text(encoding="utf-8", errors="ignore")
    ]
    assert hits == []


@pytest.mark.parametrize("url", SCREEN_URLS)
def test_screen_urls_are_routed(url: str) -> None:
    assert create_app().url_map.bind("localhost").test(url) is True, url


@pytest.mark.parametrize("url", REMOVED_URLS)
def test_removed_experimental_routes_are_gone(url: str) -> None:
    assert app_production.app.url_map.bind("localhost").test(url) is False, url
    assert app_production.app.test_client().get(url).status_code == 404


def test_plotly_is_no_longer_a_dependency() -> None:
    assert "plotly" not in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "plotly" not in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")


def test_static_url_is_unchanged() -> None:
    with app_production.app.test_request_context():
        from flask import url_for

        assert url_for("static", filename="js/ob_chart.js") == "/static/js/ob_chart.js"


def test_cli_collect_passes_args_to_collect_main(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str] | None] = []
    monkeypatch.setattr(collect, "main", lambda argv=None: seen.append(list(argv or [])))
    assert cli.main(["collect", "--screening", "20260914"]) == 0
    assert seen == [["--screening", "20260914"]]


def test_cli_collect_runs_the_same_code_as_collect_data(monkeypatch: pytest.MonkeyPatch) -> None:
    ran: list[tuple[str, str | None]] = []
    monkeypatch.setattr(
        collect, "run_and_save_screening", lambda target_date=None: ran.append(("s", target_date))
    )
    monkeypatch.setattr(
        collect,
        "run_and_save_investor_data",
        lambda target_date=None: ran.append(("i", target_date)),
    )
    assert cli.main(["collect", "20260914"]) == 0
    assert ran == [("s", "20260914"), ("i", "20260914")]


def test_cli_rejects_unknown_command_and_bad_port(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["nope"]) == 2
    assert cli.main(["serve", "--port", "abc"]) == 2
    assert cli.main([]) == 0
    assert "obscreen collect" in capsys.readouterr().out


def test_python_m_obscreen_help_runs() -> None:
    res = subprocess.run(
        [sys.executable, "-m", "obscreen", "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert "obscreen collect" in res.stdout


def _run_daily_script(tmp_path: Path, env_extra: dict[str, str]) -> str:
    """`daily_screening.sh` 사본을 가짜 파이썬으로 돌려 어떤 파이썬이 무슨 인자로 불렸나 본다."""
    import os
    import shutil

    shutil.copy(REPO_ROOT / "daily_screening.sh", tmp_path / "daily_screening.sh")
    record = tmp_path / "called.txt"
    stub = f'#!/bin/bash\necho "$0 $*" >> "{record}"\n'
    venv_py = tmp_path / "venv310" / "bin" / "python3"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text(stub)
    venv_py.chmod(0o755)
    other = tmp_path / "other-python"
    other.write_text(stub)
    other.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "OBSCREEN_PYTHON"}
    env.update({k: v.replace("{other}", str(other)) for k, v in env_extra.items()})
    subprocess.run(["bash", str(tmp_path / "daily_screening.sh")], env=env, check=True)
    assert (tmp_path / "logs" / "screening.log").exists()
    return record.read_text()


def test_daily_script_uses_repo_venv310_by_default(tmp_path: Path) -> None:
    """서버 cron은 gunicorn과 같은 venv310으로 돌아야 한다 — 시스템 python3(3.9)엔 pandas가 없다."""
    called = _run_daily_script(tmp_path, {})
    assert called.strip() == f"{tmp_path}/venv310/bin/python3 -m obscreen collect"


def test_daily_script_obscreen_python_overrides(tmp_path: Path) -> None:
    called = _run_daily_script(tmp_path, {"OBSCREEN_PYTHON": "{other}"})
    assert called.strip() == f"{tmp_path}/other-python -m obscreen collect"
