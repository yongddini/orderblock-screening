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
    "orderblock_info": "obscreen.detect.orderblock",
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


@pytest.mark.parametrize(
    "name",
    ["realtime_detector", "indicators", "obscreen.detect.realtime", "obscreen.detect.indicators"],
)
def test_old_detector_is_removed(name: str) -> None:
    """옛 자체 탐지기와 그 전용 지표 모듈은 OBS-5가 지웠다(AlphaBlock 탐지기로 교체)."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


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
    # 탐지기 값은 OBS-5가 AlphaBlock `OrderBlockParams()` 기본값으로 의도적으로 옮겼다.
    assert (s.swing_length, s.max_atr_mult, s.ob_end_method, s.combine_obs) == (
        10,
        3.5,
        "Wick",
        False,
    )
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
        if path.endswith(
            (".py", ".sh", ".toml", ".html", ".js", ".cfg", ".ini", ".service", ".in", ".timer")
        )
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


def _run_daily_script(
    tmp_path: Path, env_extra: dict[str, str], *, stub_exit: int = 0, expect_exit: int = 0
) -> str:
    """`daily_screening.sh` 사본을 가짜 파이썬으로 돌려 어떤 파이썬이 무슨 인자로 불렸나 본다."""
    import os
    import shutil

    shutil.copy(REPO_ROOT / "daily_screening.sh", tmp_path / "daily_screening.sh")
    record = tmp_path / "called.txt"
    stub = f'#!/bin/bash\necho "$0 $*" >> "{record}"\nexit {stub_exit}\n'
    venv_py = tmp_path / "venv310" / "bin" / "python3"
    venv_py.parent.mkdir(parents=True)
    venv_py.write_text(stub)
    venv_py.chmod(0o755)
    other = tmp_path / "other-python"
    other.write_text(stub)
    other.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k != "OBSCREEN_PYTHON"}
    env.update({k: v.replace("{other}", str(other)) for k, v in env_extra.items()})
    res = subprocess.run(
        ["bash", str(tmp_path / "daily_screening.sh")],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert res.returncode == expect_exit, res.stderr
    if expect_exit != 0:
        assert "수집 실패" in res.stderr
    assert (tmp_path / "logs" / "screening.log").exists()
    return record.read_text()


def test_daily_script_uses_repo_venv310_by_default(tmp_path: Path) -> None:
    """서버 cron은 gunicorn과 같은 venv310으로 돌아야 한다 — 시스템 python3(3.9)엔 pandas가 없다."""
    called = _run_daily_script(tmp_path, {})
    assert called.strip() == f"{tmp_path}/venv310/bin/python3 -m obscreen collect"


def test_daily_script_obscreen_python_overrides(tmp_path: Path) -> None:
    called = _run_daily_script(tmp_path, {"OBSCREEN_PYTHON": "{other}"})
    assert called.strip() == f"{tmp_path}/other-python -m obscreen collect"


def test_daily_script_exits_nonzero_when_collect_fails(tmp_path: Path) -> None:
    """OBS-13: 수집이 실패하면 종료 코드 1 — 옛 코드는 무조건 `exit 0`이라 실패가 안 보였다."""
    called = _run_daily_script(tmp_path, {}, stub_exit=3, expect_exit=1)
    assert called.strip() == f"{tmp_path}/venv310/bin/python3 -m obscreen collect"
    log = (tmp_path / "logs" / "screening.log").read_text()
    assert "❌ 데이터 수집 실패 (종료 코드 3)" in log
    assert "✅ 데이터 수집 완료" not in log
    # 실패해도 마무리(완료 표시)까지는 간다 — 중간에 죽은 것과 구분된다.
    assert "스크리닝 완료" in log


def test_daily_script_success_logs_completion(tmp_path: Path) -> None:
    _run_daily_script(tmp_path, {})
    log = (tmp_path / "logs" / "screening.log").read_text()
    assert "✅ 데이터 수집 완료" in log
    assert "수집 실패" not in log


def test_daily_script_is_committed_executable() -> None:
    """OBS-13: 실행 비트가 **저장소에** 있어야 한다 — crontab이 스크립트를 직접 실행한다.

    2026-09-30 서버를 git 체크아웃으로 바꾸자 644로 커밋돼 있던 이 파일이 실행 권한 없이
    내려와, cron이 2주간 `Permission denied`로 시작조차 못 했다. 작업 폴더의 권한이 아니라
    git 인덱스의 모드를 본다(서버가 받는 것은 그쪽이다).
    """
    res = subprocess.run(
        ["git", "ls-files", "-s", "--", "daily_screening.sh"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        pytest.skip("git 체크아웃이 아니다(인덱스 모드를 읽을 수 없음)")
    assert res.stdout.split()[0] == "100755", res.stdout


def test_daily_script_runs_directly_without_bash_prefix(tmp_path: Path) -> None:
    """작업 폴더 파일도 실행 가능하고 shebang으로 직접 돈다(crontab이 부르는 방식 그대로)."""
    import os

    script = REPO_ROOT / "daily_screening.sh"
    assert os.access(script, os.X_OK)
    assert script.read_text().startswith("#!/bin/bash\n")
