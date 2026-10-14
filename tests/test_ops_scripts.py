"""배포·운영 스크립트(OBS-6)가 지키는 약속 — 서버 없이 동작으로 건다.

- 모든 셸 스크립트가 `bash -n`(·있으면 shellcheck)을 통과하고 실행 비트로 커밋돼 있다.
- 비밀 값(KRX_PW·…KEY 등)은 점검·설치 출력에서 가려진다.
- `install-systemd.sh --dry-run`이 기존 유닛의 바인드를 그대로 옮기고(프록시 설정 불변),
  수집은 실행 비트에 기대지 않고 `/bin/bash`로 부르며, crontab 줄이 남아 있으면 멈춘다.
- `deploy.sh`는 `git pull → uv sync → 재시작 → /health` 순서로 돌고, 어느 단계든 실패하면
  거기서 멈춰 롤백 명령을 알린다(가짜 uv·sudo·systemctl·curl로 호출 순서를 기록한다).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
OPS_SCRIPTS = (
    "deploy.sh",
    "install-systemd.sh",
    "server-inventory.sh",
    "health-check.sh",
    "db-backup.sh",
)
ALL_SHELL = [*(SCRIPTS / name for name in OPS_SCRIPTS), SCRIPTS / "lib" / "common.sh"]


def _bash(script: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _source_lib(body: str) -> subprocess.CompletedProcess[str]:
    return _bash(f'source "{SCRIPTS / "lib" / "common.sh"}"\n{body}')


# ── 정적 점검 ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("path", ALL_SHELL, ids=lambda p: p.name)
def test_shell_syntax(path: Path) -> None:
    res = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr


def test_shellcheck_clean() -> None:
    shellcheck = shutil.which("shellcheck")
    if shellcheck is None:
        pytest.skip("shellcheck 없음(dev 의존성 shellcheck-py 가 설치한다)")
    res = subprocess.run(
        [
            shellcheck,
            "-S",
            "warning",
            "-x",
            "-P",
            "SCRIPTDIR",
            *map(str, ALL_SHELL),
            str(REPO_ROOT / "daily_screening.sh"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, res.stdout + res.stderr


@pytest.mark.parametrize("name", OPS_SCRIPTS)
def test_ops_scripts_are_committed_executable(name: str) -> None:
    """OBS-13 교훈 — 서버가 받는 것은 git 인덱스의 모드다."""
    res = subprocess.run(
        ["git", "ls-files", "-s", "--", f"scripts/{name}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0 or not res.stdout.strip():
        pytest.skip("git 인덱스에서 읽을 수 없음(커밋 전 또는 git 체크아웃 아님)")
    assert res.stdout.split()[0] == "100755", res.stdout


def test_templates_have_no_server_path() -> None:
    for tpl in (SCRIPTS / "systemd").iterdir():
        assert "/home/" not in tpl.read_text(encoding="utf-8"), tpl


# ── 공용 함수 ──────────────────────────────────────────────────────────────


def test_mask_secrets_hides_values_keeps_names() -> None:
    lines = [
        "KRX_ID=myid",
        'KRX_PW="p@ss word"',
        "DART_API_KEY='abc'",
        'Environment="DART_API_KEY=xyz" "FOO=bar"',
        "PW=1",
        "MAILTO=me@example.com",
        "EMPTY_KEY=",
        "5 20 * * * /srv/orderblock/daily_screening.sh",
    ]
    res = subprocess.run(
        ["bash", "-c", f'source "{SCRIPTS / "lib" / "common.sh"}"; mask_secrets'],
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    out = res.stdout.splitlines()
    assert out == [
        "KRX_ID=<가림>",
        "KRX_PW=<가림>",
        "DART_API_KEY=<가림>",
        'Environment="DART_API_KEY=<가림>" "FOO=bar"',
        "PW=<가림>",
        "MAILTO=me@example.com",
        "EMPTY_KEY=",  # 빈 값은 그대로 — 「설정 안 됨」이 보여야 한다
        "5 20 * * * /srv/orderblock/daily_screening.sh",
    ]
    for secret in ("myid", "p@ss", "abc", "xyz"):
        assert secret not in res.stdout


@pytest.mark.parametrize(
    ("exec_start", "bind", "workers"),
    [
        (
            "/x/venv310/bin/gunicorn -w 4 -b 127.0.0.1:5000 app_production:app",
            "127.0.0.1:5000",
            "4",
        ),
        ("gunicorn --workers 3 --bind=unix:/run/o.sock app", "unix:/run/o.sock", "3"),
        (
            "ExecStart={ path=/x/gunicorn ; "
            "argv[]=/x/gunicorn --workers 2 --bind 0.0.0.0:8000 app ; }",
            "0.0.0.0:8000",
            "2",
        ),
        ("gunicorn app_production:app", "", ""),
    ],
)
def test_extract_gunicorn_bind_and_workers(exec_start: str, bind: str, workers: str) -> None:
    res = _source_lib(
        f"extract_gunicorn_bind '{exec_start}'; echo '|'; extract_gunicorn_workers '{exec_start}'"
    )
    got_bind, got_workers = (part.strip() for part in res.stdout.split("|"))
    assert (got_bind, got_workers) == (bind, workers)


@pytest.mark.parametrize(
    ("bind", "args"),
    [
        ("0.0.0.0:5000", ["http://127.0.0.1:5000/health"]),
        ("127.0.0.1:8000", ["http://127.0.0.1:8000/health"]),
        ("[::]:80", ["http://[::1]:80/health"]),
        ("unix:/run/o.sock", ["--unix-socket", "/run/o.sock", "http://localhost/health"]),
    ],
)
def test_health_curl_args(bind: str, args: list[str]) -> None:
    res = _source_lib(f"health_curl_args '{bind}'")
    assert res.stdout.splitlines() == args


# ── 가짜 명령(PATH 앞에 둔다) ──────────────────────────────────────────────


def _stub_bin(
    tmp_path: Path, *, curl_ok: bool = True, uv_ok: bool = True, crontab: str = ""
) -> Path:
    """호출을 calls.txt 에 기록하는 가짜 uv·sudo·systemctl·curl·journalctl·crontab."""
    bin_dir = tmp_path / "stubbin"
    bin_dir.mkdir()
    calls = tmp_path / "calls.txt"
    stubs = {
        "uv": f'echo "uv $*" >> "{calls}"\nexit {0 if uv_ok else 1}',
        "sudo": f'echo "sudo $*" >> "{calls}"\nexec "$@"',
        "systemctl": (
            f'echo "systemctl $*" >> "{calls}"\n'
            'if [ "$1" = "show" ]; then\n'
            '  echo "ExecStart={ path=$OBSCREEN_FAKE_REPO/.venv/bin/gunicorn ; '
            "argv[]=$OBSCREEN_FAKE_REPO/.venv/bin/gunicorn "
            '--workers 2 --bind 127.0.0.1:5000 app ; }"\n'
            "fi\nexit 0"
        ),
        "curl": (
            f'echo "curl $*" >> "{calls}"\n'
            + ('echo \'{"status": "ok"}\'\nexit 0' if curl_ok else "exit 7")
        ),
        "journalctl": f'echo "journalctl $*" >> "{calls}"\nexit 0',
        "crontab": f"cat <<'CRON'\n{crontab}\nCRON\nexit 0",
    }
    for name, body in stubs.items():
        path = bin_dir / name
        path.write_text(f"#!/bin/bash\n{body}\n")
        path.chmod(0o755)
    return bin_dir


def _env(bin_dir: Path, repo: Path, **extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("OBSCREEN_", "GIT_"))}
    env.update(
        {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "OBSCREEN_FAKE_REPO": str(repo),
            "OBSCREEN_HEALTH_RETRIES": "2",
            "OBSCREEN_HEALTH_INTERVAL": "0",
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.com",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.com",
            "HOME": str(repo.parent),
        }
    )
    env.update(extra)
    return env


def _git(cwd: Path, *args: str, env: dict[str, str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=True
    ).stdout.strip()


def _git_init(path: Path, env: dict[str, str], *, bare: bool = False) -> None:
    """`git init -b main` 대신 — 옛 git(<2.28, 로컬 맥)은 `-b`를 모른다."""
    subprocess.run(
        ["git", "init", "-q", *(["--bare"] if bare else []), str(path)], check=True, env=env
    )
    _git(path, "symbolic-ref", "HEAD", "refs/heads/main", env=env)


def _calls(tmp_path: Path) -> list[str]:
    path = tmp_path / "calls.txt"
    return path.read_text().splitlines() if path.exists() else []


@pytest.fixture
def server_checkout(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """원격(bare) + 서버 체크아웃 + 개발자 체크아웃. 서버는 원격보다 한 커밋 뒤에 있다."""
    bin_dir = _stub_bin(tmp_path)
    origin = tmp_path / "origin.git"
    server = tmp_path / "server"
    dev = tmp_path / "dev"
    env = _env(bin_dir, server)
    _git_init(origin, env, bare=True)
    subprocess.run(["git", "clone", "-q", str(origin), str(dev)], check=True, env=env)
    _git(dev, "checkout", "-q", "-b", "main", env=env)
    shutil.copytree(SCRIPTS, dev / "scripts")
    (dev / ".gitignore").write_text("logs/\n")
    _git(dev, "add", ".", env=env)
    _git(dev, "commit", "-q", "-m", "init", env=env)
    _git(dev, "push", "-q", "origin", "main", env=env)
    subprocess.run(["git", "clone", "-q", str(origin), str(server)], check=True, env=env)
    (dev / "new.txt").write_text("new\n")
    _git(dev, "add", ".", env=env)
    _git(dev, "commit", "-q", "-m", "second", env=env)
    _git(dev, "push", "-q", "origin", "main", env=env)
    return server, dev, env


def _deploy(server: Path, env: dict[str, str], *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(server / "scripts" / "deploy.sh"), *args],
        cwd=server,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


# ── deploy.sh ─────────────────────────────────────────────────────────────


def test_deploy_pulls_syncs_restarts_and_checks_health(
    server_checkout: tuple[Path, Path, dict[str, str]], tmp_path: Path
) -> None:
    server, dev, env = server_checkout
    prev = _git(server, "rev-parse", "HEAD", env=env)
    res = _deploy(server, env)
    assert res.returncode == 0, res.stdout + res.stderr
    assert _git(server, "rev-parse", "HEAD", env=env) == _git(dev, "rev-parse", "HEAD", env=env)
    calls = _calls(tmp_path)
    order = [
        next(i for i, c in enumerate(calls) if c.startswith("uv sync --frozen --no-dev")),
        next(i for i, c in enumerate(calls) if c == "sudo systemctl restart orderblock"),
        next(i for i, c in enumerate(calls) if c.startswith("curl ")),
    ]
    assert order == sorted(order), calls
    assert any("http://127.0.0.1:5000/health" in c for c in calls)
    log = (server / "logs" / "deploy.log").read_text()
    assert f"OK {prev} -> " in log


def test_deploy_stops_on_dirty_tracked_file(
    server_checkout: tuple[Path, Path, dict[str, str]], tmp_path: Path
) -> None:
    server, _dev, env = server_checkout
    (server / "scripts" / "health-check.sh").write_text("#!/bin/bash\nexit 0\n# 손으로 고침\n")
    res = _deploy(server, env)
    assert res.returncode == 1
    assert "커밋 안 된 수정" in res.stderr
    assert "롤백(코드): ./scripts/deploy.sh --ref" in res.stderr
    assert not any(c.startswith(("uv ", "sudo ")) for c in _calls(tmp_path))


def test_deploy_stops_before_restart_when_uv_sync_fails(tmp_path: Path) -> None:
    # 별도 리포지토리를 만들되 uv 가 실패하는 가짜를 쓴다.
    sub = tmp_path / "case"
    sub.mkdir()
    bin_dir = _stub_bin(sub, uv_ok=False)
    server = sub / "server"
    env = _env(bin_dir, server)
    _git_init(server, env)
    shutil.copytree(SCRIPTS, server / "scripts")
    (server / ".gitignore").write_text("logs/\n")
    _git(server, "add", ".", env=env)
    _git(server, "commit", "-q", "-m", "init", env=env)
    origin = sub / "origin.git"
    subprocess.run(["git", "clone", "-q", "--bare", str(server), str(origin)], check=True, env=env)
    _git(server, "remote", "add", "origin", str(origin), env=env)
    res = _deploy(server, env)
    assert res.returncode == 1
    assert "uv sync 실패" in res.stderr
    assert not any("restart" in c for c in _calls(sub))


def test_deploy_reports_health_failure_with_rollback_hint(
    server_checkout: tuple[Path, Path, dict[str, str]], tmp_path: Path
) -> None:
    server, _dev, env = server_checkout
    prev = _git(server, "rev-parse", "HEAD", env=env)
    (tmp_path / "stubbin" / "curl").write_text("#!/bin/bash\nexit 7\n")
    res = _deploy(server, env)
    assert res.returncode == 1
    assert "/health 확인 실패" in res.stderr
    assert f"./scripts/deploy.sh --ref {prev}" in res.stderr
    assert any(c.startswith("journalctl -u orderblock") for c in _calls(tmp_path))
    assert "FAIL" in (server / "logs" / "deploy.log").read_text()


def test_deploy_ref_rolls_back_then_normal_deploy_returns_to_main(
    server_checkout: tuple[Path, Path, dict[str, str]],
) -> None:
    server, dev, env = server_checkout
    first = _git(server, "rev-parse", "HEAD", env=env)
    assert _deploy(server, env).returncode == 0
    res = _deploy(server, env, "--ref", first)
    assert res.returncode == 0, res.stderr
    assert _git(server, "rev-parse", "HEAD", env=env) == first
    res = _deploy(server, env)
    assert res.returncode == 0, res.stderr
    assert _git(server, "rev-parse", "--abbrev-ref", "HEAD", env=env) == "main"
    assert _git(server, "rev-parse", "HEAD", env=env) == _git(dev, "rev-parse", "HEAD", env=env)


# ── install-systemd.sh ────────────────────────────────────────────────────

OLD_UNIT = """[Unit]
Description=old
[Service]
User=svc
Group=nginx
Environment="KRX_PW=hunter2"
ExecStart=/srv/old/venv310/bin/gunicorn -w 4 -b 127.0.0.1:5000 app_production:app
"""


def _install(
    tmp_path: Path, *args: str, unit: str | None = OLD_UNIT, crontab: str = ""
) -> subprocess.CompletedProcess[str]:
    bin_dir = _stub_bin(tmp_path, crontab=crontab)
    systemd_dir = tmp_path / "systemd"
    systemd_dir.mkdir()
    if unit is not None:
        (systemd_dir / "orderblock.service").write_text(unit)
    env = _env(bin_dir, REPO_ROOT, OBSCREEN_SYSTEMD_DIR=str(systemd_dir))
    return subprocess.run(
        ["bash", str(SCRIPTS / "install-systemd.sh"), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_install_dry_run_keeps_bind_and_renders_units(tmp_path: Path) -> None:
    res = _install(tmp_path, "--dry-run")
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "@" + "REPO_DIR@" not in out
    # 바인드·워커·사용자·그룹이 옛 유닛에서 온다 — nginx 설정을 바꾸지 않아도 된다.
    assert "--workers 4 --bind 127.0.0.1:5000" in out
    assert "User=svc" in out and "Group=nginx" in out
    assert f"ExecStart={REPO_ROOT}/.venv/bin/gunicorn" in out
    # 수집은 실행 비트에 기대지 않는다(OBS-13).
    assert f"ExecStart=/bin/bash {REPO_ROOT}/daily_screening.sh" in out
    assert "OnCalendar=*-*-* 20:05:00 Asia/Seoul" in out
    assert f"EnvironmentFile=-{REPO_ROOT}/.env" in out
    # 옛 유닛의 Environment= 는 옮기지 않는다고 알리고, 값은 가린다.
    assert "Environment= 줄은 새 유닛으로 옮기지 않습니다" in res.stderr
    assert "hunter2" not in res.stdout + res.stderr
    assert not (tmp_path / "calls.txt").exists() or not any(
        c.startswith("sudo ") for c in _calls(tmp_path)
    )


def test_install_without_bind_and_without_old_unit_refuses(tmp_path: Path) -> None:
    res = _install(tmp_path, "--dry-run", unit=None)
    assert res.returncode == 1
    assert "--bind 로 지정" in res.stderr


def test_install_refuses_timer_while_cron_line_exists(tmp_path: Path) -> None:
    if not (REPO_ROOT / ".venv" / "bin" / "gunicorn").exists():
        pytest.skip(".venv 없음(uv sync 전)")
    res = _install(
        tmp_path,
        crontab=(
            "KRX_PW=hunter2\n"
            "5 20 * * * /srv/old/daily_screening.sh\n"
            "30 0 * * * /srv/alphablock/x.sh"
        ),
    )
    assert res.returncode == 1
    assert "이중으로 돕니다" in res.stderr
    assert "hunter2" not in res.stderr
    assert not any(c.startswith("sudo ") for c in _calls(tmp_path))


def test_install_dry_run_ignores_commented_cron_line(tmp_path: Path) -> None:
    res = _install(tmp_path, "--dry-run", crontab="# 5 20 * * * /srv/old/daily_screening.sh")
    assert res.returncode == 0
    assert "crontab 에 daily_screening.sh 줄이 남아" not in res.stderr
