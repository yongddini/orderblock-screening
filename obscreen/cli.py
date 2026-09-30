"""`obscreen` 명령 — `obscreen collect` · `obscreen serve`(OBS-4).

- `obscreen collect [--all|--screening|--investor] [YYYYMMDD]` — cron 수집(옛 `collect_data.py`와
  같은 인자·같은 동작·같은 종료 코드). `daily_screening.sh`가 이것을 부른다.
- `obscreen serve [--host H] [--port P]` — Flask 개발 서버(옛 `python app_production.py`).
  운영은 gunicorn(`app_production:app`)이 띄운다.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

_USAGE = """사용법:
    obscreen collect [--all|--screening|--investor] [YYYYMMDD]   데이터 수집(cron)
    obscreen serve [--host HOST] [--port PORT]                    개발 서버
"""


def serve(host: str | None = None, port: int | None = None) -> None:
    """개발 서버를 띄운다. `FLASK_ENV=development`면 먼저 스크리닝을 한 번 돌린다."""
    from obscreen.config import get_settings
    from obscreen.screening.core import run_and_save_screening
    from obscreen.web.app import app

    settings = get_settings()
    print("Application starting")
    if settings.flask_env == "development":
        run_and_save_screening()
    app.run(
        debug=True,
        host=host if host is not None else settings.host,
        port=port if port is not None else settings.port,
    )


def _serve_from_args(args: Sequence[str]) -> int:
    host: str | None = None
    port: int | None = None
    it = iter(args)
    for arg in it:
        if arg == "--host":
            host = next(it, None)
        elif arg == "--port":
            value = next(it, None)
            if value is None or not value.isdigit():
                print(f"❌ --port 값이 올바르지 않습니다: {value!r}", file=sys.stderr)
                return 2
            port = int(value)
        elif arg in ("-h", "--help"):
            print(_USAGE)
            return 0
        else:
            print(f"❌ 알 수 없는 옵션: {arg}", file=sys.stderr)
            return 2
    serve(host=host, port=port)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """`obscreen` 진입점. 종료 코드를 돌려준다."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(_USAGE)
        return 0
    command, rest = args[0], args[1:]
    if command == "collect":
        from obscreen import collect

        collect.main(rest)
        return 0
    if command == "serve":
        return _serve_from_args(rest)
    print(f"❌ 알 수 없는 명령: {command}", file=sys.stderr)
    print(_USAGE)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
