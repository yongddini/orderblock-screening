"""호환 별칭 — 수집 cron 진입점. 실제 코드는 `obscreen.collect`(OBS-4).

이 이름을 import하면 **같은 모듈 객체**를 받는다(`sys.modules`를 바꿔 끼운다). 그래서
`collect_data.X`를 바꿔 끼우는 코드(테스트의 monkeypatch 등)가 실제 구현에 그대로 닿는다.
새 코드는 `obscreen.collect`을 직접 import한다.
"""

import sys

from obscreen.collect import *  # noqa: F403 — 타입 검사기가 이름을 보게 한다

if __name__ != "__main__":
    import obscreen.collect as _impl

    sys.modules[__name__] = _impl

if __name__ == "__main__":  # `python3 collect_data.py [--all|--screening|--investor] [YYYYMMDD]`
    from obscreen.collect import main

    main()
