"""호환 별칭 — 오더블록 자료구조. 실제 코드는 `obscreen.detect.orderblock`(OBS-4).

이 이름을 import하면 **같은 모듈 객체**를 받는다(`sys.modules`를 바꿔 끼운다). 그래서
`orderblock_info.X`를 바꿔 끼우는 코드(테스트의 monkeypatch 등)가 실제 구현에 그대로 닿는다.
새 코드는 `obscreen.detect.orderblock`을 직접 import한다.
"""

import sys

from obscreen.detect.orderblock import *  # noqa: F403 — 타입 검사기가 이름을 보게 한다

if __name__ != "__main__":
    import obscreen.detect.orderblock as _impl

    sys.modules[__name__] = _impl
