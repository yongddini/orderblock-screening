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
