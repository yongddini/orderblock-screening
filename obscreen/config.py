"""설정 — 환경변수와 `.env`에서 읽는다(pydantic-settings, OBS-4).

기존 환경변수 이름(`DB_PATH`·`SCREENING_TOP_N`·`SCREENING_ETF_N`·`FLASK_ENV`)은 그대로 받는다.
새로 뺀 값은 `OBSCREEN_` 접두사를 쓴다(`HOST`·`PORT` 같은 흔한 이름이 시스템 환경변수와
부딪치지 않게). **기본값은 옮기기 전 코드에 박혀 있던 값과 같다** — 운영 결과가 바뀌면 안 된다
(`tests/test_characterization_wiring.py`가 스크리너·탐지기에 넘어가는 값을 고정한다).

`get_settings()`는 부를 때마다 새로 읽는다. 옛 코드가 `SCREENING_TOP_N` 등을 호출 시점에
`os.environ`에서 읽었으므로 그 성질을 지킨다(값이 바뀐 뒤 재시작 없이 cron이 새 값을 쓴다).
"""

from __future__ import annotations

from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: 한국시간 — 「오늘」·영업일 판단의 기준.
KST = ZoneInfo("Asia/Seoul")


def _env(*names: str) -> AliasChoices:
    return AliasChoices(*names)


class Settings(BaseSettings):
    """Obscreen 설정. 필드 설명의 괄호 안이 환경변수 이름이다."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True
    )

    # ── 저장소 · 운영 ──────────────────────────────────────────────
    db_path: str = Field("orderblock_screening.db", validation_alias=_env("DB_PATH"))
    """SQLite DB 파일 경로(`DB_PATH`)."""
    log_dir: str = Field("logs", validation_alias=_env("OBSCREEN_LOG_DIR"))
    """cron 로그 디렉터리(`OBSCREEN_LOG_DIR`). 상대 경로면 실행 폴더 기준."""
    host: str = Field("0.0.0.0", validation_alias=_env("OBSCREEN_HOST"))
    """개발 서버(`obscreen serve`) 바인드 주소(`OBSCREEN_HOST`). 운영은 gunicorn이 정한다."""
    port: int = Field(5000, validation_alias=_env("OBSCREEN_PORT"))
    """개발 서버 포트(`OBSCREEN_PORT`)."""
    flask_env: str = Field("production", validation_alias=_env("FLASK_ENV"))
    """`development`면 `obscreen serve`가 뜨기 전에 스크리닝을 한 번 돌린다(`FLASK_ENV`)."""
    experimental_routes: bool = Field(True, validation_alias=_env("OBSCREEN_EXPERIMENTAL_ROUTES"))
    """실험·구버전 라우트 등록 여부(`OBSCREEN_EXPERIMENTAL_ROUTES`) — `web/experimental.py`."""

    # ── 스크리닝 대상 ──────────────────────────────────────────────
    screening_top_n: int = Field(400, validation_alias=_env("SCREENING_TOP_N"))
    """시장별 시가총액 상위 N 종목(`SCREENING_TOP_N`)."""
    screening_etf_n: int = Field(300, validation_alias=_env("SCREENING_ETF_N"))
    """거래 상위 ETF 수(`SCREENING_ETF_N`)."""

    # ── 근접도(%) ─────────────────────────────────────────────────
    proximity_stock_daily: float = Field(
        3.0, validation_alias=_env("OBSCREEN_PROXIMITY_STOCK_DAILY")
    )
    proximity_etf_daily: float = Field(1.0, validation_alias=_env("OBSCREEN_PROXIMITY_ETF_DAILY"))
    proximity_weekly: float = Field(5.0, validation_alias=_env("OBSCREEN_PROXIMITY_WEEKLY"))

    # ── 탐지기 ────────────────────────────────────────────────────
    swing_length: int = Field(10, validation_alias=_env("OBSCREEN_SWING_LENGTH"))
    max_atr_mult: float = Field(2.0, validation_alias=_env("OBSCREEN_MAX_ATR_MULT"))
    ob_end_method: Literal["Wick", "Close"] = Field(
        "Wick", validation_alias=_env("OBSCREEN_OB_END_METHOD")
    )
    combine_obs: bool = Field(True, validation_alias=_env("OBSCREEN_COMBINE_OBS"))
    chart_max_order_blocks: int = Field(
        30, validation_alias=_env("OBSCREEN_CHART_MAX_ORDER_BLOCKS")
    )
    """차트 API가 탐지기에 넘기는 존 개수 상한."""


def get_settings() -> Settings:
    """설정을 새로 읽는다(캐시 없음 — 모듈 설명 참고)."""
    return Settings()
