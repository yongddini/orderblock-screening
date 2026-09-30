"""SQLite 저장소 스키마(OBS-4: `app_production.init_db`에서 옮김).

`investor_trading` 테이블은 여기서 만들지 않는다 — 첫 수급 수집(`obscreen.screening.core`)이
만든다. 수집 전 조회가 404를 내는 동작(OBS-9)이 그 전제 위에 있다.
"""

from __future__ import annotations

import sqlite3


def init_db(db_path: str) -> None:
    """`screening_results` 테이블과 인덱스를 만든다(이미 있으면 그대로)."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS screening_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_date VARCHAR(8) NOT NULL,
            market TEXT NOT NULL,
            code TEXT NOT NULL,
            name TEXT NOT NULL,
            current_price REAL NOT NULL,
            change_percent REAL,
            rsi REAL,
            trading_value REAL,
            zone_type TEXT NOT NULL,
            zone_position TEXT NOT NULL,
            ob_top REAL NOT NULL,
            ob_bottom REAL NOT NULL,
            distance_percent REAL,
            is_recommended INTEGER DEFAULT 0,
            timeframe TEXT DEFAULT 'daily',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(scan_date, code, timeframe)
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_scan_date ON screening_results(scan_date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_zone_type ON screening_results(zone_type)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_zone_position ON screening_results(zone_position)"
    )
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeframe ON screening_results(timeframe)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_scan_date_timeframe"
        " ON screening_results(scan_date, timeframe)"
    )

    conn.commit()
    conn.close()
    print("Database initialized")
