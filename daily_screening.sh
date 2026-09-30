#!/bin/bash

# 오더블록 + 외국인/기관 매일 스크리닝 스크립트
# 매일 저녁 실행 (서버 crontab — 2026-09-30 기준 매일 20:05)
#
# OBS-4: 서버 경로를 하드코딩하지 않는다 — 이 스크립트가 있는 폴더(저장소 체크아웃)에서 돈다.
#   OBSCREEN_LOG_DIR  로그 폴더 (기본: <스크립트 폴더>/logs)
#   OBSCREEN_PYTHON   파이썬 실행 파일 (기본: 이 폴더의 venv310/bin/python3가 있으면 그것,
#                     없으면 python3). 서버는 gunicorn과 같은 venv310으로 돈다 — 시스템
#                     python3(3.9)에는 pandas·pykrx가 없다(2026-09-30 서버 실측).
# 수집 본체는 `python3 -m obscreen collect`(= 옛 `python3 collect_data.py`, 인자·종료 코드 동일).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${OBSCREEN_LOG_DIR:-$SCRIPT_DIR/logs}"
if [ -n "$OBSCREEN_PYTHON" ]; then
    PYTHON="$OBSCREEN_PYTHON"
elif [ -x "$SCRIPT_DIR/venv310/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/venv310/bin/python3"
else
    PYTHON="python3"
fi

# 로그 파일
MAIN_LOG="$LOG_DIR/screening.log"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

echo "" >> "$MAIN_LOG"
echo "=========================================" >> "$MAIN_LOG"
echo "스크리닝 시작: $(date '+%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
echo "=========================================" >> "$MAIN_LOG"

cd "$SCRIPT_DIR" || exit 1

# 통합 수집 실행 (날짜 파라미터 없음 = 오늘/최근 영업일)
"$PYTHON" -m obscreen collect >> "$MAIN_LOG" 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 데이터 수집 완료" >> "$MAIN_LOG"
else
    echo "❌ 데이터 수집 실패" >> "$MAIN_LOG"
fi

echo "" >> "$MAIN_LOG"
echo "=========================================" >> "$MAIN_LOG"
echo "스크리닝 완료: $(date '+%Y-%m-%d %H:%M:%S')" >> "$MAIN_LOG"
echo "=========================================" >> "$MAIN_LOG"

# 로그 정리 (30일 이상 된 로그 삭제)
find "$LOG_DIR" -name "*.log" -type f -mtime +30 -delete

exit 0
