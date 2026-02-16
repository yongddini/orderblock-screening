#!/bin/bash

# 오더블록 + 외국인/기관 매일 스크리닝 스크립트
# 매일 오후 8시 30분 실행 (cron 설정)

SCRIPT_DIR="/home/rocky/orderblock"
LOG_DIR="/home/rocky/orderblock/logs"

# 로그 파일
MAIN_LOG="$LOG_DIR/screening.log"

# 로그 디렉토리 생성
mkdir -p $LOG_DIR

echo "" >> $MAIN_LOG
echo "=========================================" >> $MAIN_LOG
echo "스크리닝 시작: $(date '+%Y-%m-%d %H:%M:%S')" >> $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

cd $SCRIPT_DIR

# 통합 스크립트 실행 (날짜 파라미터 없음 = 오늘/최근 영업일)
python3 collect_data.py >> $MAIN_LOG 2>&1

if [ $? -eq 0 ]; then
    echo "✅ 데이터 수집 완료" >> $MAIN_LOG
else
    echo "❌ 데이터 수집 실패" >> $MAIN_LOG
fi

echo "" >> $MAIN_LOG
echo "=========================================" >> $MAIN_LOG
echo "스크리닝 완료: $(date '+%Y-%m-%d %H:%M:%S')" >> $MAIN_LOG
echo "=========================================" >> $MAIN_LOG

# 로그 정리 (30일 이상 된 로그 삭제)
find $LOG_DIR -name "*.log" -type f -mtime +30 -delete

exit 0