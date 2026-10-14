#!/bin/bash
# Obscreen DB 백업(OBS-6) — 서버에서 사용자가 실행한다(배포 전·스키마가 바뀌는 PR 전에 권장).
#
#   ./scripts/db-backup.sh            # DB_PATH(.env, 기본 orderblock_screening.db) → backups/db/
#   ./scripts/db-backup.sh --keep 30  # 최근 30개만 남긴다(기본 14)
#
# SQLite 온라인 백업 API로 복사한다 — 웹·수집이 돌고 있어도 일관된 사본이 나온다(cp 는 안 된다).
# 복사본을 `PRAGMA integrity_check` 로 검사해 'ok' 일 때만 최종 이름을 붙이고, 아니면 .FAILED 로
# 남기고 종료 코드 1. 복원 방법은 docs/ops/obs-6-deploy.md §8.

set -euo pipefail

# shellcheck source=scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

KEEP=14
while [ $# -gt 0 ]; do
    case "$1" in
    --keep) KEEP="${2:?--keep 값이 필요합니다}"; shift 2 ;;
    -h | --help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "알 수 없는 옵션: $1" ;;
    esac
done
case "$KEEP" in
'' | *[!0-9]*) die "--keep 은 숫자여야 합니다: $KEEP" ;;
esac

cd "$OBSCREEN_REPO_DIR"
DB="$(grep -E '^DB_PATH=' .env 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d "\"'" || true)"
DB="${DB_PATH:-${DB:-orderblock_screening.db}}"
[ -f "$DB" ] || die "DB 파일이 없습니다: $DB"

PYTHON=""
for py in "$OBSCREEN_REPO_DIR/.venv/bin/python" "$OBSCREEN_REPO_DIR/venv310/bin/python3" python3; do
    if command -v "$py" >/dev/null 2>&1; then
        PYTHON="$py"
        break
    fi
done
[ -n "$PYTHON" ] || die "파이썬을 찾지 못했습니다."

OUT_DIR="${OBSCREEN_BACKUP_DIR:-$OBSCREEN_REPO_DIR/backups/db}"
mkdir -p "$OUT_DIR"
BASE="$(basename "$DB" .db)-$(date '+%Y%m%d-%H%M%S').db"
TMP="$OUT_DIR/.$BASE.partial"

STATUS=0
"$PYTHON" - "$DB" "$TMP" <<'PY' || STATUS=$?
import sqlite3
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = sqlite3.connect(f"file:{src_path}?mode=ro", uri=True)
dst = sqlite3.connect(dst_path)
src.backup(dst)
src.close()
result = dst.execute("PRAGMA integrity_check").fetchone()[0]
dst.close()
print(result)
sys.exit(0 if result == "ok" else 1)
PY

if [ "$STATUS" -ne 0 ]; then
    [ -f "$TMP" ] && mv "$TMP" "$OUT_DIR/$BASE.FAILED"
    die "백업 검사 실패 — $OUT_DIR/$BASE.FAILED 를 확인하세요."
fi
mv "$TMP" "$OUT_DIR/$BASE"
log_ok "백업: $OUT_DIR/$BASE ($(du -h "$OUT_DIR/$BASE" | cut -f1))"

# 오래된 백업 정리 — 정상 백업(.db)만 센다. .FAILED 는 사람이 보고 지운다.
# shellcheck disable=SC2012 # 파일 이름은 이 스크립트가 만든 형식(공백 없음)
ls -1t "$OUT_DIR"/*.db 2>/dev/null | tail -n +"$((KEEP + 1))" | while read -r old; do
    rm -f -- "$old"
    log_info "오래된 백업 삭제: $old"
done
