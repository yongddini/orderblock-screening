#!/bin/bash
# Obscreen 서버 현황 파악(OBS-6) — **읽기 전용**. 서버에서 사용자가 실행하고 출력을 이슈에 붙인다.
#
#   ./scripts/server-inventory.sh            > inventory.txt
#   ./scripts/server-inventory.sh --fetch    > inventory.txt   # origin 을 먼저 fetch 해 저장소와 비교
#
# 출력: 저장소 ↔ 서버 폴더 차이 · 파이썬/패키지 버전(uv.lock 과 비교) · 도는 프로세스·포트 ·
#       systemd 유닛 · nginx(도메인·TLS 설정 위치 — 읽기만) · cron · DB · 로그 · .env(이름만) · 디스크.
#
# 아무것도 바꾸지 않는다(--fetch 는 원격 추적 브랜치만 갱신). 비밀 값(KRX_PW·…KEY·…TOKEN 등)은
# 이름만 남기고 가린다. sudo 를 쓰지 않는다 — 권한 때문에 못 읽은 곳은 「읽을 수 없음」으로 적는다.

set -uo pipefail

# shellcheck source=scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

FETCH=0
case "${1:-}" in
--fetch) FETCH=1 ;;
-h | --help) sed -n '2,/^set -uo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
"") ;;
*) die "알 수 없는 옵션: $1" ;;
esac

REPO="$OBSCREEN_REPO_DIR"
section() { printf '\n========== %s ==========\n' "$*"; }
run() {
    # 명령을 보여 주고 실행한다. 실패해도 다음으로 간다(점검은 끝까지 돈다).
    printf '$ %s\n' "$*"
    "$@" 2>&1 | mask_secrets || true
}
have() { command -v "$1" >/dev/null 2>&1; }

section "기본"
printf '시각: %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')"
printf '호스트: %s · 사용자: %s · 저장소: %s\n' "$(hostname)" "$(id -un)" "$REPO"
run uname -a
[ -r /etc/os-release ] && grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release
have systemctl && run systemctl --version
printf '시간대: %s\n' "$(timedatectl show -p Timezone --value 2>/dev/null || date +%Z)"

section "저장소 ↔ 서버 폴더"
cd "$REPO" || exit 1
if [ "$FETCH" -eq 1 ]; then
    run git fetch --prune origin
fi
run git rev-parse --abbrev-ref HEAD
run git log -1 --format='HEAD     %h %ci %s' HEAD
run git log -1 --format='origin/main %h %ci %s' origin/main
run git rev-list --left-right --count HEAD...origin/main
echo "# 추적 파일 수정(서버에서 손으로 고친 것) — 비어 있어야 deploy.sh 가 돈다"
run git status --porcelain --untracked-files=no
run git diff --stat
echo "# 추적 안 되는 파일(.gitignore 밖) — 서버 전용 파일 후보 (venv310/ 등)"
run git ls-files --others --exclude-standard
run git ls-files -s daily_screening.sh

section "파이썬 · 패키지"
have python3 && run python3 --version
for py in "$REPO/venv310/bin/python3" "$REPO/.venv/bin/python"; do
    if [ -x "$py" ]; then
        run "$py" --version
        printf '$ %s -m pip freeze (주요 패키지)\n' "$py"
        {
            "$py" -m pip freeze 2>/dev/null || { UVB="$(find_uv)" && "$UVB" pip freeze --python "$py" 2>/dev/null; }
        } | grep -iE '^(pykrx|finance-datareader|financedatareader|yfinance|pandas|numpy|flask|gunicorn|pydantic-settings|pydantic|requests)==' || echo "(pip freeze 불가)"
    else
        printf '(없음: %s)\n' "$py"
    fi
done
if UV="$(find_uv)"; then run "$UV" --version; else echo "uv: 설치 안 됨 (런북 §2)"; fi
echo "# uv.lock 이 고정한 버전"
for pkg in pykrx finance-datareader yfinance pandas flask gunicorn pydantic-settings; do
    ver="$(awk -v p="$pkg" '$0=="name = \"" p "\"" {getline; print; exit}' uv.lock 2>/dev/null | sed -E 's/version = "(.*)"/\1/')"
    printf '  %-20s %s\n' "$pkg" "${ver:-?}"
done

section "프로세스 · 포트"
echo '$ ps (gunicorn·obscreen·수집·nginx)'
ps -eo pid,user,etime,args 2>/dev/null | grep -E 'gunicorn|obscreen|collect_data|daily_screening|nginx' | grep -v grep | mask_secrets || true
if have ss; then
    run ss -ltnpx
else
    echo "ss 없음"
fi

section "systemd"
if have systemctl; then
    run systemctl cat "$OBSCREEN_SERVICE.service"
    run systemctl status --no-pager "$OBSCREEN_SERVICE.service"
    run systemctl list-units --all --no-pager 'obscreen*' "$OBSCREEN_SERVICE*"
    run systemctl list-timers --all --no-pager
    run systemctl --failed --no-pager
else
    echo "systemctl 없음"
fi

section "nginx (도메인·TLS — 읽기만, 바꾸지 않는다)"
have nginx && run nginx -v
for conf in /etc/nginx/conf.d/*.conf /etc/nginx/sites-enabled/*; do
    [ -e "$conf" ] || continue
    if [ -r "$conf" ]; then
        printf -- '--- %s\n' "$conf"
        grep -nE '^\s*(server_name|listen|proxy_pass|ssl_certificate|ssl_certificate_key|root|location|include)\b' "$conf" || true
        grep -oE 'ssl_certificate\s+[^;]+' "$conf" | awk '{print $2}' | while read -r cert; do
            if [ -r "$cert" ] && have openssl; then
                printf '    인증서 %s 만료: %s\n' "$cert" "$(openssl x509 -enddate -noout -in "$cert" 2>/dev/null | cut -d= -f2)"
            else
                printf '    인증서 %s (읽을 수 없음)\n' "$cert"
            fi
        done
    else
        printf -- '--- %s (읽을 수 없음 — sudo 필요)\n' "$conf"
    fi
done

section "cron (비밀 값은 가림)"
if have crontab; then
    run crontab -l
else
    echo "crontab 없음"
fi
grep -ilE 'orderblock|obscreen|daily_screening' /etc/cron.d/* 2>/dev/null || echo "(/etc/cron.d 에 관련 항목 없음)"

section "DB"
DB_PATH_VALUE="$(grep -E '^DB_PATH=' .env 2>/dev/null | tail -n 1 | cut -d= -f2- | tr -d "\"'")"
DB_PATH_VALUE="${DB_PATH_VALUE:-orderblock_screening.db}"
printf 'DB_PATH: %s\n' "$DB_PATH_VALUE"
run ls -l "$DB_PATH_VALUE"
run ls -l ./*.db
for py in "$REPO/.venv/bin/python" "$REPO/venv310/bin/python3" python3; do
    if have "$py" || [ -x "$py" ]; then
        "$py" - "$DB_PATH_VALUE" <<'PY' 2>&1 || true
import sqlite3, sys
path = sys.argv[1]
try:
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
except sqlite3.Error as e:
    print(f"DB 열기 실패: {e}")
    raise SystemExit(0)
tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
print("테이블:", ", ".join(tables))
if "screening_results" in tables:
    print("screening_results 마지막 scan_date:", con.execute("SELECT MAX(scan_date) FROM screening_results").fetchone()[0])
PY
        break
    fi
done

section "로그"
run ls -l logs
run tail -n 15 logs/screening.log
if have journalctl; then
    run journalctl -u "$OBSCREEN_SERVICE.service" -n 15 --no-pager
fi

section ".env (이름만 — 값은 출력하지 않는다)"
if [ -f .env ]; then
    run ls -l .env
    grep -oE '^[A-Za-z_][A-Za-z0-9_]*=' .env | sed 's/=$//' | sort | sed 's/^/  /'
else
    echo ".env 없음"
fi

section "디스크"
run df -h "$REPO"
run du -sh "$REPO/logs" "$REPO/venv310" "$REPO/.venv"

printf '\n✅ 점검 끝 — 위 출력을 이슈에 붙여 주세요(비밀 값은 가려져 있습니다. 붙이기 전에 한 번 더 확인하세요).\n'
