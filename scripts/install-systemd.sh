#!/bin/bash
# Obscreen systemd 유닛 설치(OBS-6) — 서버에서 사용자가 실행한다.
#
#   웹       <서비스>.service           gunicorn app_production:app (기존 orderblock.service 를 대체)
#   매일 수집 obscreen-collect.service    daily_screening.sh (oneshot)
#            obscreen-collect.timer      매일 20:05 KST — 옛 crontab 을 대신한다
#   로그 회전 /etc/logrotate.d/obscreen   <저장소>/logs/*.log 일자별 · 30일
#
# 사용법:
#   ./scripts/install-systemd.sh --dry-run          # 무엇을 설치할지 출력만(쓰기·sudo 없음)
#   ./scripts/install-systemd.sh                    # 설치 + 웹 재시작 + /health 확인 + 타이머 켜기
#
# 옵션:
#   --bind ADDR        gunicorn 바인드(기본: 기존 유닛의 --bind/-b 값을 그대로 — 프록시 설정 불변)
#   --workers N        워커 수(기본: 기존 유닛 값, 없으면 2)
#   --user NAME        실행 사용자(기본: 기존 유닛의 User=, 없으면 현재 사용자)
#   --service NAME     웹 서비스 이름(기본: orderblock)
#   --allow-cron       crontab 에 daily_screening.sh 줄이 남아 있어도 타이머를 켠다(이중 실행 주의)
#   --no-timer         웹만 설치하고 수집 타이머는 켜지 않는다
#   --dry-run          렌더 결과만 출력
#
# 안전장치:
#   - 기존 웹 유닛은 덮어쓰기 전에 <저장소>/backups/systemd/ 로 복사한다(롤백용 — 런북 참고).
#   - .venv(uv sync)가 없으면 멈춘다 — 새 유닛은 .venv 의 gunicorn 으로 뜬다.
#   - crontab 에 daily_screening.sh 줄이 있으면 타이머를 켜지 않고 멈춘다(이중 실행 방지).
#   - nginx·도메인·TLS 설정은 읽지도 쓰지도 않는다.

set -euo pipefail

# shellcheck source=scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

SYSTEMD_DIR="${OBSCREEN_SYSTEMD_DIR:-/etc/systemd/system}"
LOGROTATE_DIR="${OBSCREEN_LOGROTATE_DIR:-/etc/logrotate.d}"
TEMPLATE_DIR="$OBSCREEN_REPO_DIR/scripts/systemd"

BIND=""
WORKERS=""
RUN_USER=""
DRY_RUN=0
ALLOW_CRON=0
WITH_TIMER=1

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
    --bind) BIND="${2:?--bind 값이 필요합니다}"; shift 2 ;;
    --workers) WORKERS="${2:?--workers 값이 필요합니다}"; shift 2 ;;
    --user) RUN_USER="${2:?--user 값이 필요합니다}"; shift 2 ;;
    --service) OBSCREEN_SERVICE="${2:?--service 값이 필요합니다}"; shift 2 ;;
    --allow-cron) ALLOW_CRON=1; shift ;;
    --no-timer) WITH_TIMER=0; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) die "알 수 없는 옵션: $1 (--help 참고)" ;;
    esac
done

WEB_UNIT="$OBSCREEN_SERVICE.service"
EXISTING_UNIT_FILE="$SYSTEMD_DIR/$WEB_UNIT"
EXISTING=""
if [ -f "$EXISTING_UNIT_FILE" ]; then
    EXISTING="$(cat "$EXISTING_UNIT_FILE")"
fi

# ── 바인드·워커·사용자: 지정 안 했으면 기존 유닛에서 그대로 가져온다 ─────────────
if [ -z "$BIND" ]; then
    BIND="$(extract_gunicorn_bind "$EXISTING")"
    [ -n "$BIND" ] || die "gunicorn 바인드를 기존 유닛($EXISTING_UNIT_FILE)에서 못 찾았습니다 — --bind 로 지정하세요(scripts/server-inventory.sh 출력 참고)."
    log_info "바인드: $BIND (기존 유닛에서 가져옴)"
fi
if [ -z "$WORKERS" ]; then
    WORKERS="$(extract_gunicorn_workers "$EXISTING")"
    WORKERS="${WORKERS:-2}"
fi
case "$WORKERS" in
'' | *[!0-9]*) die "--workers 는 숫자여야 합니다: $WORKERS" ;;
esac
if [ -z "$RUN_USER" ]; then
    RUN_USER="$(printf '%s\n' "$EXISTING" | sed -nE 's/^User=([^[:space:]]+).*/\1/p' | head -n 1)"
    RUN_USER="${RUN_USER:-$(id -un)}"
fi
GROUP="$(printf '%s\n' "$EXISTING" | sed -nE 's/^Group=([^[:space:]]+).*/\1/p' | head -n 1)"
GROUP_LINE=""
if [ -n "$GROUP" ]; then
    GROUP_LINE="Group=$GROUP"
fi
LOGROTATE_GROUP="${GROUP:-$(id -gn "$RUN_USER" 2>/dev/null || echo "$RUN_USER")}"

LOG_DIR="${OBSCREEN_LOG_DIR:-$OBSCREEN_REPO_DIR/logs}"
case "$LOG_DIR" in
/*) ;;
*) LOG_DIR="$OBSCREEN_REPO_DIR/$LOG_DIR" ;;
esac
VENV_BIN="$(venv_bin_dir)"

render() {
    sed \
        -e "s|@REPO_DIR@|$OBSCREEN_REPO_DIR|g" \
        -e "s|@USER@|$RUN_USER|g" \
        -e "s|@GROUP@|$LOGROTATE_GROUP|g" \
        -e "s|@GROUP_LINE@|$GROUP_LINE|g" \
        -e "s|@BIND@|$BIND|g" \
        -e "s|@WORKERS@|$WORKERS|g" \
        -e "s|@VENV_BIN@|$VENV_BIN|g" \
        -e "s|@LOG_DIR@|$LOG_DIR|g" \
        -e "s|@COLLECT_UNIT@|$OBSCREEN_COLLECT_UNIT|g" \
        "$1"
}

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
render "$TEMPLATE_DIR/web.service.in" >"$STAGE/$WEB_UNIT"
render "$TEMPLATE_DIR/collect.service.in" >"$STAGE/$OBSCREEN_COLLECT_UNIT.service"
render "$TEMPLATE_DIR/collect.timer.in" >"$STAGE/$OBSCREEN_COLLECT_UNIT.timer"
render "$TEMPLATE_DIR/logrotate.conf.in" >"$STAGE/obscreen.logrotate"

# 옛 유닛에만 있고 새 유닛이 옮기지 않는 것 — 조용히 사라지면 안 된다(값은 가려서 보인다).
OLD_ENV="$(printf '%s\n' "$EXISTING" | grep -E '^Environment=' || true)"
if [ -n "$OLD_ENV" ]; then
    log_warn "기존 유닛의 Environment= 줄은 새 유닛으로 옮기지 않습니다 — 필요한 값은 .env 로 옮기세요(런북 §3):"
    printf '%s\n' "$OLD_ENV" | mask_secrets | sed 's/^/    /' >&2
fi
if printf '%s\n' "$EXISTING" | grep -E '^ExecStart=' | grep -qE -- '(^|[[:space:]])(-c|--config)([[:space:]=])'; then
    log_warn "기존 유닛이 gunicorn 설정 파일(-c/--config)을 씁니다 — 새 유닛은 쓰지 않습니다. 그 파일의 설정을 확인하세요."
fi

if grep -q '@[A-Z_]*@' "$STAGE"/*; then
    die "템플릿에 채우지 못한 자리가 남았습니다: $(grep -ho '@[A-Z_]*@' "$STAGE"/* | sort -u | tr '\n' ' ')"
fi

if [ "$DRY_RUN" -eq 1 ]; then
    for f in "$STAGE"/*; do
        printf '\n===== %s =====\n' "$(basename "$f")"
        cat "$f"
    done
    if [ -n "$EXISTING" ]; then
        printf '\n===== 기존 %s 와의 차이 (- 기존 / + 새) =====\n' "$EXISTING_UNIT_FILE"
        diff -u "$EXISTING_UNIT_FILE" "$STAGE/$WEB_UNIT" | mask_secrets || true
    else
        printf '\n(기존 %s 없음)\n' "$EXISTING_UNIT_FILE"
    fi
    if crontab_has_daily_screening >"$STAGE/cron.txt"; then
        log_warn "crontab 에 daily_screening.sh 줄이 남아 있습니다 — 지우기 전에는 타이머를 켜지 않습니다:"
        cat "$STAGE/cron.txt" >&2
    fi
    log_ok "dry-run: 아무것도 쓰지 않았습니다."
    exit 0
fi

# ── 사전 점검 ─────────────────────────────────────────────────────────────
[ -x "$VENV_BIN/gunicorn" ] || die "$VENV_BIN/gunicorn 이 없습니다 — 먼저 'uv sync --frozen --no-dev' 를 돌리세요(런북 §2)."
if [ "$WITH_TIMER" -eq 1 ] && [ "$ALLOW_CRON" -eq 0 ]; then
    if crontab_has_daily_screening >"$STAGE/cron.txt"; then
        cat "$STAGE/cron.txt" >&2
        die "crontab 에 daily_screening.sh 줄이 남아 있습니다 — 타이머와 이중으로 돕니다. 런북 §4 대로 그 줄을 지운 뒤 다시 실행하세요(--allow-cron 으로 무시 가능)."
    fi
fi

# ── 설치 ──────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"
if [ -n "$EXISTING" ]; then
    BACKUP_DIR="$OBSCREEN_REPO_DIR/backups/systemd"
    mkdir -p "$BACKUP_DIR"
    BACKUP="$BACKUP_DIR/$WEB_UNIT.$(date '+%Y%m%d-%H%M%S')"
    cp "$EXISTING_UNIT_FILE" "$BACKUP"
    log_ok "기존 유닛 백업: $BACKUP"
fi

sudo install -m 0644 "$STAGE/$WEB_UNIT" "$SYSTEMD_DIR/$WEB_UNIT"
sudo install -m 0644 "$STAGE/$OBSCREEN_COLLECT_UNIT.service" "$SYSTEMD_DIR/$OBSCREEN_COLLECT_UNIT.service"
sudo install -m 0644 "$STAGE/$OBSCREEN_COLLECT_UNIT.timer" "$SYSTEMD_DIR/$OBSCREEN_COLLECT_UNIT.timer"
sudo install -m 0644 "$STAGE/obscreen.logrotate" "$LOGROTATE_DIR/obscreen"
sudo systemctl daemon-reload
sudo systemctl enable "$WEB_UNIT"
log_info "웹 재시작: $WEB_UNIT"
sudo systemctl restart "$WEB_UNIT"

if ! "$OBSCREEN_REPO_DIR/scripts/health-check.sh" "$BIND"; then
    sudo journalctl -u "$WEB_UNIT" -n 30 --no-pager || true
    die "웹이 /health 에 응답하지 않습니다. 롤백: 런북 §7(백업 유닛 복원)."
fi

if [ "$WITH_TIMER" -eq 1 ]; then
    sudo systemctl enable --now "$OBSCREEN_COLLECT_UNIT.timer"
    systemctl list-timers --no-pager "$OBSCREEN_COLLECT_UNIT.timer" || true
fi
log_ok "설치 완료 — 상태: systemctl status $WEB_UNIT · 실패: systemctl --failed · 로그: $LOG_DIR"
