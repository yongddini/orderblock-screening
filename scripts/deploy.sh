#!/bin/bash
# Obscreen 배포(OBS-6) — 서버에서 사용자가 실행한다.
#
#   ./scripts/deploy.sh               최신 origin/main 반영: git pull → uv sync → 웹 재시작 → /health
#   ./scripts/deploy.sh --ref <커밋>   그 커밋으로 되돌린다(롤백 — 런북 §7). 다음 일반 배포가 main 으로 돌아온다.
#
# 단계마다 실패하면 거기서 멈추고(이후 단계는 안 한다) 무엇이 실패했는지·롤백 명령을 출력한다.
# 배포 기록은 <저장소>/logs/deploy.log 에 한 줄씩 남는다(이전 커밋 → 새 커밋).
#
# 환경변수:
#   OBSCREEN_SERVICE      웹 서비스 이름(기본 orderblock)
#   OBSCREEN_HEALTH_URL   /health 를 이 URL 로 확인(기본: 유닛의 gunicorn 바인드에서 만든다)
#   OBSCREEN_DEPLOY_BRANCH  따라갈 브랜치(기본 main)

set -euo pipefail

# shellcheck source=scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

BRANCH="${OBSCREEN_DEPLOY_BRANCH:-main}"
REF=""

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
    --ref) REF="${2:?--ref 값이 필요합니다}"; shift 2 ;;
    -h | --help) usage; exit 0 ;;
    *) die "알 수 없는 옵션: $1 (--help 참고)" ;;
    esac
done

cd "$OBSCREEN_REPO_DIR"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "$OBSCREEN_REPO_DIR 은 git 체크아웃이 아닙니다."

LOG_DIR="${OBSCREEN_LOG_DIR:-$OBSCREEN_REPO_DIR/logs}"
mkdir -p "$LOG_DIR"
DEPLOY_LOG="$LOG_DIR/deploy.log"
record() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$DEPLOY_LOG"; }

PREV="$(git rev-parse HEAD)"
fail() {
    record "FAIL $PREV -> $(git rev-parse --short HEAD 2>/dev/null || echo '?') : $*"
    printf '❌ 배포 실패: %s\n' "$*" >&2
    printf '   롤백(코드): ./scripts/deploy.sh --ref %s\n' "$PREV" >&2
    exit 1
}

# ── 1. 작업 트리 확인 — 서버에서 손으로 고친 추적 파일이 있으면 멈춘다 ────────────
DIRTY="$(git status --porcelain --untracked-files=no)"
if [ -n "$DIRTY" ]; then
    printf '%s\n' "$DIRTY" >&2
    fail "서버 폴더에 커밋 안 된 수정이 있습니다(위 목록). 'git diff' 로 확인 후 되돌리거나(git checkout -- <파일>) 저장소에 반영하세요."
fi

# ── 2. 코드 받기 ───────────────────────────────────────────────────────────
log_info "git fetch origin"
git fetch --prune origin || fail "git fetch 실패"
if [ -n "$REF" ]; then
    log_info "지정 커밋으로 이동: $REF"
    git checkout --quiet --detach "$REF" || fail "git checkout $REF 실패"
else
    if [ "$(git rev-parse --abbrev-ref HEAD)" != "$BRANCH" ]; then
        git checkout --quiet "$BRANCH" || fail "git checkout $BRANCH 실패"
    fi
    git merge --ff-only --quiet "origin/$BRANCH" || fail "origin/$BRANCH 로 fast-forward 할 수 없습니다(서버에 로컬 커밋이 있음?)"
fi
NEW="$(git rev-parse HEAD)"
if [ "$NEW" = "$PREV" ]; then
    log_info "코드 변경 없음 ($(git rev-parse --short HEAD)) — 의존성 동기화·재시작은 그대로 한다"
else
    log_info "코드: $(git rev-parse --short "$PREV") → $(git rev-parse --short "$NEW")"
fi

# ── 3. 의존성 — uv.lock 그대로(.venv) ─────────────────────────────────────
UV="$(find_uv)" || fail "uv 가 없습니다 — 런북 §2(uv 설치)."
log_info "uv sync --frozen --no-dev"
"$UV" sync --frozen --no-dev || fail "uv sync 실패(재시작 전에 멈췄다 — 웹은 옛 프로세스로 계속 돈다)"

EXEC_START="$(systemctl show -p ExecStart "$OBSCREEN_SERVICE.service" 2>/dev/null || true)"
if [ -n "$EXEC_START" ] && ! printf '%s' "$EXEC_START" | grep -qF "$OBSCREEN_REPO_DIR/.venv/"; then
    log_warn "$OBSCREEN_SERVICE.service 가 아직 .venv 로 돌지 않습니다(옛 venv310 유닛) — 새 의존성은 .venv 에만 깔렸습니다."
    log_warn "전환: ./scripts/install-systemd.sh (런북 §5). 전환 전이면 venv310/bin/pip install -r requirements.txt 가 필요할 수 있습니다."
fi

# ── 4. 재시작 + /health ───────────────────────────────────────────────────
log_info "sudo systemctl restart $OBSCREEN_SERVICE"
sudo systemctl restart "$OBSCREEN_SERVICE" || fail "systemctl restart $OBSCREEN_SERVICE 실패"

if ! "$OBSCREEN_REPO_DIR/scripts/health-check.sh"; then
    sudo journalctl -u "$OBSCREEN_SERVICE" -n 30 --no-pager >&2 || true
    fail "/health 확인 실패"
fi

record "OK $PREV -> $NEW"
log_ok "배포 완료: $(git rev-parse --short "$NEW") — 도메인에서도 확인하세요. 기록: $DEPLOY_LOG"
