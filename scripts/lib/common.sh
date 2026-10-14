# shellcheck shell=bash
# Obscreen 운영 스크립트 공용 함수(OBS-6). 직접 실행하지 않고 `source`로 읽는다.
#
# 서버 경로를 하드코딩하지 않는다 — 저장소 위치는 이 파일 위치에서 구한다
# (`tests/test_structure.py::test_no_hardcoded_server_path_in_code_or_scripts`).

# 저장소 루트(= scripts/ 의 부모).
OBSCREEN_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export OBSCREEN_REPO_DIR

# 기본 서비스 이름 — 서버의 기존 유닛 이름 그대로(nginx 설정이 이 서비스를 가리킨다).
OBSCREEN_SERVICE="${OBSCREEN_SERVICE:-orderblock}"
OBSCREEN_COLLECT_UNIT="${OBSCREEN_COLLECT_UNIT:-obscreen-collect}"

log_info() { printf '▶ %s\n' "$*"; }
log_ok() { printf '✅ %s\n' "$*"; }
log_warn() { printf '⚠️  %s\n' "$*" >&2; }
die() {
    printf '❌ %s\n' "$*" >&2
    exit 1
}

# 비밀 값 가리기 — 표준입력의 `NAME=value`(crontab 상단 · systemd Environment= · .env) 중
# 이름이 비밀처럼 보이는(…KEY·…PW·…PASS…·…TOKEN·…SECRET·…_ID) 값을 `<가림>`으로 바꾼다.
# 이름은 남긴다(설정 여부는 보이게). 빈 값(`NAME=`)은 그대로 둔다 — 「설정 안 됨」이 보여야 한다.
# 예) `KRX_PW=abcd` → `KRX_PW=<가림>` · `Environment="DART_API_KEY=x"` → `Environment="DART_API_KEY=<가림>"`
mask_secrets() {
    local suffix='(KEY|Key|key|PW|Pw|pw|PASS|Pass|pass|PASSWORD|Password|password|TOKEN|Token|token|SECRET|Secret|secret|_ID|_Id|_id)'
    sed -E \
        -e "s/(^|[^A-Za-z0-9_])([A-Za-z0-9_]*${suffix})=(\"[^\"]+\"|'[^']+'|[^[:space:]\"']+)/\\1\\2=<가림>/g"
}

# 유닛의 ExecStart(또는 그 문자열)에서 gunicorn `--bind`/`-b` 값을 꺼낸다. 없으면 빈 문자열
# (실패 코드를 내지 않는다 — `set -euo pipefail` 아래의 `$(...)` 대입에서 조용히 죽지 않게).
extract_gunicorn_bind() {
    local text="$1"
    printf '%s\n' "$text" | grep -oE -- '(--bind[= ]|-b )[^ ;"]+' | head -n 1 | sed -E 's/^(--bind[= ]|-b )//' || true
}

# 같은 방식으로 `--workers`/`-w` 값. 없으면 빈 문자열.
extract_gunicorn_workers() {
    local text="$1"
    printf '%s\n' "$text" | grep -oE -- '(--workers[= ]|-w )[0-9]+' | head -n 1 | sed -E 's/^(--workers[= ]|-w )//' || true
}

# gunicorn bind 값 → /health 를 부를 curl 인자(한 줄에 하나씩).
#   0.0.0.0:5000 → http://127.0.0.1:5000/health
#   [::]:5000    → http://[::1]:5000/health
#   unix:/run/x  → --unix-socket /run/x http://localhost/health
health_curl_args() {
    local bind="$1"
    case "$bind" in
    unix:*)
        printf '%s\n' "--unix-socket" "${bind#unix:}" "http://localhost/health"
        ;;
    "")
        return 1
        ;;
    *)
        local host="${bind%:*}" port="${bind##*:}"
        case "$host" in
        0.0.0.0 | "") host="127.0.0.1" ;;
        "[::]") host="[::1]" ;;
        esac
        printf '%s\n' "http://${host}:${port}/health"
        ;;
    esac
}

# `.venv`(uv) 가 만든 실행 파일 폴더.
venv_bin_dir() {
    printf '%s\n' "$OBSCREEN_REPO_DIR/.venv/bin"
}

# uv 실행 파일을 찾는다(PATH → ~/.local/bin → ~/.cargo/bin). 없으면 실패.
find_uv() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi
    local candidate
    for candidate in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

# crontab 에 daily_screening.sh 를 부르는 활성 줄이 있으면 그 줄들을 출력(가려서)하고 0.
# 없으면 1. 주석(#) 줄은 세지 않는다.
crontab_has_daily_screening() {
    local lines
    lines="$(crontab -l 2>/dev/null | grep -vE '^[[:space:]]*#' | grep -F 'daily_screening.sh' || true)"
    if [ -n "$lines" ]; then
        printf '%s\n' "$lines" | mask_secrets
        return 0
    fi
    return 1
}
