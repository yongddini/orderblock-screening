#!/bin/bash
# Obscreen 웹 /health 확인(OBS-6) — deploy.sh · install-systemd.sh 가 부른다. 직접 돌려도 된다.
#
#   ./scripts/health-check.sh [BIND]
#
# BIND(gunicorn 바인드, 예: 127.0.0.1:5000 · unix:/run/x.sock)를 안 주면 설치된 웹 유닛의
# ExecStart 에서 읽는다. OBSCREEN_HEALTH_URL 을 주면 그 URL 을 그대로 부른다(도메인 확인용 등).
# 재시도: OBSCREEN_HEALTH_RETRIES(기본 15) × OBSCREEN_HEALTH_INTERVAL(기본 2초).
# 종료 코드: 응답이 {"status": "ok"} 이면 0, 아니면 1.

set -euo pipefail

# shellcheck source=scripts/lib/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

RETRIES="${OBSCREEN_HEALTH_RETRIES:-15}"
INTERVAL="${OBSCREEN_HEALTH_INTERVAL:-2}"

CURL_ARGS=()
if [ -n "${OBSCREEN_HEALTH_URL:-}" ]; then
    CURL_ARGS=("$OBSCREEN_HEALTH_URL")
else
    BIND="${1:-}"
    if [ -z "$BIND" ]; then
        BIND="$(extract_gunicorn_bind "$(systemctl show -p ExecStart "$OBSCREEN_SERVICE.service" 2>/dev/null || true)")"
    fi
    [ -n "$BIND" ] || die "/health 주소를 정할 수 없습니다 — 바인드를 인자로 주거나 OBSCREEN_HEALTH_URL 을 설정하세요."
    while IFS= read -r arg; do
        CURL_ARGS+=("$arg")
    done < <(health_curl_args "$BIND")
fi

attempt=1
while [ "$attempt" -le "$RETRIES" ]; do
    body="$(curl -fsS --max-time 5 "${CURL_ARGS[@]}" 2>/dev/null || true)"
    if printf '%s' "$body" | grep -Eq '"status"[[:space:]]*:[[:space:]]*"ok"'; then
        log_ok "/health 응답 정상 (${CURL_ARGS[*]}, ${attempt}번째 시도)"
        exit 0
    fi
    sleep "$INTERVAL"
    attempt=$((attempt + 1))
done
printf '❌ /health 응답 없음 또는 비정상 (%s, %s회 시도) — 마지막 응답: %s\n' \
    "${CURL_ARGS[*]}" "$RETRIES" "${body:-<없음>}" >&2
exit 1
