"""옛 탐지기 → AlphaBlock 탐지기 변화 표(OBS-5).

옛 탐지기 코드는 지웠으므로 다시 돌리지 않는다. 대신 **같은 고정 입력**에서 두 탐지기가 낸
스냅샷 — 옛 것은 OBS-5 직전 커밋(`BASE`)의 `tests/snapshots/`, 새 것은 작업 트리 — 을 맞대
표를 만든다. 그래서 「표 = 스냅샷 갱신」이 정의상 성립한다(완료 기준 3).

    uv run python -m tests.tools.obs5_detector_diff > docs/obs-5-detector-change.md

읽는 것(둘 다 운영 경로 설정 — 옛: ATR 배수 2.0 · 병합 켬 / 새: AlphaBlock 기본값):

- `detector_{daily,weekly}_<종목>.json` — 스크리너가 실제로 보는 존. 옛 일봉은
  `latest_combine_True[:3]`, 옛 주봉은 자르지 않은 `latest_combine_True`(버그 후보 6),
  새 것은 `latest_default`(AlphaBlock 렌더 뷰, 방향별 최신 3개).
- `check_proximity.json` — 종목 × 근접도별 스크리닝 판정(어느 존 근처인가).
- `screen_markets.json` · `run_and_save_screening.json` — 시장 스크리닝·DB 저장 결과.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

BASE = "dffe2f8"
"""OBS-5 브랜치가 갈라진 `main` 커밋(옛 탐지기 스냅샷)."""

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = "tests/snapshots"


#: 표 앞에 싣는 설명 — 무엇이 왜 달라지는가(코드 대조로 확인한 것만).
WHY = (
    "## 0. 무엇이 달라지나",
    "",
    "| 항목 | 옛 탐지기(`realtime_detector`, 삭제) | 새 탐지기(AlphaBlock 복사본) |",
    "| -- | -- | -- |",
    "| 스윙 판정 | 좌우 대칭 창(앞뒤 10봉) 최고·최저 | FluxCharts 스윙 상태기"
    "(`ta.highest`/`ta.lowest` 10봉) |",
    "| 존 바닥(강세) 찾는 구간 | 스윙 다음 봉 ~ **돌파 봉까지** | 스윙 다음 봉 ~ "
    "**돌파 직전 봉까지** |",
    "| ATR | 단순이동평균 10 | Wilder RMA 10 |",
    "| 존 크기 상한 | ATR × 2.0 | ATR × 3.5 |",
    "| 겹치는 존 병합 | 켬(운영 설정) | 끔(AlphaBlock 기본값) |",
    "| 큰 존이 작은 존을 품으면 | 큰 존을 지움 | 둘 다 둔다 |",
    "| 보관 개수 | 방향별 30개(넘으면 오래된 것 버림) | 전체 보관, 화면은 최근 1,750봉 안 |",
    "| 스크리너가 보는 개수 | 일봉 최신 3개 · **주봉 전부** | 일봉·주봉 모두 방향별 최신 3개 |",
    "| 무효화·소멸 규칙 | 꼬리가 바닥 아래(엄격) · 소멸 = 고가가 위 | 같다 |",
    "",
    "⚠️ 고정 입력은 14종목뿐이라 §3(운영 근접도의 스크리닝 결과)은 그대로 나왔다. 실제 시장"
    "(주식 800 · ETF 300)에서의 겹침률은 배포 뒤 서버 DB의 전·후 스크리닝 결과로 다시 잴 것.",
)


def _old(name: str) -> Any:
    text = subprocess.run(
        ["git", "show", f"{BASE}:{SNAPSHOTS}/{name}.json"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(text)


def _new(name: str) -> Any:
    return json.loads((REPO_ROOT / SNAPSHOTS / f"{name}.json").read_text(encoding="utf-8"))


def _key(ob: dict[str, Any]) -> tuple[float, float, str]:
    return (float(ob["top"]), float(ob["bottom"]), str(ob["start_time"]))


def _zones(view: dict[str, list[dict[str, Any]]], limit: int | None) -> list[dict[str, Any]]:
    bull, bear = view["bull"], view["bear"]
    if limit is not None:
        bull, bear = bull[:limit], bear[:limit]
    return [*bull, *bear]


def _fmt_counts(zones: list[dict[str, Any]]) -> str:
    bull = [z for z in zones if z["type"] == "Bull"]
    bear = [z for z in zones if z["type"] == "Bear"]
    active = sum(not z["breaker"] for z in zones)
    return f"{len(bull)} / {len(bear)} (활성 {active})"


def _price(value: Any) -> str:
    x = float(value)
    return f"{x:,.0f}" if x == int(x) else f"{x:,.2f}"


def _status(result: dict[str, Any] | None) -> str:
    if not result:
        return "—"
    return f"{result['status']} {_price(result['ob_bottom'])}~{_price(result['ob_top'])}"


def _codes(rows: list[dict[str, Any]], code: str, tf: str) -> set[tuple[str, str, str]]:
    return {
        (r[code], r.get("timeframe", tf), r.get("Status") or r.get("zone_type", "")) for r in rows
    }


def render() -> str:
    """표 전체(마크다운)."""
    names = sorted(p.stem for p in (REPO_ROOT / "tests" / "fixtures" / "ohlcv").glob("*.csv"))
    out: list[str] = []
    w = out.append
    w("# OBS-5 — 옛 탐지기 대비 변화 표")
    w("")
    w(
        "생성: `uv run python -m tests.tools.obs5_detector_diff` "
        f"(옛 = `{BASE}` 스냅샷, 새 = 이 커밋)."
    )
    w("고정 입력 14종목(`tests/fixtures/ohlcv`, 기준일 2026-09-14) · 일봉 500일 · 주봉 500주.")
    w("")
    for line in WHY:
        w(line)
    w("")
    w("## 1. 종목별 존 개수 — 스크리너가 실제로 보는 존(강세 / 약세, 그중 무효화 안 된 것)")
    w("")
    w("| 종목 | 봉 | 옛 탐지기 | 새 탐지기 | 같은 존 | 옛에만 | 새에만 |")
    w("| -- | -- | --: | --: | --: | --: | --: |")
    total_same = total_old = total_new = 0
    for tf in ("daily", "weekly"):
        for t in names:
            old_view = _old(f"detector_{tf}_{t}")["latest_combine_True"]
            new_view = _new(f"detector_{tf}_{t}")["latest_default"]
            old_z = _zones(old_view, 3 if tf == "daily" else None)
            new_z = _zones(new_view, None)
            ok, nk = {_key(z) for z in old_z}, {_key(z) for z in new_z}
            same = len(ok & nk)
            total_same += same
            total_old += len(ok)
            total_new += len(nk)
            w(
                f"| {t} | {'일봉' if tf == 'daily' else '주봉'} | {_fmt_counts(old_z)} | "
                f"{_fmt_counts(new_z)} | {same} | {len(ok - nk)} | {len(nk - ok)} |"
            )
    w("")
    w(
        f"합계: 옛 {total_old}개 · 새 {total_new}개 · 위·아래·시작 봉까지 같은 존 {total_same}개 "
        f"(옛 기준 {total_same / total_old:.0%})."
    )
    w("")
    w("## 2. 종목별 스크리닝 판정(`check_proximity`) — 근접도별")
    w("")
    w("일봉 1%·3%와 주봉 5%가 운영 근접도(ETF 일봉 1% · 주식 일봉 3% · 주봉 5%)다. 50%는")
    w("「가장 가까운 존이 무엇인가」를 보려고 넓힌 참고값이다.")
    w("")
    w("| 종목 | 근접도 | 옛 판정 | 새 판정 | 변화 |")
    w("| -- | -- | -- | -- | -- |")
    old_cp, new_cp = _old("check_proximity"), _new("check_proximity")
    tally = {"그대로": 0, "존만 바뀜": 0, "새로 걸림": 0, "빠짐": 0, "둘 다 없음": 0}
    for t in names:
        for key in old_cp[t]:
            o, n = old_cp[t][key], new_cp[t][key]
            if not o and not n:
                tally["둘 다 없음"] += 1
                continue
            if o and not n:
                change = "빠짐"
            elif n and not o:
                change = "새로 걸림"
            elif (o["status"], o["ob_top"], o["ob_bottom"]) == (
                n["status"],
                n["ob_top"],
                n["ob_bottom"],
            ):
                change = "그대로"
            else:
                change = "존만 바뀜"
            tally[change] += 1
            w(f"| {t} | {key} | {_status(o)} | {_status(n)} | {change} |")
    w("")
    w("집계: " + " · ".join(f"{k} {v}" for k, v in tally.items()))
    w("")
    w("## 3. 시장 스크리닝 · DB 저장 결과(추천 종목 겹침)")
    w("")
    w("| 결과 | 옛 | 새 | 겹침 | 새로 들어옴 | 빠짐 |")
    w("| -- | --: | --: | --: | -- | -- |")
    old_sm, new_sm = _old("screen_markets"), _new("screen_markets")
    for part in ("daily_stocks", "daily_etf", "weekly"):
        o = _codes(old_sm[part], "Code", part)
        n = _codes(new_sm[part], "Code", part)
        w(_overlap_row(f"`screen_markets.{part}`", o, n))
    o = _codes(_old("run_and_save_screening"), "code", "")
    n = _codes(_new("run_and_save_screening"), "code", "")
    w(_overlap_row("`run_and_save_screening`(DB 행)", o, n))
    return "\n".join(out) + "\n"


def base_available() -> bool:
    """옛 스냅샷 커밋을 읽을 수 있나(CI의 얕은 체크아웃에는 없다)."""
    proc = subprocess.run(
        ["git", "cat-file", "-e", f"{BASE}^{{commit}}"], cwd=REPO_ROOT, capture_output=True
    )
    return proc.returncode == 0


def main() -> None:
    print(render(), end="")


def _overlap_row(label: str, old: set[Any], new: set[Any]) -> str:
    both = old & new
    rate = f"{len(both) / len(old | new):.0%}" if old | new else "—"
    added = ", ".join(sorted(f"{c[0]}({c[2]})" for c in new - old)) or "—"
    removed = ", ".join(sorted(f"{c[0]}({c[2]})" for c in old - new)) or "—"
    return f"| {label} | {len(old)} | {len(new)} | {len(both)} ({rate}) | {added} | {removed} |"


if __name__ == "__main__":
    main()
