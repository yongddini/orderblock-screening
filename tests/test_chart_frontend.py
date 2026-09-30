"""차트 프론트엔드 — Lightweight Charts 교체(OBS-10).

브라우저 없이 거는 것 셋:

1. **배선** — 화면이 벤더링한 라이브러리와 `ob_chart.js`를 싣고, ECharts는 어디에도 없다.
   정적 파일이 실제로 서빙되고 라이브러리 버전이 문서(`static/vendor/README.md`)와 같다.
2. **존 박스 좌표 규칙** — `ob_chart.js`의 순수 함수(`buildZoneBoxes` 등)를 node로 돌려
   동작으로 건다: 깨진 존은 회색 점선으로 그 봉에서 끝나고, 살아 있는 존은 마지막 봉까지,
   캔들 범위 앞에서 시작한 존은 첫 봉으로 당겨진다.
3. **실제 API 응답과의 정합** — 고정 입력으로 `/api/chart-data*`를 부르고 그 응답 그대로를
   `buildZoneBoxes`에 넣어, 박스의 가격대가 존의 위·아래와 같고 가로 시작이 존 시작 봉과
   같은지 본다(「존 박스가 캔들 위 올바른 가격대에 그려진다」의 기계 판정).

node가 없으면 2·3은 건너뛴다(GitHub Actions ubuntu 러너에는 있다).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

import app_production
from tests.support import SCREEN_DATE_COMPACT, FakeFdr

_ROOT = Path(__file__).resolve().parent.parent
_OB_CHART_JS = _ROOT / "static" / "js" / "ob_chart.js"
_VENDOR_JS = _ROOT / "static" / "vendor" / "lightweight-charts.standalone.production.js"
_NODE = shutil.which("node")
needs_node = pytest.mark.skipif(_NODE is None, reason="node 없음 — JS 순수 함수 검사 생략")


def _node(expr: str, payload: Any) -> Any:
    """`ob_chart.js`를 require한 뒤 `expr`(인자 `input`)의 결과를 JSON으로 받는다."""
    assert _NODE is not None
    script = (
        f"const ObChart = require({json.dumps(str(_OB_CHART_JS))});"
        "let buf='';process.stdin.on('data',d=>buf+=d);"
        "process.stdin.on('end',()=>{const input=JSON.parse(buf);"
        f"process.stdout.write(JSON.stringify(({expr})));}});"
    )
    out = subprocess.run(
        [_NODE, "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    return json.loads(out.stdout)


# --- 1. 배선 -----------------------------------------------------------------


def test_index_loads_lightweight_charts_not_echarts() -> None:
    html = app_production.app.test_client().get("/").get_data(as_text=True)
    assert "/static/vendor/lightweight-charts.standalone.production.js" in html
    assert "/static/js/ob_chart.js" in html
    assert "echarts" not in html.lower()
    assert "ObChart.render(" in html


def test_templates_have_no_echarts() -> None:
    for path in (_ROOT / "templates").glob("*.html"):
        assert "echarts" not in path.read_text(encoding="utf-8").lower(), path


def test_static_files_are_served() -> None:
    client = app_production.app.test_client()
    for url in (
        "/static/vendor/lightweight-charts.standalone.production.js",
        "/static/js/ob_chart.js",
    ):
        res = client.get(url)
        assert res.status_code == 200, url
        res.close()


def test_vendored_library_version_matches_docs() -> None:
    head = _VENDOR_JS.read_text(encoding="utf-8")[:400]
    assert "Lightweight Charts™ v5.2.0" in head
    assert "Apache License 2.0" in head
    readme = (_ROOT / "static" / "vendor" / "README.md").read_text(encoding="utf-8")
    assert "v5.2.0" in readme


# --- 2. 존 박스 좌표 규칙 -------------------------------------------------------

_CANDLES = [
    {"time": t, "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0} for t in range(100, 200, 10)
]  # 10봉: 100, 110, ..., 190


@needs_node
def test_zone_box_rules() -> None:
    orderblocks = {
        "bull": [
            # 살아 있는 존 → 마지막 봉(인덱스 9)까지, 방향색 실선
            {
                "top": 11.0,
                "bottom": 10.0,
                "start_time": 120,
                "break_time": None,
                "breaker": False,
                "combined": False,
            },
            # 깨진 존 → 무효화 봉(150 = 인덱스 5)에서 끝, 회색 점선
            {
                "top": 10.5,
                "bottom": 9.5,
                "start_time": 110,
                "break_time": 150,
                "breaker": False,
                "combined": True,
            },
            # 캔들 범위 앞에서 시작 → 첫 봉(0)으로 당긴다
            {
                "top": 9.8,
                "bottom": 9.2,
                "start_time": 50,
                "break_time": None,
                "breaker": False,
                "combined": False,
            },
            # start_time 없음 → 그리지 않는다
            {
                "top": 9.8,
                "bottom": 9.2,
                "start_time": None,
                "break_time": None,
                "breaker": False,
                "combined": False,
            },
        ],
        "bear": [
            # 봉 사이 시각(135) → 그 이전 봉(인덱스 3)
            {
                "top": 13.0,
                "bottom": 12.0,
                "start_time": 135,
                "break_time": None,
                "breaker": True,
                "combined": False,
            },
        ],
    }
    boxes = _node(
        "ObChart.buildZoneBoxes(input.obs, input.candles)",
        {"obs": orderblocks, "candles": _CANDLES},
    )
    theme = _node("ObChart.THEME", {})
    assert [(b["kind"], b["startIdx"], b["endIdx"], b["alive"], b["dashed"]) for b in boxes] == [
        ("bull", 2, 9, True, False),
        ("bull", 1, 5, False, True),
        ("bull", 0, 9, True, False),
        ("bear", 3, 9, True, True),
    ]
    assert boxes[0]["fill"] == theme["bullZoneFill"]
    assert boxes[1]["fill"] == theme["deadZoneFill"]
    assert boxes[3]["fill"] == theme["bearZoneFillFaded"]


@needs_node
def test_price_range_only_counts_visible_boxes() -> None:
    boxes = [
        {"startIdx": 0, "endIdx": 3, "top": 20.0, "bottom": 18.0},
        {"startIdx": 6, "endIdx": 9, "top": 5.0, "bottom": 4.0},
    ]
    both = _node("ObChart.boxesPriceRange(input, 0, 9)", boxes)
    right = _node("ObChart.boxesPriceRange(input, 5, 9)", boxes)
    none = _node("ObChart.boxesPriceRange(input, 20, 30)", boxes)
    assert both == {"minValue": 4.0, "maxValue": 20.0}
    assert right == {"minValue": 4.0, "maxValue": 5.0}
    assert none is None


@needs_node
def test_price_range_skips_far_zones() -> None:
    """보이는 캔들 범위(4~6, 높이 2)에서 높이만큼 넘게 떨어진 존은 자동 맞춤에서 뺀다."""
    boxes = [
        {"startIdx": 0, "endIdx": 9, "top": 7.5, "bottom": 7.0},  # 위로 1 — 넣는다
        {"startIdx": 0, "endIdx": 9, "top": 25.0, "bottom": 22.0},  # 위로 16 — 뺀다
        {"startIdx": 0, "endIdx": 9, "top": 2.5, "bottom": 1.0},  # 아래로 1.5 — 넣는다
    ]
    got = _node(
        "ObChart.boxesPriceRange(input, 0, 9, {minValue: 4, maxValue: 6})",
        boxes,
    )
    assert got == {"minValue": 1.0, "maxValue": 7.5}


# --- 3. 실제 API 응답과의 정합 ----------------------------------------------------


@needs_node
@pytest.mark.parametrize(
    ("route", "ticker"),
    [
        ("chart-data", "005930"),  # 코스피 대형
        ("chart-data", "228670"),  # 코스닥 소형
        ("chart-data", "069500"),  # ETF
        ("chart-data-weekly", "005930"),
        ("chart-data-weekly", "228670"),
        ("chart-data-weekly", "069500"),
    ],
)
def test_boxes_match_api_zones(route: str, ticker: str, fake_fdr: FakeFdr) -> None:
    res = app_production.app.test_client().get(f"/api/{route}/{ticker}?date={SCREEN_DATE_COMPACT}")
    body = json.loads(res.get_data(as_text=True))
    assert body["success"] is True
    data = body["data"]
    candles = data["candles"]
    zones = [z for kind in ("bull", "bear") for z in data["orderblocks"][kind]]
    assert zones, "고정 입력에 존이 하나도 없으면 이 검사가 아무것도 걸지 않는다"
    boxes = _node(
        "ObChart.buildZoneBoxes(input.orderblocks, input.candles)",
        {"orderblocks": data["orderblocks"], "candles": candles},
    )
    assert len(boxes) == len(zones)
    times = [c["time"] for c in candles]
    for zone, box in zip(zones, boxes, strict=True):
        assert box["top"] == max(zone["top"], zone["bottom"])
        assert box["bottom"] == min(zone["top"], zone["bottom"])
        # 존 시작 시각이 캔들 안이면 박스 왼쪽 봉이 정확히 그 봉이다.
        if zone["start_time"] >= times[0]:
            assert times[box["startIdx"]] == zone["start_time"]
        if zone["break_time"] is None:
            assert box["endIdx"] == len(candles) - 1 and box["alive"]
        else:
            assert times[box["endIdx"]] <= zone["break_time"] and not box["alive"]
        # 존 가격대가 실제 캔들 가격 범위 근처에 있다(단위·축 뒤바뀜 방지).
        lo = min(c["low"] for c in candles)
        hi = max(c["high"] for c in candles)
        assert lo * 0.5 <= box["bottom"] <= box["top"] <= hi * 1.5
