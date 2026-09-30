/*
 * Obscreen 종목 차트 — 캔들 + 오더블록 존 박스 + RSI 보조창 (OBS-10).
 *
 * AlphaBlock 대시보드(`~/AlphaBlock/dashboard/lightweight_chart.py`)의 렌더링 방식을
 * Flask 템플릿용 정적 JS로 옮긴 것이다. 라이브러리는 같은 TradingView Lightweight
 * Charts v5.2.0(`static/vendor/`에 벤더링)이다.
 *
 * 옮겨 온 것:
 *   - 캔들 = `chart.addSeries(CandlestickSeries, …, 0)`.
 *   - 존 박스 = **존마다 시리즈를 만들지 않는다.** 캔들 시리즈에 `ISeriesPrimitive`
 *     하나를 `attachPrimitive`로 붙이고 `draw()`에서 모든 존을 캔버스에 fillRect /
 *     strokeRect 한다(AlphaBlock에서 존마다 시리즈를 만들었더니 2,000개에서 브라우저가
 *     멈췄다 — WAN-54).
 *   - 좌측 스크롤 지연 로딩 = 이미 받은 캔들 배열에서 조각을 이어 붙인다(서버 왕복 없음).
 *   - 조작감 = 휠 줌 · 드래그 팬 · 가격축 드래그/더블클릭 · 크로스헤어 Normal ·
 *     오른쪽 여백 · 좌상단 OHLC 범례.
 *
 * 다르게 한 것(이유를 PR에 적었다):
 *   - RSI 보조창을 **남긴다**(사용자 결정 2026-09-30). v5 멀티패인 `addSeries(…, 1)`.
 *   - 존 박스 x좌표를 `timeToCoordinate`가 아니라 **캔들 인덱스 → `logicalToCoordinate`**
 *     (정수 인덱스 ± 반 봉 폭)로 구한다. 지연 로딩으로 아직 안 그린 구간에 존이 시작하면 `timeToCoordinate`가
 *     null을 내 박스가 통째로 사라지는데(AlphaBlock에도 있는 한계), 인덱스는 화면 밖이어도
 *     좌표가 나오므로 박스가 잘리지 않고 이어진다.
 *   - 가격축 자동 맞춤이 **보이는 구간의 존까지** 포함한다 — 현재가에서 먼 존도 화면에
 *     들어온다(「차트가 잘 보여야」 — 사용자 지시).
 *
 * 입력은 `/api/chart-data*` 응답의 `data` 그대로다:
 *   { candles:[{time,open,high,low,close}], rsi:[{time,value}], rsi_ema:[{time,value}],
 *     orderblocks:{ bull:[{top,bottom,start_time,break_time,breaker,combined}], bear:[…] } }
 * time은 초 단위 epoch(날짜의 UTC 자정)이다.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  } else {
    root.ObChart = api;
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  // AlphaBlock 다크 테마(트레이딩뷰 기본 다크)에 준한다. 캔들 색은 기존 스크리너 차트와
  // 같게 둔다(상승 청록 / 하락 빨강).
  const THEME = {
    background: "#131722",
    textColor: "#d1d4dc",
    gridColor: "rgba(70, 74, 86, 0.4)",
    legendBg: "rgba(30, 34, 45, 0.85)",
    upColor: "#26a69a",
    downColor: "#ef5350",
    bullZoneFill: "rgba(38, 166, 154, 0.22)",
    bullZoneFillFaded: "rgba(38, 166, 154, 0.12)",
    bullZoneLine: "rgba(38, 166, 154, 0.9)",
    bearZoneFill: "rgba(239, 83, 80, 0.22)",
    bearZoneFillFaded: "rgba(239, 83, 80, 0.12)",
    bearZoneLine: "rgba(239, 83, 80, 0.9)",
    deadZoneFill: "rgba(120, 123, 134, 0.14)",
    deadZoneLine: "rgba(120, 123, 134, 0.75)",
    rsiColor: "#7E57C2",
    rsiEmaColor: "#FFEB3B",
    rsiBandColor: "rgba(120, 123, 134, 0.8)",
  };

  /** 정렬된 times에서 `times[i] <= t`인 가장 큰 i. t가 첫 값보다 앞이면 -1. */
  function indexAtOrBefore(times, t) {
    let lo = 0;
    let hi = times.length - 1;
    let ans = -1;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (times[mid] <= t) {
        ans = mid;
        lo = mid + 1;
      } else {
        hi = mid - 1;
      }
    }
    return ans;
  }

  /**
   * 존 → 그릴 박스(캔들 **전체 배열 기준 인덱스**로 가로 범위를 잡는다).
   *
   * 색이 곧 상태다(AlphaBlock WAN-245와 같은 규칙):
   *   - `break_time`이 있는 존(깨진 존)은 **회색 점선**이고 그 봉에서 끝난다.
   *   - 살아 있는 존만 방향색(강세 청록 / 약세 빨강)으로 **마지막 봉까지** 늘인다.
   *   - `breaker`(깨진 뒤 반대로 쓰이는 존)는 옅은 채움 + 점선.
   * 박스는 봉의 **왼쪽 변(start−0.5)에서 오른쪽 변(end+0.5)** 까지 채운다.
   */
  function buildZoneBoxes(orderblocks, candles, theme) {
    const th = theme || THEME;
    const times = candles.map(function (c) { return c.time; });
    const lastIdx = times.length - 1;
    const boxes = [];
    if (lastIdx < 0 || !orderblocks) return boxes;
    ["bull", "bear"].forEach(function (kind) {
      (orderblocks[kind] || []).forEach(function (ob) {
        if (ob.start_time === null || ob.start_time === undefined) return;
        if (!(ob.top > ob.bottom || ob.top < ob.bottom)) return; // 높이 0·NaN은 못 그린다
        let startIdx = indexAtOrBefore(times, ob.start_time);
        if (startIdx < 0) startIdx = 0;
        const dead = ob.break_time !== null && ob.break_time !== undefined;
        let endIdx = dead ? indexAtOrBefore(times, ob.break_time) : lastIdx;
        if (endIdx < startIdx) endIdx = startIdx;
        const isBull = kind === "bull";
        let fill;
        let line;
        if (dead) {
          fill = th.deadZoneFill;
          line = th.deadZoneLine;
        } else if (isBull) {
          fill = ob.breaker ? th.bullZoneFillFaded : th.bullZoneFill;
          line = th.bullZoneLine;
        } else {
          fill = ob.breaker ? th.bearZoneFillFaded : th.bearZoneFill;
          line = th.bearZoneLine;
        }
        boxes.push({
          kind: kind,
          alive: !dead,
          startIdx: startIdx,
          endIdx: endIdx,
          top: Math.max(ob.top, ob.bottom),
          bottom: Math.min(ob.top, ob.bottom),
          fill: fill,
          line: line,
          dashed: dead || Boolean(ob.breaker),
        });
      });
    });
    return boxes;
  }

  /**
   * 가로 범위가 [fromIdx, toIdx](전체 인덱스)와 겹치는 박스들의 가격 범위. 없으면 null.
   *
   * `near`({minValue,maxValue} = 보이는 캔들 범위)를 주면 **그 범위에서 캔들 범위 높이만큼
   * 안쪽에 있는 존만** 넣는다 — 현재가에서 몇 배 떨어진 존 하나 때문에 캔들이 납작해지지
   * 않게(실측: 228670 주봉에서 22,000원대 약세 존이 4,000원대 캔들을 화면 1/3로 눌렀다).
   * 그보다 먼 존은 가격축 드래그로 본다.
   */
  function boxesPriceRange(boxes, fromIdx, toIdx, near) {
    let min = Infinity;
    let max = -Infinity;
    const reach = near ? Math.max(0, near.maxValue - near.minValue) : Infinity;
    boxes.forEach(function (b) {
      if (b.endIdx + 0.5 < fromIdx || b.startIdx - 0.5 > toIdx) return;
      if (near && (b.bottom > near.maxValue + reach || b.top < near.minValue - reach)) return;
      if (b.bottom < min) min = b.bottom;
      if (b.top > max) max = b.top;
    });
    return min <= max ? { minValue: min, maxValue: max } : null;
  }

  function fmtPrice(value) {
    if (value === null || value === undefined || !isFinite(value)) return "-";
    return Number(value).toLocaleString("ko-KR", { maximumFractionDigits: 2 });
  }

  function fmtDate(time) {
    // time = 날짜의 UTC 자정(초). UTC 파트로 읽어야 브라우저 시간대와 무관하게 그 날짜다.
    const d = new Date(time * 1000);
    const p2 = function (n) { return n < 10 ? "0" + n : "" + n; };
    return d.getUTCFullYear() + "-" + p2(d.getUTCMonth() + 1) + "-" + p2(d.getUTCDate());
  }

  /** 초기 화면에 보일 봉 수 — 일봉 약 6개월 · 주봉 약 2년. */
  function initialVisibleBars(timeframe, total) {
    const n = timeframe === "weekly" ? 104 : 130;
    return Math.max(1, Math.min(n, total));
  }

  /**
   * 차트를 그린다. 반환값 `{ destroy() }` — 모달을 닫거나 일봉/주봉을 바꿀 때 부른다.
   * opts: { title, timeframe: "daily"|"weekly", chunkBars, rightPadRatio }
   */
  function render(container, data, opts) {
    const LWC = (typeof window !== "undefined" && window.LightweightCharts) || null;
    if (!LWC) throw new Error("LightweightCharts 라이브러리가 로드되지 않았습니다");
    const options = opts || {};
    const th = THEME;
    const candles = (data && data.candles) || [];
    container.innerHTML = "";
    if (!candles.length) {
      container.innerHTML =
        '<div style="padding:2rem;color:#888;">표시할 데이터가 없습니다.</div>';
      return { destroy: function () {} };
    }
    container.style.position = "relative";

    const chart = LWC.createChart(container, {
      autoSize: true,
      layout: {
        background: { type: "solid", color: th.background },
        textColor: th.textColor,
        panes: { separatorColor: "rgba(70, 74, 86, 0.8)", enableResize: true },
      },
      grid: { vertLines: { color: th.gridColor }, horzLines: { color: th.gridColor } },
      localization: {
        locale: "ko-KR",
        timeFormatter: fmtDate,
        priceFormatter: fmtPrice,
      },
      // 크로스헤어 Normal = 커서 자리의 실제 가격(AlphaBlock WAN-245와 같다).
      crosshair: { mode: LWC.CrosshairMode.Normal },
      rightPriceScale: { borderVisible: false },
      handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true,
                      vertTouchDrag: false },
      handleScale: {
        mouseWheel: true,
        pinch: true,
        axisPressedMouseMove: { time: true, price: true },
        axisDoubleClickReset: { time: true, price: true },
      },
      timeScale: { borderVisible: false, timeVisible: false, secondsVisible: false },
    });

    const boxes = buildZoneBoxes(data.orderblocks, candles, th);
    const chunk = Math.max(50, options.chunkBars || 200);
    let loadedFrom = Math.max(0, candles.length - Math.max(chunk, 260));

    const candleSeries = chart.addSeries(LWC.CandlestickSeries, {
      upColor: th.upColor,
      downColor: th.downColor,
      borderVisible: false,
      wickUpColor: th.upColor,
      wickDownColor: th.downColor,
      priceLineColor: "#ff9800",
      priceLineStyle: LWC.LineStyle.Dashed,
      priceFormat: { type: "custom", formatter: fmtPrice, minMove: 0.01 },
      // 자동 맞춤이 보이는 구간의 존까지 품는다 — 현재가에서 먼 존도 화면에 들어온다.
      autoscaleInfoProvider: function (original) {
        const res = original();
        const range = chart.timeScale().getVisibleLogicalRange();
        if (!range) return res;
        const zr = boxesPriceRange(boxes, range.from + loadedFrom, range.to + loadedFrom,
                                   res && res.priceRange ? res.priceRange : null);
        if (!zr) return res;
        if (!res || !res.priceRange) return { priceRange: zr };
        return {
          priceRange: {
            minValue: Math.min(res.priceRange.minValue, zr.minValue),
            maxValue: Math.max(res.priceRange.maxValue, zr.maxValue),
          },
          margins: res.margins,
        };
      },
    }, 0);

    class OrderBlockBoxesPrimitive {
      constructor(items) {
        this._boxes = items;
        this._chart = null;
        this._series = null;
        const self = this;
        this._paneViews = [{
          renderer: function () {
            return {
              draw: function (target) {
                if (!self._chart || !self._series) return;
                const ts = self._chart.timeScale();
                const series = self._series;
                // ⚠️ `logicalToCoordinate`는 **정수 인덱스만** 제대로 받는다(v5.2.0 실측:
                // 227.5를 주면 0이 돌아온다). 봉 가장자리는 정수 좌표 ± 반 봉 폭으로 만든다.
                const base = ts.logicalToCoordinate(0);
                const next = ts.logicalToCoordinate(1);
                if (base === null || next === null) return;
                const halfBar = (next - base) / 2;
                target.useBitmapCoordinateSpace(function (scope) {
                  const ctx = scope.context;
                  for (const box of self._boxes) {
                    const c1 = ts.logicalToCoordinate(box.startIdx - loadedFrom);
                    const c2 = ts.logicalToCoordinate(box.endIdx - loadedFrom);
                    const y1 = series.priceToCoordinate(box.top);
                    const y2 = series.priceToCoordinate(box.bottom);
                    if (c1 === null || c2 === null || y1 === null || y2 === null) continue;
                    const x1 = c1 - halfBar;
                    const x2 = c2 + halfBar;
                    const left = Math.round(Math.min(x1, x2) * scope.horizontalPixelRatio);
                    const right = Math.round(Math.max(x1, x2) * scope.horizontalPixelRatio);
                    const top = Math.round(Math.min(y1, y2) * scope.verticalPixelRatio);
                    const bottom = Math.round(Math.max(y1, y2) * scope.verticalPixelRatio);
                    ctx.fillStyle = box.fill;
                    ctx.fillRect(left, top, right - left, Math.max(1, bottom - top));
                    ctx.strokeStyle = box.line;
                    ctx.lineWidth = Math.max(1, Math.round(scope.horizontalPixelRatio));
                    ctx.setLineDash(box.dashed ? [4 * scope.horizontalPixelRatio,
                                                  3 * scope.horizontalPixelRatio] : []);
                    ctx.strokeRect(left, top, right - left, Math.max(1, bottom - top));
                  }
                  ctx.setLineDash([]);
                });
              },
            };
          },
        }];
      }
      attached(param) { this._chart = param.chart; this._series = param.series; }
      detached() { this._chart = null; this._series = null; }
      updateAllViews() {}
      paneViews() { return this._paneViews; }
    }
    if (boxes.length) candleSeries.attachPrimitive(new OrderBlockBoxesPrimitive(boxes));

    // RSI 보조창(pane 1) — 사용자 결정(2026-09-30)으로 남긴다. 0~100 고정 축.
    const rsiScale = function () { return { priceRange: { minValue: 0, maxValue: 100 } }; };
    const rsiSeries = chart.addSeries(LWC.LineSeries, {
      color: th.rsiColor,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: false,
      autoscaleInfoProvider: rsiScale,
      priceFormat: { type: "price", precision: 1, minMove: 0.1 },
    }, 1);
    const rsiEmaSeries = chart.addSeries(LWC.LineSeries, {
      color: th.rsiEmaColor,
      lineWidth: 1,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: false,
      autoscaleInfoProvider: rsiScale,
      priceFormat: { type: "price", precision: 1, minMove: 0.1 },
    }, 1);
    [70, 30].forEach(function (level) {
      rsiSeries.createPriceLine({
        price: level,
        color: th.rsiBandColor,
        lineWidth: 1,
        lineStyle: LWC.LineStyle.Dashed,
        axisLabelVisible: false,
      });
    });
    // 기본 여백(위 20%·아래 10%)이면 0~100 축이 −8~120으로 보인다 — 좁혀서 30/70선이 산다.
    rsiSeries.priceScale().applyOptions({ scaleMargins: { top: 0.06, bottom: 0.06 } });
    const panes = chart.panes();
    if (panes.length > 1 && typeof panes[1].setStretchFactor === "function") {
      panes[0].setStretchFactor(3);
      panes[1].setStretchFactor(1);
    }

    const rsi = (data.rsi || []).filter(function (p) { return p && isFinite(p.value); });
    const rsiEma = (data.rsi_ema || []).filter(function (p) { return p && isFinite(p.value); });
    const rsiByTime = new Map(rsi.map(function (p) { return [p.time, p.value]; }));
    const rsiEmaByTime = new Map(rsiEma.map(function (p) { return [p.time, p.value]; }));

    function applyFrom(idx) {
      loadedFrom = Math.max(0, idx);
      const fromTime = candles[loadedFrom].time;
      candleSeries.setData(candles.slice(loadedFrom));
      rsiSeries.setData(rsi.filter(function (p) { return p.time >= fromTime; }));
      rsiEmaSeries.setData(rsiEma.filter(function (p) { return p.time >= fromTime; }));
    }
    applyFrom(loadedFrom);

    // 초기 화면 = 최근 봉 일부 + 오른쪽 여백(AlphaBlock WAN-245: 최신 봉·존이 가격축에
    // 붙지 않게 `to`를 보이는 창의 비율만큼 민다).
    const rendered = candles.length - loadedFrom;
    const n = Math.min(initialVisibleBars(options.timeframe, candles.length), rendered);
    const pad = Math.round(n * (options.rightPadRatio === undefined ? 0.08 : options.rightPadRatio));
    chart.timeScale().setVisibleLogicalRange({ from: rendered - n, to: rendered + pad });

    // 좌측 끝에 닿으면 이미 받은 배열에서 한 조각 더 붙인다(서버 왕복 없음).
    let loading = false;
    chart.timeScale().subscribeVisibleLogicalRangeChange(function (range) {
      if (!range || loading || loadedFrom <= 0) return;
      if (range.from < 20) {
        loading = true;
        const saved = chart.timeScale().getVisibleRange();
        applyFrom(loadedFrom - chunk);
        if (saved) chart.timeScale().setVisibleRange(saved);
        loading = false;
      }
    });

    // 좌상단 범례: 제목 · 커서 봉의 시/고/저/종·변동 · RSI. 커서가 나가면 마지막 봉.
    const legend = document.createElement("div");
    legend.className = "ob-chart-legend";
    legend.style.cssText =
      "position:absolute;top:8px;left:8px;z-index:5;background:" + th.legendBg + ";" +
      "padding:4px 8px;border-radius:4px;font:12px -apple-system,sans-serif;" +
      "line-height:1.6;pointer-events:none;color:" + th.textColor + ";";
    const titleRow = document.createElement("div");
    titleRow.style.cssText = "font-weight:600;";
    titleRow.textContent = options.title || "";
    const ohlcRow = document.createElement("div");
    ohlcRow.style.cssText = "font-variant-numeric:tabular-nums;";
    const zoneRow = document.createElement("div");
    zoneRow.innerHTML =
      '<span style="display:inline-block;width:10px;height:8px;background:' + th.bullZoneFill +
      ';border:1px solid ' + th.bullZoneLine + ';margin-right:4px;"></span>강세 OB ' +
      '<span style="display:inline-block;width:10px;height:8px;background:' + th.bearZoneFill +
      ';border:1px solid ' + th.bearZoneLine + ';margin:0 4px 0 8px;"></span>약세 OB ' +
      '<span style="display:inline-block;width:10px;height:8px;background:' + th.deadZoneFill +
      ';border:1px dashed ' + th.deadZoneLine + ';margin:0 4px 0 8px;"></span>무효화';
    legend.appendChild(titleRow);
    legend.appendChild(ohlcRow);
    legend.appendChild(zoneRow);
    container.appendChild(legend);

    function renderLegend(bar) {
      if (!bar) return;
      const color = bar.close >= bar.open ? th.upColor : th.downColor;
      const change = bar.close - bar.open;
      const pct = bar.open ? (change / bar.open) * 100 : 0;
      const r = rsiByTime.get(bar.time);
      const re = rsiEmaByTime.get(bar.time);
      ohlcRow.innerHTML =
        fmtDate(bar.time) + ' <span style="color:' + color + '">' +
        "시 " + fmtPrice(bar.open) + " 고 " + fmtPrice(bar.high) +
        " 저 " + fmtPrice(bar.low) + " 종 " + fmtPrice(bar.close) +
        " (" + (change >= 0 ? "+" : "") + pct.toFixed(2) + "%)</span>" +
        ' <span style="color:' + th.rsiColor + '">RSI ' +
        (r === undefined ? "-" : r.toFixed(1)) + "</span>" +
        ' <span style="color:' + th.rsiEmaColor + '">EMA ' +
        (re === undefined ? "-" : re.toFixed(1)) + "</span>";
    }
    const lastBar = candles[candles.length - 1];
    renderLegend(lastBar);
    chart.subscribeCrosshairMove(function (param) {
      const hovered = param && param.seriesData ? param.seriesData.get(candleSeries) : null;
      renderLegend(hovered || lastBar);
    });

    return {
      chart: chart,
      boxes: boxes,
      destroy: function () {
        chart.remove();
        container.innerHTML = "";
      },
    };
  }

  return {
    THEME: THEME,
    indexAtOrBefore: indexAtOrBefore,
    buildZoneBoxes: buildZoneBoxes,
    boxesPriceRange: boxesPriceRange,
    initialVisibleBars: initialVisibleBars,
    render: render,
  };
});
