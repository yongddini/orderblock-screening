"""「이중지지 + RSI과매도」를 샀다면 며칠 들고 있는 게 나았나 — 보유 기간 곡선(OBS-12).

흐름: ① 시세 받기(로컬 캐시) → ② 과거 재현(`obscreen.research.replay`, 지금 탐지기 · 운영 판정
코드 그대로) → ③ 보유 기간별 수익(신호일 **이후** 데이터만) → ④ 리포트(`reports/`).

실행(인터넷 필요 · 처음 한 번 시세를 받는다)::

    uv run python -m obscreen.research.holding --start 2025-05-01

비용: 왕복 비용(수수료 0.015% × 2 + 매도 거래세 0.20% = 0.23%)을 수익에서 뺀다
(`--cost-bps`). 지수 수익에는 비용을 빼지 않는다. G0(같은 날 대상 전체)도 같은 비용을 빼므로
G0 대비 초과수익에서 비용은 상쇄된다.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from obscreen.config import KST, get_settings
from obscreen.data import provider
from obscreen.data.provider import MARKET_SEGMENTS
from obscreen.detect.alphablock import UPSTREAM_COMMIT
from obscreen.research.replay import (
    ScreenerConfig,
    StockTask,
    build_close_panel,
    membership_dates,
    replay_stock,
    select_universe,
)

#: 보유 기간(영업일) — 신호일 h영업일 뒤 종가에 판다.
HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20, 40, 60)
#: 매수 가격 — (a) 신호일 종가(스크리닝은 장 마감 뒤라 실제로는 못 산다) · (b) 다음 영업일 시가.
ENTRIES: tuple[str, ...] = ("close", "open")
ENTRY_LABEL = {"close": "(a) 신호일 종가", "open": "(b) 다음 영업일 시가"}
#: 왕복 비용 기본값(bp) — 수수료 1.5bp × 2 + 매도 거래세 20bp.
DEFAULT_COST_BPS = 23.0
#: 이 건수 미만인 칸에는 「표본 적음 — 결론 아님」을 붙인다.
MIN_SAMPLE = 30
SAMPLE_WARNING = "⚠️ 표본 적음 — 결론 아님"
#: 시세 이력 시작일(주봉 500주 창이 측정 시작일보다 약 10년 앞까지 필요하다).
HISTORY_START = "2014-01-01"
INDEX_OF_MARKET = {"KOSPI": "KS11", "KOSDAQ": "KQ11"}
REPORT_STEM = "obs12_holding"


# ───────────────────────────────────────────────────────────── 조건(그룹)


@dataclass(frozen=True)
class Condition:
    """신호 조건 — 재현 후보 행(`replay_stock` 출력)에서 참/거짓을 낸다."""

    key: str
    label: str
    predicate: Callable[[pd.DataFrame], pd.Series]


def _dual_rsi(threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    def pred(df: pd.DataFrame) -> pd.Series:
        return df["dual_support"].astype(bool) & df[f"rsi_flag_{threshold:g}"].astype(bool)

    return pred


def _dual_rsi_weekly(weekly_threshold: float) -> Callable[[pd.DataFrame], pd.Series]:
    base = _dual_rsi(30.0)

    def pred(df: pd.DataFrame) -> pd.Series:
        return base(df) & (df["weekly_rsi"] < weekly_threshold)

    return pred


#: 주인공 — 운영 저장값 `is_recommended == 3`(이중지지 AND RSI과매도)과 같은 정의.
MAIN = Condition("G4", "이중지지 + 일봉 RSI<30 (운영 · 주인공)", _dual_rsi(30.0))

#: 비교용(사용자 결정 #4) — 참고 표. 운영 규칙은 바꾸지 않는다.
COMPARISONS: tuple[Condition, ...] = (
    MAIN,
    Condition("D35", "이중지지 + 일봉 RSI<35", _dual_rsi(35.0)),
    Condition("D40", "이중지지 + 일봉 RSI<40", _dual_rsi(40.0)),
    Condition("W30", "주인공 + 주봉 RSI<30", _dual_rsi_weekly(30.0)),
    Condition("W35", "주인공 + 주봉 RSI<35", _dual_rsi_weekly(35.0)),
    Condition("W40", "주인공 + 주봉 RSI<40", _dual_rsi_weekly(40.0)),
)

#: 참고 표(결론 대상 아님) — 주인공이 왜 그런 결과인지 설명하는 용도.
REFERENCES: tuple[Condition, ...] = (
    Condition("G1", "일봉 지지 존 내부/근접 전체", lambda df: df["ob_type"] == "Bull"),
    Condition("G2", "🔥 이중지지", lambda df: df["dual_support"].astype(bool)),
    Condition("G3", "📉 RSI과매도 배지", lambda df: df["rsi_flag_30"].astype(bool)),
)


# ───────────────────────────────────────────────────────────── 수익 계산


def forward_table(
    hist: pd.DataFrame, horizons: Sequence[int] = HORIZONS, cost: float = 0.0
) -> pd.DataFrame:
    """날짜별 「그날 신호가 났다면」의 보유 기간 수익(비용 차감)과 보유 중 최대 하락.

    신호일이 이력의 i번째 봉일 때:

    - `ret_close_h` = Close[i+h] / Close[i] − 1 − cost
    - `ret_open_h`  = Close[i+h] / Open[i+1] − 1 − cost
    - `mdd_{entry}_h` = min(Low[i+1 .. i+h]) / 매수가 − 1 (음수일수록 깊다)

    **신호일 이후의 봉만 쓴다**(매수가로 쓰는 신호일 종가는 그날 확정된 값). i+h가 이력 밖이면
    NaN — 보유 기간이 끝나지 않은 신호는 그 칸에서 빠진다.

    거래정지일(시가·고가·저가 0 · 종가는 직전 값)은 `clean_halted`로 다듬는다 — 시가가 없으면 그날
    시가 매수는 불가(NaN), 저가는 종가로 본다(가격이 움직이지 않았다).
    """
    clean = clean_halted(hist)
    close = clean["Close"]
    open_next = clean["Open"].shift(-1)
    low = clean["Low"]
    out: dict[str, pd.Series] = {}
    for h in horizons:
        exit_px = close.shift(-h)
        # min(Low[i+1..i+h]) — 뒤집어 굴린 최솟값을 한 칸 당긴다. 창이 다 차지 않으면 NaN.
        fut_low = low[::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
        out[f"ret_close_{h}"] = exit_px / close - 1 - cost
        out[f"ret_open_{h}"] = exit_px / open_next - 1 - cost
        out[f"mdd_close_{h}"] = (fut_low / close - 1).where(exit_px.notna())
        out[f"mdd_open_{h}"] = (fut_low / open_next - 1).where(exit_px.notna())
    return pd.DataFrame(out, index=hist.index)


def clean_halted(hist: pd.DataFrame) -> pd.DataFrame:
    """0 이하 가격(거래정지일 등)을 정리한 OHLC 사본.

    시가 NaN(그날 시가 매수 불가) · 저가·고가는 종가 · 종가 0은 NaN.
    """
    close = hist["Close"].astype(float)
    close = close.where(close > 0)
    open_ = hist["Open"].astype(float)
    low = hist["Low"].astype(float)
    high = hist["High"].astype(float)
    return pd.DataFrame(
        {
            "Open": open_.where(open_ > 0),
            "High": high.where(high > 0, close),
            "Low": low.where(low > 0, close),
            "Close": close,
        },
        index=hist.index,
    )


def stop_returns(
    hist: pd.DataFrame,
    date: pd.Timestamp,
    stop_price: float,
    entry: str,
    horizons: Sequence[int] = HORIZONS,
    cost: float = 0.0,
) -> dict[int, float]:
    """보조 곡선 — 보유 중 종가가 `stop_price`(존 아래끝) 아래로 내려가면 그날 종가에 판다.

    h영업일 안에 이탈이 없으면 h영업일 뒤 종가에 판다. 보유 기간이 끝나지 않으면 NaN.
    """
    i = int(hist.index.get_indexer([date])[0])
    if i < 0:
        return {h: math.nan for h in horizons}
    clean = clean_halted(hist)
    closes = clean["Close"].to_numpy(dtype=float)
    opens = clean["Open"].to_numpy(dtype=float)
    n = len(closes)
    if entry == "close":
        entry_px = float(closes[i])
    else:
        entry_px = float(opens[i + 1]) if i + 1 < n else math.nan
    result: dict[int, float] = {}
    for h in horizons:
        if i + h >= n or math.isnan(entry_px) or math.isnan(closes[i + h]):
            result[h] = math.nan
            continue
        exit_px = float(closes[i + h])
        for k in range(1, h + 1):
            if closes[i + k] < stop_price:
                exit_px = float(closes[i + k])
                break
        result[h] = exit_px / entry_px - 1 - cost
    return result


def first_day_mask(signals: pd.DataFrame, mask: pd.Series, calendar: Sequence[Any]) -> pd.Series:
    """조건이 참인 행 중 「처음 뜬 날」 — 바로 전 영업일에 같은 종목에 조건이 없던 날.

    같은 종목이 며칠 연속 뜨면 그 연속의 첫날만 참이다(끊겼다 다시 뜨면 다시 첫날).
    """
    pos = {pd.Timestamp(d): i for i, d in enumerate(calendar)}
    rows = signals.loc[mask, ["code", "date"]]
    keys = [(str(c), pos[pd.Timestamp(d)]) for c, d in zip(rows["code"], rows["date"], strict=True)]
    on = set(keys)
    first = pd.Series(False, index=signals.index)
    if keys:
        first.loc[rows.index] = [(c, p - 1) not in on for c, p in keys]
    return first


# ───────────────────────────────────────────────────────────── 집계


@dataclass
class Panel:
    """수익 계산에 필요한 것 — 재현 후보, 날짜별 G0 평균, 지수 수익, 보조 곡선."""

    signals: pd.DataFrame  # 후보 행 + 수익 열(ret_*, mdd_*, g0_*, idx_*, stop_*)
    g0: pd.DataFrame  # 대상 전체 (날짜, 종목) 쌍의 수익 열
    calendar: list[pd.Timestamp]
    midpoint: pd.Timestamp


def _cell(values: pd.Series, fmt: Callable[[float], str]) -> str:
    v = values.dropna()
    return fmt(float(v.mean())) if len(v) else "—"


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


def curve_rows(
    df: pd.DataFrame, entry: str, midpoint: pd.Timestamp, prefix: str = "ret"
) -> list[dict[str, Any]]:
    """보유 기간마다 한 행 — 신호 수 · 평균/중앙값 · 이긴 비율 · 최악 · 최대 하락 · 초과수익."""
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        col = f"{prefix}_{entry}_{h}"
        sub = df[df[col].notna()]
        r = sub[col]
        ex_g0 = r - sub[f"g0_{entry}_{h}"]
        ex_idx = r - sub[f"idx_{entry}_{h}"]
        front = sub["date"] < midpoint
        n = len(sub)
        rows.append(
            {
                "h": h,
                "n": n,
                "mean": float(r.mean()) if n else math.nan,
                "median": float(r.median()) if n else math.nan,
                "win": float((r > 0).mean()) if n else math.nan,
                "worst": float(r.min()) if n else math.nan,
                "mdd_mean": float(sub[f"mdd_{entry}_{h}"].mean()) if n else math.nan,
                "mdd_worst": float(sub[f"mdd_{entry}_{h}"].min()) if n else math.nan,
                "ex_g0": float(ex_g0.mean()) if n else math.nan,
                "ex_g0_win": float((ex_g0 > 0).mean()) if n else math.nan,
                "ex_idx": float(ex_idx.mean()) if n else math.nan,
                "ex_g0_front": float(ex_g0[front].mean()) if front.any() else math.nan,
                "ex_g0_back": float(ex_g0[~front].mean()) if (~front).any() else math.nan,
                "n_front": int(front.sum()),
                "n_back": int((~front).sum()),
            }
        )
    return rows


def _same_direction(a: float, b: float) -> str:
    if math.isnan(a) or math.isnan(b):
        return "—"
    if a > 0 and b > 0:
        return "같음(+)"
    if a < 0 and b < 0:
        return "같음(−)"
    return "엇갈림"


def _fmt(x: float, kind: str = "pct") -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if kind == "pct":
        return f"{x * 100:+.2f}%"
    if kind == "rate":
        return f"{x * 100:.1f}%"
    return f"{x}"


def render_curve(rows: list[dict[str, Any]], with_index: bool = True) -> str:
    head = (
        "| 보유(영업일) | 신호 수 | 평균 | 중앙값 | 이긴 비율 | 최악 "
        "| 보유 중 최대 하락(평균/최악) | G0 대비 초과 | 초과 이긴 비율 |"
        + (" 지수 대비 초과 |" if with_index else "")
        + " 앞 반 / 뒤 반 (G0 대비) | 비고 |"
    )
    sep = "|" + "---|" * (head.count("|") - 1)
    lines = [head, sep]
    for r in rows:
        warn = SAMPLE_WARNING if r["n"] < MIN_SAMPLE else ""
        halves = (
            f"{_fmt(r['ex_g0_front'])} ({r['n_front']}) / {_fmt(r['ex_g0_back'])} ({r['n_back']})"
            f" · {_same_direction(r['ex_g0_front'], r['ex_g0_back'])}"
        )
        line = (
            f"| {r['h']} | {r['n']} | {_fmt(r['mean'])} | {_fmt(r['median'])} | "
            f"{_fmt(r['win'], 'rate')} | {_fmt(r['worst'])} | "
            f"{_fmt(r['mdd_mean'])} / {_fmt(r['mdd_worst'])} | {_fmt(r['ex_g0'])} | "
            f"{_fmt(r['ex_g0_win'], 'rate')} |"
            + (f" {_fmt(r['ex_idx'])} |" if with_index else "")
            + f" {halves} | {warn} |"
        )
        lines.append(line)
    return "\n".join(lines)


def g0_rows(g0: pd.DataFrame, entry: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h in HORIZONS:
        r = g0[f"ret_{entry}_{h}"].dropna()
        m = g0[f"mdd_{entry}_{h}"].dropna()
        x = g0[f"ex_idx_{entry}_{h}"].dropna()
        rows.append(
            {
                "h": h,
                "n": len(r),
                "mean": float(r.mean()) if len(r) else math.nan,
                "median": float(r.median()) if len(r) else math.nan,
                "win": float((r > 0).mean()) if len(r) else math.nan,
                "worst": float(r.min()) if len(r) else math.nan,
                "mdd_mean": float(m.mean()) if len(m) else math.nan,
                "ex_idx": float(x.mean()) if len(x) else math.nan,
            }
        )
    return rows


def top5_share(df: pd.DataFrame) -> tuple[float, str]:
    """종목 편중 — 신호 상위 5종목이 차지하는 비중과 그 종목들."""
    if len(df) == 0:
        return math.nan, ""
    counts = df.groupby(["code", "name"]).size().sort_values(ascending=False)
    top = counts.head(5)
    names = ", ".join(f"{name}({n})" for (_c, name), n in top.items())
    return float(top.sum() / counts.sum()), names


# ───────────────────────────────────────────────────────────── 데이터


def _git_commit(repo: Path) -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "obscreen"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}{'-dirty' if dirty else ''}"


def _replay_source_hash(repo: Path) -> str:
    """재현 결과를 바꿀 수 있는 소스(판정·탐지·재현 코드)의 지문 — 재현 캐시 키."""
    import hashlib

    digest = hashlib.sha256()
    files = sorted(
        [
            *(repo / "obscreen" / "detect").rglob("*.py"),
            *(repo / "obscreen" / "screening").glob("*.py"),
        ]
        + [repo / "obscreen" / "research" / "replay.py", repo / "obscreen" / "data" / "provider.py"]
    )
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def load_listing(cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """오늘 KRX 목록(코스피·코스닥 · 코스닥 GLOBAL은 코스닥으로) — 캐시가 있으면 그것."""
    path = cache_dir / "listing_krx.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"Code": str})
    df = _fdr().StockListing("KRX")
    seg_to_market = {seg: m for m, segs in MARKET_SEGMENTS.items() for seg in segs}
    df = df[df["Market"].isin(seg_to_market)].copy()
    df["MarketGroup"] = df["Market"].map(seg_to_market)
    df = df[["Code", "Name", "Market", "MarketGroup", "Stocks", "Close"]]
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def _fdr() -> Any:
    """`obscreen.data.provider.fdr`(FinanceDataReader) — 테스트가 바꿔 끼우는 그 이름을 부른다."""
    return vars(provider)["fdr"]


def _read_ohlcv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_history(code: str, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """종목(또는 지수) 일봉 전체 이력 — 수정주가(FinanceDataReader). 캐시가 있으면 그것."""
    path = cache_dir / "ohlcv" / f"{code}.csv"
    if path.exists() and not refresh:
        return _read_ohlcv(path)
    df = _fdr().DataReader(code, HISTORY_START)
    if df is None or len(df) == 0:
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


# ───────────────────────────────────────────────────────────── 실행


def _replay_worker(task: StockTask) -> list[dict[str, Any]]:
    return replay_stock(task)


def run_replay(
    histories: Mapping[str, pd.DataFrame],
    listing: pd.DataFrame,
    calendar: Sequence[pd.Timestamp],
    top_n: int,
    config: ScreenerConfig,
    jobs: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """대상 종목을 날짜마다 고르고 재현한다 → (후보 행, 날짜×종목 대상 여부)."""
    closes = build_close_panel(histories, calendar)
    shares = dict(zip(listing["Code"], listing["Stocks"].astype(float), strict=True))
    markets = dict(zip(listing["Code"], listing["MarketGroup"], strict=True))
    names = dict(zip(listing["Code"], listing["Name"], strict=True))
    member = select_universe(closes, shares, markets, top_n)
    dates_by_code = membership_dates(member)
    tasks = [
        StockTask(code, str(names[code]), str(markets[code]), histories[code], dates, config)
        for code, dates in dates_by_code.items()
        if dates
    ]
    rows: list[dict[str, Any]] = []
    started = time.time()
    results: Iterable[list[dict[str, Any]]]
    if jobs <= 1:
        results = map(_replay_worker, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(_replay_worker, tasks, chunksize=4)
    for i, chunk in enumerate(results, 1):
        rows.extend(chunk)
        if i % 100 == 0 or i == len(tasks):
            print(
                f"  재현 {i}/{len(tasks)}종목 · 후보 {len(rows)}행 · {time.time() - started:.0f}초",
                flush=True,
            )
    if jobs > 1:
        pool.shutdown()
    signals = pd.DataFrame(rows)
    return signals, member


def build_panel(
    signals: pd.DataFrame,
    member: pd.DataFrame,
    histories: Mapping[str, pd.DataFrame],
    indexes: Mapping[str, pd.DataFrame],
    markets: Mapping[str, str],
    calendar: list[pd.Timestamp],
    cost: float,
) -> Panel:
    """후보 행과 대상 전체(G0)에 보유 기간 수익을 붙인다."""
    cal = pd.DatetimeIndex(calendar)
    idx_tables = {
        m: forward_table(indexes[INDEX_OF_MARKET[m]], cost=0.0).reindex(cal)
        for m in INDEX_OF_MARKET
    }
    ret_cols = [f"ret_{e}_{h}" for e in ENTRIES for h in HORIZONS]
    mdd_cols = [f"mdd_{e}_{h}" for e in ENTRIES for h in HORIZONS]
    g0_parts: list[pd.DataFrame] = []
    sig_parts: list[pd.DataFrame] = []
    by_code = {c: g for c, g in signals.groupby("code")} if len(signals) else {}
    for code in member.columns:
        dates = cal[member[code].to_numpy()]
        if len(dates) == 0:
            continue
        ft = forward_table(histories[code], cost=cost).reindex(dates)
        market = markets[code]
        idx = idx_tables[market].reindex(dates)
        part = ft[ret_cols + mdd_cols].copy()
        for e in ENTRIES:
            for h in HORIZONS:
                part[f"ex_idx_{e}_{h}"] = ft[f"ret_{e}_{h}"] - idx[f"ret_{e}_{h}"]
        part["date"] = dates
        part["code"] = code
        part["market"] = market
        g0_parts.append(part.reset_index(drop=True))
        if code in by_code:
            sig = by_code[code].copy()
            sig_dates = pd.DatetimeIndex(sig["date"])
            ftd = ft.reindex(sig_dates)
            idd = idx.reindex(sig_dates)
            for col in ret_cols + mdd_cols:
                sig[col] = ftd[col].to_numpy()
            for e in ENTRIES:
                for h in HORIZONS:
                    sig[f"idx_{e}_{h}"] = idd[f"ret_{e}_{h}"].to_numpy()
            sig_parts.append(sig)
    g0 = pd.concat(g0_parts, ignore_index=True)
    g0_mean = g0.groupby("date")[ret_cols].mean()
    out = pd.concat(sig_parts, ignore_index=True) if sig_parts else signals.copy()
    if len(out):
        gm = g0_mean.reindex(pd.DatetimeIndex(out["date"]))
        for e in ENTRIES:
            for h in HORIZONS:
                out[f"g0_{e}_{h}"] = gm[f"ret_{e}_{h}"].to_numpy()
        # 보조 곡선(존 아래끝 종가 이탈 손절)은 주인공·비교 조건 후보에만 계산한다(무겁다).
        need = _dual_rsi(40.0)(out)
        for e in ENTRIES:
            for h in HORIZONS:
                out[f"stop_{e}_{h}"] = math.nan
        for i in out.index[need]:
            row = out.loc[i]
            for e in ENTRIES:
                sr = stop_returns(
                    histories[str(row["code"])],
                    pd.Timestamp(row["date"]),
                    float(row["ob_bottom"]),
                    e,
                    cost=cost,
                )
                for h, v in sr.items():
                    out.at[i, f"stop_{e}_{h}"] = v
    return Panel(out, g0, calendar, calendar[len(calendar) // 2])


# ───────────────────────────────────────────────────────────── 리포트


@dataclass(frozen=True)
class RunMeta:
    commit: str
    start: pd.Timestamp
    end: pd.Timestamp
    data_end: pd.Timestamp
    cost_bps: float
    top_n: int
    n_candidates_pool: int
    generated_at: str
    config: ScreenerConfig


def header_lines(meta: RunMeta) -> list[str]:
    """리포트·CSV 머리 — 엔진 · 생성 커밋 · 데이터 기간 · 비용 · 대상 방식."""
    c = meta.config
    return [
        f"탐지기 = AlphaBlock `{UPSTREAM_COMMIT[:8]}` 복사본(obscreen/detect/alphablock, OBS-5)",
        f"생성 커밋 = `{meta.commit}` · 생성 시각 {meta.generated_at}",
        (
            f"신호 기간 = {meta.start.date()} ~ {meta.end.date()} (KRX 영업일) · "
            f"시세 끝 = {meta.data_end.date()} · 보유 기간이 끝나지 않은 신호는 그 칸에서 뺌"
        ),
        (
            f"비용 = 왕복 {meta.cost_bps:g}bp(수수료 1.5bp×2 + 매도 거래세 20bp)를 수익에서 뺌 · "
            "지수 수익에는 안 뺌"
        ),
        (
            f"대상 종목 = 오늘(받은 날) 코스피·코스닥 상장 {meta.n_candidates_pool}종목 중 "
            f"시장별 「그날 종가 × 오늘 상장주식수」 상위 {meta.top_n}(그날 거래 없는 종목 제외) · "
            "ETF 제외"
        ),
        (
            f"판정 = 운영과 같은 코드(StockScreener.evaluate_daily/weekly_frame · "
            f"core.rsi_oversold_flag · core.dual_support_codes) · 근접도 일봉 "
            f"{c.proximity_daily:g}% · 주봉 {c.proximity_weekly:g}% · 스윙 {c.swing_length} · "
            f"ATR×{c.max_atr_mult:g} · 무효화 {c.ob_end_method} · 병합 {c.combine_obs}"
        ),
        "시세 = FinanceDataReader 수정주가(한 번 받아 로컬 캐시, git 제외)",
    ]


def render_report(panel: Panel, meta: RunMeta) -> str:
    sig = panel.signals
    cal = panel.calendar
    out: list[str] = []
    out.append("# OBS-12 — 「이중지지 + RSI과매도」를 샀다면 며칠 들고 있는 게 나았나")
    out.append("")
    out.append("> " + "  \n> ".join(header_lines(meta)))
    out.append("")
    out.append(
        "> ⚠️ **생존 편향** — 과거 종목 목록·상장주식수는 KRX 로그인 없이 받을 수 없어 "
        "**오늘 상장된 종목**만 후보로 썼다. 그 사이 상장폐지·관리종목으로 사라진 종목이 빠져 "
        "결과가 **좋아 보이는 쪽으로** 기운다. 시장 구분(코스피/코스닥)도 오늘 기준이다. "
        "시총 순위는 「그날 종가 × 오늘 상장주식수」 근사다(그 사이 증자·감자는 반영 안 됨)."
    )
    out.append("")
    out.append(
        "> ⚠️ 이 표는 **지금 탐지기로 과거를 다시 돌린 것**이다. 서버 DB의 옛 기록(옛 탐지기 시절 "
        "화면)과 섞지 않는다."
    )
    out.append("")

    main_mask = MAIN.predicate(sig) if len(sig) else pd.Series(dtype=bool)
    main_first = first_day_mask(sig, main_mask, cal) if len(sig) else main_mask
    g0_days = int(panel.g0["date"].nunique())
    out.append("## 1. 신호 수")
    out.append("")
    out.append(f"- 신호 기간 영업일: **{len(cal)}일** (앞 반 < {panel.midpoint.date()} ≤ 뒤 반)")
    out.append(
        f"- G0(그날 대상 전체) = **{len(panel.g0):,}** (날짜, 종목) 쌍 · {g0_days}일 · "
        f"하루 평균 {len(panel.g0) / max(g0_days, 1):.0f}종목"
    )
    out.append(f"- 일봉 존 내부/근접 후보 = {len(sig):,}행")
    share_all, names_all = top5_share(sig[main_mask])
    share_first, names_first = top5_share(sig[main_first])
    out.append(
        f"- **주인공(이중지지 + RSI<30)**: 매일 따로 **{int(main_mask.sum())}건** · "
        f"처음 뜬 날 **{int(main_first.sum())}건** · "
        f"{sig.loc[main_mask, 'code'].nunique()}종목"
    )
    out.append(
        f"- 종목 편중(처음 뜬 날 기준 상위 5종목 비중): **{_fmt(share_first, 'rate')}** — "
        f"{names_first}"
    )
    out.append(f"- 종목 편중(매일 따로): {_fmt(share_all, 'rate')} — {names_all}")
    if int(main_first.sum()) < MIN_SAMPLE:
        out.append(f"- {SAMPLE_WARNING}: 처음 뜬 날 기준 신호가 {MIN_SAMPLE}건 미만이다.")
    months = sig.loc[main_first, "date"].dt.strftime("%Y-%m").value_counts().sort_index()
    out.append(
        "- 월별(처음 뜬 날): " + " · ".join(f"{m} {n}" for m, n in months.items())
        if len(months)
        else "- 월별: 없음"
    )
    out.append("")

    out.append("## 2. 주인공 보유 기간 곡선 (처음 뜬 날 기준 · 결론은 이 표)")
    out.append("")
    out.append(
        "초과수익 = 신호 수익(비용 차감) − 같은 날·같은 매수 방식 G0 평균(비용 차감). "
        "지수 대비 = 신호 수익(비용 차감) − 같은 날 소속 시장 지수(코스피 KS11 · 코스닥 KQ11, "
        "비용 없음). 보유 중 최대 하락 = 매수 뒤 그 기간 최저가 기준."
    )
    out.append("")
    first_df = sig[main_first]
    for e in ENTRIES:
        out.append(f"### {ENTRY_LABEL[e]}")
        out.append("")
        out.append(render_curve(curve_rows(first_df, e, panel.midpoint)))
        out.append("")
    out.append("### 비교 기준 G0(그날 대상 전체 · 매일)")
    out.append("")
    out.append(
        "| 보유 | 매수 | (날짜,종목) 수 | 평균 | 중앙값 | 이긴 비율 | 최대 하락(평균) | 지수 대비 |"
    )
    out.append("|---|---|---|---|---|---|---|---|")
    for e in ENTRIES:
        for r in g0_rows(panel.g0, e):
            out.append(
                f"| {r['h']} | {ENTRY_LABEL[e]} | {r['n']:,} | {_fmt(r['mean'])} | "
                f"{_fmt(r['median'])} | {_fmt(r['win'], 'rate')} | {_fmt(r['mdd_mean'])} | "
                f"{_fmt(r['ex_idx'])} |"
            )
    out.append("")

    out.append("## 3. 주인공 — 매일 따로 센 경우(참고)")
    out.append("")
    out.append(
        "같은 종목이 연속으로 뜨면 날마다 한 건으로 센다 — 신호끼리 겹쳐 독립이 아니므로 "
        "결론은 2절."
    )
    out.append("")
    out.append(render_curve(curve_rows(sig[main_mask], "open", panel.midpoint)))
    out.append("")

    out.append("## 4. 보조 — 존 아래끝 종가 이탈 시 손절 (처음 뜬 날 · (b) 다음 영업일 시가)")
    out.append("")
    out.append(
        "보유 중 종가가 일봉 존 아래끝(`ob_bottom`) 아래로 내려가면 그날 종가에 판다. G0 대비 "
        "초과의 G0은 손절 없는 보유다."
    )
    out.append("")
    out.append(render_curve(curve_rows(first_df, "open", panel.midpoint, prefix="stop")))
    out.append("")

    out.append("## 5. 비교 조건 (참고 표 · 운영 규칙 아님 · 사용자 결정 #4)")
    out.append("")
    out.append(
        "모두 처음 뜬 날 · (b) 다음 영업일 시가. 칸 = G0 대비 초과수익 평균(신호 수). "
        f"{MIN_SAMPLE}건 미만은 ⚠️. 여러 기준을 비교하면 우연히 좋아 보이는 칸이 생기기 쉽다 — "
        "아래 「주인공 대비」 표에서 **앞·뒤 반이 같은 방향일 때만** 「낫다/못하다」로 읽는다."
    )
    out.append("")
    head = "| 조건 | " + " | ".join(f"{h}일" for h in HORIZONS) + " |"
    out.append(head)
    out.append("|" + "---|" * (len(HORIZONS) + 1))
    comp_rows: dict[str, list[dict[str, Any]]] = {}
    for cond in COMPARISONS:
        m = cond.predicate(sig)
        fm = first_day_mask(sig, m, cal)
        rows = curve_rows(sig[fm], "open", panel.midpoint)
        comp_rows[cond.key] = rows
        cells = [
            f"{_fmt(r['ex_g0'])} ({r['n']}){' ⚠️' if r['n'] < MIN_SAMPLE else ''}" for r in rows
        ]
        out.append(f"| {cond.key} {cond.label} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("**주인공 대비 차이(G0 대비 초과수익의 차, 앞 반 / 뒤 반)**")
    out.append("")
    out.append(head)
    out.append("|" + "---|" * (len(HORIZONS) + 1))
    base = comp_rows[MAIN.key]
    for cond in COMPARISONS[1:]:
        cells = []
        for r, b in zip(comp_rows[cond.key], base, strict=True):
            df_ = r["ex_g0_front"] - b["ex_g0_front"]
            db_ = r["ex_g0_back"] - b["ex_g0_back"]
            small = min(r["n_front"], r["n_back"]) < MIN_SAMPLE // 2
            verdict = _same_direction(df_, db_)
            tag = (
                "낫다" if verdict == "같음(+)" else ("못하다" if verdict == "같음(−)" else verdict)
            )
            cells.append(f"{_fmt(df_)} / {_fmt(db_)} → {tag}{' ⚠️' if small else ''}")
        out.append(f"| {cond.key} | " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "「낫다/못하다」는 앞·뒤 반 둘 다 같은 부호일 때만 찍는다. ⚠️는 반쪽 표본이 "
        f"{MIN_SAMPLE // 2}건 미만 — 방향도 믿기 어렵다."
    )
    out.append("")

    out.append("## 6. 참고 그룹 (결론 대상 아님)")
    out.append("")
    out.append("처음 뜬 날 · (b) 다음 영업일 시가 · 칸 = G0 대비 초과수익 평균(신호 수).")
    out.append("")
    out.append(head)
    out.append("|" + "---|" * (len(HORIZONS) + 1))
    for cond in REFERENCES:
        m = cond.predicate(sig)
        fm = first_day_mask(sig, m, cal)
        rows = curve_rows(sig[fm], "open", panel.midpoint)
        cells = [f"{_fmt(r['ex_g0'])} ({r['n']})" for r in rows]
        out.append(f"| {cond.key} {cond.label} | " + " | ".join(cells) + " |")
    out.append("")
    out.append("## 7. 판정 줄 (기계적으로 찍음)")
    out.append("")
    out.extend(verdict_lines(curve_rows(first_df, "open", panel.midpoint)))
    out.append("")
    return "\n".join(out)


def verdict_lines(rows: list[dict[str, Any]]) -> list[str]:
    """주인공 (b) 처음 뜬 날 — 보유 기간마다 G0 대비 초과의 부호와 앞·뒤 반 일치 여부."""
    lines: list[str] = []
    for r in rows:
        if r["n"] < MIN_SAMPLE:
            lines.append(f"- {r['h']}일: 신호 {r['n']}건 — {SAMPLE_WARNING}")
            continue
        same = _same_direction(r["ex_g0_front"], r["ex_g0_back"])
        sign = "G0보다 높다" if r["ex_g0"] > 0 else "G0보다 낮다"
        if same.startswith("같음"):
            lines.append(
                f"- {r['h']}일: 신호 {r['n']}건 · 초과 {_fmt(r['ex_g0'])} — {sign}(앞·뒤 반 {same})"
            )
        else:
            lines.append(
                f"- {r['h']}일: 신호 {r['n']}건 · 초과 {_fmt(r['ex_g0'])} — 앞·뒤 반이 엇갈려 "
                "방향을 단정하지 않는다"
            )
    return lines


SIGNAL_CSV_COLUMNS = [
    "date",
    "code",
    "name",
    "market",
    "close",
    "rsi",
    "weekly_rsi",
    "status",
    "ob_top",
] + ["ob_bottom", "distance_percent", "is_recommended"]


def write_signals_csv(panel: Panel, meta: RunMeta, path: Path) -> int:
    """신호 한 건 한 건 — 이중지지 + 일봉 RSI<40 후보(비교 조건 전부의 합집합)."""
    sig = panel.signals
    keep = _dual_rsi(40.0)(sig)
    df = sig[keep].copy()
    for cond in COMPARISONS:
        m = cond.predicate(sig)
        df[f"in_{cond.key}"] = m[keep].to_numpy()
        df[f"first_{cond.key}"] = first_day_mask(sig, m, panel.calendar)[keep].to_numpy()
    flag_cols = [f"{p}_{c.key}" for c in COMPARISONS for p in ("in", "first")]
    metric_cols = [
        f"{k}_{e}_{h}"
        for k in ("ret", "mdd", "g0", "idx", "stop")
        for e in ENTRIES
        for h in HORIZONS
    ]
    df = df[SIGNAL_CSV_COLUMNS + flag_cols + metric_cols].sort_values(["date", "code"])
    df["date"] = pd.DatetimeIndex(df["date"]).strftime("%Y-%m-%d")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in header_lines(meta):
            fh.write(f"# {line}\n")
        fh.write(
            "# 수익(ret_*)·G0(g0_*)·손절(stop_*)은 비용 차감 · 지수(idx_*)는 비용 없음 · "
            "소수(0.01=1%)\n"
        )
        df.to_csv(fh, index=False, float_format="%.6g")
    return len(df)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default="2025-05-01", help="신호를 세기 시작하는 날(YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="신호를 세는 마지막 날(기본: 시세 끝)")
    parser.add_argument("--top-n", type=int, default=None, help="시장별 대상 수(기본: 운영 설정)")
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_COST_BPS)
    parser.add_argument("--cache-dir", default=".cache/obs12", help="시세 캐시(git 제외)")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--refresh", action="store_true", help="캐시를 무시하고 다시 받는다")
    parser.add_argument(
        "--reuse-replay",
        action="store_true",
        help="같은 기간·대상·커밋의 재현 결과 캐시가 있으면 다시 돌리지 않는다(수익 계산만)",
    )
    parser.add_argument("--limit", type=int, default=None, help="(시험용) 후보 종목 수 제한")
    args = parser.parse_args(argv)

    repo = Path(__file__).resolve().parents[2]
    cache_dir = Path(args.cache_dir)
    settings = get_settings()
    top_n = args.top_n or settings.screening_top_n
    config = ScreenerConfig.from_settings(settings)

    listing = load_listing(cache_dir, args.refresh)
    if args.limit:
        listing = listing.head(args.limit)
    print(f"후보 종목 {len(listing)}개 · 시세 받는 중(캐시: {cache_dir})", flush=True)
    histories: dict[str, pd.DataFrame] = {}
    for i, code in enumerate(listing["Code"], 1):
        histories[code] = load_history(code, cache_dir, args.refresh)
        if i % 250 == 0:
            print(f"  시세 {i}/{len(listing)}", flush=True)
    indexes = {sym: load_history(sym, cache_dir, args.refresh) for sym in INDEX_OF_MARKET.values()}

    kospi = indexes["KS11"]
    start = pd.Timestamp(args.start)
    data_end = pd.Timestamp(kospi.index.max())
    end = pd.Timestamp(args.end) if args.end else data_end
    calendar = [pd.Timestamp(d) for d in kospi.index if start <= d <= end]
    print(f"신호 기간 {calendar[0].date()} ~ {calendar[-1].date()} · {len(calendar)}영업일")

    replay_cache = cache_dir / (
        f"replay_{calendar[0].date()}_{calendar[-1].date()}_top{top_n}_{len(listing)}"
        f"_{_replay_source_hash(repo)}.pkl"
    )
    if args.reuse_replay and replay_cache.exists():
        print(f"재현 결과 캐시 사용: {replay_cache}")
        signals, member = pd.read_pickle(replay_cache)
    else:
        signals, member = run_replay(histories, listing, calendar, top_n, config, args.jobs)
        pd.to_pickle((signals, member), replay_cache)
    markets = dict(zip(listing["Code"], listing["MarketGroup"], strict=True))
    panel = build_panel(
        signals, member, histories, indexes, markets, calendar, cost=args.cost_bps / 10_000
    )
    meta = RunMeta(
        commit=_git_commit(repo),
        start=calendar[0],
        end=calendar[-1],
        data_end=data_end,
        cost_bps=args.cost_bps,
        top_n=top_n,
        n_candidates_pool=len(listing),
        generated_at=datetime.now(KST).strftime("%Y-%m-%d %H:%M KST"),
        config=config,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md = out_dir / f"{REPORT_STEM}_summary.md"
    md.write_text(render_report(panel, meta), encoding="utf-8")
    n = write_signals_csv(panel, meta, out_dir / f"{REPORT_STEM}_signals.csv")
    (cache_dir / "last_run.json").write_text(
        json.dumps({"md": str(md), "signals": n, "commit": meta.commit}, ensure_ascii=False)
    )
    print(f"✅ 리포트 {md} · 신호 CSV {n}행")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "COMPARISONS",
    "HORIZONS",
    "MAIN",
    "Panel",
    "curve_rows",
    "first_day_mask",
    "forward_table",
    "stop_returns",
]
