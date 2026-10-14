#!/usr/bin/env python3
"""
스크리닝 핵심 로직
"""

from __future__ import annotations

import importlib
import os
import sqlite3
import sys
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from obscreen.config import KST, get_settings
from obscreen.data.provider import KoreanStockDataProvider
from obscreen.screening.screener import StockScreener

#: 스크리닝·수급 결과를 쓰는 SQLite 경로. import 시점에 설정(`DB_PATH`)에서 한 번 읽는다.
DB_PATH = get_settings().db_path


def run_and_save_screening(target_date: str | None = None) -> None:
    """오더블록 스크리닝 실행 및 저장"""
    if target_date:
        today = datetime.strptime(target_date, "%Y%m%d").date()
        today_str = target_date
    else:
        today = datetime.now(KST).date()
        today_str = today.strftime("%Y%m%d")

    if today.weekday() >= 5:
        print(f"⏸️  {today} 주말이므로 스크리닝을 건너뜁니다.")
        return

    try:
        import holidays

        KR_HOLIDAYS = holidays.country_holidays("KR")  # = holidays.SouthKorea()
        # `bool(KR_HOLIDAYS)`로 거르지 말 것 — 연도를 지연 생성해 막 만든 객체는 비어 있어
        # 거짓이고, 그러면 평일 공휴일(추석 등)에도 스크리닝이 돈다(OBS-9).
        if today in KR_HOLIDAYS:
            holiday_name = KR_HOLIDAYS.get(today)
            print(f"🎉 {today} 공휴일({holiday_name})이므로 스크리닝을 건너뜁니다.")
            return
    except ImportError:
        pass

    settings = get_settings()

    # 시가총액 순위를 만들 수 없으면 기존 결과를 지우기 전에 멈춘다(OBS-7 §5).
    KoreanStockDataProvider.validate_market_caps(["KOSPI", "KOSDAQ"])

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM screening_results WHERE scan_date = ?", (today_str,))
    count = cursor.fetchone()[0]

    if count > 0:
        print(f"🗑️  {today} 기존 결과 {count}개 삭제...")
        cursor.execute("DELETE FROM screening_results WHERE scan_date = ?", (today_str,))
        conn.commit()
    conn.close()

    print(f"🔍 {today} 스크리닝 시작...")

    # 일반 주식용 (3%)
    screener_daily = StockScreener(
        proximity_percent=settings.proximity_stock_daily,
        swing_length=settings.swing_length,
        max_atr_mult=settings.max_atr_mult,
        ob_end_method=settings.ob_end_method,
        combine_obs=settings.combine_obs,
    )

    # ETF용 (1%)
    screener_daily_etf = StockScreener(
        proximity_percent=settings.proximity_etf_daily,
        swing_length=settings.swing_length,
        max_atr_mult=settings.max_atr_mult,
        ob_end_method=settings.ob_end_method,
        combine_obs=settings.combine_obs,
    )

    screener_weekly = StockScreener(
        proximity_percent=settings.proximity_weekly,
        swing_length=settings.swing_length,
        max_atr_mult=settings.max_atr_mult,
        ob_end_method=settings.ob_end_method,
        combine_obs=settings.combine_obs,
    )

    end_date_str = today.strftime("%Y-%m-%d")

    print("\n" + "=" * 50)
    print(f"일봉 스크리닝 시작 (기준일: {end_date_str}, 근접도: 1%)")
    print("=" * 50)

    # 일반 주식 스크리닝 (3%)
    stock_results_daily = screener_daily.screen_multiple_markets(
        markets=["KOSPI", "KOSDAQ"],
        top_n=settings.screening_top_n,
        days=500,
        end_date=end_date_str,
    )

    # ETF 스크리닝 (1%)
    etf_results_daily = screener_daily_etf.screen_etf(
        top_n=settings.screening_etf_n, days=500, end_date=end_date_str
    )

    if isinstance(stock_results_daily, pd.DataFrame):
        results_daily = stock_results_daily.to_dict("records")
    else:
        results_daily = stock_results_daily

    if isinstance(etf_results_daily, list):
        results_daily.extend(etf_results_daily)
    elif isinstance(etf_results_daily, pd.DataFrame):
        results_daily.extend(etf_results_daily.to_dict("records"))

    print("\n" + "=" * 50)
    print(f"주봉 스크리닝 시작 (기준일: {end_date_str}, 근접도: 5%)")
    print("=" * 50)

    results_weekly = []
    for market in ["KOSPI", "KOSDAQ"]:
        weekly_result = screener_weekly.screen_market_weekly(
            market=market,
            top_n=settings.screening_top_n,
            weeks=500,
            end_date=end_date_str,
        )
        results_weekly.extend(weekly_result)

    etf_weekly = screener_weekly.screen_market_weekly(
        market="ETF",
        top_n=settings.screening_etf_n,
        weeks=500,
        end_date=end_date_str,
    )
    results_weekly.extend(etf_weekly)

    all_results = []
    for r in results_daily:
        r["timeframe"] = "daily"
        all_results.append(r)

    for r in results_weekly:
        r["timeframe"] = "weekly"
        all_results.append(r)

    if not all_results or len(all_results) == 0:
        print("❌ 스크리닝 결과 없음")
        return

    results_df = pd.DataFrame(all_results)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM screening_results WHERE scan_date = ?", (today_str,))
    print(f"🗑️  기존 {today_str} 데이터 삭제 완료")

    for _, row in results_df.iterrows():
        try:
            zone_type = "지지" if row["OB_Type"] == "Bull" else "저항"
            zone_position = "내부" if "내부" in row["Status"] else "근접"

            is_recommended = 0
            if (
                row["RSI"] < 30
                and zone_type == "지지"
                and row["OB_Top"] > 0
                and row["OB_Bottom"] > 0
            ):
                ob_range_percent = ((row["OB_Top"] - row["OB_Bottom"]) / row["OB_Bottom"]) * 100

                if ob_range_percent < 10:
                    is_recommended = 1

            conn.execute(
                """
                INSERT INTO screening_results 
                (scan_date, market, code, name, current_price, change_percent, rsi, trading_value,
                 zone_type, zone_position, ob_top, ob_bottom, distance_percent, is_recommended,
                 timeframe)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    today_str,
                    row["Market"],
                    row["Code"],
                    row["Name"],
                    row["Current_Price"],
                    row["Change_Percent"],
                    row["RSI"],
                    row["trading_value"],
                    zone_type,
                    zone_position,
                    row["OB_Top"],
                    row["OB_Bottom"],
                    row["Distance_Percent"],
                    is_recommended,
                    row.get("timeframe", "daily"),
                ),
            )
        except Exception as e:
            print(f"❌ {row['Code']} 저장 오류: {e}")

    conn.commit()

    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT d.code, d.is_recommended
        FROM screening_results d
        JOIN screening_results w ON d.code = w.code AND d.scan_date = w.scan_date
        WHERE d.scan_date = ?
        AND d.timeframe = 'daily' AND w.timeframe = 'weekly'
        AND d.zone_type = '지지' AND w.zone_type = '지지'
    """,
        (today_str,),
    )

    dual_support_stocks = cursor.fetchall()

    if dual_support_stocks:
        for code, existing_flag in dual_support_stocks:
            new_flag = existing_flag | 2
            cursor.execute(
                """
                UPDATE screening_results 
                SET is_recommended = ? 
                WHERE scan_date = ? AND code = ? AND timeframe = 'daily'
            """,
                (new_flag, today_str, code),
            )
        conn.commit()
        print(f"🔥 {len(dual_support_stocks)}개 종목이 일봉+주봉 모두 지지 (추천 추가)")

    cursor.execute(
        "SELECT COUNT(*) FROM screening_results"
        ' WHERE scan_date = ? AND is_recommended > 0 AND timeframe = "daily"',
        (today_str,),
    )
    recommended_count = cursor.fetchone()[0]

    conn.close()

    print(f"✅ {len(results_df)}개 종목 저장 완료")
    print(f"⭐ {recommended_count}개 추천 종목")


def _resolve_investor_name(stock: Any, ticker: str, raw_name: Any) -> str:
    """수급 행의 종목명을 정한다 — 비었으면 티커로 조회하고, 그것도 안 되면 티커 자체.

    pandas는 CSV의 빈 칸을 NaN(float)으로 읽는데 NaN은 참으로 평가돼 `if not name`
    폴백을 건너뛰고 NOT NULL 제약에 걸려 그 행이 조용히 빠졌다 — 순위에 구멍이 났다(OBS-9).
    """
    if isinstance(raw_name, str) and raw_name.strip():
        return raw_name.strip()
    try:
        looked_up = stock.get_market_ticker_name(ticker)
    except Exception:
        looked_up = None
    if isinstance(looked_up, str) and looked_up.strip():
        return looked_up.strip()
    return str(ticker)


class InvestorDataError(RuntimeError):
    """수급(외국인·기관) 수집 실패(OBS-11).

    옛 코드는 실패를 `print`만 하고 정상 반환해서 `collect`가 「완료」·종료 코드 0으로 끝났고,
    두 달 동안 수집이 죽어 있는 것을 아무도 몰랐다. 이제 호출부까지 올린다.
    """


#: 수급 응답에 반드시 있어야 하는 열 — 로그인이 안 되면 pykrx가 이 열 없는 빈 표를 준다.
_NET_AMOUNT_COLUMN = "순매수거래대금"

_INVESTOR_CONFIGS = (
    ("외국인", "foreign"),
    ("기관합계", "institution"),  # '기관'이 아니라 '기관합계'
)


def _krx_login_problem() -> str | None:
    """KRX 로그인이 안 된 것으로 보이면 그 이유를 한 구절로 돌려준다(아니면 None).

    pykrx 1.2.x는 `KRX_ID`/`KRX_PW`로 로그인하고, 실패해도 예외 없이 안내문만 찍은 뒤 빈
    응답을 준다. 여기서는 **다시 로그인을 시도하지 않고** 이미 남은 상태만 읽는다.
    비밀 값(ID·비밀번호)은 읽어서 내보내지 않는다 — 설정 여부만 본다.
    """
    missing = [name for name in ("KRX_ID", "KRX_PW") if not os.environ.get(name)]
    if missing:
        return f"환경변수 {'·'.join(missing)} 없음"
    try:
        auth = importlib.import_module("pykrx.website.comm.auth")
    except Exception:
        return None
    if getattr(auth, "_auth_session", object()) is None:
        return "로그인 세션 없음(비밀번호 변경 요구 또는 자격 증명 오류 — pykrx 안내문 확인)"
    return None


def _investor_failure(date_str: str, reason: str) -> InvestorDataError:
    """실패 사유를 로그에 한 줄로 남기고(stderr) 올릴 예외를 만든다."""
    login = _krx_login_problem()
    if login is not None:
        print(f"❌ [KRX 로그인 실패] {date_str} 수급 수집 불가 — {login}", file=sys.stderr)
    message = f"{date_str} 수급 수집 실패: {reason}"
    print(f"❌ [수급 수집 실패] {message}", file=sys.stderr)
    return InvestorDataError(message)


def _fetch_investor_frames(stock: Any, date_str: str) -> dict[str, Any]:
    """네 분류(외국인·기관 × 순매수·순매도) 상위 100을 가져온다. 비면 `ValueError`."""
    results: dict[str, Any] = {}
    for investor_name, investor_type in _INVESTOR_CONFIGS:
        print(f"\n  📊 {investor_name} 데이터 수집 중...")
        frames = []
        for market in ("KOSPI", "KOSDAQ"):
            df = stock.get_market_net_purchases_of_equities(
                fromdate=date_str, todate=date_str, market=market, investor=investor_name
            )
            if df is None or df.empty or _NET_AMOUNT_COLUMN not in df.columns:
                raise ValueError(
                    f"{investor_name} {market} 응답이 비었거나 '{_NET_AMOUNT_COLUMN}' 열이 없음"
                    "(KRX 로그인 실패 또는 휴장일)"
                )
            df["시장"] = market
            frames.append(df)

        all_df = pd.concat(frames)
        buy_top = all_df.nlargest(100, _NET_AMOUNT_COLUMN)
        sell_top = all_df.nsmallest(100, _NET_AMOUNT_COLUMN)
        results[f"{investor_type}_buy"] = buy_top
        results[f"{investor_type}_sell"] = sell_top

        print(f"    ✅ {investor_name} 순매수 상위 {len(buy_top)}개")
        print(f"    ✅ {investor_name} 순매도 상위 {len(sell_top)}개")
    return results


def run_and_save_investor_data(target_date: str | None = None) -> None:
    """외국인/기관 매매 데이터 수집 및 저장.

    실패하면(pykrx 없음·로그인 실패·빈 응답·예외·저장 0건) `InvestorDataError`를 올린다.
    그날의 기존 행은 **새 데이터를 확보한 뒤** 같은 트랜잭션에서 지우고 다시 쓴다 — 가져오기에
    실패하면 기존 행이 그대로 남는다(OBS-11).
    """
    try:
        from pykrx import stock
    except ImportError as e:
        print(f"\n❌ pykrx import 실패: {e}")
        print(f"설치: {sys.executable} -m pip install pykrx")
        raise InvestorDataError(f"pykrx import 실패: {e}") from e

    if target_date:
        date_str = target_date
    else:
        date = datetime.now(KST).date()

        for i in range(7):
            check_date = date - timedelta(days=i)
            date_str = check_date.strftime("%Y%m%d")

            if check_date.weekday() >= 5:
                continue

            try:
                test = stock.get_market_net_purchases_of_equities(
                    fromdate=date_str, todate=date_str, market="KOSPI", investor="외국인"
                )
                if not test.empty:
                    print(f"✅ 최근 영업일: {check_date.strftime('%Y-%m-%d')}")
                    break
            except Exception:
                continue
        else:
            date_str = (datetime.now(KST).date() - timedelta(days=1)).strftime("%Y%m%d")

    print(f"\n📅 수집 날짜: {date_str}")

    # 1) 먼저 가져온다 — 여기서 실패하면 DB는 건드리지 않는다.
    print("\n🌍 외국인/기관 매매동향 수집 중...")
    try:
        results_data = _fetch_investor_frames(stock, date_str)
    except Exception as e:
        raise _investor_failure(date_str, str(e)) from e

    categories = [
        ("foreign", "buy", results_data["foreign_buy"]),
        ("foreign", "sell", results_data["foreign_sell"]),
        ("institution", "buy", results_data["institution_buy"]),
        ("institution", "sell", results_data["institution_sell"]),
    ]

    # 전체 종목 OHLCV 일괄 조회 (KOSPI + KOSDAQ 각 1회씩, 총 2회 호출)
    print(f"\n📊 {date_str} 기준 전체 종목 OHLCV 일괄 조회 중...")

    ohlcv_all = pd.DataFrame()
    for market in ["KOSPI", "KOSDAQ"]:
        try:
            market_df = stock.get_market_ohlcv(date_str, market=market)
            if market_df is not None and not market_df.empty:
                ohlcv_all = pd.concat([ohlcv_all, market_df])
                print(f"  ✅ {market}: {len(market_df)}개 종목")
            else:
                print(f"  ⚠️ {market}: 빈 응답 (휴장일 가능)")
        except Exception as e:
            print(f"  ⚠️ {market} OHLCV 조회 실패: {e}")

    # 2) 새 데이터를 손에 쥔 뒤에 지우고 쓴다 — 한 트랜잭션이라 중간에 실패하면 되돌아간다.
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investor_trading (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_date TEXT NOT NULL,
                investor_type TEXT NOT NULL,
                trade_type TEXT NOT NULL,
                rank INTEGER NOT NULL,
                code TEXT NOT NULL,
                name TEXT NOT NULL,
                market TEXT NOT NULL,
                current_price REAL,
                change_percent REAL,
                buy_amount INTEGER,
                sell_amount INTEGER,
                net_amount INTEGER,
                buy_volume INTEGER DEFAULT 0,
                sell_volume INTEGER DEFAULT 0,
                net_volume INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_investor_trading
            ON investor_trading(scan_date, investor_type, trade_type, rank)
        """)
        conn.commit()

        conn.execute("DELETE FROM investor_trading WHERE scan_date = ?", (date_str,))
        saved_count = 0

        for investor_type, trade_type, df in categories:
            if df is None or df.empty:
                continue

            print(f"\n💰 {investor_type}/{trade_type} 처리 중... ({len(df)}개 종목)")

            for rank, (ticker, row) in enumerate(df.iterrows(), 1):
                try:
                    name = _resolve_investor_name(stock, ticker, row.get("종목명"))

                    # dict lookup으로 즉시 조회 (API 호출 없음)
                    if ticker in ohlcv_all.index:
                        current = int(ohlcv_all.loc[ticker, "종가"])
                        change = float(ohlcv_all.loc[ticker, "등락률"])
                    else:
                        current = 0
                        change = 0

                    buy_amount = int(row.get("매수거래대금", 0))
                    sell_amount = int(row.get("매도거래대금", 0))
                    net_amount = int(row.get("순매수거래대금", 0))
                    buy_volume = int(row.get("매수거래량", 0))
                    sell_volume = int(row.get("매도거래량", 0))
                    net_volume = int(row.get("순매수거래량", 0))

                    conn.execute(
                        """
                        INSERT INTO investor_trading
                        (scan_date, investor_type, trade_type, rank, code, name, market,
                         current_price, change_percent, buy_amount, sell_amount, net_amount,
                         buy_volume, sell_volume, net_volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            date_str,
                            investor_type,
                            trade_type,
                            rank,
                            ticker,
                            name,
                            row.get("시장", "KOSPI"),
                            current,
                            change,
                            buy_amount,
                            sell_amount,
                            net_amount,
                            buy_volume,
                            sell_volume,
                            net_volume,
                        ),
                    )

                    saved_count += 1

                except Exception as e:
                    print(f"  ⚠️ {ticker} 저장 실패: {e}")
                    continue

        if saved_count == 0:
            conn.rollback()
            raise _investor_failure(date_str, "저장된 행이 0개(기존 행은 그대로 둠)")
        conn.commit()
    except InvestorDataError:
        raise
    except Exception as e:
        conn.rollback()
        raise _investor_failure(date_str, f"DB 저장 중 오류: {e}") from e
    finally:
        conn.close()

    print(f"\n✅ 총 {saved_count}개 종목 DB 저장 완료")


if __name__ == "__main__":
    run_and_save_screening()
