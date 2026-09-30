#!/usr/bin/env python3
"""
데이터 수집 통합 스크립트
- 오더블록 스크리닝
- 외국인/기관 매매 데이터
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from obscreen.data.provider import MarketCapUnavailableError
from obscreen.screening.core import run_and_save_investor_data, run_and_save_screening


def print_usage() -> None:
    """사용법 출력"""
    print("""
사용법:
    obscreen collect [옵션] [날짜]      (서버: python3 -m obscreen collect ...)

옵션:
    --all         전체 수집 (오더블록 + 외국인/기관) [기본값]
    --screening   오더블록 스크리닝만
    --investor    외국인/기관 데이터만
    -h, --help    도움말

날짜:
    YYYYMMDD 형식 (예: 20250212)
    생략시 오늘/최근 영업일 자동 선택

예시:
    obscreen collect                    # 전체, 오늘
    obscreen collect 20250212           # 전체, 12일
    obscreen collect --screening        # 오더블록만, 오늘
    obscreen collect --investor 20250212  # 외국인/기관만, 12일
    """)


def main(argv: Sequence[str] | None = None) -> None:
    """`obscreen collect` 본체. `argv`가 없으면 `sys.argv[1:]`을 읽는다."""
    # 파라미터 파싱
    mode = "all"  # 기본값: 전체
    target_date = None

    args = list(sys.argv[1:] if argv is None else argv)

    for arg in args:
        if arg in ["-h", "--help"]:
            print_usage()
            return
        elif arg == "--all":
            mode = "all"
        elif arg == "--screening":
            mode = "screening"
        elif arg == "--investor":
            mode = "investor"
        elif arg.isdigit() and len(arg) == 8:
            target_date = arg
        else:
            print(f"❌ 알 수 없는 옵션: {arg}")
            print_usage()
            return

    # 날짜 정보 출력
    if target_date:
        print(f"📅 지정된 날짜: {target_date}")
    else:
        print("📅 오늘/최근 영업일 데이터 수집")

    # 모드별 실행
    if mode == "all":
        print("\n" + "=" * 60)
        print("1️⃣  오더블록 스크리닝")
        print("=" * 60)
        # 시가총액 순위를 못 만들면 스크리닝만 멈추고 수급 수집은 계속한다(OBS-7 §5).
        # 끝에 종료 코드 1로 실패를 알린다(cron 로그·감시가 보도록).
        screening_error = None
        try:
            run_and_save_screening(target_date=target_date)
        except MarketCapUnavailableError as e:
            screening_error = e
            print(f"❌ 오더블록 스크리닝 중단: {e}", file=sys.stderr)

        print("\n" + "=" * 60)
        print("2️⃣  외국인/기관 매매 데이터")
        print("=" * 60)
        run_and_save_investor_data(target_date=target_date)

        if screening_error is not None:
            print("\n" + "=" * 60)
            print("❌ 수급 수집은 끝났지만 오더블록 스크리닝이 중단됐습니다.")
            print("=" * 60)
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✅ 전체 수집 완료!")
        print("=" * 60)

    elif mode == "screening":
        print("\n" + "=" * 60)
        print("📊 오더블록 스크리닝")
        print("=" * 60)
        run_and_save_screening(target_date=target_date)
        print("\n✅ 오더블록 스크리닝 완료!")

    elif mode == "investor":
        print("\n" + "=" * 60)
        print("💰 외국인/기관 매매 데이터")
        print("=" * 60)
        run_and_save_investor_data(target_date=target_date)
        print("\n✅ 외국인/기관 데이터 수집 완료!")


if __name__ == "__main__":
    main()
