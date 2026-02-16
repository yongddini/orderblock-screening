#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
데이터 수집 통합 스크립트
- 오더블록 스크리닝
- 외국인/기관 매매 데이터
"""

import sys
from screening_core import run_and_save_screening, run_and_save_investor_data

def print_usage():
    """사용법 출력"""
    print("""
사용법:
    python3 collect_data.py [옵션] [날짜]

옵션:
    --all         전체 수집 (오더블록 + 외국인/기관) [기본값]
    --screening   오더블록 스크리닝만
    --investor    외국인/기관 데이터만
    -h, --help    도움말

날짜:
    YYYYMMDD 형식 (예: 20250212)
    생략시 오늘/최근 영업일 자동 선택

예시:
    python3 collect_data.py                    # 전체, 오늘
    python3 collect_data.py 20250212           # 전체, 12일
    python3 collect_data.py --screening        # 오더블록만, 오늘
    python3 collect_data.py --investor 20250212  # 외국인/기관만, 12일
    """)

def main():
    # 파라미터 파싱
    mode = 'all'  # 기본값: 전체
    target_date = None
    
    args = sys.argv[1:]
    
    for arg in args:
        if arg in ['-h', '--help']:
            print_usage()
            return
        elif arg == '--all':
            mode = 'all'
        elif arg == '--screening':
            mode = 'screening'
        elif arg == '--investor':
            mode = 'investor'
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
        print(f"📅 오늘/최근 영업일 데이터 수집")
    
    # 모드별 실행
    if mode == 'all':
        print("\n" + "="*60)
        print("1️⃣  오더블록 스크리닝")
        print("="*60)
        run_and_save_screening(target_date=target_date)
        
        print("\n" + "="*60)
        print("2️⃣  외국인/기관 매매 데이터")
        print("="*60)
        run_and_save_investor_data(target_date=target_date)
        
        print("\n" + "="*60)
        print("✅ 전체 수집 완료!")
        print("="*60)
    
    elif mode == 'screening':
        print("\n" + "="*60)
        print("📊 오더블록 스크리닝")
        print("="*60)
        run_and_save_screening(target_date=target_date)
        print("\n✅ 오더블록 스크리닝 완료!")
    
    elif mode == 'investor':
        print("\n" + "="*60)
        print("💰 외국인/기관 매매 데이터")
        print("="*60)
        run_and_save_investor_data(target_date=target_date)
        print("\n✅ 외국인/기관 데이터 수집 완료!")

if __name__ == '__main__':
    main()