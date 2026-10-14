# OBS-13 — `daily_screening.sh` 실행 권한 · 실패 종료 코드 (서버 런북)

## 무슨 일이 있었나

- 2026-09-30 서버 폴더(`/home/rocky/orderblock`)를 git 체크아웃으로 바꿨는데, 저장소의
  `daily_screening.sh`가 mode `100644`로 커밋돼 있어 실행 권한 없이 내려왔다.
- crontab은 스크립트를 직접 실행하므로 매일 20:05에
  `/bin/sh: line 1: /home/rocky/orderblock/daily_screening.sh: Permission denied`로 시작조차 못 했다.
  `screening_results`의 마지막 날짜가 `20260929`에서 멈췄다.

## 이 변경이 하는 일

1. 실행 비트를 **저장소에** 커밋했다(`100755`) — `git pull`이 권한을 유지한다.
2. 수집(`python3 -m obscreen collect`)이 실패하면 스크립트가 **종료 코드 1**로 끝나고 stderr에
   한 줄을 낸다(로그의 `❌ 데이터 수집 실패 (종료 코드 N)`도 그대로). 성공은 0.
3. `tests/test_structure.py`가 git 인덱스 모드(`100755`)와 실패 종료 코드를 건다.

## 서버 반영 절차 (사용자 수행 — 이 저장소는 서버에 접속하지 않는다)

```bash
cd /home/rocky/orderblock
git status --short            # 이미 손으로 chmod +x 했다면 daily_screening.sh가 「M」으로 뜬다
git checkout -- daily_screening.sh   # 로컬 모드 변경만 되돌린다(내용 변경이 없을 때)
git pull
git ls-files -s daily_screening.sh   # 100755 로 시작해야 한다
ls -l daily_screening.sh             # -rwxr-xr-x
./daily_screening.sh; echo "exit=$?" # 1회 수동 실행 — 0이면 성공
tail -n 20 logs/screening.log
```

- 손으로 `chmod +x`만 해 둔 상태에서 바로 `git pull`을 하면 「local changes would be overwritten」으로
  막힐 수 있다. 위의 `git checkout -- daily_screening.sh`가 그것을 푼다(pull 뒤에는 저장소가 실행
  비트를 준다).

## 대안 — crontab을 `bash`로 부르기 (적용은 사용자 결정)

crontab 줄을 아래처럼 두면 파일 권한과 무관하게 돈다:

```
5 20 * * * bash /home/rocky/orderblock/daily_screening.sh
```

- 장점: 실행 비트가 다시 빠져도(다른 복사 경로·파일시스템) 돈다.
- 실행 비트 커밋과 함께 써도 된다(이중 안전장치). **이 PR은 crontab을 바꾸지 않는다.**

## 실패가 보이게 하려면

- 스크립트는 이제 실패 시 stderr에 한 줄을 내고 1로 끝난다. cron은 출력이 있으면 `MAILTO`로
  메일을 보내므로, crontab 맨 위에 `MAILTO=<주소>`를 두면 실패한 날만 메일이 온다(메일 전송 설정이
  된 서버에서). ⚠️ `Permission denied`처럼 **스크립트가 시작조차 못 하는** 실패는 스크립트가 알릴 수
  없다 — 그 부류는 cron 로그(`journalctl -u crond` / `/var/log/cron`)에만 남는다.
- systemd timer 전환·알림 연동은 OBS-6 소관이다.

## 범위 밖

- 놓친 날짜(9/30~10/13) 소급 스크리닝 — 지난 날짜를 넣었을 때 그날까지의 데이터만 쓰는지 확인되지
  않았다. 필요하면 별도 이슈.
