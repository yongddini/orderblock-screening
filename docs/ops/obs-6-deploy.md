# OBS-6 — 서버 배포·운영 런북

서버: `/home/rocky/orderblock`(git 체크아웃 · `origin/main`). 웹은 nginx(기존 도메인·TLS) 뒤의
gunicorn, 매일 수집은 `daily_screening.sh`. **이 저장소는 서버에 접속하지 않는다** — 아래는 전부
사용자가 서버에서 실행한다.

한눈에:

| 하고 싶은 것 | 명령 |
| -- | -- |
| 현황 파악(읽기 전용) | `./scripts/server-inventory.sh > inventory.txt` |
| 최신 main 배포 | `./scripts/deploy.sh` |
| 코드 롤백 | `./scripts/deploy.sh --ref <커밋>` |
| /health 확인 | `./scripts/health-check.sh` |
| DB 백업 | `./scripts/db-backup.sh` |
| 상태·실패 | `systemctl status orderblock` · `systemctl --failed` · `systemctl list-timers obscreen-collect.timer` |
| 수집 로그 | `tail -f logs/screening.log` · `journalctl -u obscreen-collect -n 50` |
| 웹 로그 | `tail -f logs/web.log logs/access.log` |

순서(처음 한 번): §0 → §1 기록 확인 → §2 → §3 → §4 → §5. 그 뒤로는 §6 한 줄.

---

## §0 현황 파악 (읽기 전용 — 먼저 돌려 결과를 이슈에 붙인다)

```bash
cd /home/rocky/orderblock
git pull --ff-only            # 이 스크립트들을 받기 위해 한 번(§1 상태면 깨끗이 된다)
./scripts/server-inventory.sh --fetch > ~/inventory-$(date +%Y%m%d).txt
less ~/inventory-*.txt
```

- 출력: 저장소 ↔ 서버 폴더 차이(손으로 고친 추적 파일 · 추적 안 되는 파일) · 파이썬/패키지 버전
  (venv310 · .venv · `uv.lock` 비교) · gunicorn 프로세스·포트 · `orderblock.service` 내용 ·
  nginx 설정(도메인·`proxy_pass`·인증서 만료 — **읽기만**) · crontab · DB 마지막 날짜 · 로그 · `.env`
  이름 · 디스크.
- 비밀 값(`KRX_PW`·`…KEY`·`…TOKEN`·`…_ID` 등)은 이름만 남기고 `<가림>`으로 바뀐다. **붙이기 전에 한
  번 더 눈으로 확인한다.**
- nginx 설정 파일을 읽을 권한이 없으면 「읽을 수 없음」으로 나온다 — sudo 는 쓰지 않는다.
- 확인할 것: `orderblock.service` 의 `--bind`(gunicorn 주소) · nginx `proxy_pass` 가 같은 주소인지 ·
  venv310 의 pandas·pykrx 버전이 `uv.lock` 과 같은지.

## §1 첫 전환 기록 — 구 폴더 → git 체크아웃 (2026-09-30, 이미 끝남)

새로 할 일이 아니라 **한 일의 기록**이다(같은 일을 다른 서버에 다시 할 때 참고).

1. 손으로 올리던 폴더를 `.bak` 으로 보존하고 `origin/main` 을 clone 해 같은 경로에 두었다.
   DB(`*.db`)·로그·`venv310/` 은 옮겨 왔다(전부 `.gitignore` 대상이라 git 이 건드리지 않는다).
2. 서버 전용 파일(`dart_provider.py`·`dart_cache.py`·`migrate_db.py`·`docs_cache/`·`*.bak`·
   `check_pykrx_*`, 폴더 안 `orderblock.service`·`nginx_orderblock.conf` 사본)은 사용자 결정으로 삭제.
   실제 설정은 `/etc/systemd/system/orderblock.service` · `/etc/nginx/conf.d/orderblock.conf`(실파일).
3. ⚠️ **사고**: 저장소의 `daily_screening.sh` 가 실행 비트 없이(644) 내려와 cron 이 9/30~10/13 동안
   `Permission denied` 로 멈췄다 → OBS-13 이 실행 비트를 커밋했고, 이 런북의 systemd 수집 유닛은
   `/bin/bash daily_screening.sh` 로 불러 **실행 비트에 기대지 않는다**.

교훈: 체크아웃을 바꾼 직후에는 `git ls-files -s daily_screening.sh`(100755) · `ls -l` · 수동 1회
실행을 확인한다.

## §2 uv 설치 · 파이썬 3.11 (.venv)

서버 venv310 은 파이썬 **3.10**, 저장소는 `requires-python >=3.11`. uv 가 3.11 을 받아 `.venv` 를
만든다(시스템 파이썬은 건드리지 않는다).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # ~/.local/bin/uv
exec $SHELL -l                                      # PATH 반영
cd /home/rocky/orderblock
uv sync --frozen --no-dev                           # .venv 생성 — uv.lock 그대로(3.11 자동 설치)
.venv/bin/python --version                          # 3.11.x
.venv/bin/python -c "import pandas, pykrx, flask; print(pandas.__version__, pykrx.__version__)"
```

- 이 단계는 **아무것도 바꾸지 않는다** — 웹·cron 은 아직 venv310 으로 돈다. `.venv` 는 §5 에서 쓴다.
- `venv310/` 은 지우지 않는다 — §7 롤백 경로다. 한 달 안정 뒤 지운다(별도 결정).

## §3 `.env` — 경로·비밀을 한 곳으로 (crontab 평문 비밀 제거)

지금 crontab 맨 위에 `KRX_ID`·`KRX_PW`·`DART_API_KEY` 가 **평문**으로 있다. systemd 유닛은
`.env` 를 `EnvironmentFile` 로 읽으므로 그 값들을 `.env` 로 옮긴다.

```bash
cd /home/rocky/orderblock
cp -n .env.example .env
chmod 600 .env
crontab -l > ~/crontab.backup.$(date +%Y%m%d)       # 백업(비밀 포함 — 권한 확인, 다 쓰면 지운다)
chmod 600 ~/crontab.backup.*
vi .env      # KRX_ID= · KRX_PW= · DART_API_KEY= 를 crontab 값으로 채운다
```

- 형식은 `이름=값` 한 줄(따옴표·`export` 없이). 값에 공백·`#`·따옴표가 있으면 작은따옴표로 감싼다.
- `systemctl show` 에는 `EnvironmentFile` **경로만** 나오고 값은 안 나온다.
- 비밀번호 변경·키 재발급 권고(평문으로 crontab 에 있었다)는 그대로 유효하다.

## §4 구 cron 제거 — 이중 실행 방지 (§5 전에)

timer 와 cron 이 둘 다 있으면 20:05 에 수집이 **두 번 동시에** 돈다(같은 날짜 행을 지우고 쓰는
경합). `install-systemd.sh` 는 crontab 에 `daily_screening.sh` 줄이 남아 있으면 **타이머를 켜지 않고
멈춘다**.

```bash
crontab -e
#  1) `... daily_screening.sh` 줄을 지운다(또는 맨 앞에 # — 주석 줄은 검사가 무시한다)
#  2) §3 에서 .env 로 옮긴 KRX_ID·KRX_PW·DART_API_KEY 줄을 지운다
#  ⚠️ AlphaBlock 작업(00:30 등) 줄은 그대로 둔다
crontab -l | grep -n daily_screening.sh || echo "cron 줄 없음 — OK"
```

## §5 systemd 전환 (웹 = .venv gunicorn · 매일 수집 = timer · 로그 회전)

```bash
cd /home/rocky/orderblock
./scripts/install-systemd.sh --dry-run    # 설치될 유닛 + 기존 orderblock.service 와의 차이를 본다
./scripts/install-systemd.sh              # 설치 → 웹 재시작 → /health → 타이머 켜기
```

- **바인드는 기존 유닛 값 그대로**(`--bind`/`-b` 를 읽어 옮긴다) — nginx 설정·도메인·TLS 는 손대지
  않는다. 못 찾으면 멈추고 `--bind 127.0.0.1:XXXX` 로 직접 주라고 알린다(§0 출력 참고).
  워커 수·`User=`·`Group=` 도 기존 유닛에서 가져온다.
- 기존 유닛의 `Environment=` 줄은 새 유닛으로 **옮기지 않는다**(경고가 뜬다) → §3 의 `.env` 로.
- 기존 유닛은 덮어쓰기 전에 `backups/systemd/orderblock.service.<시각>` 으로 복사된다(§7).
- 설치되는 것:
  - `/etc/systemd/system/orderblock.service` — `.venv/bin/gunicorn … app_production:app`,
    `EnvironmentFile=-<저장소>/.env`, 로그 `logs/web.log`·`logs/access.log`.
  - `/etc/systemd/system/obscreen-collect.service` — `ExecStart=/bin/bash <저장소>/daily_screening.sh`
    (oneshot · 수집 파이썬 = `.venv/bin/python`). 수집이 실패하면 종료 코드 1(OBS-11/13) → 유닛이
    **failed** 가 되어 `systemctl --failed` 에 보인다.
  - `/etc/systemd/system/obscreen-collect.timer` — 매일 **20:05 KST**(`Asia/Seoul` 고정 — 서버
    시간대와 무관). 꺼져 있던 날을 부팅 때 몰아서 돌리지 않는다(`Persistent=false`).
  - `/etc/logrotate.d/obscreen` — `logs/*.log` 일자별 · 30일 · 압축 · `copytruncate`.
- 확인:

```bash
systemctl status orderblock --no-pager
systemctl list-timers obscreen-collect.timer --no-pager     # 다음 실행 20:05
sudo systemctl start obscreen-collect.service               # (선택) 지금 한 번 수집 — 몇십 분 걸린다
systemctl status obscreen-collect.service --no-pager        # Result: success / failed
tail -n 30 logs/screening.log
sudo logrotate -d /etc/logrotate.d/obscreen                 # 회전 설정 점검(드라이런)
```

- 도메인에서 화면이 뜨는지 브라우저로 확인한다(`https://<도메인>/health` → `{"status": "ok"}`).

## §6 일상 배포 — 한 줄

```bash
cd /home/rocky/orderblock && ./scripts/deploy.sh
```

순서: 작업 트리 확인(서버에서 손으로 고친 추적 파일이 있으면 멈춤) → `git fetch` →
`origin/main` fast-forward → `uv sync --frozen --no-dev` → `sudo systemctl restart orderblock` →
`/health`(15회 × 2초). 어느 단계든 실패하면 **거기서 멈추고** 롤백 명령을 출력한다. 기록은
`logs/deploy.log`(`OK 이전커밋 -> 새커밋` / `FAIL …`).

- `uv sync` 가 실패하면 재시작 전에 멈춘다 — 웹은 옛 프로세스로 계속 돈다.
- 아직 §5 전(유닛이 venv310)이면 경고가 뜬다: 새 의존성이 `.venv` 에만 깔리므로 §5 를 하거나,
  전환 전이면 `venv310/bin/pip install -r requirements.txt` 가 필요할 수 있다.
- 도메인 확인은 `OBSCREEN_HEALTH_URL=https://<도메인>/health ./scripts/health-check.sh`.
- 수집 timer 는 재시작하지 않아도 다음 20:05 에 새 코드로 돈다(매번 새 프로세스).
- 스키마가 바뀌는 PR 이면 먼저 `./scripts/db-backup.sh`.

## §7 롤백

**(a) 코드만 되돌리기** — 배포 후 화면이 깨졌을 때:

```bash
tail -n 5 logs/deploy.log                 # 직전 OK 줄의 「이전 커밋」
./scripts/deploy.sh --ref <이전 커밋>      # 그 커밋으로 이동(detached) → uv sync → 재시작 → /health
```

다음 일반 `./scripts/deploy.sh` 가 다시 `main` 으로 돌아온다(고친 PR 이 머지된 뒤 실행).

**(b) systemd 전환(§5) 자체를 되돌리기** — `.venv` 로 웹이 안 뜰 때:

```bash
ls backups/systemd/                                         # 전환 전 유닛 백업
sudo systemctl disable --now obscreen-collect.timer
sudo cp backups/systemd/orderblock.service.<시각> /etc/systemd/system/orderblock.service
sudo systemctl daemon-reload && sudo systemctl restart orderblock
./scripts/health-check.sh
crontab -e     # ~/crontab.backup.<날짜> 의 daily_screening.sh 줄(과 필요하면 KRX 줄)을 되살린다
```

`venv310/` 이 남아 있으므로 옛 유닛은 그대로 뜬다. `/etc/logrotate.d/obscreen` 은 남겨 둬도 무해하다.

## §8 DB 백업 · 복원

```bash
./scripts/db-backup.sh            # backups/db/<이름>-<시각>.db · 최근 14개 유지(--keep N)
```

- SQLite 온라인 백업 API 로 복사해 **웹·수집이 돌고 있어도** 일관된 사본이 나온다(`cp` 는 쓰기 중이면
  깨진 사본이 나올 수 있다). 사본을 `PRAGMA integrity_check` 로 검사해 `ok` 일 때만 최종 이름을 붙이고,
  아니면 `.FAILED` 로 남기고 종료 코드 1.
- 복원:

```bash
sudo systemctl stop orderblock obscreen-collect.timer
cp orderblock_screening.db orderblock_screening.db.before-restore
cp backups/db/<백업>.db orderblock_screening.db
sudo systemctl start orderblock obscreen-collect.timer
./scripts/health-check.sh
```

- 정기 백업이 필요하면 별도 timer 로 붙인다(지금은 수동 — 결정은 사용자).

## 범위 밖

도메인·DNS·TLS(nginx 설정) 변경 · 화면 디자인 · 탐지 로직. 수집 실패 알림(메일·텔레그램) 연동은
별도 이슈 — 지금은 `systemctl --failed` 와 `journalctl -u obscreen-collect` 로 본다.
