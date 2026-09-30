"""`python -m obscreen ...` = `obscreen ...`(uv가 없는 서버 cron용)."""

import sys

from obscreen.cli import main

sys.exit(main())
