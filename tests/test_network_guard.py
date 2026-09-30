"""네트워크 차단 자체가 동작하는지 확인(OBS-3 — 「네트워크 호출 0」의 근거)."""

from __future__ import annotations

import socket

import pytest


def test_tcp_connect_is_blocked() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="네트워크 호출 금지"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


def test_dns_is_blocked() -> None:
    with pytest.raises(RuntimeError, match="DNS 조회 금지"):
        socket.getaddrinfo("example.com", 443)
