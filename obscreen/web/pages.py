"""화면 라우트 — 메인 · 수급 페이지 · 헬스체크(OBS-4: `app_production.py`에서 옮김)."""

from __future__ import annotations

from flask import Blueprint, jsonify, render_template
from flask.typing import ResponseReturnValue

bp = Blueprint("pages", __name__)


@bp.route("/")
def index() -> ResponseReturnValue:
    return render_template("index.html")


@bp.route("/health")
def health() -> ResponseReturnValue:
    """Health check endpoint"""
    return jsonify({"status": "ok"})
