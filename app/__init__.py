"""Application factory untuk web rekomendasi restok obat (Flask + MySQL)."""
from __future__ import annotations
import os
from pathlib import Path
from flask import Flask, render_template
from dotenv import load_dotenv

from .extensions import db

BASE_DIR = Path(__file__).resolve().parent.parent  # folder restock_forecasting/
load_dotenv(BASE_DIR / ".env")


def _database_uri() -> str:
    user = os.getenv("DB_USER", "root")
    pwd = os.getenv("DB_PASSWORD", "")
    host = os.getenv("DB_HOST", "127.0.0.1")
    port = os.getenv("DB_PORT", "3306")
    name = os.getenv("DB_NAME", "TugasAkhir")
    return f"mysql+pymysql://{user}:{pwd}@{host}:{port}/{name}?charset=utf8mb4"


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-secret")
    app.config["SQLALCHEMY_DATABASE_URI"] = _database_uri()
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"pool_pre_ping": True, "pool_recycle": 280}
    max_mb = int(os.getenv("MAX_UPLOAD_MB", "20"))
    app.config["MAX_CONTENT_LENGTH"] = max_mb * 1024 * 1024
    app.config["MAX_UPLOAD_MB"] = max_mb
    app.config["UPLOAD_DIR"] = str(BASE_DIR / "uploads")
    os.makedirs(app.config["UPLOAD_DIR"], exist_ok=True)

    db.init_app(app)

    # Jinja filter: parse JSON string -> object
    import json as _json

    @app.template_filter("from_json")
    def _from_json(s):
        try:
            return _json.loads(s) if s else {}
        except Exception:
            return {}

    # Blueprints
    from .blueprints.dashboard import bp as dashboard_bp
    from .blueprints.rekomendasi import bp as rekomendasi_bp
    from .blueprints.upload import bp as upload_bp
    from .blueprints.forecast import bp as forecast_bp
    from .blueprints.auth import bp as auth_bp, current_user
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(rekomendasi_bp)
    app.register_blueprint(forecast_bp)
    app.register_blueprint(upload_bp)

    # Proteksi global: semua halaman butuh login kecuali auth & static.
    from flask import session, redirect, url_for, request

    PUBLIC_ENDPOINTS = {"auth.login", "auth.logout", "static"}

    @app.before_request
    def _require_login():
        if request.endpoint in PUBLIC_ENDPOINTS:
            return None
        if not session.get("user_id"):
            return redirect(url_for("auth.login", next=request.path))
        return None

    # current_user tersedia di semua template
    @app.context_processor
    def _inject_user():
        return {"current_user": current_user()}

    # Error handlers
    @app.errorhandler(404)
    def _404(e):
        return render_template("error.html", code=404,
                               pesan="Halaman tidak ditemukan."), 404

    @app.errorhandler(500)
    def _500(e):
        return render_template("error.html", code=500,
                               pesan="Terjadi kesalahan pada server."), 500

    @app.errorhandler(413)
    def _413(e):
        return render_template("error.html", code=413,
                               pesan=f"File terlalu besar (maks {max_mb} MB)."), 413

    return app
