"""Autentikasi berbasis sesi: login, logout, dan proteksi halaman.

Sederhana & tanpa dependensi tambahan: password di-hash (werkzeug), state
login disimpan di session Flask. `login_required` melindungi route; `current_user`
disuntikkan ke template via context processor (lihat app/__init__.py).
"""
from __future__ import annotations
from datetime import datetime
from functools import wraps

from flask import (Blueprint, render_template, request, redirect, url_for,
                   session, flash)

from ..extensions import db
from ..models import User

bp = Blueprint("auth", __name__)


def login_required(view):
    """Dekorator: tolak akses bila belum login -> arahkan ke halaman login."""
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Silakan login terlebih dahulu.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def current_user() -> User | None:
    uid = session.get("user_id")
    return db.session.get(User, uid) if uid else None


@bp.route("/login", methods=["GET", "POST"])
def login():
    # sudah login -> langsung ke dashboard
    if session.get("user_id"):
        return redirect(url_for("dashboard.index"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        user = User.query.filter_by(username=username).first()
        if user and user.is_active and user.check_password(password):
            session.clear()
            session["user_id"] = user.id
            session["username"] = user.username
            session["role"] = user.role
            user.last_login = datetime.utcnow()
            db.session.commit()
            flash(f"Selamat datang, {user.nama or user.username}!", "success")
            nxt = request.args.get("next") or request.form.get("next")
            if nxt and nxt.startswith("/"):
                return redirect(nxt)
            return redirect(url_for("dashboard.index"))
        flash("Username atau password salah.", "danger")

    return render_template("login.html", next=request.args.get("next", ""))


@bp.route("/logout")
def logout():
    session.clear()
    flash("Anda telah keluar.", "success")
    return redirect(url_for("auth.login"))
