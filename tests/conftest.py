"""Fixtures pytest: aplikasi Flask + test client (anonim & sudah login).

Tes ini bersifat integrasi (memakai DB MySQL aktif & artefak model). Pastikan
`py -3 seed_users.py` sudah dijalankan agar akun apoteker tersedia.
"""
import os
import sys
import pytest

# pastikan root proyek di sys.path saat pytest dijalankan dari mana pun
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import create_app  # noqa: E402

USER = {"username": "apoteker", "password": "apoteker123"}


@pytest.fixture(scope="session")
def app():
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture
def client(app):
    """Client anonim (belum login)."""
    return app.test_client()


@pytest.fixture
def auth(app):
    """Client yang sudah login sebagai apoteker."""
    c = app.test_client()
    c.post("/login", data=USER, follow_redirects=False)
    return c
