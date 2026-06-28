"""Seed akun login awal (satu peran: apoteker).

Membuat tabel users (bila belum ada) dan menambahkan akun default:
  - apoteker / apoteker123  (role apoteker)

Jalankan: py -3 seed_users.py
Akun yang sudah ada TIDAK ditimpa. Akun lama bawaan (admin/staff) dibersihkan.
"""
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")  # aman: tak menutup buffer saat dibungkus berlapis
except (AttributeError, ValueError):
    pass

from app import create_app
from app.extensions import db
from app.models import User

DEFAULTS = [
    {"username": "apoteker", "password": "apoteker123", "nama": "Apoteker", "role": "apoteker"},
]
# akun contoh lama yang tidak dipakai lagi
OBSOLETE = ["admin", "staff"]


def main():
    app = create_app()
    with app.app_context():
        db.create_all()
        # bersihkan akun bawaan lama
        for uname in OBSOLETE:
            old = User.query.filter_by(username=uname).first()
            if old:
                db.session.delete(old)
                print(f"x {uname}: dihapus (akun lama).")
        for d in DEFAULTS:
            if User.query.filter_by(username=d["username"]).first():
                print(f"- {d['username']}: sudah ada, dilewati.")
                continue
            u = User(username=d["username"], nama=d["nama"], role=d["role"])
            u.set_password(d["password"])
            db.session.add(u)
            print(f"+ {d['username']} ({d['role']}) dibuat, password: {d['password']}")
        db.session.commit()
        print("Selesai. Total user:", User.query.count())


if __name__ == "__main__":
    main()
