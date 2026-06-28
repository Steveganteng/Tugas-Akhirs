"""Setup otomatis aplikasi web di perangkat baru.

Langkah berurutan dengan pesan error JELAS:
  1. Pastikan file .env ada (salin dari .env.example bila belum).
  2. Pastikan dependensi Python terpasang (pip install -r requirements.txt).
  3. Uji koneksi MySQL + buat database bila belum ada.
  4. Seed data (panel, model aktif, rekomendasi) + akun login.
  5. Verifikasi akhir & tampilkan cara menjalankan.

Jalankan:  python setup.py            (setup biasa)
           python setup.py --reset    (drop & buat ulang tabel + data)

Skrip ini AMAN diulang: langkah yang sudah beres akan dilewati.
"""
import sys
import os
import shutil
import subprocess
from pathlib import Path

# Catatan: skrip seed.py & pipeline_forecasting_obat.py me-reassign sys.stdout
# (TextIOWrapper). Karena itu seed dijalankan sebagai SUBPROCESS (lihat run_seed)
# agar tidak terjadi penumpukan wrapper yang menutup buffer stdout.

BASE = Path(__file__).resolve().parent
RESET = "--reset" in sys.argv


def head(n, msg):
    print(f"\n[{n}/5] {msg}")


def ok(msg):
    print("   [OK]", msg)


def warn(msg):
    print("   [!] ", msg)


def die(msg, *hints):
    print("\n   [GAGAL]", msg)
    for h in hints:
        print("      ->", h)
    print("\nSetup dihentikan. Perbaiki masalah di atas lalu jalankan ulang: python setup.py")
    sys.exit(1)


# ---------------------------------------------------------------- 1. .env
def ensure_env():
    head(1, "Memeriksa file konfigurasi .env")
    env, ex = BASE / ".env", BASE / ".env.example"
    if env.exists():
        ok(".env ditemukan")
    elif ex.exists():
        shutil.copy(ex, env)
        warn(".env belum ada -> disalin dari .env.example.")
        warn("Buka file .env dan sesuaikan DB_USER / DB_PASSWORD / DB_HOST / DB_PORT")
        warn("dengan MySQL di PERANGKAT INI bila berbeda (mis. password root / port 3307).")
    else:
        die("Tidak ada .env maupun .env.example.",
            "Pastikan Anda menjalankan skrip dari dalam folder proyek.")
    from dotenv import load_dotenv
    load_dotenv(env, override=True)


# ---------------------------------------------------------------- 2. deps
def ensure_deps():
    head(2, "Memeriksa dependensi Python")
    need = {"flask": "flask", "flask_sqlalchemy": "flask_sqlalchemy",
            "pymysql": "pymysql", "pandas": "pandas", "numpy": "numpy",
            "sklearn": "scikit-learn", "statsmodels": "statsmodels",
            "pyarrow": "pyarrow", "dotenv": "python-dotenv",
            "joblib": "joblib", "openpyxl": "openpyxl"}
    missing = [pkg for mod, pkg in need.items() if not _has(mod)]
    if not missing:
        ok("semua dependensi terpasang")
        return
    warn(f"paket belum lengkap: {', '.join(missing)} -> menjalankan pip install ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-r",
                        str(BASE / "requirements.txt")])
    if r.returncode != 0:
        die("pip install gagal.",
            "Jalankan manual: pip install -r requirements.txt")
    ok("dependensi terpasang")


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------- 3. MySQL + DB
def _cfg():
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("DB_PORT", "3306")),
        "user": os.getenv("DB_USER", "root"),
        "password": os.getenv("DB_PASSWORD", ""),
        "name": os.getenv("DB_NAME", "TugasAkhir"),
    }


def ensure_database():
    head(3, "Menguji koneksi MySQL & menyiapkan database")
    import pymysql
    c = _cfg()
    print(f"   target: {c['user']}@{c['host']}:{c['port']} (db '{c['name']}')")
    try:
        conn = pymysql.connect(host=c["host"], port=c["port"],
                               user=c["user"], password=c["password"],
                               connect_timeout=8)
    except pymysql.err.OperationalError as e:
        code = e.args[0] if e.args else None
        if code in (2003, 2002):
            die(f"Tidak bisa menghubungi server MySQL di {c['host']}:{c['port']}.",
                "Pastikan layanan MySQL menyala.",
                "Cek port: XAMPP/MariaDB sering di 3307, MySQL standar 3306.",
                "Sesuaikan DB_HOST / DB_PORT di .env.")
        elif code == 1045:
            die(f"Akses ditolak untuk user '{c['user']}' (kemungkinan password salah).",
                "Sesuaikan DB_USER / DB_PASSWORD di .env dengan kredensial MySQL perangkat ini.",
                "Catatan: password kosong hanya berlaku bila akun MySQL memang tanpa password.")
        else:
            die(f"Gagal konek MySQL (kode {code}): {e}")
    except Exception as e:
        die(f"Gagal konek MySQL: {e}")
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE DATABASE IF NOT EXISTS `{c['name']}` "
            "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit()
    conn.close()
    ok(f"server MySQL terhubung; database '{c['name']}' siap")


# ---------------------------------------------------------------- 4. seed
def run_seed():
    head(4, "Mengisi data & akun (seed)")
    # Dijalankan sebagai subprocess (interpreter terpisah) agar reassignment
    # sys.stdout di seed.py tidak mengganggu proses setup ini.
    seed_cmd = [sys.executable, str(BASE / "seed.py")] + (["--reset"] if RESET else [])
    if subprocess.run(seed_cmd, cwd=str(BASE)).returncode != 0:
        die("seed.py gagal (lihat pesan di atas).",
            "Penyebab umum: folder data/ (panel_bulanan.parquet) atau models/ "
            "tidak ikut tersalin, atau koneksi DB terputus.")
    if subprocess.run([sys.executable, str(BASE / "seed_users.py")],
                      cwd=str(BASE)).returncode != 0:
        die("seed_users.py gagal (lihat pesan di atas).")
    ok("data & akun ter-seed")


# ---------------------------------------------------------------- 5. verifikasi
def verify():
    head(5, "Verifikasi akhir")
    from app import create_app
    from app.models import PanelBulanan, Rekomendasi, ModelRegistry, User
    app = create_app()
    with app.app_context():
        p = PanelBulanan.query.count()
        r = Rekomendasi.query.count()
        u = User.query.count()
        act = ModelRegistry.query.filter_by(status="active").first()
    print(f"   panel_bulanan={p}  rekomendasi={r}  users={u}  "
          f"model_aktif={act.nama_model if act else None}")
    if p == 0 or act is None or u == 0:
        die("Verifikasi gagal: data belum lengkap.",
            "Coba jalankan ulang dengan reset: python setup.py --reset")
    ok("verifikasi sukses")


def main():
    # line-buffering: jaga urutan output skrip ini vs subprocess seed bila di-log
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    print("=" * 64)
    print("  SETUP — Web Forecasting & Rekomendasi Restok Obat")
    print("=" * 64)
    ensure_env()
    ensure_deps()
    ensure_database()
    run_seed()
    verify()
    print("\n" + "=" * 64)
    print("  SELESAI. Jalankan aplikasi dengan:  python run.py")
    print("  Login default:  apoteker / apoteker123")
    print("=" * 64)


if __name__ == "__main__":
    main()
