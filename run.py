"""Entry point web aplikasi rekomendasi restok obat.

Jalankan:  python run.py
Pastikan .env terisi & `python seed.py` sudah dijalankan.
"""
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug, use_reloader=False)
