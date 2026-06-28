"""Seed database MySQL TugasAkhir.

  1. Buat semua tabel (create_all).
  2. Muat panel_bulanan.parquet -> tabel panel_bulanan.
  3. Daftarkan model dari best_model.json sebagai versi 1 (active) di model_registry.
  4. Generate tabel rekomendasi dari model aktif untuk bulan prediksi default.

Jalankan: python seed.py   (opsi --reset untuk drop & buat ulang)
"""
import sys, json
try:
    sys.stdout.reconfigure(encoding="utf-8")  # aman: tak menutup buffer saat dibungkus berlapis
except (AttributeError, ValueError):
    pass
import warnings; warnings.filterwarnings("ignore")

import pandas as pd
import config as C
from app import create_app
from app.extensions import db
from app.models import PanelBulanan, ModelRegistry, Rekomendasi


def seed(reset=False):
    app = create_app()
    with app.app_context():
        if reset:
            print("Drop semua tabel...")
            db.drop_all()
        db.create_all()
        print("Tabel siap.")

        # ---- panel_bulanan ----
        if PanelBulanan.query.first() is None:
            panel = pd.read_parquet(C.PANEL_PATH)
            print(f"Muat panel: {len(panel)} baris.")
            for _, r in panel.iterrows():
                db.session.add(PanelBulanan(
                    obat=r[C.P_OBAT], periode=str(r[C.P_PERIOD]),
                    demand=float(r[C.P_DEMAND]), stok=float(r[C.P_STOCK]),
                    satuan=(r.get("satuan") if "satuan" in panel.columns else None),
                    is_observasi=bool(r[C.P_IS_OBS]), sumber="init"))
            db.session.commit()
            print("panel_bulanan terisi.")
        else:
            print("panel_bulanan sudah ada, lewati.")

        # ---- model_registry: versi 1 active ----
        # Model DEFAULT/AKTIF = Restock FINAL (terbaik): SES/Croston/SBA +
        # mean/ensemble + musiman + safety stock distribusi semua kategori
        # (fill rate ~95-96%). Ubah ACTIVE_MODEL bila ingin model lain.
        ACTIVE_MODEL = "restock_final"
        if ModelRegistry.query.first() is None:
            try:
                metrics = json.load(open(C.MODELS_DIR / "metrics.json", encoding="utf-8"))
            except Exception:
                metrics = {"metrics": []}
            name_map = {
                "naive": "Naive (lag-1)", "random_forest": "Random Forest",
                "gradient_boosting": "Gradient Boosting", "holt_winters": "Holt-Winters",
                "sarima": "SARIMA", "ensemble_simple": "Ensemble (simple)",
                "ensemble_weighted": "Ensemble (weighted)"}
            disp = name_map.get(ACTIVE_MODEL)
            chosen = next((m for m in metrics.get("metrics", []) if m["Model"] == disp), None)
            reg = ModelRegistry(
                nama_model=ACTIVE_MODEL, path_artefak=str(C.MODELS_DIR),
                metrics_json=json.dumps(chosen) if chosen else None,
                status="active", trained_on_upload_id=None,
                catatan=f"Model awal (seed): {ACTIVE_MODEL}.")
            db.session.add(reg)
            db.session.commit()
            print(f"model_registry: versi {reg.version_id} ({ACTIVE_MODEL}) AKTIF.")

        # ---- rekomendasi ----
        if Rekomendasi.query.first() is None:
            from app.services.forecast_service import service as FS
            from app.services.restock_service import (
                regenerate_rekomendasi_table, default_forecast_period)
            FS.load_active(force=True)
            active = ModelRegistry.query.filter_by(status="active").first()
            n = regenerate_rekomendasi_table(active.version_id, active.nama_model)
            print(f"rekomendasi terisi: {n} obat (periode {default_forecast_period()}).")
        else:
            print("rekomendasi sudah ada, lewati.")

        # ringkasan
        print("---- RINGKASAN ----")
        print("panel_bulanan :", PanelBulanan.query.count(), "baris")
        print("rekomendasi   :", Rekomendasi.query.count(), "baris")
        by = dict(db.session.query(Rekomendasi.status, db.func.count())
                  .group_by(Rekomendasi.status).all())
        print("status        :", by)
        print("Seed selesai.")


if __name__ == "__main__":
    seed(reset="--reset" in sys.argv)
