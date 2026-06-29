"""Tahap 6 — Modul rekomendasi restok.

Fungsi murni `hitung_restock` dapat dipanggil API tanpa dependensi model.
Data bulanan -> avg & std dikonversi ke basis harian; periode = label bulan
(mis. "Januari 2026"), tidak ada presisi harian palsu.
"""
from __future__ import annotations
import math
import numpy as np
import config as C


def label_periode(periode: str | None) -> str:
    """'2026-01' -> 'Januari 2026'."""
    if not periode:
        periode = C.FORECAST_START_PERIOD
    y, m = periode.split("-")
    return f"{C.BULAN_ID[int(m)]} {y}"


def hitung_restock(nama_obat: str, prediksi, stok_saat_ini: float,
                   lead_time_hari: int = C.DEFAULT_LEAD_TIME_DAYS,
                   z: float = C.DEFAULT_Z,
                   demand_historis=None, periode: str | None = None) -> dict:
    """Hitung ROP, safety stock, dan rekomendasi jumlah restok.

    prediksi        : demand bulan depan (skalar atau list -> diambil elemen pertama).
    demand_historis : opsional list demand bulanan utk estimasi variabilitas;
                      bila None dipakai aproksimasi Poisson (std_harian=sqrt(avg_harian)).
    """
    # prediksi bisa list (horizon) -> pakai periode pertama
    if isinstance(prediksi, (list, tuple, np.ndarray)):
        pred_bulan = float(prediksi[0]) if len(prediksi) else 0.0
    else:
        pred_bulan = float(prediksi)
    pred_bulan = max(0.0, pred_bulan)
    stok_saat_ini = max(0.0, float(stok_saat_ini))

    avg_harian = pred_bulan / C.DAYS_PER_MONTH

    # std harian
    if demand_historis is not None and len(demand_historis) >= 2:
        std_bulanan = float(np.std(np.asarray(demand_historis, dtype=float), ddof=1))
        std_harian = std_bulanan / math.sqrt(C.DAYS_PER_MONTH)
    else:
        std_harian = math.sqrt(max(avg_harian, 0.0))  # aproksimasi Poisson

    safety_stock = z * std_harian * math.sqrt(lead_time_hari)
    rop = avg_harian * lead_time_hari + safety_stock
    jumlah_rekomendasi = max(0.0, pred_bulan + safety_stock - stok_saat_ini)

    if stok_saat_ini <= rop:
        status = "SEGERA RESTOK"
    elif stok_saat_ini <= C.ATTENTION_FACTOR * rop:
        status = "PERLU DIPERHATIKAN"
    else:
        status = "STOK AMAN"

    return {
        "nama_obat": nama_obat,
        "rop": round(rop, 2),
        "safety_stock": round(safety_stock, 2),
        "jumlah_rekomendasi": round(jumlah_rekomendasi, 2),
        "prediksi_demand": round(pred_bulan, 2),
        "stok_saat_ini": round(stok_saat_ini, 2),
        "status": status,
        "periode": label_periode(periode),
    }


if __name__ == "__main__":
    # contoh
    print(hitung_restock("ACETYLCYSTEIN KAP", 800, 150,
                         demand_historis=[471, 1368, 570, 396, 419],
                         periode="2026-01"))
