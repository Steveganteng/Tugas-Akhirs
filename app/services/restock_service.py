"""Logika rekomendasi restok berbasis panel LIVE di DB + model aktif.

Menangani filter periode:
  - periode = bulan prediksi default (bulan setelah data terakhir) -> rekomendasi tersimpan.
  - periode masa depan lebih jauh -> horizon = selisih bulan, predict, hitung ulang.
  - periode <= data terakhir -> data historis panel (bukan prediksi).
  - > 6 bulan ke depan -> ditolak.
"""
from __future__ import annotations
import pandas as pd

import config as C
from restock import hitung_restock, label_periode
from forecaster import RestockPipelineForecaster
from ..extensions import db
from ..models import PanelBulanan, Rekomendasi
from .forecast_service import service as FS

MAX_FUTURE_MONTHS = C.FORECAST_MAX_MONTHS  # 36 bulan = 3 tahun ke depan


def _restock_fields(model, obat, pred, stok_now, demand_hist, periode) -> dict:
    """Field restok untuk satu baris. Bila model aktif = RestockPipeline, pakai
    kebijakan (s,S) Min/Max BAWAAN model (recommend_order). Selain itu pakai
    hitung_restock generik (ROP berbasis lead time + safety stock).

    Selalu mengembalikan kunci: periode, prediksi_demand, stok_saat_ini, rop,
    safety_stock, jumlah_rekomendasi, status.
    """
    if isinstance(model, RestockPipelineForecaster):
        try:
            pol = model.restock_policy(obat, stok_now)
            return {
                "periode": label_periode(periode),
                "prediksi_demand": round(float(pred), 2),
                "stok_saat_ini": round(float(stok_now), 2),
                "rop": pol["rop"],
                "safety_stock": pol["safety_stock"],
                "jumlah_rekomendasi": pol["jumlah_rekomendasi"],
                "status": pol["status"],
            }
        except KeyError:
            pass  # obat tak ada di model -> fallback generik
    return hitung_restock(obat, pred, stok_now,
                          demand_historis=demand_hist, periode=periode)


def load_panel_dict():
    """{obat: {'periods':[...], 'demand':[...], 'stok':[...]}} dari DB, urut periode."""
    rows = (PanelBulanan.query
            .order_by(PanelBulanan.obat, PanelBulanan.periode).all())
    panel = {}
    for r in rows:
        d = panel.setdefault(r.obat, {"periods": [], "demand": [], "stok": [],
                                      "satuan": None})
        d["periods"].append(r.periode)
        d["demand"].append(float(r.demand or 0))
        d["stok"].append(float(r.stok or 0))
        if r.satuan and not d["satuan"]:
            d["satuan"] = r.satuan
    return panel


def satuan_map() -> dict:
    """{obat: satuan} dari panel (first non-null per obat) utk pelabelan UI."""
    rows = PanelBulanan.query.with_entities(
        PanelBulanan.obat, PanelBulanan.satuan).all()
    out = {}
    for obat, sat in rows:
        if sat and obat not in out:
            out[obat] = sat
    return out


def last_data_period() -> str:
    r = db.session.query(db.func.max(PanelBulanan.periode)).scalar()
    return r or C.PERIOD_END


def default_forecast_period() -> str:
    """Bulan pertama yang diprediksi = bulan setelah data terakhir."""
    p = pd.Period(last_data_period(), freq="M") + 1
    return str(p)


def available_periods():
    """Daftar periode (YYYY-MM) yang bisa dipilih di filter:
    seluruh bulan historis + sampai MAX_FUTURE_MONTHS ke depan."""
    rows = db.session.query(PanelBulanan.periode).distinct().all()
    hist = sorted({r[0] for r in rows})
    last = pd.Period(last_data_period(), freq="M")
    future = [str(last + i) for i in range(1, MAX_FUTURE_MONTHS + 1)]
    return hist + future


def _segmen(obat):
    return FS.segmen(obat)


def recommendation_for_period(obat: str, panel_obat: dict, model_key: str,
                              target: str) -> dict | None:
    """Hitung satu baris rekomendasi/historis untuk obat pada periode target.
    Mengembalikan dict siap tampil, atau None bila obat tak ada di model."""
    periods = panel_obat["periods"]
    demand = panel_obat["demand"]
    stok = panel_obat["stok"]
    last = periods[-1]
    last_p = pd.Period(last, freq="M")
    tgt_p = pd.Period(target, freq="M")
    seg_label, cluster = _segmen(obat)
    base = {"nama_obat": obat, "segmen": seg_label, "cluster": cluster}

    # ---- HISTORIS (periode <= data terakhir) ----
    if tgt_p <= last_p:
        if target in periods:
            i = periods.index(target)
            base.update({
                "periode": label_periode(target),
                "prediksi_demand": round(demand[i], 2),
                "stok_saat_ini": round(stok[i], 2),
                "rop": None, "safety_stock": None,
                "jumlah_rekomendasi": None,
                "status": "HISTORIS", "tipe": "historis",
            })
            return base
        return None

    # ---- MASA DEPAN ----
    horizon = (tgt_p.year - last_p.year) * 12 + (tgt_p.month - last_p.month)
    if horizon > MAX_FUTURE_MONTHS:
        return None
    try:
        _, model = FS.get_model(model_key)
        res = model.predict(obat, horizon)
    except KeyError:
        return None
    # ambil prediksi untuk periode target
    idx = res["periode"].index(target) if target in res["periode"] else horizon - 1
    pred = res["prediksi"][idx]
    rec = _restock_fields(model, obat, pred, stok[-1], demand, target)
    base.update({
        "periode": rec["periode"],
        "prediksi_demand": rec["prediksi_demand"],
        "stok_saat_ini": rec["stok_saat_ini"],
        "rop": rec["rop"],
        "safety_stock": rec["safety_stock"],
        "jumlah_rekomendasi": rec["jumlah_rekomendasi"],
        "status": rec["status"],
        "tipe": "prediksi",
        "horizon": horizon,
        "model": res["model"],
    })
    return base


def build_recommendations(model_key: str, target: str) -> list[dict]:
    """Bangun rekomendasi semua obat untuk periode target (live, tidak menyentuh DB).
    Periode masa depan memakai prediksi BATCH (cepat untuk horizon panjang)."""
    panel = load_panel_dict()
    last_p = pd.Period(last_data_period(), freq="M")
    tgt_p = pd.Period(target, freq="M")

    # historis / <= data terakhir: tidak butuh model
    if tgt_p <= last_p:
        out = []
        for obat, po in panel.items():
            row = recommendation_for_period(obat, po, model_key, target)
            if row:
                out.append(row)
        return out

    # masa depan: satu kali prediksi batch utk semua obat
    horizon = (tgt_p.year - last_p.year) * 12 + (tgt_p.month - last_p.month)
    if horizon > MAX_FUTURE_MONTHS:
        return []
    _, model = FS.get_model(model_key)
    preds = FS.predict_many(list(panel.keys()), model_key, horizon)
    out = []
    for obat, po in panel.items():
        res = preds.get(obat)
        if not res:
            continue
        idx = res["periode"].index(target) if target in res["periode"] else horizon - 1
        rec = _restock_fields(model, obat, res["prediksi"][idx], po["stok"][-1],
                              po["demand"], target)
        seg_label, cluster = _segmen(obat)
        out.append({
            "nama_obat": obat, "segmen": seg_label, "cluster": cluster,
            "periode": rec["periode"], "prediksi_demand": rec["prediksi_demand"],
            "stok_saat_ini": rec["stok_saat_ini"], "rop": rec["rop"],
            "safety_stock": rec["safety_stock"],
            "jumlah_rekomendasi": rec["jumlah_rekomendasi"],
            "status": rec["status"], "tipe": "prediksi",
            "horizon": horizon, "model": res["model"]})
    return out


def forecast_pattern(obat: str | None, horizon: int, model_key: str | None = None):
    """Pola pengeluaran historis + forecast multi-bulan ke depan.

    obat None / '__TOTAL__' -> agregat semua obat. Mengembalikan dict berisi
    label periode, seri historis, seri forecast (tersambung), tabel per periode,
    dan ringkasan pola (tren).
    """
    panel = load_panel_dict()
    last = last_data_period()
    last_p = pd.Period(last, freq="M")
    hist_periods = sorted({p for po in panel.values() for p in po["periods"]})
    future_periods = [str(last_p + i) for i in range(1, horizon + 1)]

    rows = []
    if obat and obat != "__TOTAL__":
        po = panel.get(obat)
        if po is None:
            return None
        # seri historis (selaraskan ke hist_periods)
        dmap = dict(zip(po["periods"], po["demand"]))
        hist_vals = [round(dmap.get(p, 0), 1) for p in hist_periods]
        _, model = FS.get_model(model_key)
        res = model.predict(obat, horizon)
        fore_vals = [round(v, 1) for v in res["prediksi"]]
        stok_now = po["stok"][-1]
        for i, p in enumerate(future_periods):
            rec = _restock_fields(model, obat, res["prediksi"][i], stok_now,
                                  po["demand"], p)
            rows.append({
                "periode": rec["periode"], "prediksi_demand": rec["prediksi_demand"],
                "rop": rec["rop"], "safety_stock": rec["safety_stock"],
                "jumlah_rekomendasi": rec["jumlah_rekomendasi"],
                "status": rec["status"]})
        judul = obat
        model_name = res["model"]
        satuan = po.get("satuan")
    else:
        # agregat total semua obat
        hist_map = {p: 0.0 for p in hist_periods}
        for po in panel.values():
            for p, d in zip(po["periods"], po["demand"]):
                hist_map[p] += d
        hist_vals = [round(hist_map[p], 1) for p in hist_periods]
        totals = [0.0] * horizon
        preds = FS.predict_many(list(panel.keys()), model_key, horizon)
        n_used = len(preds)
        for r in preds.values():
            for i in range(horizon):
                totals[i] += r["prediksi"][i]
        fore_vals = [round(v, 1) for v in totals]
        for i, p in enumerate(future_periods):
            rows.append({"periode": label_periode(p),
                         "prediksi_demand": fore_vals[i],
                         "rop": None, "safety_stock": None,
                         "jumlah_rekomendasi": None, "status": None})
        judul = f"TOTAL semua obat ({n_used} obat)"
        model_name = (model_key or FS.best_model)
        satuan = None  # mode TOTAL menggabungkan beragam satuan

    # chart: sambungkan forecast ke titik historis terakhir
    labels = [label_periode(p) for p in hist_periods] + [label_periode(p) for p in future_periods]
    hist_series = hist_vals + [None] * horizon
    fore_series = [None] * (len(hist_vals) - 1) + [hist_vals[-1]] + fore_vals

    # ringkasan pola
    avg_hist = sum(hist_vals) / len(hist_vals) if hist_vals else 0
    avg_fore = sum(fore_vals) / len(fore_vals) if fore_vals else 0
    if avg_hist > 0:
        delta = (avg_fore - avg_hist) / avg_hist * 100
    else:
        delta = 0
    tren = ("meningkat" if delta > 5 else "menurun" if delta < -5 else "stabil")

    return {
        "judul": judul, "model": model_name, "satuan": satuan,
        "labels": labels, "hist": hist_series, "fore": fore_series,
        "rows": rows, "future_periods": future_periods,
        "avg_hist": round(avg_hist, 1), "avg_fore": round(avg_fore, 1),
        "delta": round(delta, 1), "tren": tren,
        "last_period": label_periode(last),
    }


def replace_panel_from_parquet(parquet_path: str, sumber: str):
    """Ganti isi tabel panel_bulanan dengan snapshot panel dari sebuah versi."""
    df = pd.read_parquet(parquet_path)
    PanelBulanan.query.delete()
    satuan_ada = "satuan" in df.columns
    for _, r in df.iterrows():
        db.session.add(PanelBulanan(
            obat=r[C.P_OBAT], periode=str(r[C.P_PERIOD]),
            demand=float(r[C.P_DEMAND]), stok=float(r[C.P_STOCK]),
            satuan=(r["satuan"] if satuan_ada else None),
            is_observasi=bool(r[C.P_IS_OBS]), sumber=sumber))
    db.session.commit()


def regenerate_rekomendasi_table(model_version: int, model_key: str):
    """Hitung ulang tabel `rekomendasi` untuk bulan prediksi default & simpan."""
    target = default_forecast_period()
    rows = build_recommendations(model_key, target)
    Rekomendasi.query.delete()
    for r in rows:
        db.session.add(Rekomendasi(
            nama_obat=r["nama_obat"], periode=r["periode"],
            prediksi_demand=r["prediksi_demand"], rop=r["rop"],
            safety_stock=r["safety_stock"], jumlah_rekomendasi=r["jumlah_rekomendasi"],
            stok_saat_ini=r["stok_saat_ini"], status=r["status"],
            segmen=r["segmen"], cluster=r["cluster"], model_version=model_version,
        ))
    db.session.commit()
    return len(rows)
