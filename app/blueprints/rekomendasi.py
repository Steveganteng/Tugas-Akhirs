"""Halaman Rekomendasi: tabel semua obat + filter (nama, status, segmen, model,
periode) + sorting + pagination + ekspor Excel.

Periode masa depan -> prediksi via model.predict + hitung_restock (live).
Periode default tersimpan -> baca tabel rekomendasi.
Periode historis -> data panel.
"""
from __future__ import annotations
import io
import pandas as pd
from flask import (Blueprint, render_template, request, send_file, flash)
from openpyxl import Workbook

import config as C
from ..extensions import db
from ..models import Rekomendasi, ModelRegistry
from ..services.forecast_service import service as FS
from ..services import restock_service as RS

bp = Blueprint("rekomendasi", __name__, url_prefix="/rekomendasi")

STATUS_ALL = ["SEGERA RESTOK", "PERLU DIPERHATIKAN", "STOK AMAN"]
PER_PAGE = 50
SORT_FIELDS = {
    "nama_obat", "segmen", "satuan", "prediksi_demand", "rop", "safety_stock",
    "stok_saat_ini", "jumlah_rekomendasi", "status",
}


def _attach_satuan(rows):
    """Lengkapi tiap baris dengan satuan obat (dari panel) untuk tampil & sort."""
    sm = RS.satuan_map()
    for r in rows:
        r["satuan"] = sm.get(r["nama_obat"])
    return rows


def _filters():
    return {
        "q": (request.args.get("q") or "").strip(),
        # buang nilai kosong ("Semua status") agar tidak menyaring habis semua baris
        "status": [s for s in request.args.getlist("status") if s],
        "segmen": (request.args.get("segmen") or "").strip(),
        "model": (request.args.get("model") or "").strip(),
        "periode": (request.args.get("periode") or "").strip(),
        "sort": request.args.get("sort") or "jumlah_rekomendasi",
        "dir": request.args.get("dir") or "desc",
    }


def _collect_rows(f):
    """Kembalikan (rows:list[dict], banner:str|None, is_future:bool)."""
    default_period = RS.default_forecast_period()
    target = f["periode"] or default_period
    model_key = f["model"] or None
    banner = None

    last = RS.last_data_period()
    tgt_p = pd.Period(target, freq="M")
    last_p = pd.Period(last, freq="M")

    # Pakai rekomendasi tersimpan hanya bila periode = bulan prediksi default
    # DAN tidak ada override model. Selain itu hitung live.
    use_stored = (target == default_period) and (not f["model"])

    if use_stored:
        rows = [_row_from_db(r) for r in Rekomendasi.query.all()]
    else:
        if tgt_p <= last_p:
            banner = (f"Menampilkan DATA HISTORIS untuk {RS_label(target)} "
                      f"(bukan prediksi).")
        else:
            h = (tgt_p.year - last_p.year) * 12 + (tgt_p.month - last_p.month)
            if h > RS.MAX_FUTURE_MONTHS:
                return [], (f"Periode {RS_label(target)} lebih dari "
                            f"{RS.MAX_FUTURE_MONTHS} bulan dari data terakhir "
                            f"({RS_label(last)}). Tidak diprediksi."), True
            banner = (f"Prediksi untuk {RS_label(target)}, {h} bulan dari data "
                      f"terakhir ({RS_label(last)}) — ketidakpastian meningkat "
                      f"seiring jarak horizon.")
        rows = RS.build_recommendations(model_key, target)

    return rows, banner, (tgt_p > last_p)


def RS_label(periode_ym):
    from restock import label_periode
    return label_periode(periode_ym)


def _row_from_db(r: Rekomendasi) -> dict:
    return {
        "nama_obat": r.nama_obat, "segmen": r.segmen, "cluster": r.cluster,
        "periode": r.periode, "prediksi_demand": r.prediksi_demand,
        "rop": r.rop, "safety_stock": r.safety_stock,
        "jumlah_rekomendasi": r.jumlah_rekomendasi,
        "stok_saat_ini": r.stok_saat_ini, "status": r.status,
    }


def _apply_filters(rows, f):
    q = f["q"].lower()
    if q:
        rows = [r for r in rows if q in (r["nama_obat"] or "").lower()]
    if f["status"]:
        rows = [r for r in rows if r["status"] in f["status"]]
    if f["segmen"]:
        rows = [r for r in rows if (r.get("segmen") or "") == f["segmen"]]
    # sort
    key = f["sort"] if f["sort"] in SORT_FIELDS else "jumlah_rekomendasi"
    reverse = f["dir"] != "asc"

    def sk(r):
        v = r.get(key)
        if v is None:
            return (1, 0) if not reverse else (0, 0)
        if isinstance(v, str):
            return (0, v.lower())
        return (0, v)
    rows = sorted(rows, key=sk, reverse=reverse)
    return rows


@bp.route("/")
def index():
    FS.load_active()
    f = _filters()
    rows, banner, is_future = _collect_rows(f)
    rows = _attach_satuan(rows)
    rows = _apply_filters(rows, f)

    total = len(rows)
    try:
        page = max(1, int(request.args.get("page", 1)))
    except ValueError:
        page = 1
    pages = max(1, (total + PER_PAGE - 1) // PER_PAGE)
    page = min(page, pages)
    start = (page - 1) * PER_PAGE
    page_rows = rows[start:start + PER_PAGE]

    segmen_opts = sorted({(r.get("segmen") or "") for r in rows if r.get("segmen")})
    if not segmen_opts:
        FS._ensure_cluster_map()
        segmen_opts = sorted({v.get("label") for v in FS.cluster_map.values() if v.get("label")})
    model_opts = ModelRegistry.query.filter(
        ModelRegistry.status.in_(["active", "archived"])).all()

    return render_template(
        "rekomendasi.html", active_page="rekomendasi", rows=page_rows,
        total=total, page=page, pages=pages, f=f, banner=banner,
        is_future=is_future, status_all=STATUS_ALL, segmen_opts=segmen_opts,
        model_opts=model_opts, periods=RS.available_periods(),
        default_period=RS.default_forecast_period(),
        label=RS_label)


@bp.route("/export")
def export():
    FS.load_active()
    f = _filters()
    rows, _, is_future = _collect_rows(f)
    rows = _attach_satuan(rows)
    rows = _apply_filters(rows, f)
    jenis = "Prediksi" if is_future else "Aktual (Historis)"

    wb = Workbook()
    ws = wb.active
    ws.title = "Rekomendasi"
    headers = ["Nama Obat", "Segmen", "Periode", "Jenis Data", "Prediksi Pemakaian/Bln",
               "Satuan", "Re-Order Point", "Stok Pengaman", "Stok Saat Ini",
               "Jumlah Re-Stok", "Status"]
    ws.append(headers)
    for r in rows:
        ws.append([r["nama_obat"], r.get("segmen"), r.get("periode"), jenis,
                   r.get("prediksi_demand"), r.get("satuan"), r.get("rop"),
                   r.get("safety_stock"), r.get("stok_saat_ini"),
                   r.get("jumlah_rekomendasi"), r["status"]])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    fname = f"rekomendasi_{f['periode'] or RS.default_forecast_period()}.xlsx"
    return send_file(buf, as_attachment=True, download_name=fname,
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
