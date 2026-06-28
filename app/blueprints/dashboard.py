"""Halaman Dashboard: kartu ringkasan + grafik + tabel butuh segera restok."""
from __future__ import annotations
import json
from collections import OrderedDict
from flask import Blueprint, render_template

import config as C
from ..extensions import db
from ..models import PanelBulanan, Rekomendasi, ModelRegistry
from ..services.forecast_service import service as FS
from ..services.restock_service import (
    last_data_period, default_forecast_period, satuan_map)

bp = Blueprint("dashboard", __name__)


@bp.route("/")
def index():
    FS.load_active()

    # ---- kartu ringkasan ----
    total_obat = db.session.query(Rekomendasi.nama_obat).distinct().count()
    by_status = dict(db.session.query(Rekomendasi.status, db.func.count())
                     .group_by(Rekomendasi.status).all())
    segera = by_status.get("SEGERA RESTOK", 0)
    perhatian = by_status.get("PERLU DIPERHATIKAN", 0)
    aman = by_status.get("STOK AMAN", 0)

    active = ModelRegistry.query.filter_by(status="active").first()
    model_aktif = active.nama_model if active else FS.best_model
    mae_aktif = None
    if active and active.metrics_json:
        try:
            mae_aktif = round(json.loads(active.metrics_json).get("MAE"), 2)
        except Exception:
            mae_aktif = None

    # ---- grafik a: total demand per bulan ----
    dm = (db.session.query(PanelBulanan.periode,
                           db.func.sum(PanelBulanan.demand))
          .group_by(PanelBulanan.periode)
          .order_by(PanelBulanan.periode).all())
    line_labels = [p for p, _ in dm]
    line_values = [round(float(v or 0), 1) for _, v in dm]

    # ---- grafik b: top 10 obat prediksi kebutuhan periode berikutnya ----
    top = (Rekomendasi.query
           .order_by(Rekomendasi.prediksi_demand.desc()).limit(10).all())
    bar_labels = [r.nama_obat for r in top]
    bar_values = [round(float(r.prediksi_demand or 0), 1) for r in top]

    # ---- grafik c: sebaran status (doughnut) ----
    status_order = ["SEGERA RESTOK", "PERLU DIPERHATIKAN", "STOK AMAN"]
    dough_labels = status_order
    dough_values = [by_status.get(s, 0) for s in status_order]

    # ---- grafik d: sebaran segmen cluster ----
    seg = (db.session.query(Rekomendasi.segmen, db.func.count())
           .group_by(Rekomendasi.segmen).all())
    seg_map = OrderedDict()
    for s, c in seg:
        seg_map[s or "Tidak terlabel"] = c
    segmen_labels = list(seg_map.keys())
    segmen_values = list(seg_map.values())

    # ---- tabel butuh segera direstok ----
    segera_rows = (Rekomendasi.query.filter_by(status="SEGERA RESTOK")
                   .order_by(Rekomendasi.jumlah_rekomendasi.desc()).limit(10).all())

    charts = {
        "line": {"labels": line_labels, "values": line_values},
        "bar": {"labels": bar_labels, "values": bar_values},
        "doughnut": {"labels": dough_labels, "values": dough_values},
        "segmen": {"labels": segmen_labels, "values": segmen_values},
    }
    return render_template(
        "dashboard.html", active_page="dashboard",
        total_obat=total_obat, segera=segera, perhatian=perhatian, aman=aman,
        model_aktif=model_aktif, mae_aktif=mae_aktif,
        last_period=last_data_period(), forecast_period=default_forecast_period(),
        charts_json=json.dumps(charts), segera_rows=segera_rows,
        satuan=satuan_map())
