"""Halaman Prediksi: pola pengeluaran historis + forecast multi-bulan ke depan
untuk semua periode mendatang, berdasarkan asumsi model aktif."""
from __future__ import annotations
import json
from flask import Blueprint, render_template, request

import config as C
from ..models import ModelRegistry
from ..services.forecast_service import service as FS
from ..services import restock_service as RS

bp = Blueprint("forecast", __name__, url_prefix="/prediksi")

# Horizon dalam bulan -> label (mendukung multi-tahun)
HORIZON_OPTS = [
    (6, "6 bulan"), (12, "12 bulan (1 tahun)"),
    (24, "24 bulan (2 tahun)"), (36, "36 bulan (3 tahun)"),
]
HORIZON_VALUES = [h for h, _ in HORIZON_OPTS]
DEFAULT_HORIZON = C.FORECAST_DEFAULT_MONTHS  # 12 = 1 tahun


@bp.route("/")
def index():
    FS.load_active()
    obat_list = FS.list_obat()
    obat = (request.args.get("obat") or "__TOTAL__").strip()
    try:
        horizon = int(request.args.get("horizon", DEFAULT_HORIZON))
    except ValueError:
        horizon = DEFAULT_HORIZON
    if horizon not in HORIZON_VALUES:
        horizon = DEFAULT_HORIZON
    model_key = (request.args.get("model") or "").strip() or None

    data = RS.forecast_pattern(obat, horizon, model_key)
    model_opts = ModelRegistry.query.filter(
        ModelRegistry.status.in_(["active", "archived"])).all()

    return render_template(
        "prediksi.html", active_page="prediksi", data=data,
        obat_list=obat_list, sel_obat=obat, horizon=horizon,
        horizon_opts=HORIZON_OPTS, model_opts=model_opts, sel_model=model_key or "",
        chart_json=json.dumps({"labels": data["labels"], "hist": data["hist"],
                               "fore": data["fore"]}) if data else "{}")
