"""Halaman Upload File: download template, upload data pengeluaran, latih
challenger (duplikat), tampilkan metrik vs model aktif, pilih pakai/tolak,
riwayat upload & versi model, rollback.
"""
from __future__ import annotations
import json
import pandas as pd
from flask import (Blueprint, render_template, request, redirect, url_for,
                   flash, send_file, jsonify, current_app)

from ..extensions import db
from ..models import UploadLog, ModelRegistry
from ..services import upload_service as US
from ..services import train_service as TS

bp = Blueprint("upload", __name__, url_prefix="/upload")

MODEL_CHOICES = [
    ("random_forest", "Random Forest (default)"),
    ("gradient_boosting", "Gradient Boosting"),
    ("holt_winters", "Holt-Winters"),
    ("sarima", "SARIMA"),
    ("ensemble_weighted", "Ensemble (weighted)"),
    ("all", "Semua model (pipeline penuh)"),
]


@bp.route("/")
def index():
    uploads = UploadLog.query.order_by(UploadLog.upload_id.desc()).limit(20).all()
    versions = ModelRegistry.query.order_by(ModelRegistry.version_id.desc()).all()
    return render_template("upload.html", active_page="upload",
                           model_choices=MODEL_CHOICES, uploads=uploads,
                           versions=versions)


@bp.route("/template")
def template():
    buf = US.generate_template()
    return send_file(buf, as_attachment=True,
                     download_name="template_data_pengeluaran.xlsx",
                     mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@bp.route("/proses", methods=["POST"])
def proses():
    model_pilihan = request.form.get("model_pilihan", "all")
    file = request.files.get("file")
    if not file or file.filename == "":
        flash("Tidak ada file dipilih.", "danger")
        return redirect(url_for("upload.index"))
    if not file.filename.lower().endswith(".xlsx"):
        flash("Format harus .xlsx", "danger")
        return redirect(url_for("upload.index"))

    try:
        df = pd.read_excel(file)
    except Exception as e:
        flash(f"Gagal membaca Excel: {e}", "danger")
        return redirect(url_for("upload.index"))

    df_valid, n_baris, n_valid, n_ditolak, errors = US.validate_dataframe(df)
    if df_valid is None:
        # catat upload gagal
        log = UploadLog(filename=file.filename, n_baris=n_baris, n_valid=n_valid,
                        n_ditolak=n_ditolak, status="gagal",
                        error_text=" | ".join(errors), model_dilatih=model_pilihan)
        db.session.add(log)
        db.session.commit()
        flash("Upload ditolak: " + " | ".join(errors), "danger")
        return redirect(url_for("upload.index"))

    uid = US.stage_upload(file.filename, df_valid, n_baris, n_valid,
                          n_ditolak, model_pilihan)
    # latih challenger di background
    TS.start_training(current_app._get_current_object(), uid, model_pilihan, df_valid)
    flash(f"File diterima ({n_valid} baris valid, {n_ditolak} ditolak). "
          f"Melatih model challenger di latar belakang...", "info")
    return redirect(url_for("upload.status_page", upload_id=uid))


@bp.route("/status/<int:upload_id>")
def status_page(upload_id):
    log = db.session.get(UploadLog, upload_id)
    if not log:
        flash("Upload tidak ditemukan.", "danger")
        return redirect(url_for("upload.index"))
    return render_template("upload_status.html", active_page="upload", log=log)


@bp.route("/status/<int:upload_id>/json")
def status_json(upload_id):
    log = db.session.get(UploadLog, upload_id)
    if not log:
        return jsonify({"status": "tidak_ada"}), 404
    return jsonify({"status": log.status, "candidate_version": log.candidate_version,
                    "error_text": (log.error_text or "")[:500]})


@bp.route("/hasil/<int:version_id>")
def hasil(version_id):
    cand = db.session.get(ModelRegistry, version_id)
    if not cand:
        flash("Versi model tidak ditemukan.", "danger")
        return redirect(url_for("upload.index"))
    try:
        cand_metrics = json.loads(open(
            f"{cand.path_artefak}/metrics.json", encoding="utf-8").read())["metrics"]
    except Exception:
        cand_metrics = []
    active = ModelRegistry.query.filter_by(status="active").first()
    active_metric = TS.active_metrics()  # metrik model aktif (chosen)
    chosen_key = cand.nama_model
    return render_template("upload_hasil.html", active_page="upload", cand=cand,
                           cand_metrics=cand_metrics, active=active,
                           active_metric=active_metric, chosen_key=chosen_key,
                           name_map=TS.KEY_TO_NAME)


@bp.route("/gunakan/<int:version_id>", methods=["POST"])
def gunakan(version_id):
    ok, msg = TS.accept_candidate(version_id)
    flash(msg, "success" if ok else "danger")
    return redirect(url_for("upload.index"))


@bp.route("/tolak/<int:version_id>", methods=["POST"])
def tolak(version_id):
    ok, msg = TS.reject_candidate(version_id)
    flash(msg, "warning" if ok else "danger")
    return redirect(url_for("upload.index"))


@bp.route("/rollback/<int:version_id>", methods=["POST"])
def rollback(version_id):
    ok, msg = TS.rollback_to(version_id)
    flash(msg, "success" if ok else "danger")
    return redirect(url_for("upload.index"))
