"""Model ORM SQLAlchemy untuk MySQL (database TugasAkhir).

Lima tabel sesuai spesifikasi:
  transaksi_raw, panel_bulanan, rekomendasi, model_registry, upload_log.
"""
from __future__ import annotations
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from .extensions import db


class User(db.Model):
    """Akun pengguna aplikasi (login berbasis sesi)."""
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    nama = db.Column(db.String(128))
    role = db.Column(db.String(32), default="apoteker", index=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)

    def set_password(self, raw: str):
        self.password_hash = generate_password_hash(raw)

    def check_password(self, raw: str) -> bool:
        return check_password_hash(self.password_hash, raw)


class TransaksiRaw(db.Model):
    __tablename__ = "transaksi_raw"
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    tanggal_masuk = db.Column(db.Date)
    register = db.Column(db.String(64))
    kode_diagnosa = db.Column(db.String(64))
    diagnosa_primer = db.Column(db.String(255))
    resep_obat = db.Column(db.String(255), index=True)
    jumlah = db.Column(db.Float)
    sisa_stok = db.Column(db.Float)
    satuan = db.Column(db.String(32))
    upload_id = db.Column(db.Integer, db.ForeignKey("upload_log.upload_id"), index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PanelBulanan(db.Model):
    __tablename__ = "panel_bulanan"
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    obat = db.Column(db.String(255), index=True, nullable=False)
    periode = db.Column(db.String(7), index=True, nullable=False)  # YYYY-MM
    demand = db.Column(db.Float, default=0)
    stok = db.Column(db.Float, default=0)
    satuan = db.Column(db.String(32))
    is_observasi = db.Column(db.Boolean, default=True)
    sumber = db.Column(db.String(32), default="init")  # 'init' | upload_id
    __table_args__ = (db.UniqueConstraint("obat", "periode", name="uq_obat_periode"),)


class Rekomendasi(db.Model):
    __tablename__ = "rekomendasi"
    id = db.Column(db.BigInteger, primary_key=True, autoincrement=True)
    nama_obat = db.Column(db.String(255), index=True, nullable=False)
    periode = db.Column(db.String(32))            # label "Januari 2026"
    prediksi_demand = db.Column(db.Float)
    rop = db.Column(db.Float)
    safety_stock = db.Column(db.Float)
    jumlah_rekomendasi = db.Column(db.Float)
    stok_saat_ini = db.Column(db.Float)
    status = db.Column(db.String(32), index=True)
    segmen = db.Column(db.String(64))
    cluster = db.Column(db.Integer)
    model_version = db.Column(db.Integer, index=True)
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)


class ModelRegistry(db.Model):
    __tablename__ = "model_registry"
    version_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    nama_model = db.Column(db.String(64))           # mis. holt_winters
    path_artefak = db.Column(db.String(512))
    metrics_json = db.Column(db.Text)               # JSON {MAE,RMSE,sMAPE,R2}
    trained_at = db.Column(db.DateTime, default=datetime.utcnow)
    trained_on_upload_id = db.Column(db.Integer, nullable=True)
    status = db.Column(db.Enum("active", "candidate", "rejected", "archived"),
                       default="candidate", index=True)
    catatan = db.Column(db.String(255))


class UploadLog(db.Model):
    __tablename__ = "upload_log"
    upload_id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    filename = db.Column(db.String(255))
    n_baris = db.Column(db.Integer, default=0)
    n_valid = db.Column(db.Integer, default=0)
    n_ditolak = db.Column(db.Integer, default=0)
    status = db.Column(db.Enum("diproses", "selesai", "gagal"),
                       default="diproses", index=True)
    error_text = db.Column(db.Text)
    model_dilatih = db.Column(db.String(64))        # pilihan model challenger
    candidate_version = db.Column(db.Integer, nullable=True)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
