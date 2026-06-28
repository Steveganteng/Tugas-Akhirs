"""Champion–Challenger: latih DUPLIKAT model pada panel baru tanpa menyentuh
artefak aktif. Simpan ke models/candidates/{version_id}/, daftarkan sebagai
'candidate', tampilkan metrik vs model aktif. User memilih pakai/tolak.

Training berjalan di background thread; status dilacak di upload_log.
"""
from __future__ import annotations
import json
import threading
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import config as C
import ml_features as F
from forecaster import (
    NaiveForecaster, RandomForestForecaster, GradientBoostingForecaster,
    HoltWintersForecaster, SarimaForecaster, EnsembleForecaster,
)
from pipeline_forecasting_obat import (
    temporal_split, evaluate, compute_ensemble_weights,
)
from ..extensions import db
from ..models import ModelRegistry, UploadLog
from .restock_service import (
    regenerate_rekomendasi_table, replace_panel_from_parquet)
from .forecast_service import service as FS

CAND_DIR = C.MODELS_DIR / "candidates"

NAME_TO_KEY = {
    "Naive (lag-1)": "naive", "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting", "Holt-Winters": "holt_winters",
    "SARIMA": "sarima", "Ensemble (simple)": "ensemble_simple",
    "Ensemble (weighted)": "ensemble_weighted",
}
KEY_TO_NAME = {v: k for k, v in NAME_TO_KEY.items()}


def _best_params(fname, default):
    try:
        return json.load(open(C.MODELS_DIR / fname, encoding="utf-8"))["best_params"]
    except Exception:
        return default


def _train_full_set(panel, cluster_map, models_dir: Path):
    """Latih SEMUA varian pada panel penuh -> simpan artefak ke models_dir."""
    models_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = F.feature_columns()
    rf_params = _best_params("rf_features.json", {"n_estimators": 300})
    gb_params = _best_params("gb_features.json", {"n_estimators": 300})

    naive = NaiveForecaster.fit(panel)
    rf = RandomForestForecaster.fit(
        panel, RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **rf_params),
        feat_cols, cluster_map)
    gb = GradientBoostingForecaster.fit(
        panel, GradientBoostingRegressor(random_state=C.RANDOM_STATE, **gb_params),
        feat_cols, cluster_map)
    hw = HoltWintersForecaster.fit(panel)
    sa = SarimaForecaster.fit(panel)
    all_periods = sorted(panel[C.P_PERIOD].unique())
    weights = compute_ensemble_weights(panel, all_periods)

    naive.save(models_dir / "naive.pkl")
    rf.save(models_dir / "random_forest.pkl")
    gb.save(models_dir / "gradient_boosting.pkl")
    hw.save(models_dir / "holtwinters.pkl")
    sa.save(models_dir / "sarima.pkl")
    json.dump(weights, open(models_dir / "ensemble_weights.json", "w"), indent=2)
    json.dump({"features": feat_cols, "best_params": rf_params},
              open(models_dir / "rf_features.json", "w"), indent=2)
    json.dump({"features": feat_cols, "best_params": gb_params},
              open(models_dir / "gb_features.json", "w"), indent=2)
    return rf_params, gb_params


def _evaluate_set(panel, cluster_map) -> list[dict]:
    """Split temporal & evaluasi semua pendekatan (metrik challenger)."""
    train, test, train_periods, test_periods = temporal_split(panel)
    H = C.TEST_MONTHS
    feat_cols = F.feature_columns()
    rf_params = _best_params("rf_features.json", {"n_estimators": 300})
    gb_params = _best_params("gb_features.json", {"n_estimators": 300})

    fcs = {}
    fcs["Naive (lag-1)"] = NaiveForecaster.fit(train)
    fcs["Random Forest"] = RandomForestForecaster.fit(
        train, RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **rf_params),
        feat_cols, cluster_map)
    fcs["Gradient Boosting"] = GradientBoostingForecaster.fit(
        train, GradientBoostingRegressor(random_state=C.RANDOM_STATE, **gb_params),
        feat_cols, cluster_map)
    hw = HoltWintersForecaster.fit(train)
    sa = SarimaForecaster.fit(train)
    fcs["Holt-Winters"] = hw
    fcs["SARIMA"] = sa
    w = compute_ensemble_weights(train, train_periods)
    fcs["Ensemble (simple)"] = EnsembleForecaster(hw, sa, w, "simple")
    fcs["Ensemble (weighted)"] = EnsembleForecaster(hw, sa, w, "weighted")

    rows = [evaluate(fc, test, test_periods, H, name) for name, fc in fcs.items()]
    rows.sort(key=lambda r: r["MAE"])
    return rows


def _set_log(uid, **kw):
    log = db.session.get(UploadLog, uid)
    for k, v in kw.items():
        setattr(log, k, v)
    db.session.commit()


def _run(app, uid, model_pilihan, df_new_raw_records):
    """Worker thread: bangun panel, latih challenger, evaluasi, registrasi."""
    with app.app_context():
        try:
            from .upload_service import build_combined_panel
            # transaksi_raw sudah berisi data upload ini (di-stage sebelum training)
            # + seluruh upload sebelumnya -> preprocessing penuh atas SEMUA data.
            panel = build_combined_panel()
            cluster_map = json.load(
                open(C.MODELS_DIR / "cluster_labels.json", encoding="utf-8"))["obat"]

            # daftarkan kandidat dulu utk dapat version_id
            reg = ModelRegistry(nama_model="(training)", path_artefak="",
                                status="candidate", trained_on_upload_id=uid,
                                catatan=f"challenger dari upload #{uid}")
            db.session.add(reg)
            db.session.flush()
            vid = reg.version_id
            cand_dir = CAND_DIR / str(vid)
            cand_dir.mkdir(parents=True, exist_ok=True)

            # simpan snapshot panel gabungan (dipakai saat kandidat diterima)
            panel.to_parquet(cand_dir / "panel.parquet", index=False)

            # latih + evaluasi
            _train_full_set(panel, cluster_map, cand_dir)
            metrics_rows = _evaluate_set(panel, cluster_map)

            # tentukan model "terpilih" untuk kandidat ini
            if model_pilihan and model_pilihan != "all":
                chosen_key = model_pilihan
            else:
                chosen_key = NAME_TO_KEY[metrics_rows[0]["Model"]]
            chosen_name = KEY_TO_NAME.get(chosen_key, chosen_key)
            chosen_metric = next((m for m in metrics_rows if m["Model"] == chosen_name),
                                 metrics_rows[0])

            json.dump({"metrics": metrics_rows}, open(cand_dir / "metrics.json", "w"), indent=2)
            json.dump({"best_model": chosen_key, "best_model_name": chosen_name,
                       "available": list(NAME_TO_KEY.values())},
                      open(cand_dir / "best_model.json", "w"), indent=2)

            reg.nama_model = chosen_key
            reg.path_artefak = str(cand_dir)
            reg.metrics_json = json.dumps(chosen_metric)
            db.session.commit()

            _set_log(uid, status="selesai", candidate_version=vid)
        except Exception:
            db.session.rollback()
            _set_log(uid, status="gagal", error_text=traceback.format_exc()[-3000:])


def start_training(app, uid, model_pilihan, df_valid_raw: pd.DataFrame):
    """Spawn background thread training challenger."""
    records = df_valid_raw.to_dict(orient="records")
    t = threading.Thread(target=_run, args=(app, uid, model_pilihan, records),
                         daemon=True)
    t.start()


# ---------------- ACCEPT / REJECT / ROLLBACK ----------------
def active_metrics() -> dict | None:
    row = ModelRegistry.query.filter_by(status="active").first()
    if row and row.metrics_json:
        try:
            return json.loads(row.metrics_json)
        except Exception:
            return None
    return None


def accept_candidate(version_id: int):
    """Jadikan kandidat aktif, arsipkan yang lama, regenerasi rekomendasi.
    Invariant: tepat satu baris 'active'."""
    cand = db.session.get(ModelRegistry, version_id)
    if not cand or cand.status != "candidate":
        return False, "Kandidat tidak ditemukan / sudah diproses."
    ModelRegistry.query.filter_by(status="active").update({"status": "archived"})
    cand.status = "active"
    db.session.commit()
    FS.load_active(force=True)
    _restore_panel(cand)
    regenerate_rekomendasi_table(cand.version_id, cand.nama_model)
    return True, f"Model versi #{version_id} ({cand.nama_model}) sekarang AKTIF."


def reject_candidate(version_id: int):
    cand = db.session.get(ModelRegistry, version_id)
    if not cand or cand.status != "candidate":
        return False, "Kandidat tidak ditemukan / sudah diproses."
    cand.status = "rejected"
    db.session.commit()
    return True, f"Model versi #{version_id} ditolak. Model aktif tidak berubah."


def rollback_to(version_id: int):
    """Aktifkan kembali versi 'archived'."""
    tgt = db.session.get(ModelRegistry, version_id)
    if not tgt or tgt.status != "archived":
        return False, "Versi tidak ditemukan / bukan arsip."
    ModelRegistry.query.filter_by(status="active").update({"status": "archived"})
    tgt.status = "active"
    db.session.commit()
    FS.load_active(force=True)
    _restore_panel(tgt)
    regenerate_rekomendasi_table(tgt.version_id, tgt.nama_model)
    return True, f"Rollback ke versi #{version_id} ({tgt.nama_model}) berhasil."


def _restore_panel(reg: ModelRegistry):
    """Pulihkan panel_bulanan ke snapshot milik versi `reg`.
    Versi seed (tanpa upload) -> panel awal; challenger -> panel.parquet kandidat."""
    from pathlib import Path
    if reg.trained_on_upload_id is None:
        snap = C.PANEL_PATH  # panel awal (seed)
        sumber = "init"
    else:
        snap = Path(reg.path_artefak) / "panel.parquet"
        sumber = f"v{reg.version_id}"
    if Path(snap).exists():
        replace_panel_from_parquet(str(snap), sumber)
