"""Lazy singleton pemuat artefak model.

Memuat SEMUA varian forecaster dari direktori versi AKTIF (model_registry.status
== 'active'). Reload hanya saat versi aktif berganti. Cluster map & label segmen
selalu dari models/ root (clustering tidak dilatih ulang per upload).
"""
from __future__ import annotations
import json
from pathlib import Path
from threading import Lock

import config as C
from forecaster import (
    NaiveForecaster, RandomForestForecaster, GradientBoostingForecaster,
    HoltWintersForecaster, SarimaForecaster, EnsembleForecaster,
    SESForecaster, RestockPipelineForecaster, RestockPipelineFinalForecaster,
)

_lock = Lock()


def _pick_best(models: dict, best: str) -> str:
    """Pilih kunci model aktif yang BENAR-BENAR termuat. Bila `best` gagal dimuat
    (mis. beda versi sklearn), jatuh ke alternatif yang tersedia."""
    if best in models:
        return best
    for k in ("restock_final", "ses", "naive"):
        if k in models:
            return k
    return next(iter(models))


def _load_dir(models_dir: Path) -> dict:
    """Muat varian forecaster dari satu direktori artefak — TOLERAN GAGAL.

    Tiap varian dimuat terpisah; bila satu gagal (mis. artefak sklearn dilatih
    dengan versi berbeda -> ModuleNotFoundError '_loss', atau file hilang), varian
    itu DILEWATI, bukan menggagalkan seluruh app. Penting untuk portabilitas:
    model aktif `restock_pipeline` (murni numpy) tetap jalan walau RF/GB/HW/SARIMA
    tak bisa di-unpickle di perangkat dengan versi pustaka berbeda.
    """
    md = Path(models_dir)
    models = {}
    skipped = {}

    def _try(key, fn):
        try:
            models[key] = fn()
        except Exception as e:  # versi pustaka beda / file hilang / korup
            skipped[key] = f"{type(e).__name__}: {e}"

    _try("naive", lambda: NaiveForecaster.load(md / "naive.pkl"))
    _try("random_forest", lambda: RandomForestForecaster.load(md / "random_forest.pkl"))
    _try("gradient_boosting", lambda: GradientBoostingForecaster.load(md / "gradient_boosting.pkl"))
    _try("holt_winters", lambda: HoltWintersForecaster.load(md / "holtwinters.pkl"))
    _try("sarima", lambda: SarimaForecaster.load(md / "sarima.pkl"))
    # Ensemble hanya bila kedua basisnya berhasil dimuat.
    if "holt_winters" in models and "sarima" in models:
        def _ens():
            weights = json.load(open(md / "ensemble_weights.json", encoding="utf-8"))
            models["ensemble_simple"] = EnsembleForecaster(
                models["holt_winters"], models["sarima"], weights, "simple")
            models["ensemble_weighted"] = EnsembleForecaster(
                models["holt_winters"], models["sarima"], weights, "weighted")
        try:
            _ens()
        except Exception as e:
            skipped["ensemble"] = f"{type(e).__name__}: {e}"
    # SES & restock_pipeline: artefak GLOBAL tunggal di models/ root (selalu
    # tersedia apa pun dir versi aktif).
    ses_path = C.MODELS_DIR / "model_ses.pkl"
    if ses_path.exists():
        _try("ses", lambda: SESForecaster.load(ses_path))
    rpf_path = C.MODELS_DIR / "restock_pipeline_model_final.pkl"
    if rpf_path.exists():
        _try("restock_final", lambda: RestockPipelineFinalForecaster.load(rpf_path))

    if skipped:
        print("  [info] varian model dilewati (gagal dimuat, kemungkinan beda "
              "versi pustaka): " + ", ".join(sorted(skipped)))
    if not models:
        raise RuntimeError(
            f"Tidak ada artefak model yang berhasil dimuat dari {md}. "
            "Periksa folder models/ dan kecocokan versi scikit-learn/statsmodels.")
    return models


class ForecastService:
    def __init__(self):
        self.models = {}
        self.cluster_map = {}
        self.metrics = {}
        self.best_model = "naive"
        self.active_version = None
        self._loaded_dir = None

    # ---- cluster/label (selalu dari root models/) ----
    def _ensure_cluster_map(self):
        if not self.cluster_map:
            p = C.MODELS_DIR / "cluster_labels.json"
            self.cluster_map = json.load(open(p, encoding="utf-8"))["obat"]

    def segmen(self, obat: str):
        self._ensure_cluster_map()
        info = self.cluster_map.get(obat, {})
        return info.get("label"), info.get("cluster")

    # ---- pemuatan versi aktif ----
    def load_active(self, force: bool = False):
        """Muat artefak dari versi 'active' di model_registry."""
        from ..models import ModelRegistry
        with _lock:
            self._ensure_cluster_map()
            row = ModelRegistry.query.filter_by(status="active").first()
            if row is None:
                # fallback: root models/ + best_model.json
                models_dir = C.MODELS_DIR
                best = json.load(open(C.MODELS_DIR / "best_model.json", encoding="utf-8"))["best_model"]
                version = None
            else:
                models_dir = Path(row.path_artefak)
                best = row.nama_model
                version = row.version_id
            if (not force) and self._loaded_dir == str(models_dir) and self.models:
                # Versi/dir sama tapi model aktif bisa berganti (mis. ses<->rf
                # yang berbagi dir root) -> selalu segarkan best_model.
                self.best_model = _pick_best(self.models, best)
                self.active_version = version
                return
            self.models = _load_dir(models_dir)
            self.best_model = _pick_best(self.models, best)
            self.active_version = version
            self._loaded_dir = str(models_dir)
            try:
                self.metrics = json.load(open(Path(models_dir) / "metrics.json", encoding="utf-8"))
            except Exception:
                self.metrics = {}

    def get_model(self, key: str | None):
        self.load_active()
        k = key or self.best_model
        if k not in self.models:
            k = self.best_model
        return k, self.models[k]

    def predict(self, obat: str, model_key: str | None = None, horizon: int = 1):
        _, model = self.get_model(model_key)
        return model.predict(obat, horizon)

    def predict_many(self, obat_list, model_key: str | None = None, horizon: int = 1):
        """Prediksi banyak obat sekaligus. Untuk model ML (RF/GB) memakai
        forecast rekursif batch (cepat untuk horizon panjang). Model lain
        (HW/SARIMA/Naive/Ensemble) sudah cepat -> loop biasa.
        Mengembalikan {obat: {'model','periode':[...],'prediksi':[...]}}."""
        import ml_features as F
        from forecaster import MLForecaster
        key, model = self.get_model(model_key)
        out = {}
        if isinstance(model, MLForecaster):
            hist = {o: model.history[o] for o in obat_list if o in model.history}
            cl = {o: model.cluster_map.get(o, {}).get("cluster", -1) for o in hist}
            periods, preds = F.recursive_forecast_batch(
                model.model, model.feature_cols, hist, cl, horizon)
            for o in hist:
                out[o] = {"model": model.name, "periode": list(periods),
                          "prediksi": preds[o]}
        else:
            for o in obat_list:
                try:
                    out[o] = model.predict(o, horizon)
                except KeyError:
                    continue
        return out

    def list_obat(self):
        """Daftar obat untuk dropdown = obat yang ADA di panel DB (sumber
        kebenaran data per periode), agar pencarian panel[obat] selalu cocok
        apa pun model aktifnya. Fallback ke history model bila panel kosong."""
        try:
            from ..models import PanelBulanan
            rows = PanelBulanan.query.with_entities(PanelBulanan.obat).distinct().all()
            obat = sorted({o for (o,) in rows})
            if obat:
                return obat
        except Exception:
            pass
        self.load_active()
        return sorted(self.models[self.best_model].history.keys())


# singleton
service = ForecastService()
