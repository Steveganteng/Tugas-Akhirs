"""Perbaikan model (P1+P2): SegmentRouter.

Memilih model TERBAIK per segmen cluster (pemetaan diturunkan dari data
VALIDASI, bukan test — lihat evaluasi_v2.py) lalu mendelegasikan prediksi.
Obat slow-moving / deret terlalu pendek dipaksa ke Naive (P2), karena terbukti
model deret-waktu/ML tidak mengalahkan naive di kelas tersebut.

Memakai ulang interface `BaseForecaster.predict(obat, horizon)`.
"""
from __future__ import annotations
import config as C
from forecaster import BaseForecaster


class SegmentRouter(BaseForecaster):
    name = "segment_router"

    def __init__(self, models: dict, cluster_map: dict, seg_to_model: dict,
                 n_obs: dict | None = None, min_obs: int = C.MIN_OBS_TS,
                 naive_key: str = "Naive", default_key: str = "Holt-Winters",
                 force_naive_prefix: str = "Slow-moving"):
        # models: {display_name: forecaster}; salah satunya dipakai utk history
        self.models = models
        self.cluster_map = cluster_map
        self.seg_to_model = seg_to_model
        self.n_obs = n_obs or {}
        self.min_obs = min_obs
        self.naive_key = naive_key
        self.default_key = default_key if default_key in models else naive_key
        self.force_naive_prefix = force_naive_prefix
        self.history = models[naive_key].history  # mandiri spt forecaster lain

    def _route(self, obat: str) -> str:
        """Tentukan model untuk satu obat (alasan dapat diaudit)."""
        seg = self.cluster_map.get(obat, {}).get("label", "")
        # P2: deret pendek -> naive
        if self.n_obs.get(obat, 99) < self.min_obs:
            return self.naive_key
        # P2: slow-moving -> naive
        if seg.startswith(self.force_naive_prefix):
            return self.naive_key
        # P1: model terbaik per segmen (dari validasi)
        return self.seg_to_model.get(seg, self.default_key)

    def predict(self, nama_obat: str, horizon_bulan: int = 1) -> dict:
        self._check(nama_obat)
        key = self._route(nama_obat)
        out = self.models[key].predict(nama_obat, horizon_bulan)
        out = dict(out)
        out["model"] = f"router:{key}"
        return out

    def routing_table(self) -> dict:
        """{obat: model_key} untuk audit/transparansi."""
        return {o: self._route(o) for o in self.history}
