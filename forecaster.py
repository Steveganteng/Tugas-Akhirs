"""Interface seragam 6 pendekatan forecasting (Tahap 5).

KONTRAK:
    class BaseForecaster:
        name: str
        def predict(self, nama_obat, horizon_bulan=1) -> dict
        def save(self, path)
        @classmethod
        def load(cls, path)

predict() mengembalikan:
    {"nama_obat":..., "model":..., "prediksi":[...], "periode":["2026-01",...]}

Setiap forecaster MANDIRI: menyimpan history per obat di dalam artefak,
sehingga web dapat memanggil predict() TANPA melatih ulang & tanpa parquet.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import joblib
import config as C
import ml_features as F

warnings.filterwarnings("ignore")


def _sanity_guard(vals, hist):
    """P3 — Guard ledakan: bila ada forecast tak hingga / > K×max(histori),
    seluruh deret forecast jatuh ke naive (lag-1). Mencegah SARIMA/HW eksplosif
    pada deret pendek menulari hasil & ensemble."""
    if not hist:
        return vals
    cap = C.FORECAST_SANITY_K * max(hist)
    if cap > 0 and any((not np.isfinite(v)) or v > cap for v in vals):
        return [hist[-1]] * len(vals)
    return vals


class BaseForecaster:
    name = "base"

    def __init__(self, history: dict):
        self.history = history  # {obat: {periods, demand, stok}}

    # ---- util ----
    def _check(self, nama_obat):
        if nama_obat not in self.history:
            raise KeyError(f"Obat tidak dikenal: {nama_obat}")

    def _last_period(self, nama_obat):
        return self.history[nama_obat]["periods"][-1]

    def _result(self, nama_obat, periods, preds):
        return {
            "nama_obat": nama_obat,
            "model": self.name,
            "prediksi": [round(float(max(0.0, p)), 2) for p in preds],
            "periode": list(periods),
        }

    def predict(self, nama_obat: str, horizon_bulan: int = 1) -> dict:
        raise NotImplementedError

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        return joblib.load(path)


# ============================ a. NAIVE ============================
class NaiveForecaster(BaseForecaster):
    name = "naive"

    @classmethod
    def fit(cls, panel: pd.DataFrame, **kw):
        return cls(F.build_history(panel))

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        self._check(nama_obat)
        last_val = self.history[nama_obat]["demand"][-1]
        periods = F.next_periods(self._last_period(nama_obat), horizon_bulan)
        return self._result(nama_obat, periods, [last_val] * horizon_bulan)


# ===================== b/c. ML (RF, GB) ==========================
class MLForecaster(BaseForecaster):
    name = "ml"

    def __init__(self, history, model, feature_cols, cluster_map):
        super().__init__(history)
        self.model = model
        self.feature_cols = feature_cols
        self.cluster_map = cluster_map

    @classmethod
    def fit(cls, panel, model, feature_cols, cluster_map, **kw):
        sup = F.make_supervised(panel, cluster_map)
        sup = sup.dropna(subset=feature_cols + ["target"])
        sup = sup.sort_values(C.P_PERIOD).reset_index(drop=True)
        X = sup[feature_cols].values
        y = sup["target"].values
        model.fit(X, y)
        return cls(F.build_history(panel), model, feature_cols, cluster_map)

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        self._check(nama_obat)
        cl = self.cluster_map.get(nama_obat, {}).get("cluster", -1)
        periods, preds = F.recursive_forecast(
            self.model, self.feature_cols, self.history[nama_obat], cl, horizon_bulan)
        return self._result(nama_obat, periods, preds)


class RandomForestForecaster(MLForecaster):
    name = "random_forest"


class GradientBoostingForecaster(MLForecaster):
    name = "gradient_boosting"


# ===================== d. HOLT-WINTERS ===========================
class HoltWintersForecaster(BaseForecaster):
    name = "holt_winters"

    def __init__(self, history, fits: dict):
        super().__init__(history)
        self.fits = fits  # {obat: {'type':'hw','params':...} | {'type':'mean','value':v}}

    @classmethod
    def fit(cls, panel, **kw):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        hist = F.build_history(panel)
        fits = {}
        for obat, h in hist.items():
            y = np.asarray(h["demand"], dtype=float)
            n_obs = int(np.sum(np.asarray(  # observasi asli
                panel[(panel[C.P_OBAT] == obat)][C.P_IS_OBS])))
            if n_obs < C.MIN_OBS_TS or np.all(y == 0):
                fits[obat] = {"type": "mean", "value": float(np.mean(y[y > 0])) if np.any(y > 0) else 0.0}
                continue
            best = None
            for trend in C.HW_TREND:
                for damped in C.HW_DAMPED:
                    if trend is None and damped:
                        continue
                    try:
                        m = ExponentialSmoothing(
                            y, trend=trend, damped_trend=damped,
                            seasonal=None, initialization_method="estimated")
                        r = m.fit()
                        aic = r.aic
                        if best is None or aic < best[0]:
                            best = (aic, trend, damped, r)
                    except Exception:
                        continue
            if best is None:
                fits[obat] = {"type": "mean", "value": float(np.mean(y))}
            else:
                fits[obat] = {"type": "hw", "trend": best[1], "damped": best[2],
                              "aic": float(best[0]), "result": best[3]}
        return cls(hist, fits)

    def _forecast_obat(self, nama_obat, horizon):
        f = self.fits[nama_obat]
        if f["type"] == "mean":
            return [f["value"]] * horizon
        hist = self.history[nama_obat]["demand"]
        try:
            vals = list(np.asarray(f["result"].forecast(horizon), dtype=float))
        except Exception:
            return [hist[-1]] * horizon
        return _sanity_guard(vals, hist)

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        self._check(nama_obat)
        periods = F.next_periods(self._last_period(nama_obat), horizon_bulan)
        return self._result(nama_obat, periods, self._forecast_obat(nama_obat, horizon_bulan))


# ===================== e. SARIMA ================================
class SarimaForecaster(BaseForecaster):
    name = "sarima"

    def __init__(self, history, fits: dict, n_fallback: int = 0):
        super().__init__(history)
        self.fits = fits
        self.n_fallback = n_fallback

    @classmethod
    def fit(cls, panel, **kw):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        hist = F.build_history(panel)
        fits = {}
        n_fallback = 0
        for obat, h in hist.items():
            y = np.asarray(h["demand"], dtype=float)
            n_obs = int(np.sum(np.asarray(panel[(panel[C.P_OBAT] == obat)][C.P_IS_OBS])))
            if n_obs < C.MIN_OBS_TS or np.all(y == 0):
                fits[obat] = {"type": "naive", "value": float(y[-1])}
                n_fallback += 1
                continue
            best = None
            for p in C.SARIMA_P:
                for d in C.SARIMA_D:
                    for q in C.SARIMA_Q:
                        if p == 0 and q == 0 and d == 0:
                            continue
                        try:
                            m = SARIMAX(y, order=(p, d, q),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
                            r = m.fit(disp=False)
                            if best is None or r.aic < best[0]:
                                best = (r.aic, (p, d, q), r)
                        except Exception:
                            continue
            if best is None:
                fits[obat] = {"type": "naive", "value": float(y[-1])}
                n_fallback += 1
            else:
                fits[obat] = {"type": "sarima", "order": best[1],
                              "aic": float(best[0]), "result": best[2]}
        return cls(hist, fits, n_fallback)

    def _forecast_obat(self, nama_obat, horizon):
        f = self.fits[nama_obat]
        if f["type"] == "naive":
            return [f["value"]] * horizon
        hist = self.history[nama_obat]["demand"]
        try:
            vals = list(np.asarray(f["result"].forecast(horizon), dtype=float))
        except Exception:
            return [hist[-1]] * horizon
        return _sanity_guard(vals, hist)

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        self._check(nama_obat)
        periods = F.next_periods(self._last_period(nama_obat), horizon_bulan)
        return self._result(nama_obat, periods, self._forecast_obat(nama_obat, horizon_bulan))


# ===================== g. SES (Single Exponential Smoothing) =====
class SESForecaster(BaseForecaster):
    """Single Exponential Smoothing per obat — artefak dict MANDIRI.

    Struktur artefak (models/model_ses.pkl):
        {'meta': {...}, 'models': {obat: {'alpha','forecast_next','history',
         'work_months','resid_std','reorder_point',...}}, 'evaluation': DataFrame}

    Sifat SES: forecast h-langkah ke depan = level smoothing terakhir
    (``forecast_next``) -> deret forecast FLAT untuk seluruh horizon.

    Lookup nama obat toleran beda kapitalisasi (nama panel DB mis.
    'BETADINE 15 ML' vs nama artefak 'Betadine 15 ml').
    """
    name = "ses"

    def __init__(self, history, models, meta=None, last_period=None):
        super().__init__(history)
        self.models = models
        self.meta = meta or {}
        self.last_period = last_period or self._infer_last_period()
        self._ci = {k.lower(): k for k in models}

    def _infer_last_period(self):
        wm = self.meta.get("work_months") or [12]
        return f"2025-{max(wm):02d}"

    def _resolve(self, nama_obat):
        if nama_obat in self.models:
            return nama_obat
        return self._ci.get(nama_obat.lower())

    def _check(self, nama_obat):
        if self._resolve(nama_obat) is None:
            raise KeyError(f"Obat tidak dikenal SES: {nama_obat}")

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        key = self._resolve(nama_obat)
        if key is None:
            raise KeyError(f"Obat tidak dikenal SES: {nama_obat}")
        fnext = float(self.models[key].get("forecast_next", 0.0))
        periods = F.next_periods(self.last_period, horizon_bulan)
        out = self._result(nama_obat, periods, [fnext] * horizon_bulan)
        out["alpha"] = float(self.models[key].get("alpha", 0.0))
        return out

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        models = d.get("models", {})
        meta = d.get("meta", {})
        wm_default = meta.get("work_months", list(range(1, 13)))
        history = {}
        for obat, m in models.items():
            hist = list(m.get("history", []))
            months = m.get("work_months", wm_default)
            periods = [f"2025-{int(mm):02d}" for mm in months][:len(hist)]
            history[obat] = {
                "periods": periods,
                "demand": [float(x) for x in hist],
                "stok": [0.0] * len(hist),
            }
        return cls(history, models, meta)


# ============ h. RESTOCK PIPELINE (SES/Croston/SBA + musiman) =====
def _norm_name(s: str) -> str:
    """Normalisasi nama obat: lower + rapatkan spasi ganda (untuk lookup toleran
    beda kapitalisasi / spasi ganda antara panel DB & artefak)."""
    return " ".join(str(s).lower().split())


class RestockPipelineForecaster(BaseForecaster):
    """Adapter artefak `restock_pipeline_model.pkl` (kelas RestockModel) ke
    kontrak BaseForecaster web.

    RestockModel menyimpan, per obat: metode (SES/Croston/SBA/TSB/aturan_stok),
    parameter terlatih, histori bersih/winsorized, dan kebijakan (s,S). Forecast
    web = level satu-langkah (deseasonalized) × faktor musiman pooled per bulan
    target -> deret bulanan untuk horizon penuh. Lookup nama toleran beda
    kapitalisasi & spasi ganda (mis. panel 'HOT IN CREAM  60G').
    """
    name = "restock_pipeline"

    def __init__(self, rm, history, last_period: str = "2025-12"):
        super().__init__(history)
        self.rm = rm  # instance RestockModel
        self.last_period = last_period
        self._norm = {_norm_name(k): k for k in rm.per_obat}

    def _resolve(self, nama_obat):
        if nama_obat in self.rm.per_obat:
            return nama_obat
        return self._norm.get(_norm_name(nama_obat))

    def _check(self, nama_obat):
        if self._resolve(nama_obat) is None:
            raise KeyError(f"Obat tidak dikenal restock_pipeline: {nama_obat}")

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        key = self._resolve(nama_obat)
        if key is None:
            raise KeyError(f"Obat tidak dikenal restock_pipeline: {nama_obat}")
        level = self.rm.forecast_next(key)            # level satu-langkah (deseasonalized)
        si = getattr(self.rm, "seasonal_index", None) or {}
        periods = F.next_periods(self.last_period, horizon_bulan)
        preds = [level * float(si.get(int(p.split("-")[1]), 1.0)) for p in periods]
        return self._result(nama_obat, periods, preds)

    def restock_policy(self, nama_obat, current_stock) -> dict:
        """Kebijakan stok (s,S) bawaan RestockModel untuk satu obat & stok terkini.

        Min (s) = titik pesan ulang (rop); Max (S) = level target. Jumlah restok
        = recommend_order = ceil(Max - stok) bila stok <= Min, selain itu 0.
        Status dipetakan ke 3 label UI (SEGERA RESTOK / PERLU DIPERHATIKAN /
        STOK AMAN) berdasarkan posisi stok terhadap Min & Max.
        """
        key = self._resolve(nama_obat)
        if key is None:
            raise KeyError(f"Obat tidak dikenal restock_pipeline: {nama_obat}")
        d = self.rm.per_obat[key]
        current_stock = max(0.0, float(current_stock))
        mn, mx, ss = d.get("Min"), d.get("Max"), d.get("safety_stock")
        has_mm = (mn is not None and mx is not None
                  and not np.isnan(mn) and not np.isnan(mx))
        order = int(self.rm.recommend_order(key, current_stock))
        if "Delisting" in str(d.get("kebijakan", "")) or not has_mm:
            status = "STOK AMAN"
        elif current_stock <= mn:
            status = "SEGERA RESTOK"
        elif current_stock <= mx:
            status = "PERLU DIPERHATIKAN"
        else:
            status = "STOK AMAN"
        return {
            "rop": round(float(mn), 2) if has_mm else None,
            "max": round(float(mx), 2) if has_mm else None,
            "safety_stock": (round(float(ss), 2)
                             if ss is not None and not np.isnan(ss) else None),
            "jumlah_rekomendasi": float(order),
            "status": status,
            "kebijakan": d.get("kebijakan"),
        }

    @classmethod
    def load(cls, path):
        import pickle
        with open(path, "rb") as fh:
            rm = pickle.load(fh)
        history = {}
        for obat, d in rm.per_obat.items():
            demand = [float(x) for x in (d.get("history_clean") or [])]
            periods = F.next_periods("2024-12", len(demand))  # 2025-01.. sepanjang histori
            history[obat] = {
                "periods": periods,
                "demand": demand,
                "stok": [float(d.get("stok_terkini") or 0.0)] * len(demand),
            }
        return cls(rm, history)


class RestockPipelineFinalForecaster(RestockPipelineForecaster):
    """Adapter artefak `restock_pipeline_model_final.pkl` — MODEL TERBAIK.
    Peramalan SES/Croston/SBA + strategi mean/ensemble + musiman (tanpa TSB), dan
    safety stock berbasis distribusi (bootstrap) untuk SEMUA kategori sehingga
    fill rate ~95-96%. Instance RestockModelV2; predict()/restock_policy() diwarisi.
    """
    name = "restock_final"


# ===================== f. ENSEMBLE HW + SARIMA ===================
class EnsembleForecaster(BaseForecaster):
    name = "ensemble"

    def __init__(self, hw: HoltWintersForecaster, sarima: SarimaForecaster,
                 weights: dict, method: str = "weighted"):
        super().__init__(hw.history)
        self.hw = hw
        self.sarima = sarima
        self.weights = weights  # {obat: {'hw':w1,'sarima':w2}}
        self.method = method  # 'simple' | 'weighted'

    def predict(self, nama_obat, horizon_bulan: int = 1) -> dict:
        self._check(nama_obat)
        ph = self.hw._forecast_obat(nama_obat, horizon_bulan)
        ps = self.sarima._forecast_obat(nama_obat, horizon_bulan)
        if self.method == "simple":
            w_hw = w_sa = 0.5
        else:
            w = self.weights.get(nama_obat, {"hw": 0.5, "sarima": 0.5})
            w_hw, w_sa = w["hw"], w["sarima"]
        preds = [w_hw * a + w_sa * b for a, b in zip(ph, ps)]
        periods = F.next_periods(self._last_period(nama_obat), horizon_bulan)
        out = self._result(nama_obat, periods, preds)
        out["model"] = f"ensemble_{self.method}"
        return out

    def save(self, path):
        joblib.dump({"hw": self.hw, "sarima": self.sarima,
                     "weights": self.weights, "method": self.method}, path)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        return cls(d["hw"], d["sarima"], d["weights"], d.get("method", "weighted"))
