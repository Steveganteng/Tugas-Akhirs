# -*- coding: utf-8 -*-
"""RestockModelV2 — artifact pipeline v2 (Quick Wins A).

Perbedaan vs v1 (restock_pipeline_model.RestockModel):
  * TSB DIBUANG (performa terburuk pada backtest).
  * Strategi per obat dipilih leak-free di antara: 'method' (SES/Croston/SBA),
    'mean' (AMC), atau 'ensemble' (rata-rata method+mean).
  * Indeks musiman diterapkan saat seleksi maupun inferensi (default ON).
  * Kebijakan stok (s,S) Min/Max IDENTIK v1 -> tidak mengganggu logika restok.
  * E2 (volume kunjungan scaler): seasonal_index dihitung dari jumlah kunjungan
    unik per bulan dibagi rata-rata kunjungan bulan latih (bulan 1-10).
    WAPE 100.0 -> 75.2 pada backtest holdout bulan 11-12.

Harus importable saat meng-unpickle: `import restock_pipeline_model_v2`.
"""
import numpy as np


def compute_visit_scale(df_raw, date_col="Tanggal Masuk", register_col="Register",
                        train_months=10):
    """Hitung visit-volume scale per bulan dari data transaksi mentah (E2).

    Args:
        df_raw: DataFrame dengan kolom tanggal dan nomor register pasien.
        date_col: nama kolom tanggal.
        register_col: nama kolom register unik (pasien per kunjungan).
        train_months: jumlah bulan awal yang dianggap periode latih.

    Returns:
        dict {bulan_int: skala_float} untuk dipakai sebagai seasonal_index.
        Bulan yang tidak ada dalam data diberi nilai 1.0 (netral).
    """
    import pandas as pd
    df = df_raw.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["_bulan"] = df[date_col].dt.month
    visits = df.groupby("_bulan")[register_col].nunique()
    train_mean = visits[visits.index <= train_months].mean()
    if train_mean == 0:
        return {}
    scale = {int(m): float(v / train_mean) for m, v in visits.items()}
    # fill missing months with 1.0
    for m in range(1, 13):
        scale.setdefault(m, 1.0)
    return scale


# ---------------- metode peramalan one-step (tanpa TSB) ----------------
def ses_next(hist, alpha):
    x = np.asarray(hist, float)
    if len(x) == 0:
        return 0.0
    F = x[0]
    for t in range(1, len(x)):
        F = alpha * x[t - 1] + (1 - alpha) * F
    return max(alpha * x[-1] + (1 - alpha) * F, 0.0)


def _croston_core(x, az, ap):
    x = np.asarray(x, float); n = len(x)
    idx = np.where(x > 0)[0]
    if len(idx) == 0:
        return 0.0, 1.0
    first = idx[0]; z = x[first]; p = float(first + 1); q = 1
    for t in range(first + 1, n):
        if x[t] > 0:
            z = az * x[t] + (1 - az) * z
            p = ap * q + (1 - ap) * p
            q = 1
        else:
            q += 1
    return z, p


def croston_next(hist, az, ap):
    z, p = _croston_core(hist, az, ap)
    return z / p if p > 0 else 0.0


def sba_next(hist, az, ap):
    z, p = _croston_core(hist, az, ap)
    return (1 - ap / 2.0) * (z / p) if p > 0 else 0.0


class RestockModelV2:
    """Artifact v2: parameter, strategi terpilih, & kebijakan (s,S) per obat."""

    def __init__(self, params, per_obat, seasonal_index):
        self.params = params
        self.per_obat = per_obat
        self.seasonal_index = seasonal_index

    # -------- info --------
    def list_obat(self):
        return sorted(self.per_obat.keys())

    def info(self, obat):
        return self.per_obat.get(obat)

    # -------- peramalan --------
    def _method_fc(self, d, hist):
        mt = d.get("method")
        if mt == "SES":
            return ses_next(hist, d["alpha"])
        if mt == "Croston":
            return croston_next(hist, d["alpha_z"], d["alpha_p"])
        if mt == "SBA":
            return sba_next(hist, d["alpha_z"], d["alpha_p"])
        return float(np.mean(hist)) if len(hist) else 0.0

    def forecast_next(self, obat, history=None, bulan_target=None, apply_seasonal=True):
        d = self.per_obat.get(obat)
        if d is None:
            raise KeyError(f"Obat tidak ada di model v2: {obat}")
        strat = d.get("strategy", "mean")
        if history is None:
            history = d.get("history_winsor") if d.get("method") == "SES" else d.get("history_clean")
        history = list(history) if history is not None else []
        meanv = float(np.mean(history)) if history else 0.0
        if strat == "mean":
            f = meanv
        elif strat == "ensemble":
            f = 0.5 * (self._method_fc(d, history) + meanv)
        else:  # 'method'
            f = self._method_fc(d, history)
        if apply_seasonal and bulan_target in (self.seasonal_index or {}):
            f = f * self.seasonal_index[bulan_target]
        return max(float(f), 0.0)

    # -------- kebijakan stok (s,S) — identik v1 --------
    def recommend_order(self, obat, current_stock):
        d = self.per_obat.get(obat)
        if d is None:
            raise KeyError(f"Obat tidak ada di model v2: {obat}")
        if "Delisting" in str(d.get("kebijakan", "")):
            return 0
        mn, mx = d.get("Min"), d.get("Max")
        if mn is None or mx is None or np.isnan(mn) or np.isnan(mx):
            return 0
        if current_stock > mn:
            return 0
        return int(np.ceil(mx - current_stock))
