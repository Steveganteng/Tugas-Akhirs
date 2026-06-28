"""Fitur supervised KAUSAL untuk model ML (RF, GB).
Semua fitur hanya memakai informasi MASA LALU (lag/rolling) -> anti-leakage.
"""
import numpy as np
import pandas as pd
import config as C


def make_supervised(panel: pd.DataFrame, cluster_map: dict,
                    feature_list=None) -> pd.DataFrame:
    """Bangun tabel supervised dari panel long.
    Satu baris per (obat, periode) dgn fitur lag & rolling dari demand & stok.
    """
    rows = []
    for obat, g in panel.groupby(C.P_OBAT):
        g = g.sort_values(C.P_PERIOD).reset_index(drop=True)
        d = g[C.P_DEMAND].values.astype(float)
        s = g[C.P_STOCK].values.astype(float)
        periods = g[C.P_PERIOD].values
        cl = cluster_map.get(obat, {}).get("cluster", -1)
        n = len(g)
        for t in range(n):
            feat = {C.P_OBAT: obat, C.P_PERIOD: periods[t],
                    "t": t, "month": int(str(periods[t])[5:7]),
                    "cluster": cl, "target": d[t]}
            for lag in C.LAGS:
                feat[f"lag_{lag}"] = d[t - lag] if t - lag >= 0 else np.nan
            for w in C.ROLL_WINDOWS:
                if t - w >= 0:
                    feat[f"rollmean_{w}"] = d[t - w:t].mean()
                    feat[f"rollstd_{w}"] = d[t - w:t].std(ddof=0)
                else:
                    feat[f"rollmean_{w}"] = np.nan
                    feat[f"rollstd_{w}"] = np.nan
            feat["stok_lag_1"] = s[t - 1] if t - 1 >= 0 else np.nan
            rows.append(feat)
    df = pd.DataFrame(rows)
    return df


def feature_columns():
    cols = ["t", "month", "cluster"]
    cols += [f"lag_{l}" for l in C.LAGS]
    for w in C.ROLL_WINDOWS:
        cols += [f"rollmean_{w}", f"rollstd_{w}"]
    cols += ["stok_lag_1"]
    return cols


def build_history(panel: pd.DataFrame) -> dict:
    """history[obat] = {'periods':[...], 'demand':[...], 'stok':[...]}.
    Disimpan dalam artefak agar predict() mandiri tanpa parquet."""
    hist = {}
    for obat, g in panel.groupby(C.P_OBAT):
        g = g.sort_values(C.P_PERIOD)
        hist[obat] = {
            "periods": g[C.P_PERIOD].tolist(),
            "demand": g[C.P_DEMAND].astype(float).tolist(),
            "stok": g[C.P_STOCK].astype(float).tolist(),
        }
    return hist


def next_periods(last_period: str, horizon: int):
    """Daftar string periode (YYYY-MM) setelah last_period."""
    p = pd.Period(last_period, freq="M")
    return [str(p + i) for i in range(1, horizon + 1)]


def recursive_forecast_batch(model, feature_cols, hist_by_obat: dict,
                             cluster_by_obat: dict, horizon: int):
    """Forecast rekursif untuk BANYAK obat sekaligus (vektorisasi).

    Memanggil model.predict() sekali per langkah horizon atas matrix semua obat,
    bukan per-obat -> jauh lebih cepat untuk horizon panjang & banyak obat.
    Asumsi: semua obat punya periode terakhir sama (panel sudah reindex seragam).
    Mengembalikan (out_periods, {obat: [pred,...]}).
    """
    obats = list(hist_by_obat.keys())
    if not obats:
        return [], {}
    demand = {o: list(hist_by_obat[o]["demand"]) for o in obats}
    stok = {o: list(hist_by_obat[o]["stok"]) for o in obats}
    last_period = hist_by_obat[obats[0]]["periods"][-1]
    out_periods = next_periods(last_period, horizon)
    preds = {o: [] for o in obats}
    for h in range(horizon):
        month = int(out_periods[h][5:7])
        X = np.empty((len(obats), len(feature_cols)), dtype=float)
        for i, o in enumerate(obats):
            d = demand[o]
            feat = {"t": len(d), "month": month,
                    "cluster": cluster_by_obat.get(o, -1)}
            for lag in C.LAGS:
                feat[f"lag_{lag}"] = d[-lag] if lag <= len(d) else d[0]
            for w in C.ROLL_WINDOWS:
                window = d[-w:] if len(d) >= w else d
                feat[f"rollmean_{w}"] = float(np.mean(window))
                feat[f"rollstd_{w}"] = float(np.std(window))
            feat["stok_lag_1"] = stok[o][-1]
            X[i] = [feat[c] for c in feature_cols]
        yhat = np.maximum(0.0, np.asarray(model.predict(X), dtype=float))
        for i, o in enumerate(obats):
            v = float(yhat[i])
            preds[o].append(round(v, 2))
            demand[o].append(v)
            stok[o].append(stok[o][-1])
    return out_periods, preds


def recursive_forecast(model, feature_cols, hist_obat: dict, cluster: int,
                       horizon: int):
    """Forecast rekursif horizon langkah untuk satu obat memakai model ML global.
    hist_obat: {'periods','demand','stok'}. Mengembalikan (periods, preds)."""
    demand = list(hist_obat["demand"])
    stok = list(hist_obat["stok"])
    periods = list(hist_obat["periods"])
    preds = []
    out_periods = next_periods(periods[-1], horizon)
    for h in range(horizon):
        t = len(demand)
        feat = {"t": t, "month": int(out_periods[h][5:7]), "cluster": cluster}
        for lag in C.LAGS:
            feat[f"lag_{lag}"] = demand[-lag] if lag <= len(demand) else demand[0]
        for w in C.ROLL_WINDOWS:
            window = demand[-w:] if len(demand) >= w else demand
            feat[f"rollmean_{w}"] = float(np.mean(window))
            feat[f"rollstd_{w}"] = float(np.std(window))
        feat["stok_lag_1"] = stok[-1]
        x = np.array([[feat[c] for c in feature_cols]], dtype=float)
        yhat = float(model.predict(x)[0])
        yhat = max(0.0, yhat)
        preds.append(yhat)
        demand.append(yhat)
        stok.append(stok[-1])  # asumsikan stok tetap (tak ada info masa depan)
    return out_periods, preds
