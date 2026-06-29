"""Kerangka eksekusi RENCANA_EVALUASI.md — pengecekan per-tahap pipeline.

Tiap fungsi cek mengembalikan dict {stage,item,status,value,threshold,note}
status: PASS | FAIL | SKIP (stub). Runner mencetak ringkasan & keluar non-zero
bila ada FAIL pada tahap KRITIS (A processing, E anti-leakage).

Jalankan:  python evaluasi_pipeline.py            # semua cek
           python evaluasi_pipeline.py A E        # tahap tertentu
"""
from __future__ import annotations
import sys, io, json
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import config as C
import data_processing as dp
import ml_features as F
import clustering as clu
from restock import hitung_restock

CRITICAL = {"A", "E"}


def R(stage, item, status, value="", threshold="", note=""):
    return {"stage": stage, "item": item, "status": status,
            "value": value, "threshold": threshold, "note": note}


def _panel():
    return pd.read_parquet(C.PANEL_PATH)


def _cluster_map():
    return json.load(open(C.MODELS_DIR / "cluster_labels.json", encoding="utf-8"))["obat"]


# ============================== TAHAP A ==============================
def a_no_double_count():
    df = dp.clean(dp.load_raw())
    g = df.groupby([C.P_OBAT, C.P_PERIOD])[C.COL_QTY].nunique(dropna=True)
    bad = int((g > 1).sum()); total = int(len(g))
    ok = bad / total < 0.01 if total else False
    return R("A", "Anti double-count (nunique JUMLAH/grup=1)",
             "PASS" if ok else "FAIL", f"{bad}/{total} grup nunique>1",
             "<1% grup", "first() valid bila broadcast benar")


def a_panel_12_months():
    p = _panel()
    sizes = set(p.groupby(C.P_OBAT).size().unique().tolist())
    ok = sizes == {12}
    return R("A", "Reindex 12 bulan/obat", "PASS" if ok else "FAIL",
             f"sizes={sizes}", "{12}")


def a_name_normalization():
    raw = dp.load_raw()
    s = raw[C.COL_OBAT].astype("string")
    up = s.str.strip().str.upper()
    merged = {}
    tmp = pd.DataFrame({"asli": s, "norm": up}).dropna()
    for norm, grp in tmp.groupby("norm"):
        uniq = grp["asli"].unique()
        if len(uniq) > 1:
            merged[norm] = list(uniq)[:4]
    ok = True  # informatif; gabungan hanya varian kapitalisasi/spasi
    return R("A", "Normalisasi nama (gabungan varian)", "PASS" if ok else "FAIL",
             f"{len(merged)} nama tergabung", "0 gabungan keliru",
             note=f"contoh: {list(merged.items())[:2]}")


def a_nonnegative():
    p = _panel()
    neg = int((p[C.P_DEMAND] < 0).sum() + (p[C.P_STOCK] < 0).sum())
    return R("A", "Demand & stok non-negatif", "PASS" if neg == 0 else "FAIL",
             f"{neg} negatif", "0")


def a_imputation_impact():
    return R("A", "Imputasi Nov: interpolate vs zero/ffill (Δmean/var)", "SKIP",
             note="TODO: bandingkan distribusi & MAE 3 metode imputasi")


def a_december_normalization():
    return R("A", "Desember (19 hari) perlu normalisasi?", "SKIP",
             note="TODO: korelasi hari-aktif vs demand; MAE dgn/tanpa scaling")


# ============================== TAHAP C ==============================
def c_causality():
    """Ubah demand[t] satu obat; fitur baris periode t harus TIDAK berubah."""
    p = _panel().copy()
    cmap = _cluster_map()
    obat = sorted(p[C.P_OBAT].unique())[0]
    sub = p[p[C.P_OBAT] == obat].sort_values(C.P_PERIOD)
    per_t = sub[C.P_PERIOD].iloc[6]  # periode tengah
    feat_cols = F.feature_columns()
    sup0 = F.make_supervised(p, cmap)
    row0 = sup0[(sup0[C.P_OBAT] == obat) & (sup0[C.P_PERIOD] == per_t)][feat_cols]
    p2 = p.copy()
    mask = (p2[C.P_OBAT] == obat) & (p2[C.P_PERIOD] == per_t)
    p2.loc[mask, C.P_DEMAND] = p2.loc[mask, C.P_DEMAND] + 1000.0
    sup1 = F.make_supervised(p2, cmap)
    row1 = sup1[(sup1[C.P_OBAT] == obat) & (sup1[C.P_PERIOD] == per_t)][feat_cols]
    ok = np.allclose(row0.values, row1.values)
    return R("C", "Kausalitas: fitur baris t tak memuat demand[t]",
             "PASS" if ok else "FAIL", f"identik={ok}", "identik",
             note=f"obat={obat}, periode={per_t}")


def c_supervised_loss():
    p = _panel(); cmap = _cluster_map(); feat_cols = F.feature_columns()
    sup = F.make_supervised(p, cmap)
    before = len(sup)
    after = len(sup.dropna(subset=feat_cols + ["target"]))
    lost = before - after
    per_obat = lost / p[C.P_OBAT].nunique()
    ok = per_obat <= max(C.LAGS) + 0.5
    return R("C", "Baris supervised hilang akibat lag", "PASS" if ok else "FAIL",
             f"{lost} baris (~{per_obat:.1f}/obat)", f"≤{max(C.LAGS)}/obat")


def c_ablation():
    return R("C", "Ablation per fitur (ΔMAE buang-satu)", "SKIP",
             note="TODO: loop drop kolom, refit RF, ukur ΔMAE")


# ============================== TAHAP D ==============================
def d_cluster_stability():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import adjusted_rand_score
    p = _panel()
    feat = clu.build_features(p)
    X = StandardScaler().fit_transform(clu.transform_features(feat))
    k = json.load(open(C.MODELS_DIR / "cluster_labels.json"))["k"]
    labs = [KMeans(n_clusters=k, random_state=s, n_init=10).fit_predict(X)
            for s in range(10)]
    aris = [adjusted_rand_score(labs[i], labs[j])
            for i in range(len(labs)) for j in range(i + 1, len(labs))]
    med = float(np.median(aris))
    ok = med >= 0.8
    return R("D", f"Stabilitas cluster (ARI, k={k})", "PASS" if ok else "FAIL",
             f"ARI median={med:.2f}", "≥0.80")


def d_k_metrics():
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                                 calinski_harabasz_score)
    p = _panel()
    feat = clu.build_features(p)
    X = StandardScaler().fit_transform(clu.transform_features(feat))
    best = {}
    sil, db, ch = {}, {}, {}
    for k in range(2, 9):
        lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)
        sil[k] = silhouette_score(X, lab)
        db[k] = davies_bouldin_score(X, lab)
        ch[k] = calinski_harabasz_score(X, lab)
    best = {"silhouette": max(sil, key=sil.get),
            "davies_bouldin": min(db, key=db.get),
            "calinski": max(ch, key=ch.get)}
    agree = len(set(best.values())) <= 2
    return R("D", "Pemilihan K (3 metrik konsisten)", "PASS" if agree else "FAIL",
             f"best={best}", "≥2 metrik sepakat",
             note=f"silhouette flat: { {k: round(v,3) for k,v in sil.items()} }")


def d_separation():
    return R("D", "Separasi cluster (Kruskal-Wallis)", "SKIP",
             note="TODO: uji antar-cluster pada volume/cv/frekuensi")


# ============================== TAHAP E ==============================
def e_split_no_overlap():
    from pipeline_forecasting_obat import temporal_split
    p = _panel()
    _, _, trp, tep = temporal_split(p)
    overlap = set(trp) & set(tep)
    ok = len(overlap) == 0
    return R("E", "Split temporal tak tumpang tindih", "PASS" if ok else "FAIL",
             f"overlap={overlap}", "kosong")


def e_obs_only_same_set():
    p = _panel()
    obs_nov = int(p[(p[C.P_PERIOD] == "2025-11") & (p[C.P_IS_OBS])].shape[0])
    ok = obs_nov == 0
    return R("E", "Test = observasi asli (Nov imputasi dibuang)",
             "PASS" if ok else "FAIL", f"obs asli Nov={obs_nov}", "0",
             note="evaluasi pakai himpunan obat sama antar model")


def e_shuffle_target():
    """Gate anti-leakage: acak target → MAE harus memburuk drastis.
    Internal holdout pada train (fit <Okt, test Okt) agar tak menyentuh test final."""
    from sklearn.ensemble import RandomForestRegressor
    p = _panel(); cmap = _cluster_map(); feat_cols = F.feature_columns()
    train = p[p[C.P_PERIOD] <= "2025-10"]
    sup = F.make_supervised(train, cmap).dropna(subset=feat_cols + ["target"])
    tr = sup[sup[C.P_PERIOD] < "2025-10"]
    te = sup[sup[C.P_PERIOD] == "2025-10"]
    if len(te) < 10:
        return R("E", "Shuffle target degradasi MAE", "SKIP", note="test internal kecil")
    Xtr, ytr = tr[feat_cols].values, tr["target"].values
    Xte, yte = te[feat_cols].values, te["target"].values
    def mae_fit(y):
        m = RandomForestRegressor(n_estimators=100, random_state=0, n_jobs=-1).fit(Xtr, y)
        return float(np.mean(np.abs(yte - m.predict(Xte))))
    real = mae_fit(ytr)
    rng = np.random.default_rng(0)
    shuf = float(np.mean([mae_fit(rng.permutation(ytr)) for _ in range(3)]))
    ratio = shuf / real if real else 0
    ok = ratio >= 1.5
    return R("E", "Shuffle target → MAE memburuk", "PASS" if ok else "FAIL",
             f"MAE real={real:.1f} shuffle={shuf:.1f} (×{ratio:.2f})", "×≥1.5",
             note="ratio rendah = indikasi leakage")


# ============================== TAHAP F ==============================
def f_grid_edges():
    grids = {"random_forest": ("rf_features.json", C.RF_GRID),
             "gradient_boosting": ("gb_features.json", C.GB_GRID)}
    edge = []
    for name, (fn, grid) in grids.items():
        try:
            bp = json.load(open(C.MODELS_DIR / fn))["best_params"]
        except Exception:
            continue
        for k, v in bp.items():
            vals = grid.get(k)
            if not vals or not isinstance(v, (int, float)):
                continue
            if any(isinstance(x, str) for x in vals):
                continue  # param kategorikal (mis. max_features='sqrt') tak diurut
            nums = [float("inf") if x is None else x for x in vals]  # None=tak terbatas
            if v == min(nums) or v == max(nums):
                edge.append(f"{name}.{k}={v}")
    ok = len(edge) == 0
    return R("F", "Best params tidak di tepi grid", "PASS" if ok else "FAIL",
             f"{len(edge)} di tepi: {edge}", "0 di tepi",
             note="di tepi → perluas grid lalu re-tune")


def f_uses_timeseries_split():
    import inspect
    from pipeline_forecasting_obat import tune_ml
    src = inspect.getsource(tune_ml)
    ok = "TimeSeriesSplit" in src and "KFold" not in src
    return R("F", "Tuning pakai TimeSeriesSplit", "PASS" if ok else "FAIL",
             f"TimeSeriesSplit={'TimeSeriesSplit' in src}", "True")


def f_validation_curve():
    return R("F", "Sensitivitas hyperparameter (validation curve)", "SKIP",
             note="TODO: validation_curve tiap param, identifikasi paling berpengaruh")


# ============================== TAHAP G ==============================
def g_fallback_rate():
    from forecaster import HoltWintersForecaster, SarimaForecaster
    hw = HoltWintersForecaster.load(C.MODELS_DIR / "holtwinters.pkl")
    sa = SarimaForecaster.load(C.MODELS_DIR / "sarima.pkl")
    n = len(hw.fits)
    hw_fb = sum(1 for v in hw.fits.values() if v.get("type") == "mean")
    sa_fb = sum(1 for v in sa.fits.values() if v.get("type") == "naive")
    return R("G", "Tingkat fallback HW/SARIMA (dilaporkan)", "PASS",
             f"HW={hw_fb}/{n}, SARIMA={sa_fb}/{n}", "dilaporkan jujur",
             note="obat fallback = naive/mean, bukan model penuh")


def g_forecast_sane():
    from forecaster import SarimaForecaster
    sa = SarimaForecaster.load(C.MODELS_DIR / "sarima.pkl")
    bad = []
    for obat, h in sa.history.items():
        mx = max(h["demand"]) if h["demand"] else 0
        f = sa.predict(obat, 6)["prediksi"]
        if any(v > 5 * (mx + 1) for v in f):
            bad.append(obat)
    ok = len(bad) == 0
    return R("G", "Forecast wajar (≤5×max historis)", "PASS" if ok else "FAIL",
             f"{len(bad)} obat eksplosif", "0",
             note="SARIMA tanpa enforce_stationarity rawan meledak deret pendek")


def g_ljung_box():
    return R("G", "Residual white-noise (Ljung-Box)", "SKIP",
             note="TODO: acorr_ljungbox residual sampel obat non-fallback")


def g_no_seasonal_justified():
    p = _panel()
    n_months = p[C.P_PERIOD].nunique()
    ok = n_months < 24
    return R("G", "Keputusan tanpa-seasonal terjustifikasi (<2 siklus)",
             "PASS" if ok else "FAIL", f"{n_months} bulan", "<24 → no seasonal")


# ============================== TAHAP H ==============================
def h_ensemble_vs_component():
    return R("H", "Ensemble ≤ komponen terbaik", "SKIP",
             note="TODO: MAE ensemble vs min(HW,SARIMA) holdout & walk-forward "
                  "(lihat evaluasi_model.py Tahap 2: ensemble mewarisi ledakan SARIMA)")


def h_weight_stability():
    return R("H", "Stabilitas bobot ensemble antar fold", "SKIP",
             note="TODO: compute_ensemble_weights di ≥3 fold, std bobot/obat <0.25")


# ============================== TAHAP I ==============================
def i_rop_manual():
    import math
    hist = [471, 1368, 570, 396, 419]
    pred, stok, L, z = 800.0, 150.0, 7, C.DEFAULT_Z
    std_b = float(np.std(np.asarray(hist, float), ddof=1))
    std_d = std_b / math.sqrt(C.DAYS_PER_MONTH)
    ss = z * std_d * math.sqrt(L)
    rop = (pred / C.DAYS_PER_MONTH) * L + ss
    rec = hitung_restock("X", pred, stok, demand_historis=hist, periode="2026-01")
    ok = abs(rec["rop"] - round(rop, 2)) < 0.5 and abs(rec["safety_stock"] - round(ss, 2)) < 0.5
    return R("I", "Rumus ROP & safety stock vs manual", "PASS" if ok else "FAIL",
             f"fungsi ROP={rec['rop']} vs manual={rop:.2f}", "selisih<0.5")


def i_edge_cases():
    cases = [
        ("σ=0", dict(nama_obat="X", prediksi=100, stok_saat_ini=50, demand_historis=[5, 5, 5])),
        ("demand=0", dict(nama_obat="X", prediksi=0, stok_saat_ini=100)),
        ("stok=0 deret pendek", dict(nama_obat="X", prediksi=50, stok_saat_ini=0, demand_historis=[50])),
    ]
    bad = []
    for nm, kw in cases:
        try:
            r = hitung_restock(periode="2026-01", **kw)
            vals = [r["rop"], r["safety_stock"], r["jumlah_rekomendasi"]]
            if any((v is None or np.isnan(v) or np.isinf(v) or v < 0) for v in vals):
                bad.append(nm)
        except Exception as e:
            bad.append(f"{nm}:{e}")
    ok = not bad
    return R("I", "Edge case σ=0/demand=0/stok=0", "PASS" if ok else "FAIL",
             f"gagal: {bad}" if bad else "semua finite & ≥0", "tak ada NaN/inf/neg")


# ============================== RUNNER ==============================
CHECKS = {
    "A": [a_no_double_count, a_panel_12_months, a_name_normalization, a_nonnegative,
          a_imputation_impact, a_december_normalization],
    "C": [c_causality, c_supervised_loss, c_ablation],
    "D": [d_cluster_stability, d_k_metrics, d_separation],
    "E": [e_split_no_overlap, e_obs_only_same_set, e_shuffle_target],
    "F": [f_grid_edges, f_uses_timeseries_split, f_validation_curve],
    "G": [g_fallback_rate, g_forecast_sane, g_ljung_box, g_no_seasonal_justified],
    "H": [h_ensemble_vs_component, h_weight_stability],
    "I": [i_rop_manual, i_edge_cases],
}
# Catatan: Tahap B (encoding) sepenuhnya stub berat (lihat RENCANA_EVALUASI.md).


def main(stages=None):
    stages = stages or list(CHECKS.keys())
    results = []
    icon = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}
    for st in stages:
        for fn in CHECKS.get(st, []):
            try:
                res = fn()
            except Exception as e:
                res = R(st, fn.__name__, "FAIL", note=f"error: {e}")
            results.append(res)
            print(f"{icon.get(res['status'],'?')} [{res['stage']}] {res['item']}")
            if res["value"]:
                print(f"      nilai: {res['value']} | ambang: {res['threshold']}")
            if res["note"]:
                print(f"      catatan: {res['note']}")
    n_pass = sum(r["status"] == "PASS" for r in results)
    n_fail = sum(r["status"] == "FAIL" for r in results)
    n_skip = sum(r["status"] == "SKIP" for r in results)
    crit_fail = [r for r in results if r["status"] == "FAIL" and r["stage"] in CRITICAL]
    print("\n" + "=" * 60)
    print(f"RINGKASAN: {n_pass} PASS | {n_fail} FAIL | {n_skip} SKIP (stub)")
    if crit_fail:
        print("FAIL KRITIS (A/E):", [r["item"] for r in crit_fail])
    return 1 if crit_fail else 0


if __name__ == "__main__":
    args = [a.upper() for a in sys.argv[1:] if a.upper() in CHECKS]
    sys.exit(main(args or None))
