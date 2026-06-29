"""Tahap 5 — Pipeline training end-to-end 6 pendekatan forecasting.

Alur:
  1. Muat panel + cluster map.
  2. Split TEMPORAL: 2 bulan terakhir = test (anti-leakage).
  3. Tuning RF & GB via GridSearchCV + TimeSeriesSplit (BUKAN KFold acak).
  4. Latih ke-6 pendekatan pada TRAIN, prediksi test, hitung metrik (SATU tabel).
  5. Pilih model terbaik (MAE test) -> best_model.json.
  6. RETRAIN semua pada SELURUH data dgn parameter tuning -> simpan artefak produksi.
  7. Tulis 03_perbandingan_model.md.

Jalankan: python pipeline_forecasting_obat.py
"""
import sys, json, time
# Set encoding stdout ke UTF-8 hanya saat dijalankan normal. Lewati di bawah
# pytest (mengganggu capture). Pakai reconfigure (bukan TextIOWrapper baru) agar
# tidak menutup buffer saat modul ini diimpor setelah skrip lain juga membungkus.
if "pytest" not in sys.modules:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

import config as C
import ml_features as F
from forecaster import (
    NaiveForecaster, RandomForestForecaster, GradientBoostingForecaster,
    HoltWintersForecaster, SarimaForecaster, EnsembleForecaster, MLForecaster,
)


# ---------------- metrik ----------------
def mae(y, yh): return float(np.mean(np.abs(y - yh)))
def rmse(y, yh): return float(np.sqrt(np.mean((y - yh) ** 2)))

def smape(y, yh):
    denom = np.abs(y) + np.abs(yh)
    mask = denom > 0
    return float(np.mean(2 * np.abs(y[mask] - yh[mask]) / denom[mask]) * 100)

def mape(y, yh):
    mask = y > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y[mask] - yh[mask]) / y[mask]) * 100)

def r2(y, yh):
    ss_res = np.sum((y - yh) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


# ---------------- split temporal ----------------
def temporal_split(panel):
    periods = sorted(panel[C.P_PERIOD].unique())
    test_periods = periods[-C.TEST_MONTHS:]
    train_periods = periods[:-C.TEST_MONTHS]
    train = panel[panel[C.P_PERIOD].isin(train_periods)].copy()
    test = panel[panel[C.P_PERIOD].isin(test_periods)].copy()
    return train, test, train_periods, test_periods


# ---------------- tuning ML ----------------
def tune_ml(train_panel, cluster_map, estimator, grid, label):
    feat_cols = F.feature_columns()
    sup = F.make_supervised(train_panel, cluster_map).dropna(
        subset=feat_cols + ["target"]).sort_values(C.P_PERIOD).reset_index(drop=True)
    X, y = sup[feat_cols].values, sup["target"].values
    tscv = TimeSeriesSplit(n_splits=C.TS_SPLITS)
    gs = GridSearchCV(estimator, grid, cv=tscv,
                      scoring="neg_mean_absolute_error", n_jobs=-1)
    t0 = time.time()
    gs.fit(X, y)
    print(f"  [{label}] tuning {len(sup)} baris, best CV MAE="
          f"{-gs.best_score_:.2f} ({time.time()-t0:.1f}s)")
    print(f"  [{label}] best params: {gs.best_params_}")
    return gs.best_params_, feat_cols


# ---------------- evaluasi satu forecaster ----------------
def collect_preds(fc, test_panel, test_periods, horizon):
    """Kumpulkan (y_true, y_pred) sejajar untuk semua obat pada periode test."""
    actual = test_panel.pivot_table(index=C.P_OBAT, columns=C.P_PERIOD,
                                    values=C.P_DEMAND)
    ys, yhs = [], []
    for obat in fc.history.keys():
        if obat not in actual.index:
            continue
        res = fc.predict(obat, horizon)
        pmap = dict(zip(res["periode"], res["prediksi"]))
        for per in test_periods:
            if per in actual.columns and not np.isnan(actual.loc[obat, per]):
                ys.append(float(actual.loc[obat, per]))
                yhs.append(float(pmap.get(per, np.nan)))
    return np.array(ys), np.array(yhs)


def evaluate(fc, test_panel, test_periods, horizon, name):
    y, yh = collect_preds(fc, test_panel, test_periods, horizon)
    valid = ~np.isnan(yh)
    y, yh = y[valid], yh[valid]
    return {
        "Model": name, "MAE": mae(y, yh), "RMSE": rmse(y, yh),
        "MAPE(%)": mape(y, yh), "sMAPE(%)": smape(y, yh),
        "R2": r2(y, yh), "n": len(y),
    }


# ---------------- bobot ensemble (inverse error validasi) ----------------
def compute_ensemble_weights(fit_panel, periods):
    """Fit HW & SARIMA pada periods[:-1], validasi pada periods[-1].
    Bobot per obat = inverse |error| dinormalisasi. Anti-leakage (tanpa test)."""
    val_period = periods[-1]
    sub = fit_panel[fit_panel[C.P_PERIOD].isin(periods[:-1])].copy()
    hw = HoltWintersForecaster.fit(sub)
    sa = SarimaForecaster.fit(sub)
    actual = fit_panel[fit_panel[C.P_PERIOD] == val_period].set_index(C.P_OBAT)[C.P_DEMAND]
    weights = {}
    for obat in hw.history.keys():
        if obat not in actual.index:
            weights[obat] = {"hw": 0.5, "sarima": 0.5}
            continue
        a = float(actual.loc[obat])
        eh = abs(hw._forecast_obat(obat, 1)[0] - a)
        es = abs(sa._forecast_obat(obat, 1)[0] - a)
        ih, is_ = 1.0 / (eh + 1e-6), 1.0 / (es + 1e-6)
        s = ih + is_
        weights[obat] = {"hw": ih / s, "sarima": is_ / s}
    return weights


# ============================ MAIN ============================
def main():
    t_start = time.time()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(C.PANEL_PATH)
    cluster_map = json.load(open(C.MODELS_DIR / "cluster_labels.json", encoding="utf-8"))["obat"]
    train, test, train_periods, test_periods = temporal_split(panel)
    H = C.TEST_MONTHS
    print(f"Train periods: {train_periods}")
    print(f"Test periods : {test_periods}  (catatan: 2025-11 imputasi; 2025-12 observasi nyata)")
    print("=" * 70)

    # ---- TUNING (pada TRAIN saja) ----
    print("TUNING (TimeSeriesSplit, hanya data train):")
    rf_params, feat_cols = tune_ml(
        train, cluster_map,
        RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1),
        C.RF_GRID, "RF")
    gb_params, _ = tune_ml(
        train, cluster_map,
        GradientBoostingRegressor(random_state=C.RANDOM_STATE),
        C.GB_GRID, "GB")
    print("=" * 70)

    # ---- FIT pada TRAIN untuk EVALUASI ----
    print("FIT pada TRAIN & evaluasi pada TEST...")
    eval_fcs = {}
    eval_fcs["Naive (lag-1)"] = NaiveForecaster.fit(train)
    eval_fcs["Random Forest"] = RandomForestForecaster.fit(
        train, RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **rf_params),
        feat_cols, cluster_map)
    eval_fcs["Gradient Boosting"] = GradientBoostingForecaster.fit(
        train, GradientBoostingRegressor(random_state=C.RANDOM_STATE, **gb_params),
        feat_cols, cluster_map)
    print("  fitting Holt-Winters per obat...")
    hw_eval = HoltWintersForecaster.fit(train)
    eval_fcs["Holt-Winters"] = hw_eval
    print("  fitting SARIMA per obat...")
    sa_eval = SarimaForecaster.fit(train)
    eval_fcs["SARIMA"] = sa_eval
    print(f"  SARIMA fallback (eval): {sa_eval.n_fallback} obat")

    # Ensemble (bobot dari validasi internal train)
    w_eval = compute_ensemble_weights(train, train_periods)
    eval_fcs["Ensemble (simple)"] = EnsembleForecaster(hw_eval, sa_eval, w_eval, "simple")
    eval_fcs["Ensemble (weighted)"] = EnsembleForecaster(hw_eval, sa_eval, w_eval, "weighted")

    rows = [evaluate(fc, test, test_periods, H, name) for name, fc in eval_fcs.items()]
    metrics = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    print("=" * 70)
    print("TABEL METRIK (test = 2 bulan terakhir):")
    print(metrics.round(3).to_string(index=False))

    # ---- ANTI-LEAKAGE CHECK ----
    print("=" * 70)
    print("VERIFIKASI ANTI-LEAKAGE:")
    print(f"  - Split temporal: train {train_periods[0]}..{train_periods[-1]}, "
          f"test {test_periods}. Tidak ada bulan test di train.")
    print(f"  - Tuning hanya pada TRAIN (TimeSeriesSplit, bukan KFold acak).")
    print(f"  - Fitur ML kausal (lag/rolling masa lalu); tidak ada fitur kontemporer target.")
    susp = metrics[metrics["R2"] > 0.95]
    print(f"  - Model dgn R²>0.95 (perlu investigasi): "
          f"{susp['Model'].tolist() if len(susp) else 'tidak ada'}")

    # ---- BEST MODEL ----
    best_row = metrics.iloc[0]
    best_name = best_row["Model"]
    name_to_key = {
        "Naive (lag-1)": "naive", "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting", "Holt-Winters": "holt_winters",
        "SARIMA": "sarima", "Ensemble (simple)": "ensemble_simple",
        "Ensemble (weighted)": "ensemble_weighted",
    }
    best_key = name_to_key[best_name]
    print("=" * 70)
    print(f"MODEL TERBAIK (MAE test): {best_name}  MAE={best_row['MAE']:.2f}")

    # ================= RETRAIN PADA SELURUH DATA =================
    print("=" * 70)
    print("RETRAIN pada SELURUH DATA (artefak produksi)...")
    naive_prod = NaiveForecaster.fit(panel)
    rf_prod = RandomForestForecaster.fit(
        panel, RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **rf_params),
        feat_cols, cluster_map)
    gb_prod = GradientBoostingForecaster.fit(
        panel, GradientBoostingRegressor(random_state=C.RANDOM_STATE, **gb_params),
        feat_cols, cluster_map)
    print("  fitting HW (full)...")
    hw_prod = HoltWintersForecaster.fit(panel)
    print("  fitting SARIMA (full)...")
    sa_prod = SarimaForecaster.fit(panel)
    print(f"  SARIMA fallback (full): {sa_prod.n_fallback} obat")
    all_periods = sorted(panel[C.P_PERIOD].unique())
    w_prod = compute_ensemble_weights(panel, all_periods)

    # ---- SIMPAN ARTEFAK ----
    print("Menyimpan artefak...")
    naive_prod.save(C.MODELS_DIR / "naive.pkl")
    rf_prod.save(C.MODELS_DIR / "random_forest.pkl")
    gb_prod.save(C.MODELS_DIR / "gradient_boosting.pkl")
    hw_prod.save(C.MODELS_DIR / "holtwinters.pkl")
    sa_prod.save(C.MODELS_DIR / "sarima.pkl")
    json.dump({"features": feat_cols, "best_params": rf_params},
              open(C.MODELS_DIR / "rf_features.json", "w"), indent=2)
    json.dump({"features": feat_cols, "best_params": gb_params},
              open(C.MODELS_DIR / "gb_features.json", "w"), indent=2)
    json.dump(w_prod, open(C.MODELS_DIR / "ensemble_weights.json", "w"), indent=2)

    metrics_out = metrics.to_dict(orient="records")
    json.dump({"metrics": metrics_out, "test_periods": test_periods,
               "train_periods": train_periods},
              open(C.MODELS_DIR / "metrics.json", "w"), indent=2)
    json.dump({"best_model": best_key, "best_model_name": best_name,
               "mae": float(best_row["MAE"]), "by": "MAE test",
               "available": list(name_to_key.values())},
              open(C.MODELS_DIR / "best_model.json", "w"), indent=2)

    # ---- verifikasi reload tanpa retrain ----
    print("Verifikasi reload artefak (tanpa retrain)...")
    sample_obat = list(naive_prod.history.keys())[0]
    for fname, cls in [("naive.pkl", NaiveForecaster),
                       ("random_forest.pkl", RandomForestForecaster),
                       ("gradient_boosting.pkl", GradientBoostingForecaster),
                       ("holtwinters.pkl", HoltWintersForecaster),
                       ("sarima.pkl", SarimaForecaster)]:
        m = cls.load(C.MODELS_DIR / fname)
        r = m.predict(sample_obat, 1)
        assert "prediksi" in r and len(r["prediksi"]) == 1
    print(f"  OK semua artefak ter-reload & predict() jalan (contoh obat: {sample_obat})")

    write_report(metrics, train_periods, test_periods, rf_params, gb_params,
                 sa_prod.n_fallback, best_name, susp, cluster_map)
    print("=" * 70)
    print(f"SELESAI dalam {time.time()-t_start:.1f}s. Artefak di {C.MODELS_DIR}")
    return metrics


def write_report(metrics, train_periods, test_periods, rf_params, gb_params,
                 sarima_fb, best_name, susp, cluster_map):
    lines = []
    lines.append("# 03 — Perbandingan Model Forecasting\n")
    lines.append(f"**Split temporal:** train `{train_periods[0]}`–`{train_periods[-1]}` "
                 f"({len(train_periods)} bln), test `{test_periods}` ({len(test_periods)} bln).\n")
    lines.append("> Catatan: 2025-11 adalah nilai **imputasi** (interpolasi linear, bulan "
                 "hilang di sumber); 2025-12 adalah **observasi nyata** dan menjadi titik "
                 "holdout paling sahih.\n")
    lines.append("\n## Tabel metrik (test set sama untuk semua pendekatan)\n")
    lines.append("| Model | MAE | RMSE | MAPE(%) | sMAPE(%) | R² | n |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in metrics.iterrows():
        lines.append(f"| {r['Model']} | {r['MAE']:.2f} | {r['RMSE']:.2f} | "
                     f"{r['MAPE(%)']:.1f} | {r['sMAPE(%)']:.1f} | {r['R2']:.3f} | {int(r['n'])} |")
    lines.append(f"\n**Model terbaik (MAE test): {best_name}.**\n")
    lines.append("\n## Parameter hasil tuning (GridSearch + TimeSeriesSplit)\n")
    lines.append(f"- **Random Forest:** `{rf_params}`")
    lines.append(f"- **Gradient Boosting:** `{gb_params}`")
    lines.append(f"- **Holt-Winters:** trend ∈ {C.HW_TREND}, damped ∈ {C.HW_DAMPED}, "
                 f"tanpa seasonal; dipilih per obat via AIC; fallback rata-rata utk deret < {C.MIN_OBS_TS} observasi.")
    lines.append(f"- **SARIMA:** p,q ∈ {C.SARIMA_P}, d ∈ {C.SARIMA_D}, tanpa seasonal; "
                 f"dipilih per obat via AIC; fallback naive: **{sarima_fb} obat**.")
    lines.append(f"- **Ensemble:** rata-rata sederhana & berbobot (inverse error validasi internal).\n")
    lines.append("\n## Verifikasi anti-leakage\n")
    lines.append("- Split **temporal** murni; tidak ada bulan test bocor ke train.")
    lines.append("- Tuning ML memakai **TimeSeriesSplit** (bukan KFold acak).")
    lines.append("- Fitur ML **kausal** (lag 1–3, rolling mean/std masa lalu, stok lag-1); "
                 "tidak ada fitur kontemporer terhadap target.")
    susp_txt = susp["Model"].tolist() if len(susp) else "tidak ada"
    lines.append(f"- Model dengan R² > 0.95 (indikasi leakage): **{susp_txt}**.\n")
    lines.append("\n## Keterbatasan\n")
    lines.append("- Hanya **11 bulan** data (Nov hilang) < 2 siklus tahunan → **komponen "
                 "musiman tahunan tidak teridentifikasi**; HW & SARIMA dijalankan **tanpa seasonal**.")
    lines.append("- 2025-11 diimputasi; metrik pada bulan itu mencerminkan imputasi, bukan realita.")
    lines.append("- Banyak obat slow-moving / intermittent → MAPE tidak stabil; sMAPE & MAE lebih andal.\n")
    lines.append("\n## Artefak produksi (retrain pada SELURUH data)\n")
    lines.append("Semua pendekatan dilatih ulang pada 12 bulan penuh dengan parameter hasil "
                 "tuning, lalu disimpan ke `models/` (naive, random_forest, gradient_boosting, "
                 "holtwinters, sarima, ensemble_weights, best_model, metrics).")
    docs_dir = C.BASE_DIR / "docs"
    docs_dir.mkdir(exist_ok=True)
    (docs_dir / "03_perbandingan_model.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Laporan -> docs/03_perbandingan_model.md")


if __name__ == "__main__":
    main()
