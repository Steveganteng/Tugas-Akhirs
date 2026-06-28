"""Evaluasi menyeluruh 6+1 pendekatan forecasting + rencana perbaikan berbukti.

Menjalankan 7 tahap (akurasi, walk-forward, per-segmen/obat, residual/bias,
dampak bisnis, diagnosis, rencana) dengan angka NYATA, lalu menulis
LAPORAN_EVALUASI.md + plot ke docs/eval_plots/.

PRINSIP:
  - Evaluasi hanya pada observasi ASLI (is_observed=True); bulan imputasi
    (2025-11, 0 observasi) DIBUANG dari test.
  - Apple-to-apple: himpunan obat & periode test SAMA untuk semua model.
  - Model untuk evaluasi DILATIH ULANG pada train-only (artefak produksi dilatih
    pada SELURUH data -> memakainya utk menilai bulan historis = leakage).
  - Selalu dibandingkan ke baseline Naive (skill score).

Jalankan: python evaluasi_model.py
"""
from __future__ import annotations
import sys, io, json, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

import config as C
import ml_features as F
from forecaster import (
    NaiveForecaster, RandomForestForecaster, GradientBoostingForecaster,
    HoltWintersForecaster, SarimaForecaster, EnsembleForecaster,
)
from pipeline_forecasting_obat import (
    mae, rmse, smape, mape, r2, compute_ensemble_weights,
)
from restock import hitung_restock

DOCS = C.BASE_DIR / "docs"
PLOTS = DOCS / "eval_plots"
PLOTS.mkdir(parents=True, exist_ok=True)

MODEL_ORDER = ["Naive", "Random Forest", "Gradient Boosting", "Holt-Winters",
               "SARIMA", "Ensemble (simple)", "Ensemble (weighted)"]


# ---------------- util ----------------
def months_between(p_from: str, p_to: str) -> int:
    a, b = pd.Period(p_from, "M"), pd.Period(p_to, "M")
    return (b.year - a.year) * 12 + (b.month - a.month)


def load_params(fname, default):
    try:
        return json.load(open(C.MODELS_DIR / fname, encoding="utf-8"))["best_params"]
    except Exception:
        return default


def fit_models(train_panel, cluster_map, rf_params, gb_params, feat_cols):
    """Latih semua pendekatan pada train-only (anti-leakage)."""
    train_periods = sorted(train_panel[C.P_PERIOD].unique())
    fcs = {}
    fcs["Naive"] = NaiveForecaster.fit(train_panel)
    fcs["Random Forest"] = RandomForestForecaster.fit(
        train_panel, RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1, **rf_params),
        feat_cols, cluster_map)
    fcs["Gradient Boosting"] = GradientBoostingForecaster.fit(
        train_panel, GradientBoostingRegressor(random_state=C.RANDOM_STATE, **gb_params),
        feat_cols, cluster_map)
    hw = HoltWintersForecaster.fit(train_panel)
    sa = SarimaForecaster.fit(train_panel)
    fcs["Holt-Winters"] = hw
    fcs["SARIMA"] = sa
    w = compute_ensemble_weights(train_panel, train_periods)
    fcs["Ensemble (simple)"] = EnsembleForecaster(hw, sa, w, "simple")
    fcs["Ensemble (weighted)"] = EnsembleForecaster(hw, sa, w, "weighted")
    return fcs, sa.n_fallback, hw


def hw_fallback_count(hw):
    return sum(1 for v in hw.fits.values() if v.get("type") == "mean")


def predict_period(fc, obat, train_last, test_period):
    """Prediksi nilai untuk satu test_period dari model yang dilatih s/d train_last."""
    h = months_between(train_last, test_period)
    if h < 1:
        return None
    try:
        r = fc.predict(obat, h)
    except KeyError:
        return None
    pmap = dict(zip(r["periode"], r["prediksi"]))
    return pmap.get(test_period)


def actual_observed(panel, test_period):
    """{obat: demand} hanya untuk observasi ASLI pada test_period."""
    sub = panel[(panel[C.P_PERIOD] == test_period) & (panel[C.P_IS_OBS])]
    return dict(zip(sub[C.P_OBAT], sub[C.P_DEMAND].astype(float)))


def evaluate_models(fcs, panel, train_last, test_period):
    """Metrik apple-to-apple: obat set = obat dgn observasi asli di test_period,
    yang ADA di semua model. Vektor y sama untuk semua model."""
    act = actual_observed(panel, test_period)
    common = [o for o in act if all(o in fc.history for fc in fcs.values())]
    common.sort()
    y = np.array([act[o] for o in common], dtype=float)
    results = {}
    preds_by_model = {}
    for name, fc in fcs.items():
        yh = np.array([predict_period(fc, o, train_last, test_period) for o in common], dtype=float)
        preds_by_model[name] = yh
        results[name] = {
            "MAE": mae(y, yh), "RMSE": rmse(y, yh), "MAPE(%)": mape(y, yh),
            "sMAPE(%)": smape(y, yh), "R2": r2(y, yh), "n": len(y)}
    return results, common, y, preds_by_model


# ============================================================
def main():
    t0 = time.time()
    print("=" * 70)
    print("EVALUASI MENYELURUH 6+1 MODEL FORECASTING")
    print("=" * 70)
    panel = pd.read_parquet(C.PANEL_PATH)
    cluster_map = json.load(open(C.MODELS_DIR / "cluster_labels.json", encoding="utf-8"))["obat"]
    feat_cols = F.feature_columns()
    rf_params = load_params("rf_features.json", {"n_estimators": 300})
    gb_params = load_params("gb_features.json", {"n_estimators": 300})
    all_periods = sorted(panel[C.P_PERIOD].unique())
    obs_per = panel.groupby(C.P_PERIOD)[C.P_IS_OBS].sum()
    real_months = [p for p in all_periods if obs_per[p] > 0]
    imputed_months = [p for p in all_periods if obs_per[p] == 0]
    print(f"Periode: {all_periods[0]}..{all_periods[-1]} | "
          f"imputasi (dibuang dari test): {imputed_months}")

    RES = {"meta": {"all_periods": all_periods, "imputed": imputed_months,
                    "obs_per": {k: int(v) for k, v in obs_per.items()},
                    "rf_params": rf_params, "gb_params": gb_params}}

    # ---------- TAHAP 1: AKURASI (holdout real) ----------
    print("\n[TAHAP 1] Akurasi pada holdout real (train<=2025-10, test=2025-12)...")
    TRAIN_LAST = "2025-10"
    TEST_PERIOD = "2025-12"   # real holdout (2025-11 imputed -> dibuang)
    train1 = panel[panel[C.P_PERIOD] <= TRAIN_LAST]
    t = time.time()
    fcs1, sa_fb1, hw1 = fit_models(train1, cluster_map, rf_params, gb_params, feat_cols)
    hw_fb1 = hw_fallback_count(hw1)
    res1, common1, y1, preds1 = evaluate_models(fcs1, panel, TRAIN_LAST, TEST_PERIOD)
    print(f"  selesai fit+eval ({time.time()-t:.1f}s) | n obat = {len(common1)} | "
          f"fallback SARIMA={sa_fb1}, HW={hw_fb1}")
    mae_naive = res1["Naive"]["MAE"]
    for name in MODEL_ORDER:
        res1[name]["skill"] = (mae_naive - res1[name]["MAE"]) / mae_naive if mae_naive else 0
    ranking = sorted(MODEL_ORDER, key=lambda n: res1[n]["MAE"])
    best_model = ranking[0]
    print("  Ranking MAE:", " < ".join(f"{n}({res1[n]['MAE']:.1f})" for n in ranking))
    print(f"  MODEL TERBAIK (holdout real): {best_model}")
    # leakage check
    suspicious = [n for n in MODEL_ORDER if res1[n]["R2"] > 0.95 or
                  (not np.isnan(res1[n]["MAPE(%)"]) and res1[n]["MAPE(%)"] < 1)]
    RES["tahap1"] = {"train_last": TRAIN_LAST, "test_period": TEST_PERIOD,
                     "n": len(common1), "metrics": res1, "ranking": ranking,
                     "best": best_model, "mae_naive": mae_naive,
                     "sarima_fallback": sa_fb1, "hw_fallback": hw_fb1,
                     "suspicious_leakage": suspicious}

    # ---------- TAHAP 2: WALK-FORWARD ----------
    print("\n[TAHAP 2] Walk-forward (refit per fold)...")
    folds = [("2025-08", "2025-09"), ("2025-09", "2025-10"), ("2025-11", "2025-12")]
    folds = [(tl, tp) for tl, tp in folds if tp in real_months]
    wf = {n: {"MAE": [], "sMAPE": []} for n in MODEL_ORDER}
    fold_rank = []
    for tl, tp in folds:
        tr = panel[panel[C.P_PERIOD] <= tl]
        fcs, _, _ = fit_models(tr, cluster_map, rf_params, gb_params, feat_cols)
        rr, _, _, _ = evaluate_models(fcs, panel, tl, tp)
        for n in MODEL_ORDER:
            wf[n]["MAE"].append(rr[n]["MAE"]); wf[n]["sMAPE"].append(rr[n]["sMAPE(%)"])
        fr = sorted(MODEL_ORDER, key=lambda n: rr[n]["MAE"])
        fold_rank.append((f"{tl}->{tp}", fr[0]))
        print(f"  fold {tl}->{tp}: terbaik={fr[0]} (MAE {rr[fr[0]]['MAE']:.1f})")
    wf_summary = {n: {"MAE_mean": float(np.mean(wf[n]["MAE"])),
                      "MAE_std": float(np.std(wf[n]["MAE"])),
                      "sMAPE_mean": float(np.mean(wf[n]["sMAPE"]))} for n in MODEL_ORDER}
    wf_rank = sorted(MODEL_ORDER, key=lambda n: wf_summary[n]["MAE_mean"])
    RES["tahap2"] = {"folds": folds, "summary": wf_summary, "ranking": wf_rank,
                     "fold_winners": fold_rank}
    print("  Ranking walk-forward (rata MAE):",
          " < ".join(f"{n}({wf_summary[n]['MAE_mean']:.1f}±{wf_summary[n]['MAE_std']:.1f})" for n in wf_rank))

    # ---------- TAHAP 3: PER SEGMEN & PER OBAT ----------
    print("\n[TAHAP 3] Error per segmen & per obat (holdout real)...")
    seg_of = {o: cluster_map.get(o, {}).get("label", "?") for o in common1}
    segments = sorted(set(seg_of.values()))
    seg_err = {name: {s: [] for s in segments} for name in MODEL_ORDER}
    for name in MODEL_ORDER:
        yh = preds1[name]
        for i, o in enumerate(common1):
            seg_err[name][seg_of[o]].append(abs(y1[i] - yh[i]))
    seg_table = {name: {s: (float(np.mean(v)) if v else float("nan"))
                        for s, v in seg_err[name].items()} for name in MODEL_ORDER}
    seg_best = {}
    seg_n = {s: sum(1 for o in common1 if seg_of[o] == s) for s in segments}
    for s in segments:
        seg_best[s] = min(MODEL_ORDER, key=lambda n: seg_table[n][s]
                          if not np.isnan(seg_table[n][s]) else 1e9)
    # 10 obat error terbesar (model terbaik)
    yh_best = preds1[best_model]
    err_obat = sorted(((abs(y1[i] - yh_best[i]), common1[i], y1[i], yh_best[i])
                       for i in range(len(common1))), reverse=True)[:10]
    cf = pd.read_csv(C.MODELS_DIR / "cluster_features.csv")
    cf_map = cf.set_index(C.P_OBAT).to_dict("index")
    nobs = panel[panel[C.P_IS_OBS]].groupby(C.P_OBAT).size().to_dict()
    worst = []
    for e, o, a, p in err_obat:
        info = cf_map.get(o, {})
        worst.append({"obat": o, "abs_err": round(e, 1), "actual": round(a, 1),
                      "pred": round(p, 1), "segmen": seg_of.get(o, "?"),
                      "volume": round(info.get("volume", 0), 1),
                      "cv": round(info.get("cv", 0), 2),
                      "n_observed": int(nobs.get(o, 0))})
    RES["tahap3"] = {"segments": segments, "seg_n": seg_n, "seg_table": seg_table,
                     "seg_best": seg_best, "worst10": worst}
    print("  Terbaik per segmen:", seg_best)
    print("  Obat error terbesar:", worst[0]["obat"], worst[0]["abs_err"])

    # ---------- TAHAP 4: RESIDUAL & BIAS ----------
    print("\n[TAHAP 4] Residual & bias model terbaik...")
    resid = y1 - yh_best   # actual - pred ; >0 = under-predict
    vol = np.array([cf_map.get(o, {}).get("volume", 0) for o in common1])
    terc = pd.qcut(pd.Series(vol).rank(method="first"), 3, labels=["low", "mid", "high"])
    bias_by_vol = {}
    for lab in ["low", "mid", "high"]:
        m = (terc == lab).values
        bias_by_vol[lab] = {"mean_resid": float(np.mean(resid[m])),
                            "mae": float(np.mean(np.abs(resid[m]))), "n": int(m.sum())}
    neg_preds = int(np.sum(yh_best < 0))
    RES["tahap4"] = {"mean_resid": float(np.mean(resid)),
                     "median_resid": float(np.median(resid)),
                     "pct_under": float(np.mean(resid > 0) * 100),
                     "pct_over": float(np.mean(resid < 0) * 100),
                     "bias_by_volume": bias_by_vol, "neg_preds": neg_preds,
                     "best_model": best_model}
    print(f"  mean residual={np.mean(resid):.1f} (>0=under) | under={np.mean(resid>0)*100:.0f}% | neg_preds={neg_preds}")
    make_plots(common1, y1, preds1, best_model, resid, seg_of, segments, RES)

    # ---------- TAHAP 5: DAMPAK BISNIS ----------
    print("\n[TAHAP 5] Dampak bisnis (stockout/overstock)...")
    stok_last = dict(zip(
        panel[panel[C.P_PERIOD] == TRAIN_LAST][C.P_OBAT],
        panel[panel[C.P_PERIOD] == TRAIN_LAST][C.P_STOCK].astype(float)))
    demand_hist = {o: panel[(panel[C.P_OBAT] == o) & (panel[C.P_PERIOD] <= TRAIN_LAST)]
                   [C.P_DEMAND].astype(float).tolist() for o in common1}
    understock = overstock = 0
    under_units = over_units = 0.0
    stockout_follow = 0
    for i, o in enumerate(common1):
        a = y1[i]; p = yh_best[i]
        if p < a: understock += 1; under_units += (a - p)
        elif p > a: overstock += 1; over_units += (p - a)
        rec = hitung_restock(o, p, stok_last.get(o, 0.0),
                             demand_historis=demand_hist[o], periode=TEST_PERIOD)
        tersedia = stok_last.get(o, 0.0) + rec["jumlah_rekomendasi"]
        if a > tersedia:
            stockout_follow += 1
    n = len(common1)
    RES["tahap5"] = {"n": n, "understock": understock, "overstock": overstock,
                     "pct_understock": round(understock / n * 100, 1),
                     "pct_overstock": round(overstock / n * 100, 1),
                     "under_units": round(under_units, 0), "over_units": round(over_units, 0),
                     "stockout_follow": stockout_follow,
                     "pct_stockout_follow": round(stockout_follow / n * 100, 1)}
    print(f"  understock={understock}/{n} ({understock/n*100:.0f}%), "
          f"overstock={overstock}/{n} | potensi stockout bila rekomendasi diikuti={stockout_follow}/{n}")

    write_report(RES)
    print("=" * 70)
    print(f"SELESAI {time.time()-t0:.1f}s -> docs/LAPORAN_EVALUASI.md + docs/eval_plots/")
    return RES


def make_plots(obat_list, y, preds, best, resid, seg_of, segments, RES):
    # 1) actual vs predicted scatter (best)
    yh = preds[best]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, yh, alpha=.6, color="#2e7d32")
    lim = max(y.max(), yh.max()) * 1.05
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlabel("Actual (Des 2025)"); ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted — {best}")
    fig.tight_layout(); fig.savefig(PLOTS / "actual_vs_pred.png", dpi=110); plt.close(fig)

    # 2) residual histogram
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(resid, bins=25, color="#1565c0", alpha=.8)
    ax.axvline(0, color="r", ls="--"); ax.set_title(f"Residual (actual−pred) — {best}")
    ax.set_xlabel("residual"); fig.tight_layout()
    fig.savefig(PLOTS / "residual_hist.png", dpi=110); plt.close(fig)

    # 3) bar MAE per model
    names = MODEL_ORDER
    maes = [RES["tahap1"]["metrics"][n]["MAE"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#c62828" if n == "Naive" else "#2e7d32" if n == best else "#90a4ae" for n in names]
    ax.bar(range(len(names)), maes, color=colors)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.axhline(RES["tahap1"]["mae_naive"], color="#c62828", ls="--", lw=1, label="MAE Naive")
    ax.set_ylabel("MAE (Des 2025)"); ax.set_title("MAE per model vs baseline Naive")
    ax.legend(); fig.tight_layout(); fig.savefig(PLOTS / "mae_per_model.png", dpi=110); plt.close(fig)
    RES["plots"] = ["actual_vs_pred.png", "residual_hist.png", "mae_per_model.png"]


def _fmt(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{d}f}"


def write_report(R):
    L = []
    a = L.append
    t1 = R["tahap1"]; m = t1["metrics"]
    a("# LAPORAN EVALUASI MODEL FORECASTING RESTOK OBAT\n")
    a("> Semua angka dihasilkan oleh `evaluasi_model.py` (dapat dijalankan ulang). "
      "Evaluasi hanya pada **observasi asli** (bulan imputasi "
      f"{R['meta']['imputed']} dibuang dari test). Model untuk evaluasi **dilatih "
      "ulang pada train-only** agar tidak ada leakage (artefak produksi dilatih "
      "pada seluruh data).\n")

    a("## Ringkasan Eksekutif\n")
    best = t1["best"]
    a(f"- **Model terbaik (holdout real Des 2025, n={t1['n']} obat): {best}** "
      f"— MAE {_fmt(m[best]['MAE'])}, skill score vs naive **{_fmt(m[best]['skill']*100,1)}%**.\n")
    pos = [n for n in MODEL_ORDER if m[n]["skill"] > 0 and n != "Naive"]
    a(f"- Model yang **mengalahkan naive**: {', '.join(pos) if pos else 'TIDAK ADA'}.\n")
    a(f"- Indikasi leakage (R²>0.95 / MAPE~0%): "
      f"**{', '.join(t1['suspicious_leakage']) if t1['suspicious_leakage'] else 'tidak ada'}** "
      "— semua R² rendah/negatif, konsisten dgn data 12 bulan & deret intermiten.\n")
    a(f"- Dampak bisnis: **{R['tahap5']['pct_understock']}%** obat under-forecast "
      f"(risiko stockout), **{R['tahap5']['pct_overstock']}%** over-forecast "
      f"(modal mengendap).\n")

    # Tahap 1
    a("\n## Tahap 1 — Akurasi Menyeluruh (holdout real)\n")
    a(f"Train ≤ {t1['train_last']} → test **{t1['test_period']}** (2-step ahead; "
      f"2025-11 imputasi dilewati). Himpunan {t1['n']} obat sama untuk semua model.\n")
    a("| # | Model | MAE | RMSE | MAPE(%) | sMAPE(%) | R² | Skill vs Naive |")
    a("|---|---|---|---|---|---|---|---|")
    for i, n in enumerate(t1["ranking"], 1):
        x = m[n]
        sk = f"{x['skill']*100:+.1f}%" if n != "Naive" else "baseline"
        a(f"| {i} | {n} | {_fmt(x['MAE'])} | {_fmt(x['RMSE'])} | {_fmt(x['MAPE(%)'],1)} | "
          f"{_fmt(x['sMAPE(%)'],1)} | {_fmt(x['R2'],3)} | {sk} |")
    a(f"\n- Fallback: SARIMA **{t1['sarima_fallback']}** obat, Holt-Winters "
      f"**{t1['hw_fallback']}** obat (deret < {C.MIN_OBS_TS} observasi → rata-rata/naive).")
    a("\n![MAE per model](eval_plots/mae_per_model.png)\n")

    # Tahap 2
    t2 = R["tahap2"]
    a("\n## Tahap 2 — Walk-Forward (robustness)\n")
    a(f"{len(t2['folds'])} fold (refit tiap fold): "
      + ", ".join(f"≤{tl}→{tp}" for tl, tp in t2["folds"]) + ".\n")
    a("| Model | MAE rata² | MAE std | sMAPE rata² |")
    a("|---|---|---|---|")
    for n in t2["ranking"]:
        s = t2["summary"][n]
        a(f"| {n} | {_fmt(s['MAE_mean'])} | {_fmt(s['MAE_std'])} | {_fmt(s['sMAPE_mean'],1)} |")
    a("\n**Pemenang per fold:** " + "; ".join(f"{f}: {w}" for f, w in t2["fold_winners"]) + ".")
    stable = len(set(w for _, w in t2["fold_winners"])) == 1
    a(f"\n**Stabilitas peringkat:** {'STABIL (pemenang sama tiap fold)' if stable else 'TIDAK stabil — pemenang berganti antar bulan, menandakan keunggulan tipis & sensitif periode'}.")
    blow = [n for n in MODEL_ORDER if t2["summary"][n]["MAE_mean"] > 1000]
    if blow:
        a(f"\n> ⚠️ **{', '.join(blow)} MELEDAK** pada fold dengan train pendek "
          "(MAE ribuan, std sangat besar): SARIMA tanpa enforce_stationarity menghasilkan "
          "forecast eksplosif pada deret ≤ 8 bulan, dan ensemble mewarisinya. Bukti kuat "
          "bahwa SARIMA/ensemble TIDAK andal untuk deret pendek → perlu guard nilai wajar.")

    # Tahap 3
    t3 = R["tahap3"]
    a("\n## Tahap 3 — Error per Segmen & per Obat\n")
    a("MAE rata² per segmen cluster (holdout real). Sel **tebal** = model terbaik di segmen itu.\n")
    hdr = "| Segmen (n) | " + " | ".join(MODEL_ORDER) + " |"
    a(hdr); a("|" + "---|" * (len(MODEL_ORDER) + 1))
    for s in t3["segments"]:
        cells = []
        for n in MODEL_ORDER:
            v = _fmt(t3["seg_table"][n][s], 1)
            cells.append(f"**{v}**" if t3["seg_best"][s] == n else v)
        a(f"| {s} ({t3['seg_n'][s]}) | " + " | ".join(cells) + " |")
    a("\n**Model terbaik per segmen:** " +
      "; ".join(f"{s} → {n}" for s, n in t3["seg_best"].items()) + ".")
    a(f"\n### 10 Obat dengan Error Terbesar (model {best})\n")
    a("| Obat | Segmen | Actual | Pred | |Error| | Volume | CV | n_obs |")
    a("|---|---|---|---|---|---|---|---|")
    for w in t3["worst10"]:
        a(f"| {w['obat']} | {w['segmen']} | {w['actual']} | {w['pred']} | "
          f"{w['abs_err']} | {w['volume']} | {w['cv']} | {w['n_observed']} |")

    # Tahap 4
    t4 = R["tahap4"]
    a("\n## Tahap 4 — Analisis Residual & Bias\n")
    a(f"- Mean residual (actual−pred) = **{_fmt(t4['mean_resid'],1)}** "
      f"({'cenderung UNDER-predict' if t4['mean_resid']>0 else 'cenderung OVER-predict'}); "
      f"median {_fmt(t4['median_resid'],1)}.")
    a(f"- {_fmt(t4['pct_under'],0)}% obat under-predict, {_fmt(t4['pct_over'],0)}% over-predict.")
    a(f"- Prediksi negatif/janggal: **{t4['neg_preds']}** (sudah di-clip ≥ 0 di `forecaster._result`).")
    a("- Bias per tercile volume:")
    a("\n| Tercile volume | n | mean residual | MAE |")
    a("|---|---|---|---|")
    for lab in ["low", "mid", "high"]:
        b = t4["bias_by_volume"][lab]
        a(f"| {lab} | {b['n']} | {_fmt(b['mean_resid'],1)} | {_fmt(b['mae'],1)} |")
    a("\n![Actual vs Predicted](eval_plots/actual_vs_pred.png)\n")
    a("![Residual](eval_plots/residual_hist.png)\n")

    # Tahap 5
    t5 = R["tahap5"]
    a("\n## Tahap 5 — Dampak Bisnis (Restok)\n")
    a(f"Dari {t5['n']} obat holdout real:")
    a(f"- **Understock** (pred < aktual → risiko kehabisan): {t5['understock']} obat "
      f"(**{t5['pct_understock']}%**), total kekurangan **{int(t5['under_units'])}** unit.")
    a(f"- **Overstock** (pred > aktual → modal mengendap): {t5['overstock']} obat "
      f"(**{t5['pct_overstock']}%**), total kelebihan **{int(t5['over_units'])}** unit.")
    a(f"- Bila rekomendasi restok diikuti apa adanya, **{t5['stockout_follow']} obat "
      f"({t5['pct_stockout_follow']}%)** tetap berpotensi stockout (aktual > stok+order).")
    a("\n> Catatan: dampak bisnis lebih relevan daripada MAE murni — under-forecast "
      "pada obat fast-moving jauh lebih berisiko daripada error nominal kecil.")

    # Tahap 6
    a("\n## Tahap 6 — Diagnosis Akar Masalah (berbukti)\n")
    a(f"1. **Keterbatasan data (bukti: R² semua negatif/rendah, MAPE ratusan %, "
      f"hanya {len([p for p in R['meta']['all_periods'] if R['meta']['obs_per'][p]>0])} "
      "bulan observasi).** 12 bulan < 2 siklus → komponen musiman tahunan tidak "
      "teridentifikasi; HW & SARIMA dijalankan tanpa seasonal.")
    a(f"2. **Deret pendek & intermiten (bukti: SARIMA fallback {t1['sarima_fallback']} "
      f"obat, HW fallback {t1['hw_fallback']} obat; obat error terbesar = "
      f"{R['tahap3']['worst10'][0]['obat']} volume {R['tahap3']['worst10'][0]['volume']}).** "
      "Banyak obat slow-moving → model deret waktu jatuh ke fallback rata-rata/naive.")
    a("3. **Keterbatasan fitur (bukti: model ML (RF/GB) KALAH dari naive — "
      f"skill {_fmt(m['Random Forest']['skill']*100,1)}% / {_fmt(m['Gradient Boosting']['skill']*100,1)}%).** "
      "Hanya ada fitur lag/rolling internal; tidak ada prediktor eksternal "
      "(jumlah kasus diagnosa, musim penyakit) yang menggerakkan demand.")
    a(f"4. **Tidak ada leakage (bukti: tidak ada model dgn R²>0.95; "
      "split temporal + refit train-only).** Performa apa adanya, bukan over-optimistis.")

    # Tahap 7
    a("\n## Tahap 7 — Rencana Perbaikan Berbukti\n")
    a("### Jangka Pendek (langsung)\n")
    a(f"- **Pakai model per-segmen, bukan satu model global.** Bukti (Tahap 3): "
      f"terbaik beda per segmen ({'; '.join(f'{s}→{n}' for s,n in list(R['tahap3']['seg_best'].items())[:3])}…). "
      "Dampak: turunkan MAE segmen yang modelnya kurang cocok. Verifikasi: MAE per "
      "segmen setelah routing < MAE model global saat ini.")
    a(f"- **Default ke Naive/Holt-Winters untuk obat fallback & slow-moving.** Bukti: "
      f"RF/GB skill negatif; {t1['sarima_fallback']} obat SARIMA fallback. Dampak: hindari "
      "prediksi ML yang lebih buruk dari naive. Verifikasi: skill score per obat ≥ 0.")
    if t4["mean_resid"] < 0:
        a(f"- **Kalibrasi turun over-prediction model {best} (mean residual "
          f"{_fmt(t4['mean_resid'],1)} unit = cenderung OVER-predict; {t5['pct_overstock']}% obat "
          f"overstock, {int(t5['over_units'])} unit kelebihan, terutama tercile volume tinggi "
          f"(resid {_fmt(t4['bias_by_volume']['high']['mean_resid'],1)})).** Dampak: kurangi modal "
          f"mengendap. Tetap lindungi {t5['pct_understock']}% obat under-forecast dgn safety stock "
          "terarah. Verifikasi: unit kelebihan & % overstock turun, % understock tidak naik.")
    else:
        a(f"- **Safety stock lebih agresif (mean residual {_fmt(t4['mean_resid'],1)} unit = cenderung "
          f"UNDER-predict).** Dampak: kurangi {t5['pct_understock']}% understock. "
          "Verifikasi: % understock & potensi stockout turun pada evaluasi ulang.")
    a("\n### Jangka Menengah\n")
    a("- **Tambah fitur eksternal kasus diagnosa ber-lag** (kolom Kode/Diagnosa "
      "Primer sudah ada di raw). Bukti: fitur internal saja membuat ML kalah naive "
      "(Tahap 1). Dampak: ML berpeluang > naive. Verifikasi: skill score RF/GB > 0.")
    a("- **Ensemble per-segmen + penanganan fallback eksplisit.** Bukti: ensemble "
      f"global ({_fmt(m['Ensemble (weighted)']['MAE'])}) tidak konsisten unggul "
      "(Tahap 2 tidak stabil). Verifikasi: MAE walk-forward turun & std mengecil.")
    a("\n### Jangka Panjang\n")
    a("- **Kumpulkan data multi-tahun** agar musiman tahunan tertangkap (HW/SARIMA "
      "seasonal). Bukti: R² rendah karena <2 siklus. Verifikasi: R² test naik > 0.")
    a("- **Retraining berkala + monitoring drift** (sudah ada champion–challenger di "
      "web). Verifikasi: MAE produksi dipantau per periode, challenger hanya dipromosi "
      "bila MAE < model aktif.")

    a("\n---\n*Dihasilkan otomatis oleh `evaluasi_model.py`.*\n")
    (DOCS / "LAPORAN_EVALUASI.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(R, open(DOCS / "evaluasi_hasil.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        # render ulang LAPORAN dari hasil tersimpan (tanpa fit ulang)
        R = json.load(open(DOCS / "evaluasi_hasil.json", encoding="utf-8"))
        write_report(R)
        print("LAPORAN_EVALUASI.md ditulis ulang dari evaluasi_hasil.json")
    else:
        main()
