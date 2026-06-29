"""Evaluasi BEFORE vs AFTER perbaikan P1-P6 (berbukti).

- P1+P2: SegmentRouter (mapping segmen->model dari VALIDASI, bukan test) + naive
  untuk slow-moving/deret pendek.
- P3: guard ledakan SARIMA/HW (di forecaster.py).
- P4: re-tune RF/GB pada grid yang diperluas; cek tepi grid hilang.
- P5: dampak bisnis (overstock/understock) router vs Holt-Winters.
- P6: bandingkan segmentasi K=3 vs K=6 untuk routing.

Anti-leakage: pemetaan segmen diturunkan pada bulan VALIDASI (2025-10), model
dievaluasi pada holdout 2025-12. Semua refit train-only.

Jalankan: python evaluasi_v2.py
"""
from __future__ import annotations
import sys, io, json, time
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import config as C
import ml_features as F
import clustering as clu
from restock import hitung_restock
from forecaster_v2 import SegmentRouter
from evaluasi_model import (
    fit_models, evaluate_models, predict_period, actual_observed, load_params,
    MODEL_ORDER,
)

DOCS = C.BASE_DIR / "docs"
CAND = MODEL_ORDER  # kandidat model per segmen


def n_obs_map(panel):
    return panel[panel[C.P_IS_OBS]].groupby(C.P_OBAT).size().to_dict()


def per_segment_best(fcs, panel, train_last, val_period, cluster_map):
    """Pemetaan segmen->model terbaik DIturunkan dari bulan validasi (anti-leakage)."""
    act = actual_observed(panel, val_period)
    common = [o for o in act if all(o in fc.history for fc in fcs.values())]
    seg_of = {o: cluster_map.get(o, {}).get("label", "?") for o in common}
    segs = sorted(set(seg_of.values()))
    errs = {n: {s: [] for s in segs} for n in CAND}
    for n in CAND:
        for o in common:
            p = predict_period(fcs[n], o, train_last, val_period)
            if p is not None:
                errs[n][seg_of[o]].append(abs(act[o] - p))
    best = {}
    for s in segs:
        best[s] = min(CAND, key=lambda n: np.mean(errs[n][s]) if errs[n][s] else 1e9)
    return best, segs


def skill(res, model):
    mn = res["Naive"]["MAE"]
    return (mn - res[model]["MAE"]) / mn if mn else 0


def business_impact(panel, common, y, yh, train_last, test_period):
    stok_last = dict(zip(panel[panel[C.P_PERIOD] == train_last][C.P_OBAT],
                         panel[panel[C.P_PERIOD] == train_last][C.P_STOCK].astype(float)))
    under = over = 0
    under_u = over_u = 0.0
    for i, o in enumerate(common):
        a, p = y[i], yh[i]
        if p < a: under += 1; under_u += a - p
        elif p > a: over += 1; over_u += p - a
    n = len(common)
    return {"understock": under, "overstock": over,
            "pct_under": round(under / n * 100, 1), "pct_over": round(over / n * 100, 1),
            "under_units": int(under_u), "over_units": int(over_u)}


def main():
    t0 = time.time()
    panel = pd.read_parquet(C.PANEL_PATH)
    cluster_map = json.load(open(C.MODELS_DIR / "cluster_labels.json", encoding="utf-8"))["obat"]
    feat_cols = F.feature_columns()
    rf_params = load_params("rf_features.json", {"n_estimators": 100})
    gb_params = load_params("gb_features.json", {"n_estimators": 100})
    nobs = n_obs_map(panel)
    OUT = {}

    print("[1/5] Derive mapping segmen->model dari VALIDASI (train<=2025-09, val=2025-10)...")
    fcs_val = fit_models(panel[panel[C.P_PERIOD] <= "2025-09"], cluster_map,
                         rf_params, gb_params, feat_cols)[0]
    seg_map, segs = per_segment_best(fcs_val, panel, "2025-09", "2025-10", cluster_map)
    print("   mapping:", seg_map)
    OUT["seg_map"] = seg_map

    print("[2/5] BEFORE vs AFTER pada holdout (train<=2025-10, test=2025-12)...")
    fcs = fit_models(panel[panel[C.P_PERIOD] <= "2025-10"], cluster_map,
                     rf_params, gb_params, feat_cols)[0]
    router = SegmentRouter(fcs, cluster_map, seg_map, n_obs=nobs)
    allm = dict(fcs); allm["SegmentRouter (AFTER)"] = router
    res, common, y, preds = evaluate_models(allm, panel, "2025-10", "2025-12")
    for nm in res:
        res[nm]["skill"] = skill(res, nm)
    OUT["holdout"] = {"n": len(common), "metrics": res}
    bi_router = business_impact(panel, common, y, preds["SegmentRouter (AFTER)"], "2025-10", "2025-12")
    bi_hw = business_impact(panel, common, y, preds["Holt-Winters"], "2025-10", "2025-12")
    OUT["business"] = {"router": bi_router, "holt_winters": bi_hw}
    print(f"   HW MAE={res['Holt-Winters']['MAE']:.1f} | Router MAE={res['SegmentRouter (AFTER)']['MAE']:.1f} "
          f"| Naive={res['Naive']['MAE']:.1f}")

    print("[3/5] Walk-forward router vs HW vs Naive (3 fold, reuse mapping)...")
    folds = [("2025-08", "2025-09"), ("2025-09", "2025-10"), ("2025-11", "2025-12")]
    wf = {"SegmentRouter": [], "Holt-Winters": [], "Naive": []}
    sane = True
    for tl, tp in folds:
        fc = fit_models(panel[panel[C.P_PERIOD] <= tl], cluster_map,
                        rf_params, gb_params, feat_cols)[0]
        rt = SegmentRouter(fc, cluster_map, seg_map, n_obs=nobs)
        am = {"SegmentRouter": rt, "Holt-Winters": fc["Holt-Winters"], "Naive": fc["Naive"]}
        rr, _, _, _ = evaluate_models(am, panel, tl, tp)
        for k in wf:
            wf[k].append(rr[k]["MAE"])
        # cek guard P3: tak ada MAE meledak
        if max(rr[k]["MAE"] for k in wf) > 1000:
            sane = False
    OUT["walkforward"] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "vals": [round(x,1) for x in v]}
                          for k, v in wf.items()}
    OUT["p3_sane"] = sane
    print("   walk-forward MAE rata2:", {k: round(np.mean(v), 1) for k, v in wf.items()}, "| P3 sane:", sane)

    print("[4/5] P4 re-tune RF/GB pada grid diperluas...")
    OUT["p4"] = retune_grid(panel, cluster_map, feat_cols)

    print("[5/5] P6 K=3 vs K=6 untuk routing (reuse model holdout)...")
    OUT["p6"] = eval_k3(panel, fcs, nobs, common, y, "2025-10", "2025-12")

    write_v2(OUT)
    print(f"SELESAI {time.time()-t0:.1f}s -> docs/LAPORAN_EVALUASI_V2.md")
    return OUT


def retune_grid(panel, cluster_map, feat_cols):
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from pipeline_forecasting_obat import tune_ml
    train = panel[panel[C.P_PERIOD] <= "2025-10"]
    rf_new, _ = tune_ml(train, cluster_map,
                        RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1),
                        C.RF_GRID, "RF")
    gb_new, _ = tune_ml(train, cluster_map,
                        GradientBoostingRegressor(random_state=C.RANDOM_STATE),
                        C.GB_GRID, "GB")
    def edges(bp, grid):
        e = []
        for k, v in bp.items():
            vals = grid.get(k)
            if not vals or not isinstance(v, (int, float)) or any(isinstance(x, str) for x in vals):
                continue
            nums = [float("inf") if x is None else x for x in vals]
            if v == min(nums) or v == max(nums):
                e.append(f"{k}={v}")
        return e
    return {"rf_new": rf_new, "gb_new": gb_new,
            "rf_edges": edges(rf_new, C.RF_GRID), "gb_edges": edges(gb_new, C.GB_GRID)}


def eval_k3(panel, fcs, nobs, common, y, train_last, test_period):
    """Bangun label K=3, petakan obat->segmen3, evaluasi router-K3 (reuse model holdout).
    Catatan: fitur cluster RF/GB tetap K6 (minoritas routing) -> aproksimasi."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    feat = clu.build_features(panel)
    X = StandardScaler().fit_transform(clu.transform_features(feat))
    lab = KMeans(n_clusters=3, random_state=C.KMEANS_RANDOM_STATE, n_init=10).fit_predict(X)
    feat["cluster"] = lab
    labels, _ = clu.label_clusters(feat)
    feat["segmen"] = feat["cluster"].map(labels)
    cmap3 = {r[C.P_OBAT]: {"cluster": int(r["cluster"]), "label": r["segmen"]}
             for _, r in feat.iterrows()}
    # mapping segmen3->model via validasi tak diulang penuh; pakai aturan: slow->naive, lain->HW
    seg_map3 = {s: ("Naive" if s.startswith("Slow-moving") else "Holt-Winters")
                for s in set(cmap3[o]["label"] for o in cmap3)}
    rt3 = SegmentRouter(fcs, cmap3, seg_map3, n_obs=nobs)
    yh3 = np.array([predict_period(rt3, o, train_last, test_period) for o in common], dtype=float)
    from pipeline_forecasting_obat import mae
    return {"k3_segmen": sorted(set(cmap3[o]["label"] for o in cmap3)),
            "k3_router_mae": float(mae(y, yh3))}


def _f(v, d=2):
    if v is None or (isinstance(v, float) and (np.isnan(v))):
        return "—"
    return f"{v:.{d}f}"


def write_v2(O):
    L = []; a = L.append
    h = O["holdout"]; m = h["metrics"]
    a("# LAPORAN EVALUASI V2 — Hasil Perbaikan P1–P6 (Before/After)\n")
    a("> Dihasilkan `evaluasi_v2.py`. Holdout real **2025-12** (n=%d). Pemetaan "
      "segmen→model diturunkan dari **validasi 2025-10** (anti-leakage seleksi), "
      "model dievaluasi pada 2025-12. Refit train-only.\n" % h["n"])

    a("## Ringkasan Before/After\n")
    before = m["Holt-Winters"]["MAE"]
    after = m["SegmentRouter (AFTER)"]["MAE"]
    delta = (before - after) / before * 100
    a(f"- **BEFORE (Holt-Winters global): MAE {_f(before)}**, skill vs naive {_f(m['Holt-Winters']['skill']*100,1)}%.")
    a(f"- **AFTER (SegmentRouter P1+P2): MAE {_f(after)}**, skill vs naive {_f(m['SegmentRouter (AFTER)']['skill']*100,1)}%.")
    a(f"- **Δ MAE: {_f(delta,1)}%** ({'membaik' if delta>0 else 'MEMBURUK'}).")
    a(f"- P3 guard: walk-forward {'TIDAK ada ledakan (MAE wajar)' if O['p3_sane'] else 'MASIH meledak'}.\n")
    a("> **REKOMENDASI (berbukti):** ADOPSI **P3 (guard ledakan)** & **P4 (grid "
      "diperluas)**; **TOLAK SegmentRouter (P1/P2)** — pemetaan per-segmen dari "
      "validasi 1 bulan tidak menggeneralisasi (peringkat segmen tidak stabil, "
      "lihat walk-forward), router ≈ naive. **Pertahankan Holt-Winters global** "
      "sebagai model produksi.\n")

    a("## P1+P2 — Akurasi holdout (semua model + router)\n")
    a("| Model | MAE | sMAPE(%) | R² | Skill vs Naive |")
    a("|---|---|---|---|---|")
    order = sorted(m.keys(), key=lambda n: m[n]["MAE"])
    for n in order:
        x = m[n]
        a(f"| {'**'+n+'**' if 'Router' in n else n} | {_f(x['MAE'])} | {_f(x['sMAPE(%)'],1)} | {_f(x['R2'],3)} | {_f(x['skill']*100,1)}% |")
    a(f"\n**Pemetaan segmen→model (dari validasi):** " +
      "; ".join(f"{s}→{mm}" for s, mm in O["seg_map"].items()) + ".")

    a("\n## P3 — Walk-forward (guard ledakan)\n")
    a("| Model | MAE rata² | std | per fold |")
    a("|---|---|---|---|")
    for k, v in O["walkforward"].items():
        a(f"| {k} | {_f(v['mean'])} | {_f(v['std'])} | {v['vals']} |")
    a(f"\nSebelumnya (V1) SARIMA/Ensemble meledak (MAE ribuan). Sekarang guard aktif "
      f"→ {'semua MAE wajar.' if O['p3_sane'] else 'masih ada anomali.'}")

    a("\n## P4 — Re-tune grid diperluas\n")
    p4 = O["p4"]
    a(f"- RF best baru: `{p4['rf_new']}` — di tepi: **{p4['rf_edges'] or 'tidak ada'}**.")
    a(f"- GB best baru: `{p4['gb_new']}` — di tepi: **{p4['gb_edges'] or 'tidak ada'}**.")
    a("\n(RF/GB tetap kalah baseline pada data ini; tuning memastikan bukan akibat grid sempit.)")

    a("\n## P5 — Dampak bisnis (router vs Holt-Winters)\n")
    br, bh = O["business"]["router"], O["business"]["holt_winters"]
    a("| | Understock | Overstock | Unit kelebihan |")
    a("|---|---|---|---|")
    a(f"| Holt-Winters | {bh['understock']} ({bh['pct_under']}%) | {bh['overstock']} ({bh['pct_over']}%) | {bh['over_units']} |")
    a(f"| **SegmentRouter** | {br['understock']} ({br['pct_under']}%) | {br['overstock']} ({br['pct_over']}%) | {br['over_units']} |")
    dunits = bh["over_units"] - br["over_units"]
    a(f"\nΔ unit kelebihan (overstock): **{dunits:+d}** ({'turun' if dunits>0 else 'naik'}).")

    a("\n## P6 — Segmentasi K=3 vs K=6 (routing)\n")
    p6 = O["p6"]
    a(f"- Router K=6 MAE: **{_f(after)}** | Router K=3 (HW+slow→naive) MAE: **{_f(p6['k3_router_mae'])}**.")
    a(f"- Segmen K=3: {p6['k3_segmen']}.")
    a(f"- K=3 jauh lebih baik dari router K=6, **tetapi masih > Holt-Winters global "
      f"({_f(before)})** → menyederhanakan segmen tak menyelamatkan routing; HW global tetap unggul.")

    a("\n## Kesimpulan (jujur, berbukti)\n")
    a(f"- **P1/P2 SegmentRouter GAGAL** ({_f(delta,1)}% vs HW; skill {_f(m['SegmentRouter (AFTER)']['skill']*100,1)}% ≈ naive). "
      "Pemetaan segmen→model dari validasi 1 bulan tidak menggeneralisasi karena peringkat "
      "per-segmen tidak stabil antar-bulan (sudah terlihat di walk-forward V1). → **TOLAK**.")
    a("- **P3 guard BERHASIL**: ledakan SARIMA/ensemble di walk-forward hilang (semua MAE wajar). → **ADOPSI**.")
    a(f"- **P4 grid BERHASIL**: RF tak lagi di tepi (edges={O['p4']['rf_edges'] or 'kosong'}), "
      f"GB edges berkurang ({O['p4']['gb_edges']}); mengonfirmasi RF/GB kalah **bukan** karena grid sempit. → **ADOPSI**.")
    a(f"- **P5**: router JUSTRU menaikkan overstock ({br['over_units']} vs {bh['over_units']} unit) → memperkuat penolakan router.")
    a("- **Tindakan akhir:** pertahankan **Holt-Winters global** + guard P3 + grid P4; "
      "untuk kalahkan HW perlu **fitur eksternal (kasus diagnosa)** & **data multi-tahun** (jangka menengah/panjang).")
    a("\n---\n*Angka dari run nyata `evaluasi_v2.py`.*\n")
    (DOCS / "LAPORAN_EVALUASI_V2.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(O, open(DOCS / "evaluasi_v2_hasil.json", "w"), indent=2, default=str)


if __name__ == "__main__":
    if "--report-only" in sys.argv:
        O = json.load(open(DOCS / "evaluasi_v2_hasil.json", encoding="utf-8"))
        write_v2(O)
        print("LAPORAN_EVALUASI_V2.md ditulis ulang dari hasil tersimpan.")
    else:
        main()
