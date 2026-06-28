# -*- coding: utf-8 -*-
"""Terapkan perbaikan audit ke nb_05_sarima & nb_06_ensemble.
Sama seperti _apply_audit_fixes.py; nb_06 ditangani khusus (eval-nya berupa loop
2 varian, tanpa variabel `metrik`/`fc_eval` tunggal)."""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "PERBAIKAN AUDIT (diterapkan)"

OLD_CLUSTER = "feat = clu.build_features(panel)"
NEW_CLUSTER = (
    "# [Perbaikan audit C-01] Cluster & scaler di-fit HANYA pada periode TRAIN\n"
    "# (anti-leakage): bulan test (Nov/Des) tidak boleh ikut menentukan fitur 'cluster'.\n"
    "_periods_all = sorted(panel[C.P_PERIOD].unique())\n"
    "_train_p = _periods_all[:-C.TEST_MONTHS]\n"
    "panel_cluster = panel[panel[C.P_PERIOD].isin(_train_p)].copy()\n"
    "feat = clu.build_features(panel_cluster)"
)

HDR = (
    "# ===== PERBAIKAN AUDIT (diterapkan) =====\n"
    "import matplotlib.pyplot as plt\n"
    "from forecaster import NaiveForecaster\n"
    "dec = test_periods[-1]                      # '2025-12' = holdout NYATA (Nov = imputasi)\n"
    "test_dec = test[test[C.P_PERIOD] == dec]\n"
)
BASELINE = (
    "m_naive = evaluate(NaiveForecaster.fit(train), test, test_periods, C.TEST_MONTHS, 'Naive (baseline)')\n"
    "print('[W-06] Baseline Naive :', {k: round(v,3) if isinstance(v,float) else v for k,v in m_naive.items()})\n"
    "print('       Model kalahkan baseline (MAE lebih kecil)?', metrik['MAE'] < m_naive['MAE'])\n"
)
DECONLY = (
    "m_dec = evaluate(fc_eval, test_dec, [dec], C.TEST_MONTHS, 'Des-only (NYATA)')\n"
    "print('[W-02] Metrik Des-only:', {k: round(v,3) if isinstance(v,float) else v for k,v in m_dec.items()})\n"
)

def backtest_block(fit_lines):
    return (
        "# [W-04] Backtest rolling-origin (expanding window) -> MAE mean +/- std (lebih robust dari 1 split)\n"
        "def _bt_fit(trp):\n"
        + fit_lines +
        "_allp = sorted(panel[C.P_PERIOD].unique())\n"
        "_cutoffs = ['2025-06','2025-07','2025-08','2025-09']   # target Jul..Okt (semua observasi NYATA)\n"
        "_bt = []\n"
        "for _co in _cutoffs:\n"
        "    _nxt = _allp[_allp.index(_co)+1]\n"
        "    _fc = _bt_fit(panel[panel[C.P_PERIOD] <= _co])\n"
        "    _act = panel[panel[C.P_PERIOD]==_nxt].set_index(C.P_OBAT)[C.P_DEMAND]\n"
        "    _e = [abs(_fc.predict(o,1)['prediksi'][0]-float(_act.loc[o])) for o in _fc.history if o in _act.index]\n"
        "    _bt.append(float(np.mean(_e)))\n"
        "print('[W-04] Backtest MAE per origin (target Jul..Okt):', [round(x,2) for x in _bt])\n"
        "print(f'       Backtest MAE = {np.mean(_bt):.2f} +/- {np.std(_bt):.2f}')\n"
    )

MD_TS = (
    "## 🔧 Perbaikan Audit (diterapkan)\n"
    "Menanggapi audit pipeline: **C-01** clustering di-fit *train-only* (sel segmentasi di atas), "
    "**W-06** baseline Naive sebagai pembanding, **W-02** metrik **Des-only** (holdout nyata; Nov = imputasi), "
    "**W-04** backtest *rolling-origin* (MAE mean ± std)."
)

# ---------------- nb_05_sarima (pola standar) ----------------
def patch_nb05():
    f = "nb_05_sarima.ipynb"
    nb = nbf.read(HERE / f, as_version=4)
    if any(MARK in c.source for c in nb.cells if c.cell_type == "code"):
        print(f"[skip] {f} sudah diperbaiki."); return
    for c in nb.cells:
        if c.cell_type == "code" and OLD_CLUSTER in c.source:
            c.source = c.source.replace(OLD_CLUSTER, NEW_CLUSTER); break
    bt = backtest_block("    return SarimaForecaster.fit(trp)\n")
    code = "\n".join([HDR, BASELINE, DECONLY, bt])
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "code" and "metrik = evaluate(" in c.source)
    nb.cells.insert(idx + 1, nbf.v4.new_markdown_cell(MD_TS))
    nb.cells.insert(idx + 2, nbf.v4.new_code_cell(code))
    nbf.write(nb, HERE / f)
    print(f"[ok] {f}: C-01 + audit (SARIMA) disisipkan.")

# ---------------- nb_06_ensemble (khusus) ----------------
def patch_nb06():
    f = "nb_06_ensemble.ipynb"
    nb = nbf.read(HERE / f, as_version=4)
    if any(MARK in c.source for c in nb.cells if c.cell_type == "code"):
        print(f"[skip] {f} sudah diperbaiki."); return
    for c in nb.cells:
        if c.cell_type == "code" and OLD_CLUSTER in c.source:
            c.source = c.source.replace(OLD_CLUSTER, NEW_CLUSTER); break
    # definisikan fc_eval & metrik (ensemble weighted = artefak produksi) lebih dulu
    prefix = (
        "# Ensemble weighted dipakai sebagai model representatif (artefak produksi)\n"
        "fc_eval = EnsembleForecaster(hw_eval, sa_eval, w_eval, 'weighted')\n"
        "metrik = evaluate(fc_eval, test, test_periods, C.TEST_MONTHS, 'Ensemble (weighted)')\n"
    )
    bt_fit = (
        "    _tp = sorted(trp[C.P_PERIOD].unique())\n"
        "    _hw = HoltWintersForecaster.fit(trp)\n"
        "    _sa = SarimaForecaster.fit(trp)\n"
        "    _w = compute_ensemble_weights(trp, _tp)\n"
        "    return EnsembleForecaster(_hw, _sa, _w, 'weighted')\n"
    )
    bt = backtest_block(bt_fit)
    code = "\n".join([prefix, HDR, BASELINE, DECONLY, bt])
    # sisipkan setelah sel evaluasi loop (mengandung 'fc_weighted = EnsembleForecaster')
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "code" and "fc_weighted = EnsembleForecaster" in c.source)
    nb.cells.insert(idx + 1, nbf.v4.new_markdown_cell(MD_TS))
    nb.cells.insert(idx + 2, nbf.v4.new_code_cell(code))
    nbf.write(nb, HERE / f)
    print(f"[ok] {f}: C-01 + audit (Ensemble) disisipkan.")

patch_nb05()
patch_nb06()
print("Selesai.")
