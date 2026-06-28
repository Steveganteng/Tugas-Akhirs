# -*- coding: utf-8 -*-
"""Terapkan perbaikan audit ke nb_01/nb_02/nb_03/nb_04.

Perbaikan:
  C-01  cluster & scaler di-fit HANYA pada periode train (anti-leakage).
  W-06  baseline Naive sebagai pembanding di dalam notebook.
  W-02  metrik khusus Desember (holdout NYATA; Nov = imputasi).
  W-05  gap MAE train vs test (deteksi overfitting) -- model ML.
  W-07  feature importance -- model ML.
  W-04  backtest rolling-origin (expanding window) -> MAE mean +/- std.
"""
import nbformat as nbf
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- C-01: cluster cell jadi train-only ----
OLD_CLUSTER = "feat = clu.build_features(panel)"
NEW_CLUSTER = (
    "# [Perbaikan audit C-01] Cluster & scaler di-fit HANYA pada periode TRAIN\n"
    "# (anti-leakage): bulan test (Nov/Des) tidak boleh ikut menentukan fitur 'cluster'.\n"
    "_periods_all = sorted(panel[C.P_PERIOD].unique())\n"
    "_train_p = _periods_all[:-C.TEST_MONTHS]\n"
    "panel_cluster = panel[panel[C.P_PERIOD].isin(_train_p)].copy()\n"
    "feat = clu.build_features(panel_cluster)"
)

# ---- blok-blok audit (string) ----
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
GAP = (
    "sup_tr = F.make_supervised(train, cluster_map).dropna(subset=feat_cols+['target'])\n"
    "mae_tr = float(np.mean(np.abs(fc_eval.model.predict(sup_tr[feat_cols].values) - sup_tr['target'].values)))\n"
    "print(f'[W-05] MAE train={mae_tr:.2f} | MAE test={metrik[\"MAE\"]:.2f} | "
    "rasio train/test={mae_tr/max(metrik[\"MAE\"],1e-9):.2f}  (rasio jauh < 1 => indikasi overfit)')\n"
)
IMP = (
    "imp = pd.Series(fc_eval.model.feature_importances_, index=feat_cols).sort_values()\n"
    "print('[W-07] Feature importance:'); print(imp.round(4).to_string())\n"
    "imp.plot.barh(figsize=(7,3), title='Feature importance'); plt.tight_layout(); plt.show()\n"
)

def backtest_block(fit_expr):
    return (
        "# [W-04] Backtest rolling-origin (expanding window) -> MAE mean +/- std (lebih robust dari 1 split)\n"
        "def _bt_fit(trp):\n"
        f"    return {fit_expr}\n"
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

# konfigurasi per-notebook: (nama, is_ml, fit_expr_backtest, md_text)
RF_FIT = ("RandomForestForecaster.fit(trp, RandomForestRegressor("
          "random_state=C.RANDOM_STATE, n_jobs=-1, **rf_params), feat_cols, cluster_map)")
GB_FIT = ("GradientBoostingForecaster.fit(trp, GradientBoostingRegressor("
          "random_state=C.RANDOM_STATE, **gb_params), feat_cols, cluster_map)")
HW_FIT = "HoltWintersForecaster.fit(trp)"
NAIVE_FIT = "NaiveForecaster.fit(trp)"

MD_ML = (
    "## 🔧 Perbaikan Audit (diterapkan)\n"
    "Menanggapi audit pipeline: **C-01** clustering di-fit *train-only* (sel segmentasi di atas), "
    "**W-06** baseline Naive sebagai pembanding, **W-02** metrik **Des-only** (holdout nyata; Nov = imputasi), "
    "**W-05** gap MAE train↔test (overfitting), **W-07** feature importance, "
    "**W-04** backtest *rolling-origin* (MAE mean ± std)."
)
MD_TS = (
    "## 🔧 Perbaikan Audit (diterapkan)\n"
    "Menanggapi audit pipeline: **C-01** clustering di-fit *train-only* (sel segmentasi di atas), "
    "**W-06** baseline Naive sebagai pembanding, **W-02** metrik **Des-only** (holdout nyata; Nov = imputasi), "
    "**W-04** backtest *rolling-origin* (MAE mean ± std)."
)
MD_NAIVE = (
    "## 🔧 Perbaikan Audit (diterapkan)\n"
    "Menanggapi audit pipeline: **C-01** clustering di-fit *train-only* (sel segmentasi di atas), "
    "**W-02** metrik **Des-only** (holdout nyata; Nov = imputasi), "
    "**W-04** backtest *rolling-origin* (MAE mean ± std). (Naive sendiri adalah baseline.)"
)

CONFIG = {
    "nb_01_naive.ipynb":            dict(kind="naive", fit=NAIVE_FIT, md=MD_NAIVE),
    "nb_02_random_forest.ipynb":    dict(kind="ml",    fit=RF_FIT,    md=MD_ML),
    "nb_03_gradient_boosting.ipynb":dict(kind="ml",    fit=GB_FIT,    md=MD_ML),
    "nb_04_holt_winters.ipynb":     dict(kind="ts",    fit=HW_FIT,    md=MD_TS),
}

MARK = "PERBAIKAN AUDIT (diterapkan)"

for fname, cfg in CONFIG.items():
    nb = nbf.read(HERE / fname, as_version=4)

    # idempotent: jika sudah pernah diterapkan, lewati
    if any(MARK in c.source for c in nb.cells if c.cell_type == "code"):
        print(f"[skip] {fname} sudah berisi perbaikan audit.")
        continue

    # 1) C-01: edit cluster cell
    edited_cluster = False
    for c in nb.cells:
        if c.cell_type == "code" and OLD_CLUSTER in c.source:
            c.source = c.source.replace(OLD_CLUSTER, NEW_CLUSTER)
            edited_cluster = True
            break
    assert edited_cluster, f"{fname}: cluster cell tidak ditemukan"

    # 2) susun isi sel audit
    parts = [HDR]
    if cfg["kind"] != "naive":
        parts.append(BASELINE)
    parts.append(DECONLY)
    if cfg["kind"] == "ml":
        parts.append(GAP)
        parts.append(IMP)
    parts.append(backtest_block(cfg["fit"]))
    audit_code = "\n".join(parts)

    # 3) sisipkan setelah sel evaluasi (yang mengandung 'metrik = evaluate(')
    idx = next(i for i, c in enumerate(nb.cells)
               if c.cell_type == "code" and "metrik = evaluate(" in c.source)
    md_cell = nbf.v4.new_markdown_cell(cfg["md"])
    code_cell = nbf.v4.new_code_cell(audit_code)
    nb.cells.insert(idx + 1, md_cell)
    nb.cells.insert(idx + 2, code_cell)

    nbf.write(nb, HERE / fname)
    print(f"[ok] {fname}: C-01 + {len(parts)} blok audit disisipkan (kind={cfg['kind']}).")

print("Selesai.")
