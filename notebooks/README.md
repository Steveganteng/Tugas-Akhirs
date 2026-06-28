# Notebook Per-Model (Forecasting Restok Obat)

Setiap notebook **mandiri** dan menjalankan seluruh tahapan untuk **satu pendekatan**,
dari *preprocessing* sampai *modeling*:

1. Setup & import (memakai ulang modul `config`, `data_processing`, `clustering`,
   `ml_features`, `forecaster`, `restock`, `pipeline_forecasting_obat`)
2. Muat data mentah → `load_raw()`
3. Pembersihan → `clean()`
4. Panel bulanan obat×bulan → `build_panel()` + EDA grafik
5. Segmentasi K-Means (silhouette)
6. Split temporal (anti-leakage; 2 bulan terakhir = test)
7. Feature engineering & training model
8. Evaluasi (MAE / RMSE / MAPE / sMAPE / R²)
9. Simpan artefak ke `../models_nb/`
10. Contoh prediksi 3 bulan + rekomendasi restok (`hitung_restock`)

| Notebook | Model |
|---|---|
| `nb_01_naive.ipynb` | Naive (lag-1) — baseline |
| `nb_02_random_forest.ipynb` | Random Forest — **model aktif di web** |
| `nb_03_gradient_boosting.ipynb` | Gradient Boosting |
| `nb_04_holt_winters.ipynb` | Holt-Winters |
| `nb_05_sarima.ipynb` | SARIMA |
| `nb_06_ensemble.ipynb` | Ensemble (HW + SARIMA) |

## Menjalankan
```bash
# dari folder ini
jupyter notebook            # lalu buka & Run All
# atau headless:
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 nb_02_random_forest.ipynb
```

## Catatan
- Artefak notebook ditulis ke `../models_nb/` agar **tidak menimpa** artefak
  produksi di `../models/` yang dipakai aplikasi web.
- RF & GB menjalankan `GridSearchCV` + `TimeSeriesSplit` (butuh ~1–2 menit).
- HW & SARIMA di-fit per obat (SARIMA paling lama). Notebook ensemble melatih
  HW+SARIMA dua kali (eval + penuh) sehingga paling lama (~3–4 menit).
- Untuk perbandingan ke-6 model dalam satu tabel, jalankan
  `python pipeline_forecasting_obat.py` (lihat `03_perbandingan_model.md`).
