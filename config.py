"""Konfigurasi terpusat pipeline forecasting restok obat.
Semua parameter di sini — TIDAK ada hardcode tersebar di modul lain.
"""
from pathlib import Path

# ---------------- PATH ----------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "Rekap_2025_Gabungan_Terisi.xlsx"
SHEET_NAME = "Rekap Kunjungan"

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PANEL_PATH = DATA_DIR / "panel_bulanan.parquet"

# ---------------- KOLOM ----------------
COL_DATE = "Tanggal Masuk"
COL_OBAT = "Resep Obat"
COL_QTY = "JUMLAH"        # demand bulanan (broadcast per baris) -> target
COL_STOCK = "SISA_STOK"   # sisa stok bulanan (broadcast per baris)
COL_UNIT = "SATUAN"
COL_REGISTER = "Register"

# Nama kolom panel hasil olahan
P_OBAT = "obat"
P_PERIOD = "periode"       # Period[M]
P_DEMAND = "demand"
P_STOCK = "stok"
P_IS_OBS = "is_observed"   # True jika demand asli (bukan hasil isi)

# ---------------- PANEL ----------------
# Rentang penuh; November 2025 hilang di sumber -> akan di-reindex & diisi.
PERIOD_START = "2025-01"
PERIOD_END = "2025-12"
# Strategi isi celah: interpolasi linear hanya untuk celah INTERNAL (mis. Nov),
# leading/trailing di luar masa aktif obat -> 0.
FILL_INTERNAL = "interpolate"   # interpolate | zero
FILL_EDGE = "zero"

# ---------------- CLUSTERING ----------------
K_RANGE = range(2, 11)          # silhouette dievaluasi pada k = 2..10
KMEANS_RANDOM_STATE = 42
CLUSTER_FEATURES = ["volume", "cv", "frekuensi", "stok"]

# ---------------- SPLIT TEMPORAL ----------------
TEST_MONTHS = 2                 # 2 bulan terakhir = test
MIN_OBS_TS = 5                  # < 5 titik -> fallback utk HW/SARIMA

# ---------------- FORECAST DEFAULT ----------------
FORECAST_START_PERIOD = "2026-01"   # periode pertama yang diprediksi web
DEFAULT_HORIZON = 1                  # horizon default untuk perhitungan restok 1 bulan
# Forecast jangka panjang (pola pengeluaran ke depan):
FORECAST_DEFAULT_MONTHS = 12        # default tampilan = 1 tahun
FORECAST_MAX_MONTHS = 36           # maksimum = 3 tahun (multi-tahun)
# Guard ledakan forecast (P3): nilai > K×max(histori) dianggap tak wajar -> fallback naive
FORECAST_SANITY_K = 5

# ---------------- TUNING ----------------
RANDOM_STATE = 42
TS_SPLITS = 3                   # n_splits TimeSeriesSplit

# Grid DIPERLUAS (P4): best params lama menyentuh tepi grid
#   (RF n_est=100; GB lr=0.1, depth=2, n_est=100, subsample=1.0) -> beri ruang ke luar.
RF_GRID = {
    "n_estimators": [50, 100, 300, 500],            # +50 (tepi bawah lama)
    "max_depth": [5, 10, 15, None],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", 0.5, 1.0],
}
GB_GRID = {
    "n_estimators": [50, 100, 300, 500],            # +50
    "learning_rate": [0.01, 0.05, 0.1, 0.2],        # +0.2
    "max_depth": [1, 2, 3, 5],                       # +1
    "subsample": [0.6, 0.8, 1.0],                    # +0.6
    "min_samples_leaf": [1, 3, 5],
}

# Holt-Winters (tanpa seasonal — data < 2 siklus)
HW_TREND = [None, "add"]
HW_DAMPED = [True, False]

# SARIMA (tanpa seasonal)
SARIMA_P = [0, 1, 2]
SARIMA_D = [0, 1]
SARIMA_Q = [0, 1, 2]

# Fitur lag/rolling untuk model ML (kausal — hanya masa lalu)
LAGS = [1, 2, 3]
ROLL_WINDOWS = [3]

# ---------------- RESTOCK ----------------
DEFAULT_LEAD_TIME_DAYS = 7
DEFAULT_Z = 1.65
DAYS_PER_MONTH = 30
ATTENTION_FACTOR = 1.5          # status PERLU DIPERHATIKAN jika stok <= 1.5*ROP

BULAN_ID = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April", 5: "Mei", 6: "Juni",
    7: "Juli", 8: "Agustus", 9: "September", 10: "Oktober", 11: "November",
    12: "Desember",
}
