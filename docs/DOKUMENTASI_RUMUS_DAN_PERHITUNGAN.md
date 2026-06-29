# Dokumentasi Rumus dan Perhitungan Pipeline Forecasting Restock Obat
## Klinik Del — Institut Teknologi Del

> **File referensi:** `notebooks/Forecasting_Pipeline_Lengkap.ipynb`  
> **Alur kerja:** Data Mentah → X\_t Matrix → Klasifikasi ADI/CV² → SES / Croston / SBA → E2 Scaler → Evaluasi → Kebijakan Stok (Min/Max/Safety Stock) → ABC-VEN → Model `.pkl`

---

## Daftar Isi

1. [Dataset dan Fitur](#1-dataset-dan-fitur)
2. [Pemrosesan Data (Data Processing)](#2-pemrosesan-data-data-processing)
3. [Klasifikasi Pola Permintaan (ADI / CV²)](#3-klasifikasi-pola-permintaan-adi--cv)
4. [Rekayasa Fitur (Feature Engineering)](#4-rekayasa-fitur-feature-engineering)
5. [Model Forecasting](#5-model-forecasting)
   - 5.1 [SES — Single Exponential Smoothing](#51-ses--single-exponential-smoothing)
   - 5.2 [Croston's Method](#52-crostons-method)
   - 5.3 [SBA — Syntetos-Boylan Approximation](#53-sba--syntetos-boylan-approximation)
   - 5.4 [E2 — Visit Scaler (Penyesuai Kunjungan)](#54-e2--visit-scaler-penyesuai-kunjungan)
6. [Metrik Evaluasi](#6-metrik-evaluasi)
7. [Kebijakan Stok (Safety Stock, Min, Max)](#7-kebijakan-stok-safety-stock-min-max)
8. [Klasifikasi ABC](#8-klasifikasi-abc)
9. [Klasifikasi VEN](#9-klasifikasi-ven)
10. [Skor Prioritas ABC-VEN](#10-skor-prioritas-abc-ven)
11. [Analisis Musiman (Seasonal Index)](#11-analisis-musiman-seasonal-index)
12. [Definisi Seluruh Variabel](#12-definisi-seluruh-variabel)

---

## 1. Dataset dan Fitur

### 1.1 Data Mentah (`Rekap_2025_Gabungan_Terisi.xlsx`, sheet `Rekap Kunjungan`)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `Tanggal Masuk` | datetime | Tanggal kunjungan pasien |
| `Register` | string | ID unik kunjungan (digunakan menghitung jumlah kunjungan) |
| `Kode Diagnosa` | string | Kode ICD-10 diagnosis primer |
| `Diagnosa Primer` | string | Teks diagnosis primer |
| `Resep Obat` | string | Nama obat yang diresepkan |
| `JUMLAH` | float | Jumlah unit obat yang diberikan |
| `SISA_STOK` | float | Sisa stok obat pada saat kunjungan |
| `SATUAN` | string | Satuan unit obat (tablet, kapsul, dll.) |

### 1.2 Data Panel Bulanan (`data/panel_bulanan.parquet`)

| Kolom | Tipe | Deskripsi |
|---|---|---|
| `periode` | string | Periode format `YYYY-MM` |
| `obat` | string | Nama obat |
| `demand` | float | Total pemakaian (permintaan) per bulan |
| `stok` | float | Level stok pada periode tersebut |
| `satuan` | string | Satuan unit |
| `is_observed` | bool | Apakah data berasal dari pengamatan nyata |
| `bulan` | int | Nomor bulan (1–12) |

### 1.3 Fitur yang Digunakan untuk Machine Learning (`FEATURE_COLS`)

| Fitur | Deskripsi |
|---|---|
| `demand_lag1` | Permintaan 1 bulan sebelumnya |
| `demand_lag2` | Permintaan 2 bulan sebelumnya |
| `rolling_mean3` | Rata-rata bergulir permintaan 3 bulan terakhir (t-1, t-2, t-3) |
| `visit_count` | Jumlah kunjungan unik pada bulan tersebut |
| `stok_prev` | Stok bulan sebelumnya |
| `AMC` | Rata-rata konsumsi bulanan (Average Monthly Consumption) |
| `sigma_bln` | Standar deviasi permintaan bulanan |
| `ADI` | Average Demand Interval |
| `CV2` | Coefficient of Variation kuadrat |
| `bulan` | Nomor bulan (variabel musiman) |

**Target variabel:** `demand_winsor` — permintaan yang telah di-winsorize (P5–P95) untuk meredam outlier ekstrem.

---

## 2. Pemrosesan Data (Data Processing)

### 2.1 Pembangunan Matriks X\_t (Pivot Drug × Bulan)

Dari data mentah, dibentuk matriks **X** berukuran **175 obat × 12 bulan**:

```python
# Agregasi: ambil nilai pertama per (obat, bulan)
obs = raw.dropna(subset=['Resep Obat','JUMLAH'])
       .groupby(['Resep Obat','bulan'])['JUMLAH'].first()

# Pivot ke matriks obat × bulan
X = obs.pivot(index='obat', columns='bulan', values='X_t')
       .reindex(columns=range(1, 13))
```

- **Hasil:** 175 obat × 12 bulan; 42.8% sel memiliki nilai
- Sel kosong (NaN) terjadi karena obat tidak muncul di bulan tertentu

### 2.2 Penanganan Nilai Hilang: `no_demand` vs `data_gap`

Dua jenis nilai hilang dibedakan secara eksplisit:

| Tipe | Kondisi | Perlakuan |
|---|---|---|
| `no_demand` | Obat tidak ada di resep bulan itu | Diisi `0.0` (memang tidak ada pemakaian) |
| `data_gap` | Obat ada di resep tapi `JUMLAH` kosong | Tetap `NaN` (data rusak/tidak tercatat) |

```python
presc = set of (obat, bulan) where obat was prescribed

for ob in X.index:
    for b in range(1, 13):
        if NaN and (ob, b) NOT in presc:
            X_clean[ob, b] = 0.0    # no_demand → 0
        # else: data_gap → NaN tetap
```

**Hasil:** 893 sel `no_demand` → diisi 0, 308 sel `data_gap` → tetap NaN.

### 2.3 Winsorization Per Obat (Metode IQR)

Winsorization meredam outlier dengan memotong nilai di luar batas IQR:

```
Batas bawah = max(Q1 − 1.5 × IQR, 0)
Batas atas  = Q3 + 1.5 × IQR
IQR         = Q3 − Q1
```

Diterapkan **per baris (per obat)** pada matriks `X_clean`:

```python
def winsor_row(s):
    v = s.dropna()
    if len(v) < 4: return s           # skip jika data terlalu sedikit
    q1, q3 = v.quantile(.25), v.quantile(.75)
    iqr = q3 - q1
    lower = max(q1 - 1.5*iqr, 0)     # tidak boleh negatif
    upper = q3 + 1.5*iqr
    return s.clip(lower=lower, upper=upper)

X_winsor = X_clean.apply(winsor_row, axis=1)
```

**Hasil:** 462 sel berubah setelah winsorization.

### 2.4 Winsorization Target pada Data Panel (P5–P95)

Untuk data panel bulanan (target ML), winsorization dilakukan per obat menggunakan persentil P5 dan P95:

```python
def winsor_series(s, limits=(0.05, 0.05)):
    if len(s) < 4: return s
    return pd.Series(scipy_winsorize(s.values, limits=limits), index=s.index)

df_pre['demand_winsor'] = df_pre.groupby('obat')['demand'].transform(winsor_series)
```

### 2.5 Split Data Train / Test

| Set | Bulan | Keterangan |
|---|---|---|
| Train | 1–10 (Jan–Okt) | Digunakan untuk fitting model |
| Validasi | 7–10 (Jul–Okt) | Subset train untuk tuning hyperparameter (alpha) |
| Test | 11–12 (Nov–Des) | Evaluasi akhir model |

### 2.6 Standarisasi Fitur (StandardScaler)

Semua fitur numerik (kecuali `bulan`) dinormalisasi menggunakan:

$$z = \frac{x - \mu_{train}}{\sigma_{train}}$$

```python
SCALE_COLS = ['demand_lag1','demand_lag2','rolling_mean3','visit_count',
              'stok_prev','AMC','sigma_bln','ADI','CV2']

scaler = StandardScaler()
X_train_sc[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
X_test_sc[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])
```

> **Penting:** Scaler hanya di-*fit* pada data train (bulan 1–10) untuk menghindari data leakage.

---

## 3. Klasifikasi Pola Permintaan (ADI / CV²)

### 3.1 Rumus ADI (Average Demand Interval)

ADI mengukur seberapa jarang suatu obat diminta. Semakin besar ADI, semakin jarang:

$$ADI = \frac{\text{Total periode pengamatan}}{\text{Jumlah periode dengan permintaan} > 0}$$

**Contoh:** Jika obat muncul di 6 dari 12 bulan → ADI = 12/6 = 2.0

### 3.2 Rumus CV² (Coefficient of Variation Squared)

CV² mengukur variabilitas besaran permintaan pada periode yang ada permintaan:

$$CV^2 = \left(\frac{\sigma_{non-zero}}{\mu_{non-zero}}\right)^2$$

di mana $\sigma$ = standar deviasi dan $\mu$ = rata-rata dari nilai permintaan yang tidak nol.

### 3.3 Matriks Klasifikasi (Syntetos et al., 2005)

|  | **CV² < 0.49** | **CV² ≥ 0.49** |
|---|---|---|
| **ADI < 1.32** | `smooth` — permintaan halus/reguler | `erratic` — permintaan tak beraturan |
| **ADI ≥ 1.32** | `intermittent` — permintaan jarang tapi konsisten | `lumpy` — permintaan jarang dan tidak beraturan |

**Khusus:** Jika jumlah data non-zero < 2 atau total periode < 4, dikategorikan `data tidak cukup`.

**Implementasi:**
```python
ADI_CUT = 1.32
CV2_CUT = 0.49

def klasifikasi(row):
    v = row.dropna()
    nz = v[v > 0]
    if len(nz) < 2 or len(v) < 4:
        return 'data tidak cukup', np.nan, np.nan

    adi = len(v) / len(nz)
    cv2 = (nz.std() / nz.mean())**2 if nz.mean() > 0 else np.nan

    if   adi < ADI_CUT and cv2 < CV2_CUT: c = 'smooth'
    elif adi < ADI_CUT:                    c = 'erratic'
    elif cv2 < CV2_CUT:                    c = 'intermittent'
    else:                                  c = 'lumpy'
    return c, adi, cv2
```

**Distribusi hasil (175 obat):**
- 66 obat: `data tidak cukup`
- 48 obat: `intermittent`
- 33 obat: `lumpy`
- 15 obat: `erratic`
- 13 obat: `smooth`

---

## 4. Rekayasa Fitur (Feature Engineering)

### 4.1 AMC (Average Monthly Consumption)

$$AMC = \frac{\sum_{t=1}^{T} X_t}{T}$$

di mana $T$ = jumlah bulan yang memiliki data (tidak NaN).

```python
amc_map = X_clean.apply(
    lambda s: s.dropna().mean() if s.dropna().size else 0.0, axis=1)
```

**Digunakan pada dataset:** Setiap obat memiliki satu nilai AMC yang dimerge ke seluruh 12 baris panel, sebagai fitur statis yang merepresentasikan rata-rata konsumsi historis.

### 4.2 Sigma Bulanan (σ)

$$\sigma = \sqrt{\frac{\sum_{t=1}^{T}(X_t - \mu)^2}{T}} \quad \text{(populasi, ddof=0)}$$

```python
sig_map = X_clean.apply(
    lambda s: s.dropna().std(ddof=0) if s.dropna().size > 1 else 0.0, axis=1)
```

**Digunakan pada dataset:** Nilai σ per obat di-merge ke panel dan digunakan sebagai fitur `sigma_bln`, serta menjadi input utama untuk menghitung Safety Stock.

### 4.3 Lag Permintaan

```python
feat['demand_lag1'] = feat.groupby('obat')['demand'].shift(1)  # bulan t-1
feat['demand_lag2'] = feat.groupby('obat')['demand'].shift(2)  # bulan t-2
```

**Digunakan pada dataset:** Memberikan informasi historis langsung ke model. Misalnya, untuk baris obat A bulan 3, `demand_lag1` = permintaan obat A bulan 2, `demand_lag2` = bulan 1.

### 4.4 Rolling Mean 3 Bulan

$$\text{rolling\_mean3}_t = \frac{X_{t-1} + X_{t-2} + X_{t-3}}{3}$$

```python
feat['rolling_mean3'] = feat.groupby('obat')['demand'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean())
```

**Digunakan pada dataset:** Menggunakan `.shift(1)` terlebih dahulu memastikan tidak ada data leakage — nilai pada bulan t tidak masuk ke rolling.

### 4.5 Stok Bulan Sebelumnya

```python
feat['stok_prev'] = feat.groupby('obat')['stok'].shift(1)
```

**Digunakan pada dataset:** Memberikan konteks apakah stok sebelumnya sudah rendah atau masih aman, yang bisa mempengaruhi pola permintaan yang tercatat.

### 4.6 Visit Count (Jumlah Kunjungan Bulanan)

```python
visit_bln = raw.groupby('bulan')['Register'].nunique()
```

**Digunakan pada dataset:** Di-merge ke panel berdasarkan nomor bulan. Karena data kunjungan agregat (bukan per obat), semua obat di bulan yang sama mendapat nilai `visit_count` yang identik.

---

## 5. Model Forecasting

Tidak digunakan model ML klasik (Random Forest, XGBoost, dll.). Model yang digunakan adalah metode deret waktu yang dipilih berdasarkan **kategori permintaan** obat:

| Kategori | Metode yang digunakan |
|---|---|
| `smooth`, `erratic` | SES (Single Exponential Smoothing) |
| `intermittent`, `lumpy` | Croston atau SBA (dipilih yang terbaik) |

### 5.1 SES — Single Exponential Smoothing

#### Rumus

$$F_{t+1} = \alpha \cdot X_t + (1 - \alpha) \cdot F_t$$

di mana:
- $F_{t+1}$ = nilai ramalan untuk periode berikutnya
- $X_t$ = nilai aktual pada periode $t$
- $F_t$ = nilai ramalan pada periode $t$
- $\alpha \in (0, 1)$ = faktor pemulusan (semakin besar = semakin reaktif terhadap data terbaru)

#### Implementasi

```python
def ses_next(train, alpha):
    x = np.asarray(train, float)
    F = x[0]                                          # inisialisasi dengan nilai pertama
    for t in range(1, len(x)):
        F = alpha * x[t-1] + (1-alpha) * F           # update rekursif
    return max(alpha * x[-1] + (1-alpha) * F, 0.0)   # ramalan periode berikutnya, minimal 0
```

#### Tuning Hyperparameter α

Dilakukan **grid search** dengan validasi rolling-origin pada bulan 7–10:

```
GRID_SES = [0.01, 0.03, 0.05, ..., 0.99]  (49 nilai, step 0.02)
```

Untuk setiap obat dan setiap nilai α, dihitung WAPE pada bulan validasi. Nilai α dengan WAPE terkecil dipilih.

#### Penerapan pada Dataset

- Diterapkan pada 28 obat dengan kategori `smooth` atau `erratic`
- Setiap obat memiliki nilai α optimal yang berbeda (median α = 0.41)
- Median WAPE pada test set: 83.2%

---

### 5.2 Croston's Method

Metode ini dirancang khusus untuk permintaan **intermittent** (jarang muncul). Ia memisahkan forecasting untuk dua komponen: **besaran permintaan** dan **interval antar permintaan**.

#### Rumus

$$\hat{Z}_q = \alpha_z \cdot X_{t_q} + (1-\alpha_z) \cdot \hat{Z}_{q-1}$$

$$\hat{P}_q = \alpha_p \cdot q + (1-\alpha_p) \cdot \hat{P}_{q-1}$$

$$F_{Croston} = \frac{\hat{Z}}{\hat{P}}$$

di mana:
- $X_{t_q}$ = nilai permintaan non-nol ke-$q$
- $\hat{Z}_q$ = estimasi besaran permintaan (diperbarui hanya saat ada permintaan)
- $\hat{P}_q$ = estimasi interval antar permintaan
- $q$ = jarak (bulan) sejak permintaan terakhir
- $\alpha_z, \alpha_p \in (0, 1)$ = faktor pemulusan masing-masing komponen

#### Implementasi

```python
def _croston_core(x, az, ap):
    x = np.asarray(x, float)
    idx = np.where(x > 0)[0]               # indeks bulan yang ada permintaan
    if len(idx) == 0: return 0.0, 1.0

    first = idx[0]
    z = x[first]                            # init Z dengan permintaan non-nol pertama
    p = float(first + 1)                    # init P dengan jarak ke permintaan pertama
    q = 1

    for t in range(first + 1, len(x)):
        if x[t] > 0:
            z = az * x[t] + (1 - az) * z   # update Z hanya saat ada permintaan
            p = ap * q + (1 - ap) * p       # update P
            q = 1                           # reset counter interval
        else:
            q += 1                          # tambah counter jika tidak ada permintaan
    return z, p

def croston_next(tr, az, ap):
    z, p = _croston_core(tr, az, ap)
    return z / p if p > 0 else 0.0
```

#### Tuning Hyperparameter

```
GRID_INT = [0.05, 0.10, 0.15, ..., 0.50]  (10 nilai)
Grid search: 10 × 10 = 100 kombinasi (α_z, α_p)
```

#### Penerapan pada Dataset

- Diterapkan pada 81 obat dengan kategori `intermittent` atau `lumpy`
- Median WAPE test: 100% (permintaan sangat tidak menentu)

---

### 5.3 SBA — Syntetos-Boylan Approximation

SBA adalah koreksi dari Croston yang mengurangi **bias ke atas** (*upward bias*) yang inheren dalam metode Croston.

#### Rumus

$$F_{SBA} = \left(1 - \frac{\alpha_p}{2}\right) \cdot \frac{\hat{Z}}{\hat{P}}$$

Faktor koreksi $\left(1 - \frac{\alpha_p}{2}\right)$ sedikit mengecilkan estimasi Croston untuk mengurangi over-estimation.

#### Implementasi

```python
def sba_next(tr, az, ap):
    z, p = _croston_core(tr, az, ap)        # sama seperti Croston
    return (1 - ap / 2.0) * (z / p) if p > 0 else 0.0
```

#### Pemilihan Antara Croston dan SBA

Setelah keduanya difit, metode dengan **WAPE terkecil pada validation set** dipilih sebagai `metode_terbaik` per obat.

---

### 5.4 E2 — Visit Scaler (Penyesuai Kunjungan)

E2 (*External Event*) adalah faktor penyesuaian yang memodulasi ramalan berdasarkan **proporsi jumlah kunjungan** bulan target terhadap rata-rata kunjungan training.

#### Rumus

$$F^{E2}_t = F_t \times \frac{kunjungan(t)}{\overline{kunjungan}_{train}}$$

di mana $\overline{kunjungan}_{train}$ = rata-rata kunjungan unik pada bulan 1–10.

#### Implementasi

```python
visit_raw = raw.groupby('bulan')['Register'].nunique()
visit_train_mean = float(visit_raw[visit_raw.index <= 10].mean())

# visit_scale: dict {bulan: skala}
visit_scale = {
    int(m): float(v / visit_train_mean)
    for m, v in visit_raw.items()
}

# Diterapkan pada setiap ramalan:
forecast_e2 = max(fc(train) * visit_scale.get(target_month, 1.0), 0)
```

#### Interpretasi pada Dataset

- Jika bulan November memiliki 120 kunjungan sedangkan rata-rata training 100, maka `visit_scale[11] = 1.20`, artinya forecast dinaikkan 20%.
- Korelasi agregat demand vs kunjungan = **r = 0.87** (sangat kuat), mendukung penggunaan E2.

---

## 6. Metrik Evaluasi

### 6.1 WAPE (Weighted Absolute Percentage Error) — Metrik Utama

$$WAPE = \frac{\sum_{t} |A_t - F_t|}{\sum_{t} |A_t|} \times 100\%$$

WAPE tahan terhadap permintaan nol dan lebih stabil dari MAPE untuk data intermittent.

```python
def WAPE(a, f):
    den = np.abs(a).sum()
    return np.abs(a - f).sum() / den * 100 if den > 0 else np.nan
```

### 6.2 MAD (Mean Absolute Deviation)

$$MAD = \frac{1}{n}\sum_{t=1}^{n}|A_t - F_t|$$

Mengukur rata-rata kesalahan absolut dalam satuan unit obat.

```python
def MAD(a, f):
    return float(np.mean(np.abs(a - f))) if len(a) else np.nan
```

### 6.3 MSE (Mean Squared Error)

$$MSE = \frac{1}{n}\sum_{t=1}^{n}(A_t - F_t)^2$$

Memberikan penalti lebih besar pada kesalahan besar.

```python
def MSE(a, f):
    return float(np.mean((a - f)**2)) if len(a) else np.nan
```

### 6.4 MAPE (Mean Absolute Percentage Error)

$$MAPE = \frac{1}{n}\sum_{t: A_t \neq 0} \left|\frac{A_t - F_t}{A_t}\right| \times 100\%$$

Dihitung hanya untuk periode dengan $A_t \neq 0$ (menghindari pembagian nol).

```python
def MAPE(a, f):
    m = a != 0
    return float(np.mean(np.abs((a[m] - f[m]) / a[m])) * 100) if m.sum() else np.nan
```

### 6.5 Kategori Mutu Berdasarkan WAPE

| WAPE | Kategori Mutu |
|---|---|
| ≤ 20% | Excellent |
| 21% – 50% | Good |
| 51% – 100% | Bad |
| > 100% | Tidak Layak |

---

## 7. Kebijakan Stok (Safety Stock, Min, Max)

### Parameter Global

| Parameter | Nilai | Keterangan |
|---|---|---|
| `LT` | 1.0 bulan | Lead Time — waktu tunggu pengadaan |
| `R` | 1.0 bulan | Review Period — interval pemeriksaan stok |
| `Z` | 1.65 | Z-score untuk service level 95% |

### 7.1 Safety Stock (Stok Pengaman)

Safety stock adalah buffer untuk mengantisipasi variabilitas permintaan selama lead time.

$$SS = Z \cdot \sigma \cdot \sqrt{LT}$$

$$SS = 1.65 \times \sigma \times \sqrt{1} = 1.65\sigma$$

Dibulatkan ke atas:

$$SS = \lceil 1.65 \times \sigma \rceil$$

**Implementasi:**
```python
inv['safety_stock'] = (Z * SIG_ser * math.sqrt(LT)).apply(lambda x: int(math.ceil(x)))
```

**Penerapan pada data:** Nilai `SIG_ser` (σ) dihitung dari `X_clean` per obat dengan `ddof=0`. Untuk obat dengan permintaan sangat stabil (σ rendah), safety stock kecil; untuk obat dengan permintaan fluktuatif (σ besar), safety stock lebih besar.

**Contoh:** Obat A dengan σ = 10 tablet → SS = ceil(1.65 × 10) = ceil(16.5) = 17 tablet

### 7.2 Min Stock / Reorder Point (ROP)

Min stock adalah titik pemicu pemesanan ulang. Ketika stok turun ke atau di bawah angka ini, pemesanan harus dilakukan.

$$Min = AMC \times LT + SS = AMC \times LT + Z \cdot \sigma \cdot \sqrt{LT}$$

$$Min = AMC \times 1 + 1.65 \times \sigma \times \sqrt{1} = AMC + 1.65\sigma$$

Dibulatkan ke atas:

$$Min = \lceil AMC + 1.65\sigma \rceil$$

**Interpretasi:**
- $AMC \times LT$ = perkiraan kebutuhan selama menunggu pesanan datang
- $Z \cdot \sigma \cdot \sqrt{LT}$ = buffer untuk lonjakan permintaan selama lead time

**Implementasi:**
```python
inv['Min'] = (AMC_ser * LT + Z * SIG_ser * math.sqrt(LT)).apply(lambda x: int(math.ceil(x)))
```

**Penerapan pada data:** AMC diambil dari `X_clean` (rata-rata historis), bukan dari forecast. Ini memastikan Min/Max didasarkan pada konsumsi yang sudah terbukti, bukan proyeksi masa depan.

**Contoh:** Obat A dengan AMC = 50, σ = 10 → Min = ceil(50×1 + 1.65×10×√1) = ceil(50 + 16.5) = 67 tablet

### 7.3 Max Stock

Max stock adalah target stok setelah pemesanan. Pemesanan dilakukan hingga mencapai level ini.

$$Max = AMC \times (LT + R) + Z \cdot \sigma \cdot \sqrt{LT + R}$$

$$Max = AMC \times (1 + 1) + 1.65 \times \sigma \times \sqrt{1 + 1}$$

$$Max = 2 \times AMC + 1.65 \times \sigma \times \sqrt{2}$$

Dibulatkan ke atas:

$$Max = \lceil 2 \times AMC + 1.65\sigma\sqrt{2} \rceil$$

**Interpretasi:**
- $AMC \times (LT + R)$ = kebutuhan selama lead time + satu periode review
- $Z \cdot \sigma \cdot \sqrt{LT + R}$ = buffer untuk variabilitas permintaan selama periode tersebut

**Implementasi:**
```python
inv['Max'] = (AMC_ser * (LT + R) + Z * SIG_ser * math.sqrt(LT + R)).apply(
    lambda x: int(math.ceil(x)))
```

**Contoh:** Obat A dengan AMC = 50, σ = 10 → Max = ceil(2×50 + 1.65×10×√2) = ceil(100 + 23.3) = 124 tablet

### 7.4 Stok Terkini dan Status

```python
# Ambil sisa stok terakhir yang tercatat per obat
ks = raw.dropna(subset=['Resep Obat','SISA_STOK']).sort_values('bulan')
inv['stok_terkini'] = ks.groupby('Resep Obat')['SISA_STOK'].last()

# Tentukan status
inv['status'] = np.where(
    inv.stok_terkini.isna(),          'tidak diketahui',
    np.where(inv.stok_terkini <= inv.Min, 'DI BAWAH MIN',   'aman'))
```

### 7.5 Jumlah yang Perlu Dipesan

$$\text{Pesan} = \max(0,\ Max - \text{stok\_terkini}) \quad \text{jika stok\_terkini} \leq Min$$

```python
inv['perlu_pesan'] = inv.apply(
    lambda r: int(max(0, r.Max - r.stok_terkini))
              if pd.notna(r.stok_terkini) and r.stok_terkini <= r.Min
              else 0, axis=1)
```

**Interpretasi:** Pemesanan hanya dipicu jika stok sudah di bawah atau sama dengan Min. Jumlah yang dipesan adalah selisih antara Max dan stok saat ini, sehingga stok kembali ke level maksimum.

---

## 8. Klasifikasi ABC

Klasifikasi ABC menggunakan **analisis Pareto** berdasarkan total konsumsi tahunan.

### Rumus

1. Hitung total konsumsi tahunan per obat: $TOT_i = \sum_{t=1}^{12} X_{i,t}$
2. Urutkan dari tertinggi ke terendah
3. Hitung persentase kumulatif: $KUM_i = \frac{\sum_{j=1}^{i} TOT_j}{\sum_{j} TOT_j} \times 100$
4. Tetapkan kategori berdasarkan kumulatif

| Kategori | Kumulatif | Interpretasi |
|---|---|---|
| **A** | ≤ 80% | Sedikit obat yang menyumbang sebagian besar pemakaian — prioritas tinggi |
| **B** | 80% – 95% | Pemakaian menengah |
| **C** | 95% – 100% | Banyak obat tapi kontribusi kecil — prioritas rendah |

**Implementasi:**
```python
ab  = inv.sort_values('total_thn', ascending=False)
kum = ab['total_thn'].cumsum() / ab['total_thn'].sum() * 100
inv['ABC'] = kum.map(lambda c: 'A' if c <= 80 else ('B' if c <= 95 else 'C'))
```

**Penerapan pada data:** `total_thn` dihitung dari `X_clean` (`TOT_ser`), yaitu total unit yang benar-benar terpakai sepanjang tahun, bukan dari forecast. Obat kelas A memerlukan kontrol stok yang lebih ketat.

---

## 9. Klasifikasi VEN

Klasifikasi VEN menentukan tingkat kebutuhan klinis obat berdasarkan **huruf pertama kode diagnosis ICD-10** yang paling sering muncul bersama obat tersebut.

### Pemetaan ICD-10 → VEN

| VEN | Arti | Kode ICD-10 (huruf pertama) | Contoh Chapter ICD-10 |
|---|---|---|---|
| **V** (Vital) | Obat tidak tergantikan untuk kondisi mengancam jiwa | A, B, C, D, I, O, P, S, T | Infeksi, Neoplasma ganas, Kardiovaskular, Trauma, Kehamilan |
| **E** (Essential) | Obat penting tapi ada alternatif | E, F, G, H, J, K, M, N, Q | Endokrin, Mental, Saraf, Pernapasan, Pencernaan |
| **N** (Non-essential) | Obat pendukung/kenyamanan | L, R, Z | Kulit, Gejala tak spesifik, Status kesehatan |

### Implementasi

```python
VEN_BAB = {
    **dict.fromkeys(list('ABCDIOPS T'), 'V'),
    **dict.fromkeys(list('EFGHJKMNQ'), 'E'),
    **dict.fromkeys(list('LRZ'),       'N'),
}

kd = raw.dropna(subset=['Resep Obat','Kode Diagnosa']).copy()
kd['Kode Diagnosa'] = kd['Kode Diagnosa'].astype(str).str.strip()

# Ambil kode ICD-10 yang paling sering muncul per obat
dom = kd.groupby('Resep Obat')['Kode Diagnosa'].agg(
    lambda s: s.value_counts().index[0])

# Peta huruf pertama kode ke VEN
inv['VEN'] = dom.reindex(inv.index).str[0].map(VEN_BAB).fillna('N')
```

**Penerapan pada data:** Setiap obat dapat diresepkan untuk berbagai diagnosis. Kode ICD-10 **dominan** (paling sering) yang menentukan klasifikasi VEN. Jika tidak ada kode diagnosis yang bisa dipetakan, default ke 'N'.

---

## 10. Skor Prioritas ABC-VEN

Skor gabungan ABC-VEN menghasilkan angka prioritas yang menggabungkan nilai ekonomi dan nilai klinis.

### Rumus

$$\text{skor\_prioritas} = w_{ABC} + w_{VEN}$$

| Kelas | Bobot |
|---|---|
| A | 3 |
| B | 2 |
| C | 1 |
| V | 3 |
| E | 2 |
| N | 1 |

**Rentang nilai:** 2 (C+N, prioritas terendah) hingga 6 (A+V, prioritas tertinggi)

```python
wA = {'A': 3, 'B': 2, 'C': 1}
wV = {'V': 3, 'E': 2, 'N': 1}

master['skor_prioritas'] = (
    master['ABC'].map(wA).fillna(1) +
    master['VEN'].map(wV).fillna(1)
)
```

**Tabel Skor:**
| Skor | Kombinasi | Tindakan yang Disarankan |
|---|---|---|
| 6 | A + V | Pemantauan harian, tidak boleh stockout |
| 5 | A + E atau B + V | Pemantauan ketat, stok cadangan tinggi |
| 4 | A + N atau B + E atau C + V | Monitoring reguler |
| 3 | B + N atau C + E | Monitoring bulanan |
| 2 | C + N | Monitoring triwulanan |

**Pengurutan master table:**
```python
master = master.sort_values(
    ['kritis', 'skor_prioritas', 'perlu_pesan'],
    ascending=[False, False, False]
)
```

---

## 11. Analisis Musiman (Seasonal Index)

### 11.1 Uji ANOVA Musiman

Untuk mendeteksi apakah ada pola musiman yang signifikan secara statistik:

1. Z-score normalisasi per obat:

$$z_{i,t} = \frac{X_{i,t} - \bar{X}_i}{\sigma_i}$$

2. Uji ANOVA satu arah: apakah rata-rata z antar bulan berbeda nyata?

$$F = \frac{MSB}{MSW} \quad (\text{Between Groups / Within Groups})$$

```python
prows = []
for ob in X.index:
    v = X.loc[ob].dropna()
    if len(v) < 3 or v.std(ddof=0) == 0: continue
    z = (v - v.mean()) / v.std(ddof=0)
    for m, val in z.items():
        prows.append((m, val))

pnl = pd.DataFrame(prows, columns=['bulan','z'])
F, pval = stats.f_oneway(*[g['z'].values for _, g in pnl.groupby('bulan')])
```

### 11.2 Seasonal Index (Rasio terhadap Mean)

$$SI_t = \frac{\bar{r}_t}{1} \quad \text{di mana} \quad r_{i,t} = \frac{X_{i,t}}{\bar{X}_i}$$

- $r_{i,t}$ = rasio permintaan obat $i$ pada bulan $t$ terhadap rata-rata tahunannya
- $\bar{r}_t$ = rata-rata rasio semua obat pada bulan $t$

Nilai $SI_t > 1$ menunjukkan bulan dengan permintaan di atas rata-rata; $SI_t < 1$ menunjukkan bulan di bawah rata-rata.

---

## 12. Definisi Seluruh Variabel

### Variabel Data dan DataFrame

| Variabel | Tipe | Definisi |
|---|---|---|
| `raw` | DataFrame | Data mentah seluruh kunjungan dan resep dari Excel |
| `visit_bln` | Series | Jumlah kunjungan unik per bulan (`Register.nunique()`) |
| `obs` | DataFrame | Format panjang: obat-bulan-jumlah setelah agregasi |
| `X` | DataFrame (175×12) | Matriks pivot obat × bulan mentah; NaN jika tidak ada catatan |
| `X_clean` | DataFrame | X setelah `no_demand` → 0, `data_gap` tetap NaN |
| `X_winsor` | DataFrame | X_clean setelah winsorization IQR per obat |
| `presc` | set | Himpunan `(obat, bulan)` di mana obat ada dalam resep |
| `panel` | DataFrame | Data panel bulanan dari file parquet |
| `feat` | DataFrame | Panel + fitur tambahan (1572 baris × 17 kolom) |
| `df_pre` | DataFrame | `feat` setelah imputasi dan winsorize target |
| `df_split` | DataFrame | Subset `df_pre` untuk train/test split |
| `meta` | DataFrame | Metadata per obat: AMC, σ, ADI, CV², kategori |
| `inv` | DataFrame | Tabel kebijakan inventori: SS, Min, Max, ABC, VEN, status |
| `master` | DataFrame | Tabel master final: forecast + inventori + ABC-VEN + prioritas |

### Variabel Konfigurasi

| Variabel | Nilai | Definisi |
|---|---|---|
| `ADI_CUT` | 1.32 | Ambang batas ADI untuk klasifikasi smooth vs intermittent |
| `CV2_CUT` | 0.49 | Ambang batas CV² untuk klasifikasi smooth vs erratic |
| `LT` | 1.0 | Lead Time dalam bulan |
| `R` | 1.0 | Review Period dalam bulan |
| `Z` | 1.65 | Z-score untuk service level 95% |
| `TRAIN_M` | [1..10] | Daftar bulan training (Januari–Oktober) |
| `VAL_M` | [7,8,9,10] | Daftar bulan validasi (Juli–Oktober) |
| `TEST_M` | [11,12] | Daftar bulan test (November–Desember) |
| `GRID_SES` | [0.01..0.99] | Grid alpha untuk tuning SES (49 nilai) |
| `GRID_INT` | [0.05..0.50] | Grid alpha untuk tuning Croston/SBA (10 nilai) |
| `FEATURE_COLS` | list (10) | Nama kolom fitur ML |
| `TARGET_COL` | `demand_winsor` | Variabel target |
| `SCALE_COLS` | list (9) | Kolom yang dinormalisasi dengan StandardScaler |

### Variabel Statistik Per Obat

| Variabel | Definisi |
|---|---|
| `AMC_ser` | Average Monthly Consumption per obat (dari `X_clean`) |
| `SIG_ser` | Standar deviasi permintaan bulanan per obat (populasi, ddof=0) |
| `TOT_ser` | Total konsumsi tahunan per obat (dari `X_clean`) |
| `ADI_map` | Average Demand Interval per obat |
| `CV2_map` | Coefficient of Variation kuadrat per obat |
| `kat` | Kategori permintaan per obat (smooth/erratic/intermittent/lumpy) |
| `amc_map` | Alias `AMC_ser` (digunakan saat merge ke `feat`) |
| `sig_map` | Alias `SIG_ser` (digunakan saat merge ke `feat`) |

### Variabel Model

| Variabel | Definisi |
|---|---|
| `scope_ses` | Daftar obat yang menggunakan SES (smooth/erratic) |
| `SCOPE_INT` | Daftar obat yang menggunakan Croston/SBA (intermittent/lumpy) |
| `ses_df` | Hasil fitting SES: alpha optimal, WAPE, MAD, MAPE per obat |
| `cro` | Hasil fitting Croston: alpha_z, alpha_p, WAPE, MAD per obat |
| `sba` | Hasil fitting SBA: alpha_z, alpha_p, WAPE, MAD per obat |
| `final` | Tabel seleksi model terbaik per obat dengan parameter dan WAPE |
| `eval_df` | Evaluasi komprehensif dengan E2: WAPE, MAPE, MAD, MSE, mutu |

### Variabel E2 Scaler

| Variabel | Definisi |
|---|---|
| `visit_train_mean` | Rata-rata kunjungan unik per bulan pada bulan 1–10 (training) |
| `visit_scale` | Dict `{bulan: visits(bulan) / visit_train_mean}` — faktor skala E2 |
| `df_e2` | Tabel analisis dampak E2: WAPE_base, WAPE_E2, delta per obat |

### Variabel Inventori

| Variabel | Definisi |
|---|---|
| `inv['safety_stock']` | Stok pengaman = $\lceil Z \cdot \sigma \cdot \sqrt{LT} \rceil$ |
| `inv['Min']` | Titik pemesanan ulang = $\lceil AMC \cdot LT + Z \cdot \sigma \cdot \sqrt{LT} \rceil$ |
| `inv['Max']` | Stok maksimum = $\lceil AMC \cdot (LT+R) + Z \cdot \sigma \cdot \sqrt{LT+R} \rceil$ |
| `inv['stok_terkini']` | Sisa stok terakhir yang tercatat (dari kolom `SISA_STOK`) |
| `inv['status']` | `aman` / `DI BAWAH MIN` / `tidak diketahui` |
| `inv['perlu_pesan']` | Jumlah unit yang perlu dipesan = $\max(0, Max - \text{stok\_terkini})$ |
| `inv['ABC']` | Kategori ABC berdasarkan analisis Pareto konsumsi |
| `inv['VEN']` | Kategori VEN berdasarkan ICD-10 diagnosis dominan |

### Variabel Prioritas

| Variabel | Definisi |
|---|---|
| `wA` | Bobot ABC: `{'A':3, 'B':2, 'C':1}` |
| `wV` | Bobot VEN: `{'V':3, 'E':2, 'N':1}` |
| `master['skor_prioritas']` | Skor prioritas = `wA[ABC] + wV[VEN]` (rentang 2–6) |
| `master['kritis']` | Flag boolean: `True` jika status `DI BAWAH MIN` dan `perlu_pesan > 0` |
| `VEN_BAB` | Kamus pemetaan huruf ICD-10 → kategori VEN |
| `dom` | Kode ICD-10 dominan per obat (frekuensi tertinggi) |

### Variabel Output dan Path File

| Variabel | Definisi |
|---|---|
| `RAW_EXCEL` | Path: `../Rekap_2025_Gabungan_Terisi.xlsx` |
| `PANEL_PARQUET` | Path: `../data/panel_bulanan.parquet` |
| `OUT_DIR` | Path output: `../output_analisis` |
| `PKL_OUT` | Path model: `../models/restock_pipeline_model_final.pkl` |
| `model_v2` | Objek `RestockModelV2` yang di-serialize ke `.pkl` |
| `per_obat` | Dict artifact model per obat (metode, parameter, forecast) |
| `params` | Metadata global model: versi, parameter inventori, info visit scaler |
| `scaler` | Objek `StandardScaler` yang di-fit pada training data |

---

*Dokumen ini dihasilkan dari `notebooks/Forecasting_Pipeline_Lengkap.ipynb` — Pipeline Forecasting Restock Obat Klinik Del, Institut Teknologi Del.*
