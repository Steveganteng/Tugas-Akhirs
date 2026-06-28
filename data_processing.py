

"""Tahap 3 — Data processing & pembentukan panel bulanan obat x bulan.

Prinsip kunci (lihat 01_audit_data.md):
  - JUMLAH & SISA_STOK adalah AGREGAT BULANAN yang di-broadcast ke tiap baris
    resep. Maka panel diambil dengan first() non-null per (obat, bulan),
    BUKAN sum().
  - Nama obat dinormalisasi strip().upper().
  - Reindex ke 12 bulan (Jan-Des 2025); November hilang -> diisi.
"""
import pandas as pd
import numpy as np
import config as C


def load_raw() -> pd.DataFrame:
    if not C.DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {C.DATASET_PATH}")
    df = pd.read_excel(C.DATASET_PATH, sheet_name=C.SHEET_NAME)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Pembersihan dasar: tanggal, normalisasi nama obat, buang baris tak terpakai."""
    df = df.copy()
    df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE], errors="coerce")
    df = df.dropna(subset=[C.COL_DATE])

    # Normalisasi nama obat: strip + upper (gabung kapitalisasi/spasi ganda)
    df[C.P_OBAT] = df[C.COL_OBAT].astype("string").str.strip().str.upper()
    df = df.dropna(subset=[C.P_OBAT])
    df = df[df[C.P_OBAT] != ""]

    # SATUAN strip
    if C.COL_UNIT in df.columns:
        df[C.COL_UNIT] = df[C.COL_UNIT].astype("string").str.strip()

    # Numeric
    for col in [C.COL_QTY, C.COL_STOCK]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Buang duplikat penuh
    df = df.drop_duplicates()

    df[C.P_PERIOD] = df[C.COL_DATE].dt.to_period("M")
    return df


def build_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Bentuk panel obat x bulan dengan first() non-null (anti double-count).

    Mengembalikan long DataFrame: [obat, periode, demand, stok, is_observed].
    """
    # SATU nilai demand & stok per (obat, bulan) = first non-null
    grp = df.groupby([C.P_OBAT, C.P_PERIOD])
    demand = grp[C.COL_QTY].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    stock = grp[C.COL_STOCK].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
    unit = grp[C.COL_UNIT].apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None)

    panel = pd.DataFrame({
        C.P_DEMAND: demand, C.P_STOCK: stock, "satuan": unit
    }).reset_index()

    # Hanya pertahankan obat yang punya >=1 demand valid (target forecasting)
    obat_valid = panel.dropna(subset=[C.P_DEMAND])[C.P_OBAT].unique()
    panel = panel[panel[C.P_OBAT].isin(obat_valid)].copy()

    # Reindex tiap obat ke rentang penuh 12 bulan
    full_idx = pd.period_range(C.PERIOD_START, C.PERIOD_END, freq="M")
    out = []
    for obat, g in panel.groupby(C.P_OBAT):
        g = g.set_index(C.P_PERIOD).reindex(full_idx)
        g[C.P_OBAT] = obat
        g.index.name = C.P_PERIOD
        # tandai observasi asli sebelum diisi
        g[C.P_IS_OBS] = g[C.P_DEMAND].notna()

        # --- isi demand ---
        d = g[C.P_DEMAND]
        if C.FILL_INTERNAL == "interpolate":
            d = d.interpolate(method="linear", limit_area="inside")
        else:
            d = d.fillna(0)
        d = d.fillna(0)  # leading/trailing (luar masa aktif) -> 0
        d = d.clip(lower=0).round().astype(float)
        g[C.P_DEMAND] = d

        # --- isi stok: forward lalu backward fill (stok persisten) ---
        s = g[C.P_STOCK].ffill().bfill()
        g[C.P_STOCK] = s.fillna(0).clip(lower=0)

        g["satuan"] = g["satuan"].ffill().bfill()
        out.append(g.reset_index())

    panel_full = pd.concat(out, ignore_index=True)
    panel_full[C.P_PERIOD] = panel_full[C.P_PERIOD].astype(str)
    panel_full = panel_full.sort_values([C.P_OBAT, C.P_PERIOD]).reset_index(drop=True)
    return panel_full


def main():
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    print(f"Raw rows: {len(raw)}")
    df = clean(raw)
    print(f"Setelah clean: {len(df)} baris | obat unik: {df[C.P_OBAT].nunique()}")
    panel = build_panel(df)
    n_obat = panel[C.P_OBAT].nunique()
    n_period = panel[C.P_PERIOD].nunique()
    print(f"Panel: {len(panel)} baris = {n_obat} obat x {n_period} bulan")
    print(f"Observasi asli (is_observed=True): {panel[C.P_IS_OBS].sum()} "
          f"({100*panel[C.P_IS_OBS].mean():.1f}%)")
    print("Periode:", sorted(panel[C.P_PERIOD].unique()))

    # Validasi anti double-count: total demand bulanan harus realistis (ribuan)
    chk = panel.groupby(C.P_PERIOD)[C.P_DEMAND].sum()
    print("\nTotal demand per bulan (panel):")
    print(chk.to_string())

    panel.to_parquet(C.PANEL_PATH, index=False)
    print(f"\nTersimpan -> {C.PANEL_PATH}")
    return panel


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    main()
