"""Tahap 4 — Segmentasi obat via K-Means.

Fitur per obat (dihitung dari OBSERVASI ASLI, bukan nilai isian):
  volume    = rata-rata demand bulanan
  cv        = koefisien variasi demand (std/mean) -> stabilitas
  frekuensi = jumlah bulan aktif (ada observasi demand)
  stok      = rata-rata sisa stok
K optimal dipilih via silhouette score pada k = 2..10.
Label cluster interpretatif berdasar karakter centroid (volume & cv).
"""
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import config as C


def build_features(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for obat, g in panel.groupby(C.P_OBAT):
        obs = g[g[C.P_IS_OBS]]
        d = obs[C.P_DEMAND] if len(obs) else g[C.P_DEMAND]
        mean_d = float(d.mean())
        std_d = float(d.std(ddof=0)) if len(d) > 1 else 0.0
        cv = std_d / mean_d if mean_d > 0 else 0.0
        rows.append({
            C.P_OBAT: obat,
            "volume": mean_d,
            "cv": cv,
            "frekuensi": int(g[C.P_IS_OBS].sum()),
            "stok": float(g[C.P_STOCK].mean()),
        })
    return pd.DataFrame(rows)


def transform_features(feat: pd.DataFrame) -> np.ndarray:
    """volume & stok di-log1p (sangat skewed) agar jarak tak didominasi outlier;
    cv & frekuensi dipakai apa adanya. Lalu StandardScaler."""
    X = feat[C.CLUSTER_FEATURES].copy()
    X["volume"] = np.log1p(X["volume"])
    X["stok"] = np.log1p(X["stok"])
    return X.values


def choose_k(X: np.ndarray):
    scores = {}
    for k in C.K_RANGE:
        if k >= len(X):
            continue
        km = KMeans(n_clusters=k, random_state=C.KMEANS_RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        scores[k] = silhouette_score(X, labels)
    best_k = max(scores, key=scores.get)
    return best_k, scores


def label_clusters(feat: pd.DataFrame) -> dict:
    """Label interpretatif RELATIF antar-cluster.
    - Tingkat volume: cluster diurut by volume rata-rata -> Fast/Medium/Slow-moving
      (3+ cluster) atau Fast/Slow-moving (2 cluster).
    - Stabilitas: cv di atas median antar-cluster -> 'fluktuatif', selain itu 'stabil'.
    """
    agg = feat.groupby("cluster").agg(
        volume=("volume", "mean"), cv=("cv", "mean"),
        frekuensi=("frekuensi", "mean"), stok=("stok", "mean"),
        n=("volume", "size"),
    )
    k = len(agg)
    # urut volume menaik -> tetapkan tingkat
    order = agg["volume"].sort_values().index.tolist()
    if k <= 2:
        tier_names = ["Slow-moving", "Fast-moving"]
        tiers = {order[i]: tier_names[i] for i in range(k)}
    else:
        # bagi jadi 3 tingkat: slow / medium / fast
        tiers = {}
        for rank, cl in enumerate(order):
            frac = rank / (k - 1)
            tiers[cl] = ("Slow-moving" if frac < 1/3
                         else "Medium-moving" if frac < 2/3 else "Fast-moving")
    cv_med = agg["cv"].median()
    labels = {}
    for cl, r in agg.iterrows():
        stab = "fluktuatif" if r["cv"] > cv_med else "stabil"
        labels[int(cl)] = f"{tiers[cl]} {stab}"
    return labels, agg


def main():
    panel = pd.read_parquet(C.PANEL_PATH)
    feat = build_features(panel)
    Xcols = C.CLUSTER_FEATURES
    Xraw = transform_features(feat)
    scaler = StandardScaler()
    X = scaler.fit_transform(Xraw)

    best_k, scores = choose_k(X)
    print("Silhouette per k:")
    for k, s in scores.items():
        mark = "  <== terpilih" if k == best_k else ""
        print(f"  k={k}: {s:.4f}{mark}")

    km = KMeans(n_clusters=best_k, random_state=C.KMEANS_RANDOM_STATE, n_init=10)
    feat["cluster"] = km.fit_predict(X)
    labels, agg = label_clusters(feat)
    feat["segmen"] = feat["cluster"].map(labels)

    print(f"\nK optimal = {best_k}")
    print("\nRingkasan cluster:")
    agg2 = agg.copy()
    agg2["label"] = agg2.index.map(labels)
    print(agg2.round(2).to_string())
    print("\nSebaran segmen:")
    print(feat["segmen"].value_counts().to_string())

    # Simpan artefak
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(km, C.MODELS_DIR / "kmeans.pkl")
    joblib.dump(scaler, C.MODELS_DIR / "kmeans_scaler.pkl")

    mapping = {
        row[C.P_OBAT]: {
            "cluster": int(row["cluster"]),
            "label": row["segmen"],
            "volume": round(row["volume"], 2),
            "cv": round(row["cv"], 3),
            "frekuensi": int(row["frekuensi"]),
            "stok": round(row["stok"], 2),
        }
        for _, row in feat.iterrows()
    }
    out = {
        "k": int(best_k),
        "features": Xcols,
        "cluster_label_map": {str(k): v for k, v in labels.items()},
        "silhouette": {str(k): round(v, 4) for k, v in scores.items()},
        "obat": mapping,
    }
    with open(C.MODELS_DIR / "cluster_labels.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    feat.to_csv(C.MODELS_DIR / "cluster_features.csv", index=False)
    print(f"\nArtefak tersimpan: kmeans.pkl, kmeans_scaler.pkl, cluster_labels.json")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    main()
