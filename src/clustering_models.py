import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"

import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


FEATURES_FOR_CLUSTERING = [
    "Recency", "Frequency", "Monetary",
    "total_items", "avg_basket_size",
    "avg_unit_price", "unique_products", "unique_days",
    "avg_basket_value", "purchase_span_days", "avg_days_between_orders",
]


def _load_selected_features(features: pd.DataFrame) -> list[str]:
    """Varsa korelasyon seçimi dosyasından model feature listesini okur."""
    selected_features_path = "outputs/reports/selected_features.txt"
    if os.path.exists(selected_features_path):
        with open(selected_features_path, "r", encoding="utf-8") as file:
            selected = [line.strip() for line in file if line.strip()]
        cols = [col for col in selected if col in features.columns]
        print(f"  Seçilmiş feature listesi okundu → {selected_features_path}")
    else:
        cols = [c for c in FEATURES_FOR_CLUSTERING if c in features.columns]
        print("  selected_features.txt bulunamadı; varsayılan feature listesi kullanılacak.")

    if not cols:
        raise ValueError("Kümeleme için kullanılabilecek sayısal feature bulunamadı.")
    return cols


def scale_data(features: pd.DataFrame):
    cols = _load_selected_features(features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features[cols])
    joblib.dump(scaler, "outputs/models/scaler.pkl")
    print(f"  Ölçekleme yapıldı ({len(cols)} özellik): {', '.join(cols)}")
    return X_scaled


# ──────────────────────────────────────────
# METRİKLER
# ──────────────────────────────────────────

def compute_metrics(X, labels: np.ndarray, name: str) -> dict:
    mask = labels != -1
    X_m  = X[mask]
    l_m  = labels[mask]
    n_clusters = len(set(l_m))

    if n_clusters < 2:
        print(f"  {name:<12} → Yeterli küme yok (n_clusters={n_clusters})")
        return {"algorithm": name, "n_clusters": n_clusters,
                "silhouette": -1, "davies_bouldin": 999,
                "calinski_harabasz": 0, "noise_points": int((labels == -1).sum())}

    sil = silhouette_score(X_m, l_m)
    db  = davies_bouldin_score(X_m, l_m)
    ch  = calinski_harabasz_score(X_m, l_m)
    noise = (labels == -1).sum()

    print(f"  {name:<12} → "
          f"Silhouette: {sil:.4f}  |  "
          f"Davies-Bouldin: {db:.4f}  |  "
          f"Calinski-Harabasz: {ch:.1f}"
          + (f"  |  Gürültü: {noise}" if noise > 0 else ""))

    return {
        "algorithm":         name,
        "n_clusters":        n_clusters,
        "silhouette":        sil,
        "davies_bouldin":    db,
        "calinski_harabasz": ch,
        "noise_points":      int(noise),
    }


# ──────────────────────────────────────────
# ELBOW YÖNTEMİ (WCSS)
# ──────────────────────────────────────────

def find_best_k_elbow(X, K=range(2, 11)) -> int:
    """
    WCSS (Within-Cluster Sum of Squares) değerlerini hesaplar,
    Elbow grafiğini kaydeder ve dirsek noktasını döndürür.
    """
    wcss = []
    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X)
        wcss.append(model.inertia_)

    # Dirsek noktası: eğim değişiminin en büyük olduğu yer
    deltas     = np.diff(wcss)
    delta2     = np.diff(deltas)
    elbow_idx  = np.argmax(np.abs(delta2)) + 1  # +1 offset
    best_k     = list(K)[elbow_idx]

    # Grafik
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K, wcss, marker="o", color="coral")
    ax.axvline(x=best_k, color="red", linestyle="--",
               label=f"Dirsek k={best_k}")
    ax.set_title("Elbow Yöntemi (WCSS)")
    ax.set_xlabel("Küme Sayısı (k)")
    ax.set_ylabel("WCSS")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/elbow.png")
    plt.close()

    print(f"  Elbow → En iyi k: {best_k}")
    return best_k, wcss


# ──────────────────────────────────────────
# K-MEANS İÇİN ÇOK KRİTERLİ K SEÇİMİ
# ──────────────────────────────────────────

def select_best_k_kmeans(X, K=range(2, 11)) -> tuple[int, pd.DataFrame]:
    """K-Means için k değerini üç metrik rank toplamına göre seçer."""
    rows = []
    for k in K:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        rows.append({
            "k": k,
            "wcss": model.inertia_,
            "silhouette": silhouette_score(X, labels),
            "davies_bouldin": davies_bouldin_score(X, labels),
            "calinski_harabasz": calinski_harabasz_score(X, labels),
        })

    df_k = pd.DataFrame(rows)
    df_k["rank_silhouette"] = df_k["silhouette"].rank(ascending=False, method="min")
    df_k["rank_davies_bouldin"] = df_k["davies_bouldin"].rank(ascending=True, method="min")
    df_k["rank_calinski_harabasz"] = df_k["calinski_harabasz"].rank(ascending=False, method="min")
    df_k["total_rank"] = (
        df_k["rank_silhouette"] +
        df_k["rank_davies_bouldin"] +
        df_k["rank_calinski_harabasz"]
    )
    df_k.to_csv("outputs/reports/kmeans_k_selection.csv", index=False)

    best_row = df_k.sort_values(
        ["total_rank", "rank_silhouette", "rank_davies_bouldin", "rank_calinski_harabasz"]
    ).iloc[0]
    best_k = int(best_row["k"])
    print(f"  K-Means çok kriterli k seçimi → En iyi k: {best_k}")
    return best_k, df_k


# ──────────────────────────────────────────
# BIC/AIC (GMM İÇİN)
# ──────────────────────────────────────────

def find_best_k_bic(X, K=range(2, 11)) -> int:
    """
    GMM için BIC ve AIC değerlerini hesaplar.
    Düşük BIC = daha iyi model.
    """
    bic_scores = []
    aic_scores = []

    for k in K:
        gmm = GaussianMixture(n_components=k, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    best_k_bic = list(K)[np.argmin(bic_scores)]
    best_k_aic = list(K)[np.argmin(aic_scores)]

    pd.DataFrame({
        "k": list(K),
        "bic": bic_scores,
        "aic": aic_scores,
    }).to_csv("outputs/reports/gmm_k_selection.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(K, bic_scores, marker="o", color="purple",  label="BIC")
    ax.plot(K, aic_scores, marker="s", color="orange",  label="AIC")
    ax.axvline(x=best_k_bic, color="purple", linestyle="--",
               label=f"BIC min k={best_k_bic}")
    ax.set_title("GMM — BIC / AIC Analizi")
    ax.set_xlabel("Bileşen Sayısı (k)")
    ax.set_ylabel("Skor (düşük = iyi)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/bic_aic.png")
    plt.close()

    print(f"  GMM BIC → En iyi k: {best_k_bic}  |  "
          f"AIC minimum k: {best_k_aic}")
    return best_k_bic, bic_scores, aic_scores


# ──────────────────────────────────────────
# KARŞILAŞTIRMA GRAFİĞİ
# ──────────────────────────────────────────

def plot_k_comparison(K, wcss, sil_scores, bic_scores, aic_scores):
    """Elbow, Silhouette, BIC/AIC grafiklerini tek panelde gösterir."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(K, wcss, marker="o", color="coral")
    axes[0].set_title("Elbow (WCSS) ↓")
    axes[0].set_xlabel("k")

    axes[1].plot(K, sil_scores, marker="o", color="steelblue")
    axes[1].set_title("Silhouette ↑")
    axes[1].set_xlabel("k")

    axes[2].plot(K, bic_scores, marker="o", color="purple", label="BIC")
    axes[2].plot(K, aic_scores, marker="s", color="orange", label="AIC")
    axes[2].set_title("BIC / AIC ↓")
    axes[2].set_xlabel("k")
    axes[2].legend()

    plt.suptitle("K Seçim Yöntemleri Karşılaştırması", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/figures/k_selection_comparison.png")
    plt.close()
    print("  K seçim karşılaştırma grafiği kaydedildi.")


def plot_metrics(metrics_list: list):
    df_m = pd.DataFrame(metrics_list)
    report_cols = [
        "algorithm", "silhouette", "davies_bouldin",
        "calinski_harabasz", "n_clusters", "noise_points", "topsis_score",
    ]
    for col in report_cols:
        if col not in df_m.columns:
            df_m[col] = 0
    df_m[report_cols].to_csv("outputs/reports/clustering_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(df_m["algorithm"], df_m["silhouette"],        color="steelblue")
    axes[0].set_title("Silhouette ↑")
    axes[1].bar(df_m["algorithm"], df_m["davies_bouldin"],    color="coral")
    axes[1].set_title("Davies-Bouldin ↓")
    axes[2].bar(df_m["algorithm"], df_m["calinski_harabasz"], color="mediumseagreen")
    axes[2].set_title("Calinski-Harabasz ↑")
    plt.suptitle("Kümeleme Algoritmaları Karşılaştırması", fontsize=13)
    plt.tight_layout()
    plt.savefig("outputs/figures/clustering_comparison.png")
    plt.close()
    print("  Algoritma karşılaştırma grafiği kaydedildi.")


# ──────────────────────────────────────────
# ALGORİTMALAR
# ──────────────────────────────────────────

def select_best_algorithm_topsis(metrics_list: list) -> tuple[dict, pd.DataFrame]:
    """Algoritmaları TOPSIS yöntemiyle sıralar ve en iyisini seçer."""
    df_m = pd.DataFrame(metrics_list).copy()
    criteria = ["silhouette", "davies_bouldin", "calinski_harabasz"]
    matrix = df_m[criteria].astype(float).to_numpy()

    # Vektör normalizasyonu ve eşit kriter ağırlıkları
    denominator = np.sqrt((matrix ** 2).sum(axis=0))
    denominator[denominator == 0] = 1
    weights = np.array([1 / 3, 1 / 3, 1 / 3])
    weighted = (matrix / denominator) * weights

    # Fayda kriterleri: silhouette, calinski_harabasz; maliyet kriteri: davies_bouldin
    ideal = np.array([weighted[:, 0].max(), weighted[:, 1].min(), weighted[:, 2].max()])
    negative_ideal = np.array([weighted[:, 0].min(), weighted[:, 1].max(), weighted[:, 2].min()])

    distance_to_ideal = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    distance_to_negative = np.sqrt(((weighted - negative_ideal) ** 2).sum(axis=1))
    denominator_score = distance_to_ideal + distance_to_negative
    denominator_score[denominator_score == 0] = 1
    df_m["topsis_score"] = distance_to_negative / denominator_score

    topsis_cols = ["algorithm", "silhouette", "davies_bouldin", "calinski_harabasz", "topsis_score"]
    df_m[topsis_cols].to_csv("outputs/reports/topsis_results.csv", index=False)

    best_row = df_m.sort_values("topsis_score", ascending=False).iloc[0]
    print("  En iyi algoritma TOPSIS yöntemi ile seçildi.")
    return best_row.to_dict(), df_m


def run_kmeans(X, k):
    model  = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X)
    joblib.dump(model, "outputs/models/kmeans.pkl")
    metrics = compute_metrics(X, labels, "K-Means")
    return labels, metrics, model


def run_dbscan(X):
    model  = DBSCAN(eps=0.8, min_samples=5)
    labels = model.fit_predict(X)
    joblib.dump(model, "outputs/models/dbscan.pkl")
    metrics = compute_metrics(X, labels, "DBSCAN")
    return labels, metrics, model


def run_gmm(X, k):
    model  = GaussianMixture(n_components=k, random_state=42)
    labels = model.fit_predict(X)
    joblib.dump(model, "outputs/models/gmm.pkl")
    metrics = compute_metrics(X, labels, "GMM")
    return labels, metrics, model


# ──────────────────────────────────────────
# ANA FONKSİYON
# ──────────────────────────────────────────

def run_clustering(features: pd.DataFrame):

    X = scale_data(features)
    K = range(2, 11)

    print("\n  K Seçim Yöntemleri çalıştırılıyor...")
    best_k_elbow, wcss = find_best_k_elbow(X, K)
    best_k_kmeans, kmeans_k_df = select_best_k_kmeans(X, K)
    best_k_gmm, bic_scores, aic_scores = find_best_k_bic(X, K)

    # Elbow grafiği yalnızca görsel destek olarak korunur;
    # K-Means k seçimi rank toplamıyla, GMM k seçimi BIC ile yapılır.
    plot_k_comparison(
        list(K),
        wcss,
        kmeans_k_df["silhouette"].tolist(),
        bic_scores,
        aic_scores,
    )

    print(f"\n  K-Means best_k={best_k_kmeans}  |  "
          f"GMM best_k={best_k_gmm}  |  "
          f"DBSCAN k kullanmaz (eps=0.8, min_samples=5)")

    print("\n  Algoritmalar çalıştırılıyor...")
    labels_km,     metrics_km,     _ = run_kmeans(X, best_k_kmeans)
    labels_dbscan, metrics_dbscan, _ = run_dbscan(X)
    labels_gmm,    metrics_gmm,    _ = run_gmm(X, best_k_gmm)

    metrics_list = [metrics_km, metrics_dbscan, metrics_gmm]

    # En iyi algoritma → TOPSIS skoru en yüksek olan algoritma
    best, topsis_df = select_best_algorithm_topsis(metrics_list)
    plot_metrics(topsis_df.to_dict("records"))
    best_name = best["algorithm"].lower().replace("-", "").replace(" ", "")

    label_map = {
        "kmeans": labels_km,
        "dbscan": labels_dbscan,
        "gmm":    labels_gmm,
    }
    best_display_map = {
        "kmeans": "KMeans",
        "dbscan": "DBSCAN",
        "gmm":    "GMM",
    }
    best_algorithm_name = best_display_map.get(best_name, "KMeans")
    best_labels = label_map.get(best_name, labels_km)

    with open("outputs/reports/best_clustering_model.txt", "w", encoding="utf-8") as file:
        file.write(f"{best_algorithm_name}\n")

    print(f"\n  ✅ En iyi algoritma: {best_algorithm_name} "
          f"(Silhouette: {best['silhouette']:.4f}  |  "
          f"DB: {best['davies_bouldin']:.4f}  |  "
          f"CH: {best['calinski_harabasz']:.1f})")
    print("  En iyi algoritma kaydedildi → outputs/reports/best_clustering_model.txt")

    # Sonuçları kaydet
    results_df = pd.DataFrame({
        "CustomerID":     features.index,
        "kmeans_label":   labels_km,
        "dbscan_label":   labels_dbscan,
        "gmm_label":      labels_gmm,
        "best_label":     best_labels,
        "best_algorithm": best_algorithm_name,
    })
    results_df.to_csv("outputs/reports/clustering_results.csv", index=False)

    segment_df = features.copy()
    segment_df["segment"] = best_labels
    segment_df.to_csv("data/processed/customer_segments.csv")

    joblib.dump(
        {"algorithm": best_name, "labels": best_labels},
        "outputs/models/best_clustering.pkl"
    )

    print("  Kümeleme tamamlandı.")
    return segment_df
