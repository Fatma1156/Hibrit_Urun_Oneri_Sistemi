import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPORTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/figures"

MODEL_FEATURE_CANDIDATES = [
    "Recency",
    "Frequency",
    "Monetary",
    "total_items",
    "avg_basket_size",
    "avg_unit_price",
    "unique_products",
    "unique_days",
    "avg_basket_value",
    "purchase_span_days",
    "avg_days_between_orders",
]

FEATURE_PRIORITY = [
    "Recency",
    "Frequency",
    "Monetary",
    "unique_products",
    "total_items",
    "avg_basket_value",
    "avg_basket_size",
    "avg_unit_price",
    "unique_days",
    "purchase_span_days",
    "avg_days_between_orders",
]


def _ensure_output_dirs():
    """Rapor ve grafik klasörlerini yoksa oluşturur."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _safe_qcut(series: pd.Series, labels):
    """Tekrarlı değerlerde qcut hata vermesin diye güvenli skor üretir."""
    try:
        return pd.qcut(series, q=5, labels=labels, duplicates="drop").astype(int)
    except ValueError:
        return pd.Series(3, index=series.index, dtype=int)


def _build_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Verilen DataFrame için RFM + gelişmiş davranışsal özellikler üretir."""

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
    )

    extra = df.groupby("CustomerID").agg(
        total_items=("Quantity", "sum"),
        avg_basket_size=("Quantity", "mean"),
        avg_unit_price=("UnitPrice", "mean"),
        unique_products=("Description", "nunique"),
        unique_days=("InvoiceDate", lambda x: x.dt.date.nunique()),
        first_purchase_date=("InvoiceDate", "min"),
        last_purchase_date=("InvoiceDate", "max"),
    )

    features = rfm.join(extra)

    # Davranışsal özellikler: sıfıra bölme hatasına karşı güvenli hesaplama yapılır.
    safe_frequency = features["Frequency"].replace(0, np.nan)
    features["avg_basket_value"] = (features["Monetary"] / safe_frequency).fillna(0)
    features["purchase_span_days"] = (
        features["last_purchase_date"] - features["first_purchase_date"]
    ).dt.days.clip(lower=0)
    features["avg_days_between_orders"] = (
        features["purchase_span_days"] / safe_frequency
    ).replace([np.inf, -np.inf], np.nan).fillna(0)
    features = features.drop(columns=["first_purchase_date", "last_purchase_date"])

    # RFM skorları raporlama/yorumlama için korunur; model feature listesine dahil edilmez.
    features["R_score"] = _safe_qcut(features["Recency"], labels=[5, 4, 3, 2, 1])
    features["F_score"] = _safe_qcut(features["Frequency"].rank(method="first"), labels=[1, 2, 3, 4, 5])
    features["M_score"] = _safe_qcut(features["Monetary"].rank(method="first"), labels=[1, 2, 3, 4, 5])

    features["RFM_score"] = (
        features["R_score"].astype(str) +
        features["F_score"].astype(str) +
        features["M_score"].astype(str)
    )

    features["RFM_total"] = features["R_score"] + features["F_score"] + features["M_score"]

    def rfm_segment(score):
        if score >= 13:
            return "Champions"
        if score >= 10:
            return "Loyal"
        if score >= 7:
            return "Potential"
        if score >= 4:
            return "At Risk"
        return "Lost"

    features["rfm_label"] = features["RFM_total"].apply(rfm_segment)

    return features


def _select_features_by_correlation(features: pd.DataFrame, correlation_threshold: float = 0.80) -> list[str]:
    """Model aday feature listesini korelasyon önceliğine göre daraltır."""
    candidate_features = [c for c in MODEL_FEATURE_CANDIDATES if c in features.columns]
    corr = features[candidate_features].corr(numeric_only=True)
    corr.to_csv(f"{REPORTS_DIR}/feature_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Model Feature Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/feature_correlation_matrix.png")
    plt.close()

    priority_rank = {feature: idx for idx, feature in enumerate(FEATURE_PRIORITY)}
    selected = candidate_features.copy()
    dropped_rows = []

    for i, feature_a in enumerate(candidate_features):
        for feature_b in candidate_features[i + 1:]:
            if feature_a not in selected or feature_b not in selected:
                continue
            correlation = corr.loc[feature_a, feature_b]
            if pd.isna(correlation) or abs(correlation) < correlation_threshold:
                continue

            if priority_rank.get(feature_a, 999) <= priority_rank.get(feature_b, 999):
                kept_feature, dropped_feature = feature_a, feature_b
            else:
                kept_feature, dropped_feature = feature_b, feature_a

            if dropped_feature in selected:
                selected.remove(dropped_feature)
                dropped_rows.append({
                    "dropped_feature": dropped_feature,
                    "kept_feature": kept_feature,
                    "correlation": correlation,
                    "reason": f"|korelasyon| >= {correlation_threshold:.2f}; öncelik sıralamasında {kept_feature} korundu.",
                })

    pd.DataFrame(
        dropped_rows,
        columns=["dropped_feature", "kept_feature", "correlation", "reason"],
    ).to_csv(f"{REPORTS_DIR}/dropped_correlated_features.csv", index=False)

    with open(f"{REPORTS_DIR}/selected_features.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(selected))
        file.write("\n")

    print(f"  Korelasyon eşiği       : {correlation_threshold:.2f}")
    print(f"  Aday model feature     : {len(candidate_features)}")
    print(f"  Seçilen model feature  : {len(selected)}")
    if dropped_rows:
        print("  Korelasyon nedeniyle çıkarılan feature'lar:")
        for row in dropped_rows:
            print(f"    - {row['dropped_feature']} (korunan: {row['kept_feature']}, corr={row['correlation']:.3f})")
    else:
        print("  Korelasyon nedeniyle çıkarılan feature yok.")

    return selected


def create_customer_features():
    """
    Train verisi üzerinden özellik üretir; test müşterileri için ayrı özellik seti oluşturur.
    Feature engineering sonrası korelasyon tabanlı model feature seçimini raporlar.
    """
    _ensure_output_dirs()

    train_df = pd.read_csv("data/processed/online_retail_train.csv", parse_dates=["InvoiceDate"])
    test_df = pd.read_csv("data/processed/online_retail_test.csv", parse_dates=["InvoiceDate"])
    full_df = pd.read_csv("data/processed/online_retail_clean.csv", parse_dates=["InvoiceDate"])

    snapshot = full_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    train_features = _build_features(train_df, snapshot)
    test_features = _build_features(test_df, snapshot)

    selected_features = _select_features_by_correlation(train_features, correlation_threshold=0.80)

    train_features.to_csv("data/processed/customer_features.csv")
    train_features.to_csv("data/processed/customer_features_train.csv")
    test_features.to_csv("data/processed/customer_features_test.csv")

    print(f"  Train müşteri   : {len(train_features):,}  ({train_features.shape[1]} özellik)")
    print(f"  Test müşteri    : {len(test_features):,}")
    print(f"  Modelde kullanılacak feature listesi: {', '.join(selected_features)}")
    print(f"  RFM Segment dağılımı (train):\n{train_features['rfm_label'].value_counts().to_string()}")
    print("  Özellik mühendisliği tamamlandı.")

    return train_features
