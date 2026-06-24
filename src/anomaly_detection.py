import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# Modele verilecek sayısal sütunlar — kategorik/metin sütunlar dahil edilmez
NUMERIC_COLS = [
    "Recency", "Frequency", "Monetary",
    "total_items", "avg_basket_size",
    "avg_unit_price", "unique_products", "unique_days",
    "R_score", "F_score", "M_score", "RFM_total",
]


def run_anomaly_detection(features=None):
    """
    Isolation Forest ile anormal müşterileri tespit eder.
    Yalnızca sayısal sütunlar modele verilir;
    'rfm_label', 'RFM_score' gibi kategorik/metin sütunlar dışlanır.

    Returns
    -------
    features_clean : DataFrame
        Anomaliler çıkarılmış, orijinal tüm sütunları koruyan DataFrame.
    """

    if features is None:
        features = pd.read_csv("data/processed/customer_features.csv",
                               index_col="CustomerID")

    # Sadece mevcut sayısal sütunları seç
    num_cols = [c for c in NUMERIC_COLS if c in features.columns]

    # Yukarıdaki listede olmayan ama sayısal olan sütunları da ekle
    extra_numeric = features.select_dtypes(include="number").columns.tolist()
    all_numeric = list(dict.fromkeys(num_cols + extra_numeric))  # sıra koru, tekrar etme

    X = features[all_numeric].fillna(0).values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # contamination=0.02 → veriyi %2 anormal varsayar
    model = IsolationForest(contamination=0.02, random_state=42)
    features = features.copy()
    features["anomaly"] = model.fit_predict(X_scaled)

    joblib.dump(model,  "outputs/models/isolation_forest.pkl")
    joblib.dump(scaler, "outputs/models/anomaly_scaler.pkl")

    # Tüm sonuçları kaydet (kategorik sütunlar dahil)
    features.to_csv("outputs/reports/anomalies.csv")

    n_anomaly = (features["anomaly"] == -1).sum()
    n_total   = len(features)
    print(f"  Sayısal özellik sayısı : {len(all_numeric)}")
    print(f"  Tespit edilen anomali  : {n_anomaly} / {n_total} "
          f"({n_anomaly / n_total:.1%})")

    # Anomalileri çıkar, orijinal sütun yapısını koru (anomaly kolonu hariç)
    features_clean = (
        features[features["anomaly"] == 1]
        .drop(columns=["anomaly"])
    )

    print(f"  Temiz müşteri sayısı   : {len(features_clean)}")
    print("  Anomali analizi tamamlandı.")

    return features_clean
