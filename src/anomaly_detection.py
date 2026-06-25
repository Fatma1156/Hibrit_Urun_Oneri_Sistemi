import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# RFM skorları model girdisi değil; yalnızca seçilmiş model feature'ları kullanılır.
FALLBACK_MODEL_FEATURES = [
    "Recency", "Frequency", "Monetary",
    "total_items", "avg_basket_size",
    "avg_unit_price", "unique_products", "unique_days",
    "avg_basket_value", "purchase_span_days", "avg_days_between_orders",
]


def _load_selected_features(features: pd.DataFrame) -> list[str]:
    """Isolation Forest için selected_features.txt içindeki feature listesini okur."""
    try:
        with open("outputs/reports/selected_features.txt", "r", encoding="utf-8") as file:
            selected = [line.strip() for line in file if line.strip()]
    except FileNotFoundError:
        selected = FALLBACK_MODEL_FEATURES

    cols = [col for col in selected if col in features.columns]
    if not cols:
        raise ValueError("Anomali tespiti için kullanılabilecek seçilmiş feature bulunamadı.")
    return cols


def run_anomaly_detection(features=None):
    """
    Isolation Forest ile anormal müşterileri tespit eder.
    Model girdisi olarak yalnızca selected_features.txt içindeki feature'lar kullanılır;
    R_score, F_score, M_score ve RFM_total modele dahil edilmez.
    """

    if features is None:
        features = pd.read_csv("data/processed/customer_features.csv",
                               index_col="CustomerID")

    model_cols = _load_selected_features(features)
    X = features[model_cols].fillna(0)

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
    print(f"  Seçilmiş model feature sayısı : {len(model_cols)}")
    print(f"  Kullanılan feature'lar        : {', '.join(model_cols)}")
    print(f"  Tespit edilen anomali         : {n_anomaly} / {n_total} "
          f"({n_anomaly / n_total:.1%})")

    # Anomalileri çıkar, orijinal sütun yapısını koru (anomaly kolonu hariç)
    features_clean = (
        features[features["anomaly"] == 1]
        .drop(columns=["anomaly"])
    )

    print(f"  Temiz müşteri sayısı          : {len(features_clean)}")
    print("  Anomali analizi tamamlandı.")

    return features_clean
