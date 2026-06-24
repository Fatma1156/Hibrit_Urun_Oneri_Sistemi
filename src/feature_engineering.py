import pandas as pd
import numpy as np


def _build_features(df: pd.DataFrame, snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """Verilen DataFrame için RFM + gelişmiş özellikler üretir."""

    rfm = df.groupby("CustomerID").agg(
        Recency   = ("InvoiceDate",  lambda x: (snapshot_date - x.max()).days),
        Frequency = ("InvoiceNo",    "nunique"),
        Monetary  = ("TotalPrice",   "sum"),
    )

    extra = df.groupby("CustomerID").agg(
        total_items        = ("Quantity",    "sum"),
        avg_basket_size    = ("Quantity",    "mean"),
        avg_unit_price     = ("UnitPrice",   "mean"),
        unique_products    = ("Description", "nunique"),
        unique_days        = ("InvoiceDate", lambda x: x.dt.date.nunique()),
    )

    features = rfm.join(extra)

    # RFM skorları (1–5)
    features["R_score"] = pd.qcut(
        features["Recency"], q=5,
        labels=[5, 4, 3, 2, 1], duplicates="drop"
    ).astype(int)

    features["F_score"] = pd.qcut(
        features["Frequency"].rank(method="first"), q=5,
        labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    features["M_score"] = pd.qcut(
        features["Monetary"].rank(method="first"), q=5,
        labels=[1, 2, 3, 4, 5], duplicates="drop"
    ).astype(int)

    features["RFM_score"] = (
        features["R_score"].astype(str) +
        features["F_score"].astype(str) +
        features["M_score"].astype(str)
    )

    features["RFM_total"] = (
        features["R_score"] + features["F_score"] + features["M_score"]
    )

    def rfm_segment(score):
        if score >= 13:  return "Champions"
        elif score >= 10: return "Loyal"
        elif score >= 7:  return "Potential"
        elif score >= 4:  return "At Risk"
        else:             return "Lost"

    features["rfm_label"] = features["RFM_total"].apply(rfm_segment)

    return features


def create_customer_features():
    """
    Train verisi üzerinden özellik üretir.
    Test müşterileri için ayrı özellik seti de oluşturur.
    """
    train_df = pd.read_csv("data/processed/online_retail_train.csv",
                           parse_dates=["InvoiceDate"])
    test_df  = pd.read_csv("data/processed/online_retail_test.csv",
                           parse_dates=["InvoiceDate"])

    # Snapshot: tüm verinin en son tarihi + 1 gün (tutarlı referans)
    full_df  = pd.read_csv("data/processed/online_retail_clean.csv",
                           parse_dates=["InvoiceDate"])
    snapshot = full_df["InvoiceDate"].max() + pd.Timedelta(days=1)

    # Train özellikleri
    train_features = _build_features(train_df, snapshot)
    train_features.to_csv("data/processed/customer_features.csv")
    train_features.to_csv("data/processed/customer_features_train.csv")

    # Test özellikleri
    test_features = _build_features(test_df, snapshot)
    test_features.to_csv("data/processed/customer_features_test.csv")

    print(f"  Train müşteri   : {len(train_features):,}  "
          f"({train_features.shape[1]} özellik)")
    print(f"  Test müşteri    : {len(test_features):,}")
    print(f"  RFM Segment dağılımı (train):\n"
          f"{train_features['rfm_label'].value_counts().to_string()}")
    print("  Özellik mühendisliği tamamlandı.")

    return train_features
