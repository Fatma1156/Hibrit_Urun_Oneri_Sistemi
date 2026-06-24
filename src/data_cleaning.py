import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data():

    df = pd.read_excel("data/raw/Online Retail.xlsx")

    # Eksik müşteri kaldır
    df = df.dropna(subset=["CustomerID"])

    # İptaller
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Negatif değerler
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    # Tip dönüşümü
    df["CustomerID"] = df["CustomerID"].astype(int)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # Duplicate sil
    df = df.drop_duplicates()

    # Yeni özellik
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # ── Train / Test Split (müşteri bazlı %80 / %20) ──────────────
    unique_customers = df["CustomerID"].unique()

    train_customers, test_customers = train_test_split(
        unique_customers,
        test_size=0.2,
        random_state=42
    )

    train_df = df[df["CustomerID"].isin(train_customers)]
    test_df  = df[df["CustomerID"].isin(test_customers)]

    # Kaydet
    df.to_csv("data/processed/online_retail_clean.csv",       index=False)
    train_df.to_csv("data/processed/online_retail_train.csv", index=False)
    test_df.to_csv("data/processed/online_retail_test.csv",   index=False)

    print(f"  Toplam müşteri  : {len(unique_customers):,}")
    print(f"  Train müşteri   : {len(train_customers):,}  "
          f"({len(train_df):,} işlem)")
    print(f"  Test müşteri    : {len(test_customers):,}  "
          f"({len(test_df):,} işlem)")
    print("  Veri temizleme tamamlandı.")
