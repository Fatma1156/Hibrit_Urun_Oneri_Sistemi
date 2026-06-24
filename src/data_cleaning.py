import os

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = "data/raw/Online Retail.xlsx"
PROCESSED_DIR = "data/processed"
REPORTS_DIR = "outputs/reports"


def _ensure_output_dirs():
    """Çıktı klasörlerini yoksa oluşturur."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def _log_step(log_rows: list[dict], step: str, before: int, after: int, explanation: str):
    """Temizlik adımını hem listeye hem terminale yazar."""
    removed = before - after
    log_rows.append({
        "step": step,
        "rows_before": before,
        "rows_after": after,
        "rows_removed": removed,
        "explanation": explanation,
    })
    print(f"  {step:<45} | Silinen: {removed:>8,} | Kalan: {after:>8,}")


def _iqr_outlier_mask(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.Series:
    """IQR yöntemiyle tek bir sütundaki aykırı değer maskesini üretir."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    return (df[column] < lower) | (df[column] > upper)


def clean_data(remove_iqr_outliers: bool = True):
    """
    Ham Online Retail verisini bilimsel temizlik sırasıyla işler.
    Train/test ayrımı tüm temizlik adımları bittikten sonra müşteri bazlı yapılır.
    """
    _ensure_output_dirs()
    log_rows = []

    print("\n  Veri temizleme ham veri dosyasını okuyor...")
    df = pd.read_excel(RAW_DATA_PATH)
    print(f"  Başlangıç satır sayısı: {len(df):,}")

    before = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    _log_step(log_rows, "CustomerID eksik kayıtları kaldır", before, len(df), "CustomerID olmayan işlemler müşteri bazlı modelleme için kullanılamaz.")

    before = len(df)
    df = df.dropna(subset=["Description"]).copy()
    _log_step(log_rows, "Description eksik kayıtları kaldır", before, len(df), "Ürün açıklaması olmayan kayıtlar ürün analizi için kullanılamaz.")

    before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    _log_step(log_rows, "İade/iptal faturalarını kaldır", before, len(df), "InvoiceNo değeri C ile başlayan kayıtlar iptal/iade işlemidir.")

    before = len(df)
    df = df[df["Quantity"] > 0].copy()
    _log_step(log_rows, "Quantity <= 0 kayıtları kaldır", before, len(df), "Pozitif olmayan miktarlar satış işlemi değildir.")

    before = len(df)
    df = df[df["UnitPrice"] > 0].copy()
    _log_step(log_rows, "UnitPrice <= 0 kayıtları kaldır", before, len(df), "Pozitif olmayan fiyatlar ciro ve sepet değerini bozar.")

    before = len(df)
    df = df.drop_duplicates().copy()
    _log_step(log_rows, "Duplicate kayıtları kaldır", before, len(df), "Tamamen aynı tekrar kayıtlar tekilleştirildi.")

    before = len(df)
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    _log_step(log_rows, "Description standardizasyonu", before, len(df), "Ürün açıklamaları strip ve upper ile standartlaştırıldı; satır silinmedi.")

    invalid_description_mask = (df["Description"].str.len() < 2) | (df["Description"].str.match(r"^\d+$", na=False))
    invalid_descriptions = df[invalid_description_mask].copy()
    invalid_descriptions.to_csv(f"{REPORTS_DIR}/removed_invalid_descriptions.csv", index=False)
    before = len(df)
    df = df[~invalid_description_mask].copy()
    _log_step(log_rows, "Anlamsız Description kayıtları kaldır", before, len(df), "Sadece rakam olan veya 2 karakterden kısa açıklamalar raporlandı ve kaldırıldı.")

    before = len(df)
    df["CustomerID"] = df["CustomerID"].astype(int)
    _log_step(log_rows, "CustomerID int dönüşümü", before, len(df), "CustomerID tam sayıya dönüştürüldü; satır silinmedi.")

    before = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    _log_step(log_rows, "InvoiceDate datetime dönüşümü", before, len(df), "InvoiceDate datetime tipine dönüştürüldü; satır silinmedi.")

    before = len(df)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    _log_step(log_rows, "TotalPrice oluştur", before, len(df), "TotalPrice = Quantity * UnitPrice; satır silinmedi.")

    if remove_iqr_outliers:
        before = len(df)
        quantity_mask = _iqr_outlier_mask(df, "Quantity")
        unit_price_mask = _iqr_outlier_mask(df, "UnitPrice")
        outlier_mask = quantity_mask | unit_price_mask
        removed_outliers = df[outlier_mask].copy()
        removed_outliers.to_csv(f"{REPORTS_DIR}/removed_iqr_outliers.csv", index=False)
        df = df[~outlier_mask].copy()
        _log_step(log_rows, "IQR aykırı değerlerini kaldır", before, len(df), "Quantity ve UnitPrice için IQR yöntemiyle tespit edilen aykırı kayıtlar kaldırıldı.")
    else:
        pd.DataFrame().to_csv(f"{REPORTS_DIR}/removed_iqr_outliers.csv", index=False)
        _log_step(log_rows, "IQR aykırı değerlerini atla", len(df), len(df), "remove_iqr_outliers=False olduğu için aykırı değer silinmedi.")

    pd.DataFrame(log_rows).to_csv(f"{REPORTS_DIR}/cleaning_log.csv", index=False)

    unique_customers = df["CustomerID"].unique()
    train_customers, test_customers = train_test_split(unique_customers, test_size=0.2, random_state=42)
    train_df = df[df["CustomerID"].isin(train_customers)].copy()
    test_df = df[df["CustomerID"].isin(test_customers)].copy()

    df.to_csv(f"{PROCESSED_DIR}/online_retail_clean.csv", index=False)
    train_df.to_csv(f"{PROCESSED_DIR}/online_retail_train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/online_retail_test.csv", index=False)

    print("\n  ── Final Temizlik Özeti ──")
    print(f"  Final satır sayısı   : {len(df):,}")
    print(f"  Final müşteri sayısı : {df['CustomerID'].nunique():,}")
    print(f"  Final ürün sayısı    : {df['Description'].nunique():,}")
    print(f"  Final fatura sayısı  : {df['InvoiceNo'].nunique():,}")
    print(f"  Train müşteri sayısı : {len(train_customers):,} ({len(train_df):,} işlem)")
    print(f"  Test müşteri sayısı  : {len(test_customers):,} ({len(test_df):,} işlem)")
    print("  Temizlik raporu kaydedildi → outputs/reports/cleaning_log.csv")
    print("  Veri temizleme tamamlandı.")

    return df
