import os

import pandas as pd


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


def clean_data(**_ignored_options):
    """
    Ham Online Retail verisini yalnızca kesin veri kalitesi problemleri için temizler.
    Aykırı değer temizliği, feature seçimi ve train/test ayrımı bu aşamada yapılmaz.
    """
    _ensure_output_dirs()
    log_rows = []

    print("\n  Veri temizleme ham veri dosyasını okuyor...")
    df = pd.read_excel(RAW_DATA_PATH)

    print("\n  ── Başlangıç Veri Özeti ──")
    print(f"  Başlangıç satır sayısı   : {len(df):,}")
    print(f"  Başlangıç müşteri sayısı : {df['CustomerID'].nunique():,}")
    print(f"  Başlangıç ürün sayısı    : {df['Description'].nunique():,}")
    print(f"  Başlangıç fatura sayısı  : {df['InvoiceNo'].nunique():,}")

    before = len(df)
    df = df.dropna(subset=["CustomerID"]).copy()
    _log_step(
        log_rows,
        "CustomerID eksik kayıtları kaldır",
        before,
        len(df),
        "CustomerID olmayan işlemler müşteri bazlı analiz için kullanılamaz.",
    )

    before = len(df)
    df = df.dropna(subset=["Description"]).copy()
    _log_step(
        log_rows,
        "Description eksik kayıtları kaldır",
        before,
        len(df),
        "Ürün açıklaması olmayan kayıtlar ürün bazlı analiz için kullanılamaz.",
    )

    before = len(df)
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")].copy()
    _log_step(
        log_rows,
        "İade/iptal faturalarını kaldır",
        before,
        len(df),
        "InvoiceNo değeri C ile başlayan kayıtlar iptal/iade işlemidir.",
    )

    before = len(df)
    df = df[df["Quantity"] > 0].copy()
    _log_step(
        log_rows,
        "Quantity <= 0 kayıtları kaldır",
        before,
        len(df),
        "Pozitif olmayan miktarlar gerçek satış işlemi değildir.",
    )

    before = len(df)
    df = df[df["UnitPrice"] > 0].copy()
    _log_step(
        log_rows,
        "UnitPrice <= 0 kayıtları kaldır",
        before,
        len(df),
        "Pozitif olmayan fiyatlar ciro hesaplaması için geçerli değildir.",
    )

    before = len(df)
    df = df.drop_duplicates().copy()
    _log_step(
        log_rows,
        "Duplicate kayıtları kaldır",
        before,
        len(df),
        "Tamamen aynı tekrar kayıtlar tekilleştirildi.",
    )

    before = len(df)
    df["Description"] = (
        df["Description"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\s+", " ", regex=True)
    )
    _log_step(
        log_rows,
        "Description standardizasyonu",
        before,
        len(df),
        "Description strip, upper ve çoklu boşlukları tek boşluğa indirme ile standartlaştırıldı.",
    )

    before = len(df)
    df = df[df["Description"] != ""].copy()
    _log_step(
        log_rows,
        "Boş Description kayıtlarını kaldır",
        before,
        len(df),
        "strip sonrasında boş kalan ürün açıklamaları kaldırıldı.",
    )

    before = len(df)
    df = df[~df["Description"].str.match(r"^\d+$", na=False)].copy()
    _log_step(
        log_rows,
        "Sayısal Description kayıtlarını kaldır",
        before,
        len(df),
        "Sadece rakamlardan oluşan açıklamalar gerçek ürün adı olmadığı için kaldırıldı.",
    )

    special_stockcodes = {"POST", "DOT", "C2", "M", "BANK CHARGES", "CRUK", "PADS"}
    stockcode_clean = df["StockCode"].astype(str).str.strip().str.upper()
    before = len(df)
    df = df[~stockcode_clean.isin(special_stockcodes)].copy()
    _log_step(
        log_rows,
        "Özel StockCode kayıtlarını kaldır",
        before,
        len(df),
        "Lojistik, manuel işlem ve muhasebe StockCode kayıtları kaldırıldı; D koduna dokunulmadı.",
    )

    non_product_descriptions = {
        "POSTAGE",
        "DOTCOM POSTAGE",
        "CARRIAGE",
        "MANUAL",
        "BANK CHARGES",
        "CRUK COMMISSION",
        "PADS TO MATCH ALL CUSHIONS",
        "EBAY",
    }
    before = len(df)
    df = df[~df["Description"].isin(non_product_descriptions)].copy()
    _log_step(
        log_rows,
        "Ürün olmayan Description kayıtlarını kaldır",
        before,
        len(df),
        "Öneri sistemi ürünü olmayan açıklamalar büyük-küçük harf duyarsız şekilde kaldırıldı.",
    )

    non_product_keywords = ["EBAY", "UNSALEABLE", "DESTROYED"]
    keyword_pattern = "|".join(non_product_keywords)
    before = len(df)
    df = df[~df["Description"].str.contains(keyword_pattern, na=False)].copy()
    _log_step(
        log_rows,
        "Description anahtar kelime filtresi",
        before,
        len(df),
        "EBAY, UNSALEABLE veya DESTROYED içeren açıklamalar kaldırıldı.",
    )

    before = len(df)
    df["CustomerID"] = df["CustomerID"].astype(int)
    _log_step(
        log_rows,
        "CustomerID int dönüşümü",
        before,
        len(df),
        "CustomerID tam sayıya dönüştürüldü; satır silinmedi.",
    )

    before = len(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    _log_step(
        log_rows,
        "InvoiceDate datetime dönüşümü",
        before,
        len(df),
        "InvoiceDate datetime tipine dönüştürüldü; satır silinmedi.",
    )

    before = len(df)
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    _log_step(
        log_rows,
        "TotalPrice oluştur",
        before,
        len(df),
        "TotalPrice = Quantity * UnitPrice; satır silinmedi.",
    )

    # Zaman bazlı split: cutoff tarihi veri içindeki %80 satır konumundan otomatik belirlenir.
    df = df.sort_values("InvoiceDate").reset_index(drop=True)
    split_index = min(int(len(df) * 0.80), len(df) - 1)
    cutoff_date = df.iloc[split_index]["InvoiceDate"]

    train_df = df[df["InvoiceDate"] <= cutoff_date].copy()
    test_df = df[df["InvoiceDate"] > cutoff_date].copy()
    train_ratio = len(train_df) / len(df) if len(df) else 0
    test_ratio = len(test_df) / len(df) if len(df) else 0

    pd.DataFrame(log_rows).to_csv(f"{REPORTS_DIR}/cleaning_log.csv", index=False)
    df.to_csv(f"{PROCESSED_DIR}/online_retail_clean.csv", index=False)
    train_df.to_csv(f"{PROCESSED_DIR}/online_retail_train.csv", index=False)
    test_df.to_csv(f"{PROCESSED_DIR}/online_retail_test.csv", index=False)

    print("\n  ── Zaman Bazlı Train/Test Ayrımı ──")
    print(f"  Cutoff tarihi            : {cutoff_date.date()}")
    print(f"  Train satır sayısı       : {len(train_df):,}")
    print(f"  Test satır sayısı        : {len(test_df):,}")
    print(f"  Train oranı              : {train_ratio:.2%}")
    print(f"  Test oranı               : {test_ratio:.2%}")
    print(f"  Train tarih aralığı      : {train_df['InvoiceDate'].min()} → {train_df['InvoiceDate'].max()}")
    print(f"  Test tarih aralığı       : {test_df['InvoiceDate'].min()} → {test_df['InvoiceDate'].max()}")

    print("\n  ── Final Temizlik Özeti ──")
    print(f"  Final satır sayısı   : {len(df):,}")
    print(f"  Final müşteri sayısı : {df['CustomerID'].nunique():,}")
    print(f"  Final ürün sayısı    : {df['Description'].nunique():,}")
    print(f"  Final fatura sayısı  : {df['InvoiceNo'].nunique():,}")
    print("  Temizlik raporu kaydedildi → outputs/reports/cleaning_log.csv")
    print("  Temiz veri kaydedildi → data/processed/online_retail_clean.csv")
    print("  Zaman bazlı train/test dosyaları kaydedildi → data/processed/")
    print("  Veri temizleme tamamlandı.")

    return df
