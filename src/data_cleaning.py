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
    df["Description"] = df["Description"].astype(str).str.strip().str.upper()
    _log_step(
        log_rows,
        "Description standardizasyonu",
        before,
        len(df),
        "Description için yalnızca strip ve upper standardizasyonu yapıldı; ek filtre uygulanmadı.",
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

    pd.DataFrame(log_rows).to_csv(f"{REPORTS_DIR}/cleaning_log.csv", index=False)
    df.to_csv(f"{PROCESSED_DIR}/online_retail_clean.csv", index=False)

    print("\n  ── Final Temizlik Özeti ──")
    print(f"  Final satır sayısı   : {len(df):,}")
    print(f"  Final müşteri sayısı : {df['CustomerID'].nunique():,}")
    print(f"  Final ürün sayısı    : {df['Description'].nunique():,}")
    print(f"  Final fatura sayısı  : {df['InvoiceNo'].nunique():,}")
    print("  Temizlik raporu kaydedildi → outputs/reports/cleaning_log.csv")
    print("  Temiz veri kaydedildi → data/processed/online_retail_clean.csv")
    print("  Veri temizleme tamamlandı.")

    return df
