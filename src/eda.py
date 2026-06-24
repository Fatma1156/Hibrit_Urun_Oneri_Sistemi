import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os


# ──────────────────────────────────────────
# DESCRIPTION TEMİZLİĞİ
# ──────────────────────────────────────────

def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Description sütununu temizler:
    - Boş / NaN description satırlarını kaldırır
    - Büyük harfe çevirir, baştaki/sondaki boşlukları siler
    - Sayısal veya anlamsız kısa açıklamaları filtreler
    """
    df = df.dropna(subset=["Description"])
    df["Description"] = df["Description"].str.strip().str.upper()

    # Sadece rakam veya 2 karakterden kısa açıklamaları çıkar
    df = df[df["Description"].str.len() > 2]
    df = df[~df["Description"].str.match(r"^\d+$")]

    return df


# ──────────────────────────────────────────
# AYKIRI DEĞER ANALİZİ
# ──────────────────────────────────────────

def detect_outliers_iqr(df: pd.DataFrame,
                         col: str,
                         factor: float = 3.0) -> pd.Series:
    """
    IQR yöntemiyle aykırı değerleri tespit eder.
    factor=3.0 → çok uç aykırılar (perakende verisi için önerilir)
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - factor * IQR
    upper = Q3 + factor * IQR
    return (df[col] < lower) | (df[col] > upper)


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantity ve UnitPrice için aykırı değerleri kaldırır.
    """
    before = len(df)

    mask_qty   = detect_outliers_iqr(df, "Quantity")
    mask_price = detect_outliers_iqr(df, "UnitPrice")
    outlier_mask = mask_qty | mask_price

    df_clean = df[~outlier_mask].copy()
    after = len(df_clean)

    print(f"  Aykırı değer: {before - after} satır kaldırıldı "
          f"({(before - after) / before:.1%})")

    # Aykırı değerleri raporla
    outlier_df = df[outlier_mask].copy()
    outlier_df.to_csv("outputs/reports/outliers.csv", index=False)

    return df_clean


# ──────────────────────────────────────────
# VERİ KEŞFİ (EDA)
# ──────────────────────────────────────────

def run_eda():
    """
    Temizlenmiş veri üzerinde keşifsel veri analizi yapar.
    Temel istatistikler ve görselleştirmeler üretir.
    """
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/reports", exist_ok=True)

    df = pd.read_csv("data/processed/online_retail_clean.csv",
                     parse_dates=["InvoiceDate"])

    print("\n  ── Temel İstatistikler ──")
    print(f"  Toplam satır       : {len(df):,}")
    print(f"  Benzersiz müşteri  : {df['CustomerID'].nunique():,}")
    print(f"  Benzersiz ürün     : {df['Description'].nunique():,}")
    print(f"  Benzersiz fatura   : {df['InvoiceNo'].nunique():,}")
    print(f"  Ülke sayısı        : {df['Country'].nunique()}")
    print(f"  Tarih aralığı      : {df['InvoiceDate'].min().date()} "
          f"→ {df['InvoiceDate'].max().date()}")
    print(f"  Toplam ciro (£)    : {df['TotalPrice'].sum():,.2f}")

    # Özet istatistikler
    summary = df[["Quantity", "UnitPrice", "TotalPrice"]].describe()
    summary.to_csv("outputs/reports/eda_summary.csv")

    # ── Görsel 1: Aylık Ciro ──
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["TotalPrice"].sum()

    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Aylık Toplam Ciro (£)")
    ax.set_xlabel("Ay")
    ax.set_ylabel("Ciro (£)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/figures/eda_monthly_revenue.png")
    plt.close()

    # ── Görsel 2: En Çok Satan 10 Ürün ──
    top_products = (
        df.groupby("Description")["Quantity"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    top_products.plot(kind="barh", ax=ax, color="coral")
    ax.set_title("En Çok Satan 10 Ürün")
    ax.set_xlabel("Toplam Satış Adedi")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("outputs/figures/eda_top_products.png")
    plt.close()

    # ── Görsel 3: Ülke Bazlı Ciro (UK hariç) ──
    country_rev = (
        df[df["Country"] != "United Kingdom"]
        .groupby("Country")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    country_rev.plot(kind="bar", ax=ax, color="mediumseagreen")
    ax.set_title("Ülke Bazlı Ciro — UK Hariç (İlk 10)")
    ax.set_ylabel("Ciro (£)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("outputs/figures/eda_country_revenue.png")
    plt.close()

    # ── Görsel 4: Aykırı Değer Boxplot ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    df["Quantity"].plot(kind="box", ax=axes[0], title="Quantity Dağılımı")
    df["UnitPrice"].plot(kind="box", ax=axes[1], title="UnitPrice Dağılımı")
    plt.tight_layout()
    plt.savefig("outputs/figures/eda_outlier_boxplot.png")
    plt.close()

    print("  EDA grafikleri kaydedildi → outputs/figures/")
    print("  EDA tamamlandı.")

    return df


def run_cleaning_with_eda():
    """
    data_cleaning sonrasında Description temizliği +
    aykırı değer kaldırma uygular, temizlenmiş veriyi günceller.
    """
    df = pd.read_csv("data/processed/online_retail_clean.csv",
                     parse_dates=["InvoiceDate"])

    print(f"  EDA öncesi satır: {len(df):,}")

    df = clean_descriptions(df)
    df = remove_outliers(df)

    print(f"  EDA sonrası satır: {len(df):,}")

    df.to_csv("data/processed/online_retail_clean.csv", index=False)
    print("  Temizlenmiş veri güncellendi.")

    return df
