import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


RAW_DATA_PATH = "data/raw/Online Retail.xlsx"
REPORTS_DIR = "outputs/reports"
FIGURES_DIR = "outputs/figures"


def _ensure_output_dirs():
    """Rapor ve grafik klasörlerini yoksa oluşturur."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _sample_values(series: pd.Series, limit: int = 5) -> str:
    """Kolon profili için ilk benzersiz değerleri string olarak döndürür."""
    samples = []
    seen = set()
    for value in series:
        display_value = "<NA>" if pd.isna(value) else str(value)
        if display_value in seen:
            continue
        seen.add(display_value)
        samples.append(display_value)
        if len(samples) == limit:
            break
    return ", ".join(samples)


def _clean_description(value) -> str | None:
    """Sepet analizleri için geçici Description_clean değerini üretir."""
    if pd.isna(value):
        return None
    cleaned = str(value).strip().upper()
    return cleaned if cleaned else None


def _basket_products(values) -> list[str]:
    """Satır silmeden sepet içindeki benzersiz Description_clean ürünleri listeler."""
    products = set()
    for value in values:
        if value is not None:
            products.add(value)
    return sorted(products)


def _description_quality_report(df: pd.DataFrame):
    """Description kalitesini satır silmeden analiz eder ve inceleme raporlarını kaydeder."""
    description_as_text = df["Description"].map(lambda value: "" if pd.isna(value) else str(value))
    stripped_description = description_as_text.str.strip()

    numeric_only_mask = stripped_description.str.match(r"^\d+$", na=False)
    short_description_mask = stripped_description.str.len() < 2
    blank_after_strip_mask = stripped_description == ""

    description_quality = pd.DataFrame([
        {
            "total_description_count": len(df["Description"]),
            "unique_description_count": len({
                str(value) for value in df["Description"] if not pd.isna(value)
            }),
            "numeric_only_count": int(numeric_only_mask.sum()),
            "short_description_count": int(short_description_mask.sum()),
            "blank_after_strip_count": int(blank_after_strip_mask.sum()),
        }
    ])
    description_quality.to_csv(f"{REPORTS_DIR}/eda_description_quality.csv", index=False)

    df.loc[numeric_only_mask].to_csv(
        f"{REPORTS_DIR}/numeric_only_descriptions.csv", index=False
    )
    df.loc[short_description_mask].to_csv(
        f"{REPORTS_DIR}/short_descriptions.csv", index=False
    )
    df.loc[blank_after_strip_mask].to_csv(
        f"{REPORTS_DIR}/blank_descriptions_after_strip.csv", index=False
    )


def _special_stockcodes_report(df: pd.DataFrame):
    """Gerçek ürün olmayabilecek özel StockCode kayıtlarını satır silmeden raporlar."""
    known_special_codes = {
        "M",
        "POST",
        "DOT",
        "CRUK",
        "C2",
        "BANK CHARGES",
        "D",
        "AMAZONFEE",
        "S",
    }

    stockcode_clean = df["StockCode"].map(
        lambda value: "" if pd.isna(value) else str(value).strip().upper()
    )
    normal_product_mask = stockcode_clean.str.match(r"^\d{5}[A-Z]?$", na=False)
    letters_only_mask = stockcode_clean.str.match(r"^[A-Z]+$", na=False)
    known_special_mask = stockcode_clean.isin(known_special_codes)
    special_stockcode_mask = known_special_mask | letters_only_mask | ~normal_product_mask

    special_examples = df.loc[special_stockcode_mask].copy()
    special_examples.to_csv(
        f"{REPORTS_DIR}/special_stockcodes_examples.csv", index=False
    )

    report_source = special_examples.copy()
    report_source["StockCode"] = stockcode_clean.loc[special_stockcode_mask].values
    report = (
        report_source
        .groupby(["StockCode", "Description"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["count", "StockCode", "Description"], ascending=[False, True, True])
    )
    report.to_csv(f"{REPORTS_DIR}/special_stockcodes_report.csv", index=False)
    return {
        "record_count": len(special_examples),
        "stockcode_count": report["StockCode"].nunique() if not report.empty else 0,
    }


def _save_histogram(df: pd.DataFrame, column: str, path: str, title: str):
    """Sayısal değişken histogramını kaydeder."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df[column].plot(kind="hist", bins=50, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frekans")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _get_time_split_cutoff(analysis_df: pd.DataFrame) -> pd.Timestamp | None:
    """80/20 zaman bazlı cutoff tarihini rapordan veya ham veriden geçici hesaplar."""
    split_summary_path = f"{REPORTS_DIR}/time_split_summary.csv"
    if os.path.exists(split_summary_path):
        split_summary = pd.read_csv(split_summary_path)
        if "cutoff_date" in split_summary.columns and not split_summary.empty:
            cutoff = pd.to_datetime(split_summary.loc[0, "cutoff_date"], errors="coerce")
            if pd.notna(cutoff):
                return cutoff

    dated_df = analysis_df[analysis_df["InvoiceDate"].notna()].sort_values("InvoiceDate")
    if dated_df.empty:
        return None
    split_index = min(int(len(dated_df) * 0.80), len(dated_df) - 1)
    return dated_df.iloc[split_index]["InvoiceDate"]


def _save_monthly_purchase_summary(analysis_df: pd.DataFrame) -> bool:
    """Ham veriden aylık satın alma hacmi raporu ve grafiği üretir."""
    monthly_df = analysis_df[analysis_df["InvoiceDate"].notna()].copy()
    if monthly_df.empty:
        return False

    monthly_df["month"] = monthly_df["InvoiceDate"].dt.to_period("M").dt.to_timestamp()
    monthly_summary = (
        monthly_df
        .groupby("month")
        .agg(
            row_count=("InvoiceNo", "size"),
            invoice_count=("InvoiceNo", "nunique"),
            total_quantity=("Quantity", "sum"),
            total_revenue=("TotalPrice", "sum"),
        )
        .reset_index()
        .sort_values("month")
    )

    report_df = monthly_summary.copy()
    report_df["month"] = report_df["month"].dt.strftime("%Y-%m")
    report_df.to_csv(f"{REPORTS_DIR}/eda_monthly_purchase_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(monthly_summary["month"], monthly_summary["row_count"],
            marker="o", label="Satır sayısı")
    ax.plot(monthly_summary["month"], monthly_summary["invoice_count"],
            marker="s", label="Fatura sayısı")

    cutoff_date = _get_time_split_cutoff(analysis_df)
    if cutoff_date is not None:
        ax.axvline(cutoff_date, color="red", linestyle="--", label="80/20 cutoff")

    ax.set_title("Aylık Satın Alma Hacmi")
    ax.set_xlabel("Ay")
    ax.set_ylabel("İşlem/Fatura Sayısı")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/eda_monthly_purchase_volume.png")
    plt.close()
    return cutoff_date is not None


def run_eda():
    """
    Ham veri üzerinde keşifsel veri analizi yapar.
    Bu fonksiyon kalıcı veri temizliği yapmaz ve hiçbir satır silmez.
    """
    _ensure_output_dirs()

    print("\n  EDA ham veri dosyasını okuyor...")
    df = pd.read_excel(RAW_DATA_PATH)

    print("\n  ── Ham Veri Genel Bakış ──")
    print(f"  Satır sayısı      : {len(df):,}")
    print(f"  Sütun sayısı      : {df.shape[1]:,}")
    print(f"  Sütun isimleri    : {', '.join(df.columns.astype(str))}")
    print(f"  Duplicate kayıt   : {df.duplicated().sum():,}")

    dataset_overview = pd.DataFrame([
        {"metric": "row_count", "value": len(df)},
        {"metric": "column_count", "value": df.shape[1]},
        {"metric": "duplicate_count", "value": int(df.duplicated().sum())},
    ])
    dataset_overview.to_csv(f"{REPORTS_DIR}/eda_dataset_overview.csv", index=False)

    column_profile = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(dtype) for dtype in df.dtypes],
        "missing_count": df.isna().sum().values,
        "missing_rate": df.isna().mean().values,
        "unique_count": [len(pd.unique(df[col])) for col in df.columns],
        "sample_values": [_sample_values(df[col]) for col in df.columns],
    })
    column_profile.to_csv(f"{REPORTS_DIR}/eda_column_profile.csv", index=False)

    _description_quality_report(df)
    special_stockcode_summary = _special_stockcodes_report(df)

    # TotalPrice yalnızca EDA içindeki sayısal özet ve korelasyon için geçici analiz sütunudur.
    analysis_df = df.copy()
    analysis_df["InvoiceDate"] = pd.to_datetime(analysis_df["InvoiceDate"], errors="coerce")
    analysis_df["TotalPrice"] = analysis_df["Quantity"] * analysis_df["UnitPrice"]

    numeric_summary = analysis_df[["Quantity", "UnitPrice", "TotalPrice"]].describe().T
    numeric_summary.to_csv(f"{REPORTS_DIR}/eda_numeric_summary.csv")

    correlation = analysis_df[["Quantity", "UnitPrice", "TotalPrice"]].corr(numeric_only=True)
    correlation.to_csv(f"{REPORTS_DIR}/eda_correlation_matrix.csv")
    cutoff_added = _save_monthly_purchase_summary(analysis_df)

    # Sepet analizleri için Description kalıcı değiştirilmez; geçici temiz alan kullanılır.
    basket_df = df.copy()
    basket_df["Description_clean"] = basket_df["Description"].map(_clean_description)

    customer_baskets = (
        basket_df
        .groupby("CustomerID")["Description_clean"]
        .apply(_basket_products)
        .reset_index(name="products")
    )
    customer_baskets["unique_product_count"] = customer_baskets["products"].apply(len)
    customer_baskets = customer_baskets[["CustomerID", "unique_product_count", "products"]].head(20)
    customer_baskets.to_csv(f"{REPORTS_DIR}/customer_product_baskets_preview.csv", index=False)

    invoice_baskets_all = (
        basket_df
        .groupby("InvoiceNo")["Description_clean"]
        .apply(_basket_products)
        .reset_index(name="products")
    )
    invoice_baskets_all["product_count"] = invoice_baskets_all["products"].apply(len)
    invoice_baskets_all[["InvoiceNo", "product_count", "products"]].head(20).to_csv(
        f"{REPORTS_DIR}/invoice_product_baskets_preview.csv", index=False
    )

    missing = df.isna().sum().sort_values(ascending=False)
    missing_nonzero = missing[missing > 0]
    print("\n  ── Eksik Veri Özeti ──")
    if missing_nonzero.empty:
        print("  Eksik değer bulunamadı.")
    else:
        print(missing_nonzero.to_string())

    print("\n  ── Duplicate Özeti ──")
    print(f"  Duplicate kayıt sayısı: {df.duplicated().sum():,}")

    print("\n  ── Special StockCode Özeti ──")
    print(
        "  Gerçek ürün olmayabilecek kayıt sayısı: "
        f"{special_stockcode_summary['record_count']:,}"
    )
    print(
        "  Farklı özel StockCode sayısı: "
        f"{special_stockcode_summary['stockcode_count']:,}"
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    missing.plot(kind="bar", ax=ax, color="coral")
    ax.set_title("Eksik Değer Sayıları")
    ax.set_xlabel("Sütun")
    ax.set_ylabel("Eksik Kayıt Sayısı")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/eda_missing_values.png")
    plt.close()

    _save_histogram(df, "Quantity", f"{FIGURES_DIR}/eda_quantity_histogram.png", "Quantity Histogram")
    _save_histogram(df, "UnitPrice", f"{FIGURES_DIR}/eda_unitprice_histogram.png", "UnitPrice Histogram")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Quantity, UnitPrice, TotalPrice Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/eda_correlation_matrix.png")
    plt.close()

    print("\n  EDA raporları kaydedildi → outputs/reports/")
    print("  EDA grafikleri kaydedildi → outputs/figures/")
    print("  Aylık satın alma özeti kaydedildi.")
    print("  Aylık satın alma hacmi grafiği kaydedildi.")
    if cutoff_added:
        print("  80/20 cutoff çizgisi grafiğe eklendi.")
    print("  EDA tamamlandı; ham veri üzerinde satır silinmedi.")
    return df


def run_cleaning_with_eda():
    """
    Geriye uyumluluk için bırakılmıştır.
    Artık veri temizliği yapmaz; yalnızca ham veri EDA akışını çalıştırır.
    """
    print("  run_cleaning_with_eda artık veri silmez; run_eda() çağrılıyor.")
    return run_eda()
