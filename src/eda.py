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


def _iqr_report(df: pd.DataFrame, columns: list[str], factor: float = 1.5) -> pd.DataFrame:
    """EDA için satır silmeden IQR aykırı değer özet raporu üretir."""
    rows = []
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append({
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_rate": float(mask.mean()),
        })
    return pd.DataFrame(rows)


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


def _save_boxplot(df: pd.DataFrame, column: str, path: str, title: str):
    """Sayısal değişken boxplot grafiğini kaydeder."""
    fig, ax = plt.subplots(figsize=(8, 4))
    df[column].plot(kind="box", ax=ax, vert=False)
    ax.set_title(title)
    ax.set_xlabel(column)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


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
    _special_stockcodes_report(df)

    # TotalPrice yalnızca EDA içindeki sayısal özet ve korelasyon için geçici analiz sütunudur.
    analysis_df = df.copy()
    analysis_df["TotalPrice"] = analysis_df["Quantity"] * analysis_df["UnitPrice"]

    numeric_summary = analysis_df[["Quantity", "UnitPrice", "TotalPrice"]].describe().T
    numeric_summary.to_csv(f"{REPORTS_DIR}/eda_numeric_summary.csv")
    print("\n  Quantity describe():")
    print(df["Quantity"].describe().to_string())
    print("\n  UnitPrice describe():")
    print(df["UnitPrice"].describe().to_string())

    correlation = analysis_df[["Quantity", "UnitPrice", "TotalPrice"]].corr(numeric_only=True)
    correlation.to_csv(f"{REPORTS_DIR}/eda_correlation_matrix.csv")

    outlier_report = _iqr_report(df, ["Quantity", "UnitPrice"])
    outlier_report.to_csv(f"{REPORTS_DIR}/eda_outlier_report.csv", index=False)

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
    _save_boxplot(df, "Quantity", f"{FIGURES_DIR}/eda_quantity_boxplot.png", "Quantity Boxplot")
    _save_boxplot(df, "UnitPrice", f"{FIGURES_DIR}/eda_unitprice_boxplot.png", "UnitPrice Boxplot")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Quantity, UnitPrice, TotalPrice Korelasyon Matrisi")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/eda_correlation_matrix.png")
    plt.close()

    print("\n  EDA raporları kaydedildi → outputs/reports/")
    print("  EDA grafikleri kaydedildi → outputs/figures/")
    print("  EDA tamamlandı; ham veri üzerinde satır silinmedi.")
    return df


def run_cleaning_with_eda():
    """
    Geriye uyumluluk için bırakılmıştır.
    Artık veri temizliği yapmaz; yalnızca ham veri EDA akışını çalıştırır.
    """
    print("  run_cleaning_with_eda artık veri silmez; run_eda() çağrılıyor.")
    return run_eda()
