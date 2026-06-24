import itertools
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


def _save_histogram(df: pd.DataFrame, column: str, path: str, title: str):
    """Sayısal değişken histogramını kaydeder."""
    fig, ax = plt.subplots(figsize=(9, 5))
    df[column].dropna().plot(kind="hist", bins=50, ax=ax, color="steelblue", edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frekans")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _save_boxplot(df: pd.DataFrame, column: str, path: str, title: str):
    """Sayısal değişken boxplot grafiğini kaydeder."""
    fig, ax = plt.subplots(figsize=(8, 4))
    df[column].dropna().plot(kind="box", ax=ax, vert=False)
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
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

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
        "unique_count": df.nunique(dropna=True).values,
    })
    column_profile.to_csv(f"{REPORTS_DIR}/eda_column_profile.csv", index=False)

    numeric_summary = df[["Quantity", "UnitPrice", "TotalPrice"]].describe().T
    numeric_summary.to_csv(f"{REPORTS_DIR}/eda_numeric_summary.csv")
    print("\n  Quantity describe():")
    print(df["Quantity"].describe().to_string())
    print("\n  UnitPrice describe():")
    print(df["UnitPrice"].describe().to_string())

    correlation = df[["Quantity", "UnitPrice", "TotalPrice"]].corr(numeric_only=True)
    correlation.to_csv(f"{REPORTS_DIR}/eda_correlation_matrix.csv")

    outlier_report = _iqr_report(df, ["Quantity", "UnitPrice"])
    outlier_report.to_csv(f"{REPORTS_DIR}/eda_outlier_report.csv", index=False)

    customer_baskets = (
        df.dropna(subset=["CustomerID", "Description"])
        .groupby("CustomerID")["Description"]
        .apply(lambda x: sorted(set(map(str, x))))
        .head(20)
        .reset_index(name="products")
    )
    customer_baskets.to_csv(f"{REPORTS_DIR}/customer_product_baskets_preview.csv", index=False)

    invoice_baskets_all = (
        df.dropna(subset=["InvoiceNo", "Description"])
        .groupby("InvoiceNo")["Description"]
        .apply(lambda x: sorted(set(map(str, x))))
        .reset_index(name="products")
    )
    invoice_baskets_all.head(20).to_csv(
        f"{REPORTS_DIR}/invoice_product_baskets_preview.csv", index=False
    )

    pair_counts = {}
    for products in invoice_baskets_all["products"]:
        for a, b in itertools.combinations(products, 2):
            pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    top_pairs = pd.DataFrame(
        [{"product_1": a, "product_2": b, "count": count} for (a, b), count in pair_counts.items()]
    ).sort_values("count", ascending=False).head(20) if pair_counts else pd.DataFrame(columns=["product_1", "product_2", "count"])
    top_pairs.to_csv(f"{REPORTS_DIR}/top_product_pairs_preview.csv", index=False)

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
