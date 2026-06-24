import pandas as pd
import joblib


_PREDICTION_PRINT_LIMIT = 10
_prediction_print_count = 0
_prediction_limit_message_printed = False


def get_customer_segment(customer_id: int) -> int:
    segments = pd.read_csv("data/processed/customer_segments.csv",
                           index_col="CustomerID")
    if customer_id in segments.index:
        return int(segments.loc[customer_id, "segment"])
    return int(segments["segment"].value_counts().idxmax())


def _load_selected_features() -> list[str]:
    """Eğitimde kullanılan seçilmiş feature sırasını dosyadan okur."""
    with open("outputs/reports/selected_features.txt", "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def _safe_divide(numerator: float, denominator: float) -> float:
    """Sıfıra bölme olmadan oran hesaplar."""
    return numerator / denominator if denominator else 0.0


def _build_new_customer_features(num_transactions: int,
                                 total_items: int,
                                 total_spent: float,
                                 recency: int | None = None,
                                 unique_products: int | None = None,
                                 unique_days: int | None = None,
                                 purchase_span_days: int | None = None,
                                 avg_unit_price: float | None = None) -> dict:
    """Yeni müşteri için eğitimdeki feature mantığına uyumlu değerleri üretir."""
    frequency = max(int(num_transactions), 0)
    items = max(int(total_items), 0)
    monetary = float(total_spent)

    if avg_unit_price is None:
        avg_unit_price = _safe_divide(monetary, items)
    if unique_products is None:
        # Ürün çeşitliliği doğrudan verilmediyse alışveriş hacminden temkinli tahmin edilir.
        unique_products = min(items, max(1, frequency * 2)) if items else 0
    if purchase_span_days is None:
        # Siparişler arası gün hesabı için minimum anlamlı alışveriş aralığı varsayılır.
        purchase_span_days = max(frequency - 1, 0)
    if unique_days is None:
        unique_days = min(frequency, purchase_span_days + 1) if frequency else 0
    if recency is None:
        recency = 30

    return {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "total_items": items,
        "avg_basket_size": _safe_divide(items, frequency),
        "avg_unit_price": float(avg_unit_price),
        "unique_products": int(unique_products),
        "unique_days": int(unique_days),
        "avg_basket_value": _safe_divide(monetary, frequency),
        "purchase_span_days": int(purchase_span_days),
        "avg_days_between_orders": _safe_divide(float(purchase_span_days), frequency),
    }


def _print_prediction_example(row: pd.DataFrame, segment: int):
    """Terminalde yalnızca ilk 10 yeni müşteri tahmin detayını gösterir."""
    global _prediction_print_count, _prediction_limit_message_printed

    if _prediction_print_count < _PREDICTION_PRINT_LIMIT:
        print("  Yeni müşteri feature değerleri:")
        print(row.to_string(index=False))
        print(f"  Tahmin edilen segment: {segment}")
        _prediction_print_count += 1
    elif not _prediction_limit_message_printed:
        print("  ... diğer müşteriler gösterilmedi")
        _prediction_limit_message_printed = True


def predict_segment_for_features(num_transactions: int,
                                  total_items: int,
                                  total_spent: float,
                                  recency: int | None = None,
                                  unique_products: int | None = None,
                                  unique_days: int | None = None,
                                  purchase_span_days: int | None = None,
                                  avg_unit_price: float | None = None) -> int:
    """Yeni müşteriyi eğitimdeki scaler + K-Means pipeline'ı ile segmentler."""
    selected_features = _load_selected_features()
    scaler = joblib.load("outputs/models/scaler.pkl")
    kmeans = joblib.load("outputs/models/kmeans.pkl")

    feature_values = _build_new_customer_features(
        num_transactions=num_transactions,
        total_items=total_items,
        total_spent=total_spent,
        recency=recency,
        unique_products=unique_products,
        unique_days=unique_days,
        purchase_span_days=purchase_span_days,
        avg_unit_price=avg_unit_price,
    )
    row = pd.DataFrame([{feature: feature_values.get(feature, 0.0)
                         for feature in selected_features}])

    X = scaler.transform(row[selected_features])
    segment = int(kmeans.predict(X)[0])
    _print_prediction_example(row, segment)
    return segment


def parse_consequents(val) -> list:
    if isinstance(val, (set, frozenset)):
        return list(val)
    val = str(val)
    val = val.replace("frozenset(", "").replace(")", "")
    val = val.replace("{", "").replace("}", "")
    val = val.replace("'", "").replace('"', "")
    return [v.strip() for v in val.split(",") if v.strip()]


def recommend_products(customer_id: int = None,
                       segment_id: int = None,
                       top_n: int = 5,
                       already_bought: list = None,
                       algo_name: str = None) -> pd.DataFrame:
    """
    Hibrit öneri:
    - Segmentin birliktelik kurallarını hybrid_score'a göre sıralar
    - already_bought listesindeki ürünleri dışlar
    - General + Personalized kuralları dengeler
    """
    if segment_id is not None:
        segment = segment_id
    elif customer_id is not None:
        segment = get_customer_segment(customer_id)
    else:
        raise ValueError("customer_id veya segment_id verilmeli.")

    # Algoritma bazlı kural dosyası varsa onu kullan, yoksa genel
    if algo_name:
        path = f"outputs/reports/association_rules_{algo_name.lower()}.csv"
    else:
        path = "outputs/reports/association_rules.csv"

    try:
        rules = pd.read_csv(path)
    except FileNotFoundError:
        rules = pd.read_csv("outputs/reports/association_rules.csv")

    seg_rules = rules[rules["segment"] == segment].copy()

    if seg_rules.empty:
        print(f"  ⚠️  Segment {segment} için birliktelik kuralı bulunamadı.")
        return pd.DataFrame()

    # Consequents'i parse et
    seg_rules["product"] = seg_rules["consequents"].apply(
        lambda x: parse_consequents(x)[0] if parse_consequents(x) else ""
    )
    seg_rules = seg_rules[seg_rules["product"] != ""]

    # Satın alınanları çıkar
    if already_bought:
        bought_upper = [b.strip().upper() for b in already_bought]
        seg_rules = seg_rules[
            ~seg_rules["product"].str.upper().isin(bought_upper)
        ]

    # Hybrid score'a göre sırala (yoksa lift'e göre)
    sort_col = "hybrid_score" if "hybrid_score" in seg_rules.columns else "lift"
    seg_rules = seg_rules.sort_values(sort_col, ascending=False)
    seg_rules = seg_rules.drop_duplicates(subset="product")

    # General ve Personalized dengeleme: ilk yarı General, ikinci yarı Personalized
    if "rule_type" in seg_rules.columns:
        general      = seg_rules[seg_rules["rule_type"] == "General"].head(top_n // 2 + 1)
        personalized = seg_rules[seg_rules["rule_type"] == "Personalized"].head(top_n)
        combined     = pd.concat([general, personalized]).drop_duplicates(
            subset="product").head(top_n)
        if len(combined) < top_n:
            combined = seg_rules.head(top_n)
        seg_rules = combined

    cols = ["product", "confidence", "lift", "segment"]
    if "hybrid_score" in seg_rules.columns:
        cols.append("hybrid_score")
    if "rule_type" in seg_rules.columns:
        cols.append("rule_type")

    recommendations = seg_rules.head(top_n)[cols].reset_index(drop=True)
    recommendations.index += 1
    return recommendations


def _print_recs(recs: pd.DataFrame):
    if recs.empty:
        print("    Öneri bulunamadı.")
    else:
        for i, row in recs.iterrows():
            score = f"  hybrid={row['hybrid_score']:.3f}" \
                if "hybrid_score" in row else ""
            rtype = f"  [{row['rule_type']}]" \
                if "rule_type" in row else ""
            print(f"    {i}. {str(row['product']):<45} "
                  f"lift={row['lift']:.2f}  conf={row['confidence']:.2f}"
                  f"{score}{rtype}")


def run_recommendation_demo():
    print("\n  ===== ÖRNEK ÖNERİLER =====")

    print("\n  [A] Mevcut Müşteriler")
    segments = pd.read_csv("data/processed/customer_segments.csv",
                            index_col="CustomerID")
    sample_ids = segments.index[:3].tolist()

    for cid in sample_ids:
        seg = get_customer_segment(cid)
        print(f"\n  Müşteri {cid}  →  Segment {seg}")
        recs = recommend_products(customer_id=cid, top_n=5)
        _print_recs(recs)

    print("\n  [B] Yeni Müşteri Tahmini")
    new_customers = [
        {"num_transactions": 2,  "total_items": 15,  "total_spent": 45.0,
         "recency": 90, "unique_products": 6, "unique_days": 2,
         "purchase_span_days": 14, "label": "Az Harcayan (Yeni)"},
        {"num_transactions": 20, "total_items": 300, "total_spent": 1800.0,
         "recency": 7, "unique_products": 85, "unique_days": 18,
         "purchase_span_days": 180, "label": "Sık Alışveriş Yapan"},
        {"num_transactions": 5,  "total_items": 80,  "total_spent": 500.0,
         "recency": 35, "unique_products": 25, "unique_days": 5,
         "purchase_span_days": 60, "label": "Orta Segment"},
    ]

    for nc in new_customers:
        seg = predict_segment_for_features(
            num_transactions=nc["num_transactions"],
            total_items=nc["total_items"],
            total_spent=nc["total_spent"],
            recency=nc.get("recency"),
            unique_products=nc.get("unique_products"),
            unique_days=nc.get("unique_days"),
            purchase_span_days=nc.get("purchase_span_days")
        )
        print(f"\n  {nc['label']}  →  Tahmin Edilen Segment: {seg}")
        recs = recommend_products(segment_id=seg, top_n=5)
        _print_recs(recs)

    print("\n  Öneri sistemi testi tamamlandı.")
