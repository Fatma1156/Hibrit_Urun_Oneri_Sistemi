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


def _print_prediction_example(row: pd.DataFrame, segment: int, verbose: bool = True):
    """Terminalde yalnızca ilk 10 yeni müşteri tahmin detayını gösterir."""
    if not verbose:
        return

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
                                  avg_unit_price: float | None = None,
                                  verbose: bool = True) -> int:
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
    _print_prediction_example(row, segment, verbose=verbose)
    return segment


def parse_itemset(val) -> list:
    """Association rule itemset alanlarını listeye çevirir."""
    if isinstance(val, (set, frozenset)):
        return [str(item).strip().upper() for item in val if str(item).strip()]
    val = str(val)
    val = val.replace("frozenset(", "").replace(")", "")
    val = val.replace("{", "").replace("}", "")
    val = val.replace("'", "").replace('"', "")
    return [v.strip().upper() for v in val.split(",") if v.strip()]


def parse_consequents(val) -> list:
    """Geriye uyumluluk için consequents parse işlemini korur."""
    return parse_itemset(val)


def _load_customer_train_history(customer_id: int) -> list[str]:
    """Mevcut müşterinin yalnızca TRAIN dönemindeki satın aldığı ürünleri okur."""
    train_path = "data/processed/online_retail_train.csv"
    train_df = pd.read_csv(train_path, usecols=["CustomerID", "Description"])
    customer_products = train_df.loc[
        train_df["CustomerID"].astype(str) == str(customer_id), "Description"
    ]
    return (
        customer_products
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .drop_duplicates()
        .tolist()
    )


def _prepare_recommendation_scores(seg_rules: pd.DataFrame,
                                   already_bought_upper: set[str]) -> pd.DataFrame:
    """Kuralları müşteri geçmişiyle eşleştirip kişiselleştirilmiş skor üretir."""
    seg_rules = seg_rules.copy()
    seg_rules["antecedents_parsed"] = seg_rules["antecedents"].apply(parse_itemset)
    seg_rules["product"] = seg_rules["consequents"].apply(
        lambda x: parse_itemset(x)[0] if parse_itemset(x) else ""
    )
    seg_rules = seg_rules[seg_rules["product"] != ""]

    seg_rules["history_match"] = seg_rules["antecedents_parsed"].apply(
        lambda items: sum(1 for item in items if item in already_bought_upper)
    )
    seg_rules["history_match_ratio"] = seg_rules.apply(
        lambda row: row["history_match"] / len(row["antecedents_parsed"])
        if row["antecedents_parsed"] else 0.0,
        axis=1,
    )

    if "hybrid_score" in seg_rules.columns:
        base_score = seg_rules["hybrid_score"].fillna(0)
    elif "lift" in seg_rules.columns:
        lift = seg_rules["lift"].fillna(0)
        lift_min = lift.min()
        lift_max = lift.max()
        base_score = (lift - lift_min) / (lift_max - lift_min) if lift_max > lift_min else 1.0
    else:
        base_score = 0.0

    seg_rules["personalized_score"] = (
        0.70 * base_score + 0.30 * seg_rules["history_match_ratio"]
    )

    if "rule_type" in seg_rules.columns:
        seg_rules["rule_type_priority"] = seg_rules["rule_type"].map({
            "Personalized": 0,
            "General": 1,
        }).fillna(2)
    else:
        seg_rules["rule_type_priority"] = 2

    return seg_rules


def recommend_products(customer_id: int = None,
                       segment_id: int = None,
                       top_n: int = 5,
                       already_bought: list = None,
                       algo_name: str = None) -> pd.DataFrame:
    """
    Hibrit öneri:
    - Müşteri geçmişindeki antecedent eşleşmelerini önceliklendirir
    - TRAIN döneminde satın alınan ürünleri tekrar önermez
    - Segment kurallarını kişiselleştirilmiş skora göre sıralar
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

    if already_bought is None and customer_id is not None:
        already_bought = _load_customer_train_history(customer_id)

    already_bought_upper = {
        str(product).strip().upper()
        for product in (already_bought or [])
        if str(product).strip()
    }

    seg_rules = _prepare_recommendation_scores(seg_rules, already_bought_upper)

    # TRAIN döneminde satın alınan ürünleri tekrar önerme.
    if already_bought_upper:
        seg_rules = seg_rules[
            ~seg_rules["product"].str.upper().isin(already_bought_upper)
        ]

    if seg_rules.empty:
        return pd.DataFrame()

    # Önce geçmişle eşleşen kurallar, sonra Personalized, en son General fallback.
    seg_rules["has_history_match"] = seg_rules["history_match"] > 0
    seg_rules = seg_rules.sort_values(
        ["has_history_match", "rule_type_priority", "personalized_score", "confidence", "lift"],
        ascending=[False, True, False, False, False],
    )
    seg_rules = seg_rules.drop_duplicates(subset="product")

    optional_cols = ["hybrid_score", "personalized_score", "history_match", "rule_type"]
    cols = ["product", "confidence", "lift", "segment"]
    cols.extend([col for col in optional_cols if col in seg_rules.columns])

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
            purchase_span_days=nc.get("purchase_span_days"),
            verbose=True
        )
        print(f"\n  {nc['label']}  →  Tahmin Edilen Segment: {seg}")
        recs = recommend_products(segment_id=seg, top_n=5)
        _print_recs(recs)

    print("\n  Öneri sistemi testi tamamlandı.")
